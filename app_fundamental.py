import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="IBEX 35 Expert - Fundamental v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #333;
        padding: 12px;
        border-radius: 8px;
    }
    .score-excelente { color: #00ff88; font-size: 2rem; font-weight: bold; }
    .score-bueno { color: #88cc00; font-size: 2rem; font-weight: bold; }
    .score-neutro { color: #ffcc00; font-size: 2rem; font-weight: bold; }
    .score-malo { color: #ff4444; font-size: 2rem; font-weight: bold; }
    .tag-compra { background: #00ff8822; border: 1px solid #00ff88; border-radius: 6px; padding: 4px 12px; color: #00ff88; font-weight: bold; }
    .tag-mantener { background: #ffcc0022; border: 1px solid #ffcc00; border-radius: 6px; padding: 4px 12px; color: #ffcc00; font-weight: bold; }
    .tag-venta { background: #ff444422; border: 1px solid #ff4444; border-radius: 6px; padding: 4px 12px; color: #ff4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📈 IBEX 35 Expert — Análisis Fundamental Avanzado v2.0")

# =============================================================================
# DATOS ESTÁTICOS
# =============================================================================
SECTORES = {
    "Utilities & Energy": ["IBE.MC", "ELE.MC", "REP.MC", "NTGY.MC", "ENG.MC", "RED.MC", "ANE.MC", "SLR.MC"],
    "Bancos & Seguros":   ["BBVA.MC", "SAN.MC", "CABK.MC", "SAB.MC", "BKT.MC", "UNI.MC", "MAP.MC"],
    "Industria & Construcción": ["ACS.MC", "FER.MC", "ANA.MC", "SCYR.MC", "IDR.MC", "MTS.MC", "FDR.MC", "ACX.MC"],
    "Consumo & Retail":   ["ITX.MC", "ROVI.MC", "PUIG.MC", "LOG.MC", "GRF.MC", "IAG.MC", "AMS.MC"],
    "Real Estate & Telco": ["TEF.MC", "MRL.MC", "COL.MC", "CLNX.MC"],
    "Aeropuertos":        ["AENA.MC"]
}

NOMBRES_IBEX = {
    "ACS.MC": "ACS", "ACX.MC": "Acerinox", "AENA.MC": "Aena", "AMS.MC": "Amadeus",
    "ANA.MC": "Acciona", "ANE.MC": "Acciona Energía", "BBVA.MC": "BBVA", "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank", "CLNX.MC": "Cellnex", "COL.MC": "Colonial", "ELE.MC": "Endesa",
    "ENG.MC": "Enagás", "FDR.MC": "Fluidra", "FER.MC": "Ferrovial", "GRF.MC": "Grifols",
    "IAG.MC": "IAG (Iberia)", "IBE.MC": "Iberdrola", "IDR.MC": "Indra", "ITX.MC": "Inditex",
    "LOG.MC": "Logista", "MAP.MC": "Mapfre", "MRL.MC": "Merlin Prop.", "MTS.MC": "ArcelorMittal",
    "NTGY.MC": "Naturgy", "PUIG.MC": "Puig Brands", "RED.MC": "Redeia", "REP.MC": "Repsol",
    "ROVI.MC": "Rovi", "SAB.MC": "Sabadell", "SAN.MC": "Santander", "SCYR.MC": "Sacyr",
    "SLR.MC": "Solaria", "TEF.MC": "Telefónica", "UNI.MC": "Unicaja"
}

# Tasa impositiva aproximada por sector para cálculo de NOPAT
TAX_RATE_SECTOR = {
    "Utilities & Energy": 0.25,
    "Bancos & Seguros": 0.25,
    "Industria & Construcción": 0.23,
    "Consumo & Retail": 0.23,
    "Real Estate & Telco": 0.25,
    "Aeropuertos": 0.25,
    "General": 0.25
}

def get_sector(ticker):
    for sector, tickers in SECTORES.items():
        if ticker in tickers:
            return sector
    return "General"

def safe_get(d, key, default=0):
    """Extrae valor de dict con fallback seguro a número."""
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v

def safe_loc(df, key, col=0, default=None):
    """Extrae valor de DataFrame con fallback seguro."""
    try:
        if key in df.index:
            val = df.loc[key].iloc[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)
    except Exception:
        pass
    return default

# =============================================================================
# 1. F-SCORE DE PIOTROSKI (CORREGIDO — nombre correcto)
# =============================================================================
def calcular_piotroski_f_score(fin, bal, cashflow):
    """
    Calcula el F-Score de Piotroski (0-9).
    Devuelve (score, detalle_dict) para mostrar cada componente en la UI.
    """
    detalle = {}
    score = 0

    # Datos base
    net_income   = safe_loc(fin, 'Net Income', 0)
    net_income_p = safe_loc(fin, 'Net Income', 1)
    total_assets   = safe_loc(bal, 'Total Assets', 0)
    total_assets_p = safe_loc(bal, 'Total Assets', 1)
    revenue   = safe_loc(fin, 'Total Revenue', 0)
    revenue_p = safe_loc(fin, 'Total Revenue', 1)
    gross_profit   = safe_loc(fin, 'Gross Profit', 0)
    gross_profit_p = safe_loc(fin, 'Gross Profit', 1)

    cfo = safe_loc(cashflow, 'Operating Cash Flow', 0, default=0)
    lt_debt   = safe_loc(bal, 'Long Term Debt', 0, default=0)
    lt_debt_p = safe_loc(bal, 'Long Term Debt', 1, default=0)

    curr_assets   = safe_loc(bal, 'Current Assets', 0)
    curr_liab     = safe_loc(bal, 'Current Liabilities', 0)
    curr_assets_p = safe_loc(bal, 'Current Assets', 1)
    curr_liab_p   = safe_loc(bal, 'Current Liabilities', 1)

    shares   = safe_loc(bal, 'Share Issued', 0, default=0)
    shares_p = safe_loc(bal, 'Share Issued', 1, default=0)

    # --- RENTABILIDAD ---
    # 1. ROA > 0
    roa   = (net_income / total_assets) if total_assets and total_assets != 0 else None
    roa_p = (net_income_p / total_assets_p) if total_assets_p and total_assets_p != 0 else None
    p1 = bool(roa is not None and roa > 0)
    if p1: score += 1
    detalle['1. ROA > 0'] = ('✅', f'{roa*100:.2f}%') if p1 else ('❌', f'{roa*100:.2f}%' if roa is not None else 'N/D')

    # 2. CFO > 0
    p2 = bool(cfo is not None and cfo > 0)
    if p2: score += 1
    detalle['2. CFO > 0'] = ('✅', f'{cfo/1e6:.0f}M€') if p2 else ('❌', f'{cfo/1e6:.0f}M€' if cfo else 'N/D')

    # 3. ROA creciente
    p3 = bool(roa is not None and roa_p is not None and roa > roa_p)
    if p3: score += 1
    detalle['3. ROA Creciente'] = ('✅', f'{roa*100:.2f}% vs {roa_p*100:.2f}% año ant.') if p3 else ('❌', f'{roa*100:.2f}% vs {roa_p*100:.2f}% año ant.' if roa is not None and roa_p is not None else 'N/D')

    # 4. CFO > Net Income (Calidad de beneficios — accrual)
    p4 = bool(net_income is not None and cfo is not None and cfo > net_income)
    if p4: score += 1
    detalle['4. CFO > Beneficio Neto'] = ('✅', 'Beneficios respaldados por caja') if p4 else ('❌', 'Beneficios contables > caja operativa')

    # --- APALANCAMIENTO / LIQUIDEZ ---
    # 5. Deuda LP decreciente
    p5 = bool(lt_debt is not None and lt_debt_p is not None and lt_debt <= lt_debt_p)
    if p5: score += 1
    detalle['5. Deuda LP Decreciente'] = ('✅', f'{lt_debt/1e6:.0f}M€ vs {lt_debt_p/1e6:.0f}M€') if p5 else ('❌', f'{lt_debt/1e6:.0f}M€ vs {lt_debt_p/1e6:.0f}M€' if lt_debt is not None else 'N/D')

    # 6. Current Ratio creciente
    cr   = (curr_assets / curr_liab) if curr_liab and curr_liab != 0 else None
    cr_p = (curr_assets_p / curr_liab_p) if curr_liab_p and curr_liab_p != 0 else None
    p6 = bool(cr is not None and cr_p is not None and cr > cr_p)
    if p6: score += 1
    detalle['6. Liquidez Creciente (CR)'] = ('✅', f'{cr:.2f}x vs {cr_p:.2f}x') if p6 else ('❌', f'{cr:.2f}x vs {cr_p:.2f}x' if cr is not None and cr_p is not None else 'N/D')

    # 7. Sin dilución
    p7 = bool(shares is not None and shares_p is not None and shares_p > 0 and shares <= shares_p)
    if p7: score += 1
    detalle['7. Sin Dilución Accionistas'] = ('✅', f'{shares/1e6:.0f}M acc.') if p7 else ('❌', f'{shares/1e6:.0f}M vs {shares_p/1e6:.0f}M acc.' if shares and shares_p else 'N/D')

    # --- EFICIENCIA OPERATIVA ---
    # 8. Margen Bruto creciente
    gm   = (gross_profit / revenue) if revenue and revenue != 0 else None
    gm_p = (gross_profit_p / revenue_p) if revenue_p and revenue_p != 0 else None
    p8 = bool(gm is not None and gm_p is not None and gm > gm_p)
    if p8: score += 1
    detalle['8. Margen Bruto Creciente'] = ('✅', f'{gm*100:.1f}% vs {gm_p*100:.1f}%') if p8 else ('❌', f'{gm*100:.1f}% vs {gm_p*100:.1f}%' if gm is not None and gm_p is not None else 'N/D')

    # 9. Rotación de Activos creciente
    at   = (revenue / total_assets) if total_assets and total_assets != 0 else None
    at_p = (revenue_p / total_assets_p) if total_assets_p and total_assets_p != 0 else None
    p9 = bool(at is not None and at_p is not None and at > at_p)
    if p9: score += 1
    detalle['9. Rotación Activos Creciente'] = ('✅', f'{at:.3f}x vs {at_p:.3f}x') if p9 else ('❌', f'{at:.3f}x vs {at_p:.3f}x' if at is not None and at_p is not None else 'N/D')

    return score, detalle


# =============================================================================
# 2. GRAHAM NUMBER
# =============================================================================
def calcular_graham_number(eps, bvps):
    """
    Número de Graham: √(22.5 × EPS × BookValuePerShare)
    Solo válido con EPS y BVPS positivos. Representa el precio máximo
    defensivo que Graham pagaría. No aplica a bancos ni empresas con
    beneficios negativos.
    """
    if eps is not None and bvps is not None and eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return None


# =============================================================================
# 3. ROIC CORRECTO (NOPAT / Capital Invertido)
# =============================================================================
def calcular_roic_nopat(fin, bal, sector):
    """
    ROIC = NOPAT / Capital Invertido
    NOPAT = EBIT × (1 - Tasa Impositiva)  [No Net Income, que incluye apalancamiento]
    Capital Invertido = Equity + Deuda Total - Caja Excedente
    
    Diferencia vs v1: v1 usaba EBIT sin impuestos y sin restar caja, sobreestimando ROIC.
    """
    tax_rate = TAX_RATE_SECTOR.get(sector, 0.25)

    ebit = safe_loc(fin, 'EBIT', 0)
    if ebit is None:
        ebit = safe_loc(fin, 'Operating Income', 0)
    if ebit is None:
        return None

    nopat = ebit * (1 - tax_rate)

    equity = safe_loc(bal, 'Stockholders Equity', 0, default=0)
    debt   = safe_loc(bal, 'Total Debt', 0, default=0)
    cash   = safe_loc(bal, 'Cash And Cash Equivalents', 0, default=0)

    # Capital Invertido: se resta la caja excedente (no es capital "trabajando")
    invested_capital = equity + debt - cash
    if invested_capital is None or invested_capital <= 0:
        return None

    return (nopat / invested_capital) * 100


# =============================================================================
# 4. MOMENTUM TÉCNICO (sin cambios, ya era correcto)
# =============================================================================
def analizar_momentum(hist, score_max=20):
    """
    3 factores de momentum técnico con pesos definidos:
    - SMA200 (50%), SMA50 (25%), Proximidad 52w máx (25%)
    Recibe el DataFrame ya descargado para evitar llamadas redundantes a la API.
    """
    reasons = []
    score = 0.0

    if hist is None or hist.empty or len(hist) < 200:
        return 0, ["⚠️ Histórico insuficiente para calcular momentum"]

    current_price = hist['Close'].iloc[-1]

    # 1. SMA 200
    sma200 = hist['Close'].rolling(200).mean().iloc[-1]
    w = score_max * 0.5
    if current_price > sma200:
        score += w
        dist = (current_price / sma200 - 1) * 100
        reasons.append(f"✅ Tendencia Alcista LP (Precio > SMA200, +{dist:.1f}%) (+{w:g}p)")
    else:
        dist = (current_price / sma200 - 1) * 100
        reasons.append(f"⚠️ Tendencia Bajista LP (Precio < SMA200, {dist:.1f}%)")

    # 2. SMA 50
    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
    w = score_max * 0.25
    if current_price > sma50:
        score += w
        reasons.append(f"✅ Fortaleza Medio Plazo (Precio > SMA50) (+{w:g}p)")
    else:
        reasons.append("⚠️ Precio por debajo de SMA50")

    # 3. Proximidad a máximos de 52 semanas
    high_52w = hist['Close'].max()
    drawdown = (current_price / high_52w - 1) * 100
    w = score_max * 0.25
    if drawdown > -10:
        score += w
        reasons.append(f"✅ Cerca de Máximos Anuales ({drawdown:.1f}%) (+{w:g}p)")
    elif drawdown > -20:
        reasons.append(f"⚪ A {drawdown:.1f}% de máximos anuales")
    else:
        reasons.append(f"⚠️ Lejos de máximos anuales ({drawdown:.1f}%)")

    return round(score, 1), reasons


# =============================================================================
# 5. SCORING GENERAL (NO BANCOS) — MAX 100 puntos
# =============================================================================
# Distribución real de puntos (suma exacta = 100):
# Rentabilidad:    30 pts  (ROIC 15 + Margen 15)
# Salud:           30 pts  (Piotroski 15 + Deuda 15)
# Valoración:      20 pts  (PEG 10 + FCF Yield 10)
# Crecimiento:     10 pts
# Momentum:        10 pts
# TOTAL:          100 pts

def analizar_general(ticker, info, fin, bal, cashflow, hist):
    sector = get_sector(ticker)
    score = 0.0
    razones = []

    # ── RENTABILIDAD (30 pts) ──────────────────────────────────────────────
    # ROIC con NOPAT (corrección principal vs v1)
    roic = calcular_roic_nopat(fin, bal, sector)

    # Fallback: si no hay EBIT, usar ROE (menos preciso, se indica)
    if roic is None:
        roe = safe_get(info, 'returnOnEquity')
        roic = roe * 100 if roe else None
        roic_label = "ROE (proxy)" if roic else None
    else:
        roic_label = "ROIC (NOPAT)"

    umbral_exc = 15 if sector not in ("Utilities & Energy",) else 10
    umbral_sol = umbral_exc - 5

    if roic is not None:
        if roic > umbral_exc:
            score += 15
            razones.append(f"✅ Rentabilidad Excelente ({roic_label}: {roic:.1f}% > {umbral_exc}%) (+15p)")
        elif roic > umbral_sol:
            score += 8
            razones.append(f"🟡 Rentabilidad Sólida ({roic_label}: {roic:.1f}%) (+8p)")
        else:
            razones.append(f"⚪ Rentabilidad Baja ({roic_label}: {roic:.1f}%)")
    else:
        razones.append("⚠️ No disponible: ROIC/ROE")

    # Expansión de Margen Neto
    ni_0 = safe_loc(fin, 'Net Income', 0)
    ni_1 = safe_loc(fin, 'Net Income', 1)
    rev_0 = safe_loc(fin, 'Total Revenue', 0)
    rev_1 = safe_loc(fin, 'Total Revenue', 1)

    if ni_0 and rev_0 and ni_1 and rev_1 and rev_0 != 0 and rev_1 != 0:
        nm_0 = ni_0 / rev_0
        nm_1 = ni_1 / rev_1
        if nm_0 > nm_1:
            score += 15
            razones.append(f"✅ Expansión de Margen Neto ({nm_1*100:.1f}% → {nm_0*100:.1f}%) (+15p)")
        else:
            razones.append(f"⚠️ Contracción de Margen Neto ({nm_1*100:.1f}% → {nm_0*100:.1f}%)")
    else:
        razones.append("⚠️ No disponible: datos de margen")

    # ── SALUD FINANCIERA (30 pts) ─────────────────────────────────────────
    f_score, f_detalle = calcular_piotroski_f_score(fin, bal, cashflow)
    if f_score >= 7:
        score += 15
        razones.append(f"✅ Calidad Financiera Impecable (Piotroski F-Score: {f_score}/9) (+15p)")
    elif f_score >= 5:
        score += 8
        razones.append(f"🟡 Calidad Financiera Aceptable (Piotroski F-Score: {f_score}/9) (+8p)")
    else:
        score -= 5
        razones.append(f"❌ Deterioro Financiero (Piotroski F-Score: {f_score}/9) (-5p)")

    # Deuda Neta / EBITDA con tolerancias sectoriales
    total_debt = safe_loc(bal, 'Total Debt', 0, default=0)
    cash_eq    = safe_loc(bal, 'Cash And Cash Equivalents', 0, default=0)
    ebitda     = safe_loc(fin, 'EBITDA', 0)

    if ebitda and ebitda > 0:
        net_debt_ebitda = (total_debt - cash_eq) / ebitda
        tolerancia = {"Utilities & Energy": 4.5, "Real Estate & Telco": 5.5}.get(sector, 2.5)
        if net_debt_ebitda < tolerancia:
            score += 15
            razones.append(f"✅ Deuda Controlada ({net_debt_ebitda:.1f}x ≤ {tolerancia}x tolerancia sectorial) (+15p)")
        elif net_debt_ebitda < tolerancia * 1.3:
            score += 5
            razones.append(f"🟡 Deuda Elevada pero Manejable ({net_debt_ebitda:.1f}x) (+5p)")
        else:
            razones.append(f"❌ Deuda Excesiva para Sector ({net_debt_ebitda:.1f}x, límite {tolerancia}x)")
    else:
        razones.append("⚠️ No disponible: EBITDA para ratio deuda")

    # ── VALORACIÓN (20 pts) ───────────────────────────────────────────────
    # PEG: P/E / Crecimiento esperado de BPA
    pe = safe_get(info, 'trailingPE')
    # earningsGrowth viene en decimal (0.15 = 15%)
    eg_raw = safe_get(info, 'earningsGrowth', default=None)
    if eg_raw is not None and eg_raw > 0 and pe > 0:
        eg_pct = eg_raw * 100  # convertir a porcentaje para el ratio PEG estándar
        peg = pe / eg_pct
        if peg < 1.0:
            score += 10
            razones.append(f"✅ Barata vs Crecimiento (PEG {peg:.2f} < 1.0) (+10p)")
        elif peg < 1.5:
            score += 5
            razones.append(f"🟡 Valoración Justa (PEG {peg:.2f}) (+5p)")
        else:
            razones.append(f"⚪ Cara vs Crecimiento (PEG {peg:.2f})")
    elif pe > 0:
        razones.append(f"⚪ PEG incalculable (crecimiento no disponible, P/E: {pe:.1f}x)")
    else:
        razones.append("⚪ P/E negativo o no disponible")

    # FCF Yield = FCF / Market Cap
    mkt_cap = safe_get(info, 'marketCap', default=1)
    fcf = safe_loc(cashflow, 'Free Cash Flow', 0)
    if fcf is not None and mkt_cap > 0:
        fcf_yield = (fcf / mkt_cap) * 100
        if fcf_yield > 6:
            score += 10
            razones.append(f"✅ FCF Yield Muy Atractivo ({fcf_yield:.1f}%) (+10p)")
        elif fcf_yield > 4:
            score += 5
            razones.append(f"🟡 FCF Yield Sólido ({fcf_yield:.1f}%) (+5p)")
        elif fcf_yield > 0:
            razones.append(f"⚪ FCF Yield Bajo ({fcf_yield:.1f}%)")
        else:
            razones.append(f"⚠️ FCF Yield Negativo ({fcf_yield:.1f}%) — quemar caja")
    else:
        razones.append("⚪ No disponible: FCF Yield")

    # ── CRECIMIENTO (10 pts) ──────────────────────────────────────────────
    rev_growth = safe_get(info, 'revenueGrowth', default=None)
    if rev_growth is not None:
        if rev_growth > 0.10:
            score += 10
            razones.append(f"✅ Crecimiento Ventas Doble Dígito ({rev_growth*100:.1f}%) (+10p)")
        elif rev_growth > 0:
            score += 5
            razones.append(f"🟡 Crecimiento Ventas Positivo ({rev_growth*100:.1f}%) (+5p)")
        else:
            razones.append(f"⚠️ Ventas Decrecientes ({rev_growth*100:.1f}%)")
    else:
        razones.append("⚪ No disponible: crecimiento de ventas")

    # ── MOMENTUM TÉCNICO (10 pts) ─────────────────────────────────────────
    score_mom, razones_mom = analizar_momentum(hist, score_max=10)
    score += score_mom
    razones.extend(razones_mom)

    return round(min(score, 100)), razones, f_score, f_detalle


# =============================================================================
# 6. SCORING BANCOS (MAX 100 puntos — completamente reescrito)
# =============================================================================
# Distribución real:
# ROE:                    25 pts
# Valoración P/B vs ROE:  25 pts
# Dividendo sostenible:   15 pts
# Ratio de Eficiencia:    15 pts  ← NUEVO (era 0 en v1)
# Solidez capital (CET1): 10 pts  ← NUEVO
# Momentum:               10 pts
# TOTAL:                 100 pts

def analizar_banco(ticker, info, fin, bal, hist):
    score = 0.0
    razones = []

    roe = safe_get(info, 'returnOnEquity', default=None)
    pb  = safe_get(info, 'priceToBook', default=None)
    div_yield_raw = safe_get(info, 'dividendYield', default=None)
    payout_raw    = safe_get(info, 'payoutRatio', default=None)
    mkt_cap = safe_get(info, 'marketCap', default=1)

    # ── ROE (25 pts) ──────────────────────────────────────────────────────
    if roe is not None:
        if roe > 0.12:
            score += 25
            razones.append(f"✅ ROE Líder de Sector ({roe*100:.1f}% > 12%) (+25p)")
        elif roe > 0.08:
            score += 15
            razones.append(f"🟡 ROE Sano ({roe*100:.1f}%) (+15p)")
        elif roe > 0:
            score += 5
            razones.append(f"⚪ ROE Positivo pero Bajo ({roe*100:.1f}%) (+5p)")
        else:
            razones.append(f"❌ ROE Negativo ({roe*100:.1f}%)")
    else:
        razones.append("⚠️ No disponible: ROE")

    # ── VALORACIÓN P/B vs ROE (25 pts) ───────────────────────────────────
    # Un banco con ROE > 10% cotizando a P/B < 0.8 es una clara oportunidad de valor
    if pb is not None and roe is not None:
        if roe > 0.10 and pb < 0.8:
            score += 25
            razones.append(f"✅ Ganga Fundamental: Alto ROE ({roe*100:.1f}%) + P/B Bajo ({pb:.2f}x) (+25p)")
        elif pb < 1.0:
            score += 15
            razones.append(f"🟡 Cotiza Bajo Valor en Libros (P/B {pb:.2f}x) (+15p)")
        elif pb < 1.5:
            score += 8
            razones.append(f"⚪ Valoración Razonable (P/B {pb:.2f}x) (+8p)")
        else:
            razones.append(f"⚠️ Cotiza con Prima Elevada (P/B {pb:.2f}x)")
    else:
        razones.append("⚠️ No disponible: P/B Ratio")

    # ── DIVIDENDO SOSTENIBLE (15 pts) ────────────────────────────────────
    # CORRECCIÓN v2: No puntuar dividendo alto si el payout es insostenible
    if div_yield_raw is not None and div_yield_raw > 0:
        # yfinance devuelve dividendYield en decimal (0.05 = 5%)
        dy_pct = div_yield_raw * 100 if div_yield_raw < 1 else div_yield_raw

        # Verificar payout ratio para sostenibilidad
        payout_ok = True
        payout_msg = ""
        if payout_raw is not None and payout_raw > 0:
            if payout_raw > 1.0:  # Payout > 100% = insostenible
                payout_ok = False
                payout_msg = f" ⚠️ Payout {payout_raw*100:.0f}% — INSOSTENIBLE"
            elif payout_raw > 0.80:
                payout_msg = f" (Payout {payout_raw*100:.0f}% — elevado)"

        if dy_pct > 5 and payout_ok:
            score += 15
            razones.append(f"✅ Dividendo Potente y Sostenible ({dy_pct:.1f}%{payout_msg}) (+15p)")
        elif dy_pct > 3 and payout_ok:
            score += 8
            razones.append(f"🟡 Dividendo Atractivo ({dy_pct:.1f}%{payout_msg}) (+8p)")
        elif not payout_ok:
            razones.append(f"❌ Dividendo Alto pero Insostenible ({dy_pct:.1f}%{payout_msg})")
        else:
            razones.append(f"⚪ Dividendo Bajo ({dy_pct:.1f}%)")
    else:
        razones.append("⚪ Sin dividendo o no disponible")

    # ── RATIO DE EFICIENCIA (15 pts) ────────────────────────────────────
    # Cost-to-Income = Gastos Operativos / Ingresos Operativos
    # < 50%: excelente; 50-60%: aceptable; > 65%: problemático
    op_expense = safe_loc(fin, 'Operating Expense', 0)
    op_income  = safe_loc(fin, 'Operating Income', 0)
    total_rev  = safe_loc(fin, 'Total Revenue', 0)

    cost_to_income = None
    if op_expense and total_rev and total_rev != 0:
        cost_to_income = abs(op_expense) / total_rev
    elif op_income and total_rev and total_rev != 0:
        # Alternativa: 1 - margen operativo
        cost_to_income = 1 - (op_income / total_rev)

    if cost_to_income is not None:
        cti_pct = cost_to_income * 100
        if cti_pct < 50:
            score += 15
            razones.append(f"✅ Ratio Eficiencia Excelente (Cost-to-Income: {cti_pct:.1f}% < 50%) (+15p)")
        elif cti_pct < 60:
            score += 8
            razones.append(f"🟡 Ratio Eficiencia Aceptable (Cost-to-Income: {cti_pct:.1f}%) (+8p)")
        else:
            razones.append(f"❌ Banco Ineficiente (Cost-to-Income: {cti_pct:.1f}% > 60%)")
    else:
        razones.append("⚪ No disponible: Ratio de Eficiencia (Cost-to-Income)")

    # ── SOLIDEZ DE CAPITAL (10 pts) ──────────────────────────────────────
    # yfinance no expone CET1 directamente. Usamos Equity/Total Assets como proxy
    # (Tier 1 Capital Ratio simplificado)
    equity = safe_loc(bal, 'Stockholders Equity', 0)
    assets = safe_loc(bal, 'Total Assets', 0)
    if equity and assets and assets > 0:
        cap_ratio = (equity / assets) * 100
        if cap_ratio > 8:
            score += 10
            razones.append(f"✅ Base de Capital Sólida (Equity/Assets: {cap_ratio:.1f}%) (+10p)")
        elif cap_ratio > 5:
            score += 5
            razones.append(f"🟡 Capital Adecuado (Equity/Assets: {cap_ratio:.1f}%) (+5p)")
        else:
            razones.append(f"⚠️ Capital Escaso (Equity/Assets: {cap_ratio:.1f}%)")
    else:
        razones.append("⚪ No disponible: Ratio de Capital")

    # ── MOMENTUM TÉCNICO (10 pts) ─────────────────────────────────────────
    score_mom, razones_mom = analizar_momentum(hist, score_max=10)
    score += score_mom
    razones.extend(razones_mom)

    return round(min(score, 100)), razones, 0, {}  # bancos no usan F-Score


# =============================================================================
# 7. FUNCIÓN DE CARGA DE DATOS
# =============================================================================

@st.cache_data(ttl=3600)  # CORRECCIÓN: 1h, no 60s — los fundamentales son trimestrales
def cargar_datos_fundamentales(tickers_list):
    data = []
    detalles = {}
    f_scores_detalle = {}
    errores = []

    for ticker in tickers_list:
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info

            # Datos financieros — preferir anuales para consistencia con Piotroski
            fin      = stock.income_stmt      # Cuenta de resultados anual
            bal      = stock.balance_sheet    # Balance anual
            cashflow = stock.cashflow         # Flujo de caja anual

            if fin is None or fin.empty:
                fin = stock.financials  # Fallback

            # Histórico para momentum (descargamos aquí para reutilizarlo)
            hist = yf.Ticker(ticker).history(period="1y")

            sector = get_sector(ticker)

            if sector == "Bancos & Seguros":
                score, razones, f_sc, f_det = analizar_banco(ticker, info, fin, bal, hist)
            else:
                score, razones, f_sc, f_det = analizar_general(ticker, info, fin, bal, cashflow, hist)

            # Recomendación basada en score normalizado a 100
            if score >= 75:
                rec = "🟢 COMPRA FUERTE"
            elif score >= 58:
                rec = "🟡 COMPRA"
            elif score >= 42:
                rec = "⚪ MANTENER"
            else:
                rec = "🔴 VENTA"

            # Graham Number
            eps  = safe_get(info, 'trailingEps', default=None)
            bvps = safe_get(info, 'bookValue', default=None)
            graham = calcular_graham_number(eps, bvps)

            current_price = safe_get(info, 'currentPrice') or safe_get(info, 'regularMarketPrice')
            upside_graham = None
            if graham and current_price and current_price > 0:
                upside_graham = (graham / current_price - 1) * 100

            # Dividendo — fix de escala (decimal vs porcentaje)
            dy_raw = safe_get(info, 'dividendYield', default=None)
            if dy_raw is not None:
                dy_str = f"{dy_raw*100:.2f}%" if dy_raw < 1 else f"{dy_raw:.2f}%"
            else:
                dy_str = "—"

            # PEG para visualización
            pe     = safe_get(info, 'trailingPE')
            eg     = safe_get(info, 'earningsGrowth', default=None)
            peg_display = f"{pe / (eg * 100):.2f}" if pe and eg and eg > 0 else "N/D"

            data.append({
                "Ticker":      ticker.replace(".MC", ""),
                "Empresa":     NOMBRES_IBEX.get(ticker, ticker),
                "Sector":      sector,
                "Score /100":  score,
                "Rec.":        rec,
                "Precio":      f"{current_price:.2f}€" if current_price else "—",
                "Graham №":   f"{graham:.2f}€" if graham else "N/A",
                "Upside Graham": f"{upside_graham:.1f}%" if upside_graham is not None else "N/A",
                "Piotroski":  f"{f_sc}/9" if sector != "Bancos & Seguros" else "N/A",
                "PEG":        peg_display,
                "Div Yield":  dy_str,
                "_score_raw": score
            })

            detalles[ticker.replace(".MC", "")] = razones
            f_scores_detalle[ticker.replace(".MC", "")] = f_det

        except Exception as e:
            errores.append(f"{ticker}: {str(e)[:60]}")
            continue

    df = pd.DataFrame(data).sort_values("_score_raw", ascending=False) if data else pd.DataFrame()
    return df, detalles, f_scores_detalle, errores


# =============================================================================
# 8. GRÁFICO RADAR REAL (basado en dimensiones calculadas, no aleatorio)
# =============================================================================
def construir_radar(ticker_full, info, fin, bal, cashflow, hist, sector):
    """
    Radar de 5 dimensiones con valores REALES (0-10), no aleatorios.
    Dimensiones: Calidad, Valor, Crecimiento, Momentum, Seguridad Financiera
    """
    dimensiones = ['Calidad\n(ROIC/ROE)', 'Valor\n(PEG+FCF)', 'Crecimiento\n(Ventas)', 'Momentum\n(Técnico)', 'Seguridad\n(F-Score+Deuda)']
    valores = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 1. Calidad
    if sector == "Bancos & Seguros":
        roe = safe_get(info, 'returnOnEquity', default=0)
        valores[0] = min(10, (roe * 100) / 2)  # 20% ROE = 10/10
    else:
        roic = calcular_roic_nopat(fin, bal, sector)
        if roic is not None:
            umbral = 10 if sector == "Utilities & Energy" else 15
            valores[0] = min(10, (roic / umbral) * 10)

    # 2. Valor
    pe = safe_get(info, 'trailingPE')
    eg = safe_get(info, 'earningsGrowth', default=None)
    mkt_cap = safe_get(info, 'marketCap', default=1)
    fcf = safe_loc(cashflow, 'Free Cash Flow', 0) if cashflow is not None else None
    peg_score = 0
    if pe and eg and eg > 0:
        peg = pe / (eg * 100)
        peg_score = max(0, min(5, (2 - peg) * 5))  # PEG 0=5pts, PEG 1=5pts, PEG 2=0pts
    fcf_score = 0
    if fcf and mkt_cap > 0:
        fy = (fcf / mkt_cap) * 100
        fcf_score = min(5, fy / 2)  # 10% FCF yield = 5pts
    valores[1] = peg_score + fcf_score

    # 3. Crecimiento
    rg = safe_get(info, 'revenueGrowth', default=None)
    if rg is not None:
        valores[2] = min(10, max(0, rg * 50))  # 20% crecimiento = 10/10

    # 4. Momentum
    mom_score, _ = analizar_momentum(hist, score_max=10)
    valores[3] = mom_score

    # 5. Seguridad
    if sector != "Bancos & Seguros":
        f_sc, _ = calcular_piotroski_f_score(fin, bal, cashflow)
        fs_pts = (f_sc / 9) * 5  # 0-5
        ebitda = safe_loc(fin, 'EBITDA', 0)
        total_debt = safe_loc(bal, 'Total Debt', 0, default=0)
        cash_eq = safe_loc(bal, 'Cash And Cash Equivalents', 0, default=0)
        tol = {"Utilities & Energy": 4.5, "Real Estate & Telco": 5.5}.get(sector, 2.5)
        if ebitda and ebitda > 0:
            nd_eb = max(0, (total_debt - cash_eq) / ebitda)
            debt_pts = min(5, max(0, 5 - (nd_eb / tol) * 5))
        else:
            debt_pts = 2.5
        valores[4] = fs_pts + debt_pts
    else:
        equity = safe_loc(bal, 'Stockholders Equity', 0)
        assets = safe_loc(bal, 'Total Assets', 0)
        cap_r = (equity / assets * 100) if equity and assets and assets > 0 else 0
        valores[4] = min(10, cap_r * 0.8)

    valores = [round(max(0, min(10, v)), 1) for v in valores]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores + [valores[0]],
        theta=dimensiones + [dimensiones[0]],
        fill='toself',
        fillcolor='rgba(0, 200, 120, 0.2)',
        line=dict(color='#00cc78', width=2),
        name=ticker_full.replace(".MC", "")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickvals=[2, 4, 6, 8, 10]),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=30, b=30, l=60, r=60),
        height=350
    )
    return fig, dict(zip(dimensiones, valores))


# =============================================================================
# 9. INTERFAZ PRINCIPAL
# =============================================================================

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.info("""
    **v2.0 — Mejoras clave:**
    - ROIC con NOPAT (correcto)
    - Dividendo con payout sostenibilidad
    - Ratio Eficiencia bancos real
    - Radar Chart con datos reales
    - Piotroski (nombre correcto)
    - TTL de caché corregido (1h)
    - Score normalizado a 100
    """)

    sectores_filtro = st.multiselect(
        "Filtrar por Sector",
        options=list(SECTORES.keys()),
        default=list(SECTORES.keys())
    )
    min_score_filtro = st.slider("Score mínimo a mostrar", 0, 100, 40, 5)
    mostrar_errores = st.toggle("Mostrar errores de carga", value=False)

# Recopilar tickers según filtro de sector
tickers_filtrados = []
for sec in sectores_filtro:
    tickers_filtrados.extend(SECTORES.get(sec, []))
tickers_filtrados = list(set(tickers_filtrados))

# Tabs
tab_ranking, tab_detalle, tab_metodologia = st.tabs([
    "🏆 Ranking & Oportunidades",
    "🔬 Análisis Detallado",
    "📘 Metodología v2.0"
])

# ---------------------------------------------------------------------------
# TAB 1 — RANKING
# ---------------------------------------------------------------------------
with tab_ranking:
    st.markdown("### 📊 Análisis Fundamental IBEX 35")
    st.caption("Los datos fundamentales (balance, P&L, flujos) son anuales. El momentum usa 252 sesiones diarias.")

    if st.button("🔍 Analizar IBEX 35", type="primary"):
        with st.spinner("Descargando y procesando datos fundamentales..."):
            df, detalles, f_det_all, errores = cargar_datos_fundamentales(tickers_filtrados)
            st.session_state['fund_df'] = df
            st.session_state['fund_detalles'] = detalles
            st.session_state['fund_f_det'] = f_det_all
            st.session_state['fund_errores'] = errores

    if mostrar_errores and 'fund_errores' in st.session_state and st.session_state['fund_errores']:
        with st.expander(f"⚠️ {len(st.session_state['fund_errores'])} errores de carga"):
            for e in st.session_state['fund_errores']:
                st.caption(e)

    if 'fund_df' in st.session_state and not st.session_state['fund_df'].empty:
        df_view = st.session_state['fund_df']
        df_view = df_view[df_view['_score_raw'] >= min_score_filtro]

        cols_tabla = ['Ticker', 'Empresa', 'Sector', 'Score /100', 'Rec.', 'Precio',
                      'Graham №', 'Upside Graham', 'Piotroski', 'PEG', 'Div Yield']

        def color_rec(val):
            if '🟢' in str(val): return 'color: #00ff88; font-weight: bold'
            if '🟡' in str(val): return 'color: #ffcc00'
            if '🔴' in str(val): return 'color: #ff4444'
            return ''

        st.dataframe(
            df_view[cols_tabla].style
            .background_gradient(subset=['Score /100'], cmap='RdYlGn', vmin=20, vmax=100)
            .applymap(color_rec, subset=['Rec.']),
            use_container_width=True,
            height=600
        )

        # Gráfico de barras score por empresa
        fig_bar = go.Figure()
        colors = ['#00ff88' if s >= 75 else '#ffcc00' if s >= 55 else '#ff6655'
                  for s in df_view['_score_raw']]
        fig_bar.add_trace(go.Bar(
            x=df_view['Empresa'],
            y=df_view['_score_raw'],
            marker_color=colors,
            text=df_view['_score_raw'],
            textposition='outside'
        ))
        fig_bar.add_hline(y=75, line=dict(color='#00ff88', dash='dot'), annotation_text="Compra Fuerte")
        fig_bar.add_hline(y=58, line=dict(color='#ffcc00', dash='dot'), annotation_text="Compra")
        fig_bar.add_hline(y=42, line=dict(color='#ff6655', dash='dot'), annotation_text="Mantener")
        fig_bar.update_layout(
            template="plotly_dark", height=400,
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, 110]),
            margin=dict(t=20, b=100)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — DETALLE
# ---------------------------------------------------------------------------
with tab_detalle:
    st.markdown("### 🔬 Análisis Detallado por Empresa")

    if 'fund_df' not in st.session_state or st.session_state['fund_df'].empty:
        st.info("Ejecuta primero el análisis en la pestaña Ranking.")
    else:
        df_view = st.session_state['fund_df']
        detalles = st.session_state['fund_detalles']
        f_det_all = st.session_state['fund_f_det']

        sel = st.selectbox(
            "Selecciona empresa:",
            df_view.sort_values('_score_raw', ascending=False)['Ticker'].tolist()
        )

        row = df_view[df_view['Ticker'] == sel].iloc[0]
        ticker_full = sel + ".MC"
        sector = row['Sector']

        # Header métricas
        col_s, col_g, col_p, col_d = st.columns(4)
        score_val = row['_score_raw']
        col_s.metric("Expert Score", f"{score_val}/100", delta=row['Rec.'])
        col_g.metric("Valor Graham", row['Graham №'], delta=f"Upside: {row['Upside Graham']}")
        col_p.metric("Piotroski F-Score", row['Piotroski'])
        col_d.metric("Dividendo", row['Div Yield'])

        st.divider()

        col_tesis, col_radar = st.columns([1, 1])

        with col_tesis:
            st.subheader("📋 Tesis de Inversión")
            for r in detalles.get(sel, []):
                st.write(r)

        with col_radar:
            st.subheader("🕸️ Perfil de Calidad (Datos Reales)")
            # Recalcular radar con datos frescos para la empresa seleccionada
            try:
                stk  = yf.Ticker(ticker_full)
                info = stk.info
                fin  = stk.income_stmt
                bal  = stk.balance_sheet
                cf   = stk.cashflow
                hist = stk.history(period="1y")
                fig_r, dim_vals = construir_radar(ticker_full, info, fin, bal, cf, hist, sector)
                st.plotly_chart(fig_r, use_container_width=True)

                # Tabla de dimensiones del radar
                dim_df = pd.DataFrame([
                    {"Dimensión": k.replace("\n", " "), "Puntuación /10": v}
                    for k, v in dim_vals.items()
                ])
                st.dataframe(dim_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar el radar: {e}")

        # Piotroski detalle
        if sector != "Bancos & Seguros" and sel in f_det_all and f_det_all[sel]:
            st.divider()
            st.subheader("🧬 Piotroski F-Score — Desglose Componente a Componente")
            f_det = f_det_all[sel]
            rows_f = []
            for criterio, (icono, valor) in f_det.items():
                rows_f.append({
                    "Criterio": criterio,
                    "Estado": icono,
                    "Valor": valor,
                    "Puntos": "+1" if icono == "✅" else "0"
                })
            df_f = pd.DataFrame(rows_f)
            st.table(df_f)

            total_ok = sum(1 for _, (ic, _) in f_det.items() if ic == "✅")
            interpretacion = "Alta calidad — posible compra" if total_ok >= 7 else "Calidad aceptable" if total_ok >= 5 else "Fundamentales deteriorándose — cuidado"
            st.info(f"**F-Score total: {total_ok}/9** — {interpretacion}")

        # Gráfico histórico precio + SMA
        st.divider()
        st.subheader(f"📈 Evolución Técnica: {sel}")
        try:
            hist_plot = yf.Ticker(ticker_full).history(period="2y")
            if not hist_plot.empty:
                hist_plot['SMA50']  = hist_plot['Close'].rolling(50).mean()
                hist_plot['SMA200'] = hist_plot['Close'].rolling(200).mean()

                fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      row_heights=[0.75, 0.25], vertical_spacing=0.04)
                fig_h.add_trace(go.Candlestick(
                    x=hist_plot.index, open=hist_plot['Open'], high=hist_plot['High'],
                    low=hist_plot['Low'], close=hist_plot['Close'], name='Precio',
                    increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
                ), row=1, col=1)
                fig_h.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['SMA50'],
                    line=dict(color='orange', width=1.5), name='SMA50'), row=1, col=1)
                fig_h.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['SMA200'],
                    line=dict(color='royalblue', width=2), name='SMA200'), row=1, col=1)
                fig_h.add_trace(go.Bar(x=hist_plot.index, y=hist_plot['Volume'],
                    marker_color='rgba(100,150,255,0.4)', name='Volumen'), row=2, col=1)
                fig_h.update_layout(height=550, template="plotly_dark",
                                    xaxis_rangeslider_visible=False,
                                    margin=dict(t=20, b=20))
                st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.warning(f"Gráfico no disponible: {e}")

# ---------------------------------------------------------------------------
# TAB 3 — METODOLOGÍA
# ---------------------------------------------------------------------------
with tab_metodologia:
    st.header("📘 Metodología v2.0 — Cambios y Fundamentos")

    st.markdown("""
    ## Cambios críticos respecto a v1.0

    ### 1. ROIC corregido: NOPAT en lugar de EBIT

    **v1.0 (incorrecto):** `ROIC = EBIT / (Equity + Total Debt)`

    **v2.0 (correcto):** `ROIC = NOPAT / Capital Invertido`

    donde `NOPAT = EBIT × (1 - Tasa Impositiva)` y `Capital Invertido = Equity + Deuda − Caja Excedente`.

    El EBIT sin impuestos sobreestima el retorno real entre un 20-30% dependiendo del sector. El capital invertido debe restar la caja excedente porque ese efectivo no está "trabajando" en el negocio operativo. Esta corrección puede cambiar significativamente el ranking: una empresa con ROIC aparente del 18% (v1) puede tener un ROIC real del 13% (v2) una vez aplicados impuestos del 25%.

    ---

    ### 2. Nombre correcto: **Piotroski** (no "Pietroski")

    El F-Score fue desarrollado por **Joseph D. Piotroski** (Universidad de Chicago, 2000) en su paper *"Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers"*.

    ---

    ### 3. Dividendo: verificación de sostenibilidad

    **v1.0:** +15 puntos si `DividendYield > 5%` (sin más comprobaciones)

    **v2.0:** Solo puntúa positivo si además `PayoutRatio ≤ 100%`. Un dividendo del 8% con payout del 110% es una trampa de valor clásica — la empresa está pagando más de lo que gana, lo que conduce invariablemente a un recorte del dividendo.

    | Dividendo | Payout | Interpretación v2.0 |
    |-----------|--------|---------------------|
    | 7% | 65% | ✅ Potente y sostenible |
    | 7% | 115% | ❌ Penalización — trampa de valor |
    | 4% | 45% | 🟡 Atractivo y sólido |

    ---

    ### 4. Ratio de Eficiencia para Bancos (Cost-to-Income)

    **v1.0:** La función reconocía explícitamente en un comentario que esta métrica no estaba implementada.

    **v2.0:** Se calcula como `Gastos Operativos / Ingresos Totales`. Es la métrica más importante de eficiencia bancaria junto con el ROE:

    - **< 50%:** Banco muy eficiente (Inditex entre los bancos)
    - **50-60%:** Aceptable, margen de mejora
    - **> 65%:** Estructuralmente ineficiente — señal de riesgo

    ---

    ### 5. Radar Chart con datos reales

    **v1.0:** Los 5 valores del radar se generaban con `np.random.uniform()` basado vagamente en el score total. Cada recarga mostraba valores diferentes para la misma empresa.

    **v2.0:** Cada dimensión del radar (0-10) se calcula a partir de métricas reales:

    | Dimensión | Métrica base |
    |-----------|-------------|
    | Calidad | ROIC/ROE normalizado por sector |
    | Valor | PEG + FCF Yield combinados |
    | Crecimiento | Revenue Growth |
    | Momentum | SMA200 + SMA50 + Proximidad 52w |
    | Seguridad | F-Score + Deuda/EBITDA (o Capital Ratio en bancos) |

    ---

    ### 6. Score normalizado a 100 puntos exactos

    **v1.0:** La documentación decía "0-100" pero la suma real de puntos era 110 (error silencioso que distorsionaba umbrales de recomendación).

    **v2.0:** Cada función de scoring tiene sus componentes perfectamente documentados y sumando exactamente 100 puntos máximos. El cap `min(score, 100)` evita overflow en casos límite.

    | Pilar | Empresas Generales | Bancos |
    |-------|--------------------|--------|
    | Rentabilidad | 30 pts | ROE: 25 pts |
    | Salud / Capital | 30 pts | P/B + Capital: 35 pts |
    | Valoración | 20 pts | Dividendo: 15 pts |
    | Crecimiento | 10 pts | Eficiencia: 15 pts |
    | Momentum | 10 pts | Momentum: 10 pts |
    | **Total** | **100 pts** | **100 pts** |

    ---

    ### 7. Caché TTL corregido

    **v1.0:** `@st.cache_data(ttl=60)` — recalculaba todo cada minuto generando decenas de llamadas redundantes a yfinance (rate limiting garantizado en el mercado real).

    **v2.0:** `@st.cache_data(ttl=3600)` — 1 hora, coherente con la frecuencia real de actualización de datos fundamentales.

    ---

    ## El Número de Graham

    `V = √(22.5 × EPS × BookValuePerShare)`

    Fórmula de Benjamin Graham del capítulo 14 de *"The Intelligent Investor"* (1949). El factor 22.5 proviene de los criterios defensivos de Graham: máximo P/E de 15 × máximo P/B de 1.5 = 22.5.

    **Limitaciones importantes** (que la v1.0 no mencionaba):
    - No aplica a empresas con EPS o Book Value negativos
    - No aplica a bancos (el balance es su materia prima, no una comparativa simple)
    - Es un precio "máximo defensivo", no un precio objetivo de crecimiento
    - No incorpora tipos de interés (fue diseñada en un entorno de tipos muy distintos)

    ---

    > ⚠️ **Aviso legal:** Esta herramienta es exclusivamente educativa. No constituye asesoramiento financiero regulado. Los datos provienen de yfinance (Yahoo Finance) y pueden contener errores o retrasos. Consulte siempre fuentes primarias (CNMV, informes anuales auditados) antes de tomar decisiones de inversión.
    """)
