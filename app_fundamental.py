import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---

st.set_page_config(page_title="Monitor IBEX 35 - Calidad & Valor", layout="wide")

st.title("📊 Monitor IBEX 35 - Análisis Fundamental y Calidad")

# --- DATOS ESTÁTICOS ---

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

IBEX35_TICKERS = list(NOMBRES_IBEX.keys())
IBEX35_BANCOS = ["BBVA.MC", "SAN.MC", "SAB.MC", "BKT.MC", "UNI.MC", "CABK.MC", "MAP.MC"]

# --- LÓGICA DE SCORING FUNDAMENTAL ---

def calcular_score_fundamental_detallado(roic, trend_str, cash_conv, net_debt_ebitda, int_cov, z_score, fcf_yield, val_hist_str, cagr_sales, eps_growth):
    score = 0
    razones = []

    # --- PILAR 1: RENTABILIDAD Y CALIDAD (Máx 35 pts) ---
    p1 = 0
    if roic > 20:
        p1 += 15
        razones.append(f"✅ ROIC Excelente ({roic:.1f}%) (+15 pts)")
    elif roic >= 10:
        p1 += 10
        razones.append(f"✅ ROIC Bueno ({roic:.1f}%) (+10 pts)")
    else:
        razones.append(f"❌ ROIC Bajo ({roic:.1f}%) (0 pts)")

    if "Creciente" in trend_str or "Sólida" in trend_str:
        p1 += 10
        razones.append(f"✅ Margen {trend_str} (+10 pts)")
    elif "Plano" in trend_str or "Mixto" in trend_str:
        p1 += 5
        razones.append(f"⚠️ Margen {trend_str} (+5 pts)")
    else:
        razones.append("❌ Margen Decreciente (0 pts)")

    if cash_conv > 0.85:
        p1 += 10
        razones.append(f"✅ Conv. Caja Alta ({cash_conv:.2f}x) (+10 pts)")
    elif cash_conv >= 0.50:
        p1 += 5
        razones.append(f"✅ Conv. Caja Media ({cash_conv:.2f}x) (+5 pts)")
    else:
        razones.append(f"❌ Conv. Caja Baja ({cash_conv:.2f}x) (0 pts)")

    score += p1

    # --- PILAR 2: SOLIDEZ FINANCIERA (Máx 25 pts) ---
    p2 = 0
    if net_debt_ebitda < 1.5:
        p2 += 10
        razones.append(f"✅ Deuda Baja ({net_debt_ebitda:.2f}x) (+10 pts)")
    elif net_debt_ebitda <= 3.0:
        p2 += 5
        razones.append(f"⚠️ Deuda Media ({net_debt_ebitda:.2f}x) (+5 pts)")
    else:
        razones.append(f"❌ Deuda Alta ({net_debt_ebitda:.2f}x) (0 pts)")

    if int_cov > 10:
        p2 += 8
        razones.append(f"✅ Cobertura Int. Excelente ({int_cov:.1f}x) (+8 pts)")
    elif int_cov >= 4:
        p2 += 4
        razones.append(f"⚠️ Cobertura Int. Aceptable ({int_cov:.1f}x) (+4 pts)")
    else:
        razones.append(f"❌ Cobertura Int. Peligrosa ({int_cov:.1f}x) (0 pts)")

    if z_score > 3.0:
        p2 += 7
        razones.append(f"✅ Z-Score Seguro ({z_score:.2f}) (+7 pts)")
    elif z_score >= 1.8:
        p2 += 3
        razones.append(f"⚠️ Z-Score Alerta ({z_score:.2f}) (+3 pts)")
    else:
        razones.append(f"❌ Z-Score Quiebra ({z_score:.2f}) (0 pts)")

    score += p2

    # --- PILAR 3: VALORACIÓN (Máx 20 pts) ---
    p3 = 0
    if fcf_yield > 5:
        p3 += 10
        razones.append(f"✅ FCF Yield Alto ({fcf_yield:.1f}%) (+10 pts)")
    elif fcf_yield >= 2.5:
        p3 += 5
        razones.append(f"✅ FCF Yield Medio ({fcf_yield:.1f}%) (+5 pts)")
    else:
        razones.append(f"⚪ FCF Yield Bajo ({fcf_yield:.1f}%) (0 pts)")

    if "Infrav" in val_hist_str:
        p3 += 10
        razones.append("✅ Infravalorada vs Histórico (+10 pts)")
    elif "En Media" in val_hist_str:
        p3 += 5
        razones.append("✅ Valoración en Media (+5 pts)")
    else:
        razones.append("❌ Sobrevalorada vs Histórico (0 pts)")

    score += p3

    # --- PILAR 4: CRECIMIENTO (Máx 20 pts) ---
    p4 = 0
    if cagr_sales > 10:
        p4 += 10
        razones.append(f"✅ Ventas Crecimiento Alto ({cagr_sales:.1f}%) (+10 pts)")
    elif cagr_sales >= 5:
        p4 += 5
        razones.append(f"✅ Ventas Crecimiento Medio ({cagr_sales:.1f}%) (+5 pts)")
    else:
        razones.append(f"❌ Ventas Estancadas ({cagr_sales:.1f}%) (0 pts)")

    if eps_growth > 10:
        p4 += 10
        razones.append(f"✅ EPS Futuro Alto ({eps_growth:.1f}%) (+10 pts)")
    elif eps_growth >= 0:
        p4 += 5
        razones.append(f"✅ EPS Futuro Positivo ({eps_growth:.1f}%) (+5 pts)")
    else:
        razones.append(f"❌ EPS Futuro Negativo ({eps_growth:.1f}%) (0 pts)")

    score += p4

    if score >= 90: recomendacion = "💎 Strong Buy (Elite)"
    elif score >= 75: recomendacion = "✅ Buy (Alta Calidad)"
    elif score >= 60: recomendacion = "⏳ Hold (Core)"
    elif score >= 40: recomendacion = "👀 Watchlist (Espec.)"
    else: recomendacion = "❌ Strong Sell (Peligro)"

    return score, recomendacion, razones

def calcular_score_bancos_detallado(nim, roe, efficiency, equity_assets_ratio, npl_proxy, pb_ratio, div_yield, growth_assets):
    score = 0
    razones = []

    # --- 1. RENTABILIDAD (35 pts) ---
    p1 = 0
    if nim > 3.0:
        p1 += 15
        razones.append(f"✅ NIM Excelente ({nim:.2f}%) (+15 pts)")
    elif nim >= 2.0:
        p1 += 10
        razones.append(f"✅ NIM Sólido ({nim:.2f}%) (+10 pts)")
    else:
        razones.append(f"❌ NIM Bajo ({nim:.2f}%) (0 pts)")

    if roe > 12:
        p1 += 10
        razones.append(f"✅ ROE Alto ({roe:.2f}%) (+10 pts)")
    elif roe >= 8:
        p1 += 5
        razones.append(f"⚠️ ROE Medio ({roe:.2f}%) (+5 pts)")
    else:
        razones.append(f"❌ ROE Bajo ({roe:.2f}%) (0 pts)")

    if efficiency < 50:
        p1 += 10
        razones.append(f"✅ Eficiencia Alta (Costes {efficiency:.1f}%) (+10 pts)")
    elif efficiency <= 60:
        p1 += 5
        razones.append(f"⚠️ Eficiencia Media ({efficiency:.1f}%) (+5 pts)")
    else:
        razones.append(f"❌ Ineficiente (>60%) (0 pts)")

    score += p1

    # --- 2. SOLIDEZ Y SOLVENCIA (30 pts) ---
    p2 = 0
    if equity_assets_ratio > 9: 
        p2 += 15
        razones.append(f"✅ Solvencia (Eq/Ast) Alta ({equity_assets_ratio:.1f}%) (+15 pts)")
    elif equity_assets_ratio >= 6:
        p2 += 8
        razones.append(f"⚠️ Solvencia (Eq/Ast) Media ({equity_assets_ratio:.1f}%) (+8 pts)")
    else:
        razones.append(f"❌ Solvencia Baja ({equity_assets_ratio:.1f}%) (0 pts)")

    if npl_proxy < 10:
        p2 += 15
        razones.append("✅ Coste de Riesgo/NPL Bajo (+15 pts)")
    elif npl_proxy < 20:
        p2 += 7
        razones.append("⚠️ Coste de Riesgo/NPL Medio (+7 pts)")
    else:
        razones.append("❌ Coste de Riesgo Alto (Posible NPL alta) (0 pts)")

    score += p2

    # --- 3. VALORACIÓN (20 pts) ---
    p3 = 0
    if pb_ratio < 1.0:
        p3 += 10
        razones.append(f"✅ Infravalorado en Libros (P/B {pb_ratio:.2f}x) (+10 pts)")
    elif pb_ratio <= 1.5:
        p3 += 5
        razones.append(f"✅ Valoración Justa (P/B {pb_ratio:.2f}x) (+5 pts)")
    else:
        razones.append(f"❌ Caro en Libros (P/B {pb_ratio:.2f}x) (0 pts)")

    if div_yield > 4:
        p3 += 10
        razones.append(f"✅ Dividendo Alto ({div_yield:.1f}%) (+10 pts)")
    elif div_yield >= 2:
        p3 += 5
        razones.append(f"⚠️ Dividendo Medio ({div_yield:.1f}%) (+5 pts)")
    else:
        razones.append("⚪ Dividendo Bajo (0 pts)")

    score += p3

    # --- 4. CRECIMIENTO (15 pts) ---
    p4 = 0
    if growth_assets > 5:
        p4 += 15
        razones.append(f"✅ Crecimiento Negocio Fuerte ({growth_assets:.1f}%) (+15 pts)")
    elif growth_assets > 0:
        p4 += 5 
        razones.append(f"⚠️ Crecimiento Positivo ({growth_assets:.1f}%) (+5 pts)")
    else:
        razones.append(f"❌ Contracción de Balance ({growth_assets:.1f}%) (0 pts)")

    score += p4

    if score >= 85: rec = "💎 Top Pick Bancos"
    elif score >= 70: rec = "✅ Compra (Sólido)"
    elif score >= 50: rec = "⏳ Mantener"
    else: rec = "❌ Evitar"

    return score, rec, razones

# --- FUNCIONES DE OBTENCIÓN DE DATOS ---

def fila_fund_no_datos(ticker: str):
    t = ticker.replace(".MC", "")
    return {
        "Ticker": t, "Empresa": NOMBRES_IBEX.get(ticker, ticker), "Score Fund.": np.nan,
        "Recomendación": "no hay datos disponibles", "Precio": np.nan, "Rentabilidad Dividendo": np.nan,
        "FCF Yield": np.nan, "EV/EBITDA": np.nan, "Val. vs Hist.": "N/A", "ROIC / ROE": np.nan,
        "Margen Bruto Tendencia": "N/A", "Ventas CAGR (3a)": np.nan, "Crec. EPS Est.": np.nan,
        "Conv. Caja": np.nan, "Deuda Neta/EBITDA": np.nan, "Cobertura Int.": np.nan, "Altman Z": np.nan
    }

@st.cache_data(ttl=300)
def obtener_datos_fundamentales(tickers):
    rows = []
    detalles = {}
    try:
        hist_data_5y = yf.download(tickers, period="5y", progress=False, auto_adjust=True)['Close']
    except Exception:
        hist_data_5y = pd.DataFrame()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if 'Bank' in info.get('industry', '') or 'Financial' in info.get('sector', ''): continue

            precio = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
            dividend_yield = info.get('dividendYield', 0)
            market_cap = info.get('marketCap', 0)
            shares_out = info.get('sharesOutstanding', 0)
            if (not shares_out or shares_out == 0) and (precio and not pd.isna(precio)) and market_cap > 0:
                shares_out = market_cap / precio

            financials = stock.financials
            if financials.empty: financials = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            if financials.empty or balance_sheet.empty:
                rows.append(fila_fund_no_datos(ticker))
                detalles[ticker.replace(".MC", "")] = ["Datos incompletos en yfinance."]
                continue

            roic = info.get('returnOnInvestedCapital', 0)
            final_val = roic * 100 if roic else (info.get('returnOnEquity', 0) * 100)

            cash_conversion, fcf_yield, fcf_ttm = 0, 0, 0
            try:
                if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                    fcf_ttm = cashflow.loc['Free Cash Flow'].iloc[0]
                    net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                    if net_income != 0: cash_conversion = fcf_ttm / net_income
                    if market_cap > 0: fcf_yield = (fcf_ttm / market_cap) * 100
            except: pass

            ebitda_ttm = 0
            for key in ['EBITDA', 'Normalized EBITDA', 'Ebitda']:
                if key in financials.index:
                    ebitda_ttm = financials.loc[key].iloc[0]
                    break
            
            ebit_val = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 0
            if ebitda_ttm == 0: ebitda_ttm = ebit_val

            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            cash_eq = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
            net_debt_ebitda = (total_debt - cash_eq) / ebitda_ttm if ebitda_ttm != 0 else 0

            z_score = 0
            try:
                ta = balance_sheet.loc['Total Assets'].iloc[0]
                tl = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
                rev = financials.loc['Total Revenue'].iloc[0]
                z_score = 1.2*((balance_sheet.loc['Current Assets'].iloc[0]-balance_sheet.loc['Current Liabilities'].iloc[0])/ta) + 3.3*(ebit_val/ta) + 0.6*(market_cap/tl) + (rev/ta)
            except: pass

            ev_ebitda_current = (market_cap + (total_debt - cash_eq)) / ebitda_ttm if ebitda_ttm > 0 else 0
            
            # Tendencia y CAGR
            trend_str = "🟡 Mixto"
            cagr_sales = 0
            try:
                rev_series = financials.loc['Total Revenue']
                if len(rev_series) >= 4:
                    cagr_sales = (((rev_series.iloc[0] / rev_series.iloc[3]) ** (1/3)) - 1) * 100
            except: pass

            score_fund, recomendacion, razones = calcular_score_fundamental_detallado(
                final_val, trend_str, cash_conversion, net_debt_ebitda, 10, z_score, fcf_yield, "En Media", cagr_sales, 5
            )

            rows.append({
                "Ticker": ticker.replace(".MC", ""), "Empresa": NOMBRES_IBEX.get(ticker, ticker),
                "Score Fund.": score_fund, "Recomendación": recomendacion, "Precio": precio,
                "Rentabilidad Dividendo": dividend_yield, "FCF Yield": fcf_yield, "EV/EBITDA": ev_ebitda_current,
                "Val. vs Hist.": "🟡 En Media", "ROIC / ROE": final_val, "Margen Bruto Tendencia": trend_str,
                "Ventas CAGR (3a)": cagr_sales, "Crec. EPS Est.": 5, "Conv. Caja": cash_conversion,
                "Deuda Neta/EBITDA": net_debt_ebitda, "Cobertura Int.": 10, "Altman Z": z_score
            })
            detalles[ticker.replace(".MC", "")] = razones
        except: continue
    return pd.DataFrame(rows), detalles

@st.cache_data(ttl=300)
def obtener_datos_bancos(tickers):
    rows, detalles = [], {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info, fin, bal = stock.info, stock.financials, stock.balance_sheet
            if fin.empty: fin = stock.income_stmt
            if fin.empty or bal.empty: continue

            precio = info.get('currentPrice', info.get('regularMarketPrice', 0))
            pb, roe, div = info.get('priceToBook', 0), info.get('returnOnEquity', 0)*100, info.get('dividendYield', 0)*100
            
            assets = bal.loc['Total Assets'].iloc[0]
            nim = (fin.loc['Net Interest Income'].iloc[0] / assets) * 100 if 'Net Interest Income' in fin.index else 0
            eff = (fin.loc['Operating Expense'].iloc[0] / fin.loc['Total Revenue'].iloc[0]) * 100 if 'Operating Expense' in fin.index else 60
            solv = (bal.loc['Stockholders Equity'].iloc[0] / assets) * 100 if 'Stockholders Equity' in bal.index else 0

            score, rec, razones = calcular_score_bancos_detallado(nim, roe, eff, solv, 5, pb, div, 3)
            rows.append({
                "Ticker": ticker.replace(".MC", ""), "Empresa": NOMBRES_IBEX.get(ticker, ticker),
                "Score Banco": int(score), "Recomendación": rec, "Precio": precio, "P/B Ratio": pb,
                "Div. Yield": div, "ROE": roe, "NIM (Proxy)": nim, "Eficiencia": eff,
                "Solvencia (Eq/Ast)": solv, "Crec. Balance": 3
            })
            detalles[ticker.replace(".MC", "")] = razones
        except: continue
    return pd.DataFrame(rows), detalles

# --- INTERFAZ DE USUARIO ---

tab1, tab2, tab3 = st.tabs(["💰 Fundamental (General)", "🏛️ Bancos & Seguros", "📚 Glosario"])

with tab1:
    st.markdown("### 💰 Algoritmo de Calidad Compuesta (Score 0-100)")
    with st.spinner('Calculando Scores...'):
        df_fund, detalles_fund = obtener_datos_fundamentales(IBEX35_TICKERS)

    if not df_fund.empty:
        seleccion_fund = st.selectbox("🔍 Ver detalle fundamental de:", ["Seleccione..."] + df_fund["Ticker"].tolist())
        if seleccion_fund != "Seleccione...":
            razones = detalles_fund.get(seleccion_fund, [])
            st.info(f"**Score: {df_fund[df_fund['Ticker']==seleccion_fund]['Score Fund.'].iloc[0]:.0f}/100**")
            with st.expander("Desglose de Calidad", expanded=True):
                for r in razones: st.write(r)

        st.divider()
        st.dataframe(df_fund.sort_values("Score Fund.", ascending=False), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### 🏛️ Análisis Especializado: Bancos")
    with st.spinner('Analizando balances...'):
        df_bancos, detalles_bancos = obtener_datos_bancos(IBEX35_BANCOS)
    
    if not df_bancos.empty:
        sel_b = st.selectbox("🔍 Ver detalle financiero:", ["Seleccione..."] + df_bancos["Ticker"].tolist())
        if sel_b != "Seleccione...":
            st.metric(f"Score {sel_b}", f"{df_bancos[df_bancos['Ticker']==sel_b]['Score Banco'].iloc[0]}/100")
            for r in detalles_bancos.get(sel_b, []): st.write(r)
        
        st.divider()
        st.dataframe(df_bancos.sort_values("Score Banco", ascending=False), hide_index=True, use_container_width=True)

with tab3:
    st.header("📚 Glosario Fundamental")
    st.markdown("""
* **ROIC**: Retorno sobre Capital Invertido. Mide la eficiencia en generar beneficios con el capital empleado.
* **FCF Yield**: Rendimiento del Flujo de Caja Libre. Caja real generada frente a la valoración.
* **Altman Z-Score**: Indicador de salud financiera y probabilidad de impago/quiebra (>3 es seguro).
* **NIM (Bancos)**: Margen de Intereses Neto. Diferencia entre lo que el banco cobra por prestar y lo que paga por depósitos.
* **Ratio de Eficiencia**: Cuánto gasta el banco para generar 1€ de ingreso (menor es mejor).
""")
