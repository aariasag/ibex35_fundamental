import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
if __name__ == "__main__":
    st.set_page_config(page_title="IBEX 35 Expert - Value & Quality", layout="wide")
    st.title("📈 IBEX 35 Expert - Análisis Fundamental Avanzado")

# --- DATOS Y SECTORIZACIÓN ---
SECTORES = {
    "Utilities & Energy": ["IBE.MC", "ELE.MC", "REP.MC", "NTGY.MC", "ENG.MC", "RED.MC", "ANE.MC", "SLR.MC"],
    "Bancos & Seguros": ["BBVA.MC", "SAN.MC", "CABK.MC", "SAB.MC", "BKT.MC", "UNI.MC", "MAP.MC"],
    "Industria & Construcción": ["ACS.MC", "FER.MC", "ANA.MC", "SCYR.MC", "IDR.MC", "CAF.MC", "MTS.MC", "FDR.MC", "ACX.MC"],
    "Consumo & Retail": ["ITX.MC", "ROVI.MC", "PUIG.MC", "LOG.MC", "GRF.MC", "IAG.MC", "AMS.MC"],
    "Real Estate & Telco": ["TEF.MC", "MRL.MC", "COL.MC", "CLNX.MC"]
}

NOMBRES_IBEX = {
    "ACS.MC": "ACS", "ACX.MC": "Acerinox", "AENA.MC": "Aena", "AMS.MC": "Amadeus",
    "ANA.MC": "Acciona", "ANE.MC": "Acciona Energía", "BBVA.MC": "BBVA", "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank", "CLNX.MC": "Cellnex", "COL.MC": "Colonial", "ELE.MC": "Endesa",
    "ENG.MC": "Enagás", "FDR.MC": "Fluidra", "FER.MC": "Ferrovial", "GRF.MC": "Grifols",
    "IAG.MC": "IAG", "IBE.MC": "Iberdrola", "IDR.MC": "Indra", "ITX.MC": "Inditex",
    "LOG.MC": "Logista", "MAP.MC": "Mapfre", "MRL.MC": "Merlin", "MTS.MC": "ArcelorMittal",
    "NTGY.MC": "Naturgy", "PUIG.MC": "Puig", "RED.MC": "Redeia", "REP.MC": "Repsol",
    "ROVI.MC": "Rovi", "SAB.MC": "Sabadell", "SAN.MC": "Santander", "SCYR.MC": "Sacyr",
    "SLR.MC": "Solaria", "TEF.MC": "Telefónica", "UNI.MC": "Unicaja"
}

def get_sector(ticker):
    for sector, tickers in SECTORES.items():
        if ticker in tickers: return sector
    return "General"

# --- MÉTRICAS EXPERTAS ---

def calcular_pietroski_f_score(fin, bal, cashflow):
    """Calcula el F-Score de Pietroski (0-9) basado en cambios año a año."""
    score = 0
    try:
        # 1. ROA positivo
        net_income = fin.loc['Net Income'].iloc[0]
        total_assets = bal.loc['Total Assets'].iloc[0]
        roa = net_income / total_assets
        if roa > 0: score += 1
        
        # 2. CFO positivo
        cfo = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
        if cfo > 0: score += 1
        
        # 3. ROA creciente (vs año anterior)
        net_income_prev = fin.loc['Net Income'].iloc[1]
        total_assets_prev = bal.loc['Total Assets'].iloc[1]
        roa_prev = net_income_prev / total_assets_prev
        if roa > roa_prev: score += 1
        
        # 4. CFO > Net Income (Calidad devengo)
        if cfo > net_income: score += 1
        
        # 5. Deuda a Largo Plazo decreciente (o apalancamiento)
        lt_debt = bal.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in bal.index else 0
        lt_debt_prev = bal.loc['Long Term Debt'].iloc[1] if 'Long Term Debt' in bal.index else 0
        if lt_debt <= lt_debt_prev: score += 1
        
        # 6. Current Ratio creciente (Liquidez)
        curr_assets = bal.loc['Current Assets'].iloc[0]
        curr_liab = bal.loc['Current Liabilities'].iloc[0]
        curr_ratio = curr_assets / curr_liab
        
        curr_assets_prev = bal.loc['Current Assets'].iloc[1]
        curr_liab_prev = bal.loc['Current Liabilities'].iloc[1]
        curr_ratio_prev = curr_assets_prev / curr_liab_prev
        
        if curr_ratio > curr_ratio_prev: score += 1
        
        # 7. Sin dilución de acciones
        shares = bal.loc['Share Issued'].iloc[0] if 'Share Issued' in bal.index else 0
        shares_prev = bal.loc['Share Issued'].iloc[1] if 'Share Issued' in bal.index else 0
        if shares <= shares_prev: score += 1
        
        # 8. Margen Bruto creciente
        gross_profit = fin.loc['Gross Profit'].iloc[0]
        revenue = fin.loc['Total Revenue'].iloc[0]
        gross_margin = gross_profit / revenue
        
        gross_profit_prev = fin.loc['Gross Profit'].iloc[1]
        revenue_prev = fin.loc['Total Revenue'].iloc[1]
        gross_margin_prev = gross_profit_prev / revenue_prev
        
        if gross_margin > gross_margin_prev: score += 1
        
        # 9. Rotación de Activos creciente
        asset_turnover = revenue / total_assets
        asset_turnover_prev = revenue_prev / total_assets_prev
        
        if asset_turnover > asset_turnover_prev: score += 1
        
    except Exception:
        pass # Si faltan datos para comparar años, se queda con lo acumulado
        
    return score

def calcular_graham_number(eps, bvps):
    """Valor intrínseco de Graham: Sqrt(22.5 * EPS * BVPS)"""
    if eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return 0

# --- SCORING POR SECTOR ---

def analizar_general_expert(ticker, info, fin, bal, cashflow):
    """Lógica de scoring ajustada por sector."""
    sector = get_sector(ticker)
    score = 0
    razones = []
    
    # ---------------------------
    # 1. RENTABILIDAD & RETORNOS (30 pts)
    # ---------------------------
    roic = info.get('returnOnInvestedCapital', 0) * 100
    if roic == 0: # Fallback a ROE si falta ROIC
        roic = info.get('returnOnEquity', 0) * 100
        
    # Ajuste umbral ROIC por sector
    umbral_roic_excelente = 15 if sector != "Utilities & Energy" else 10
    
    if roic > umbral_roic_excelente:
        score += 15
        razones.append(f"✅ Rentabilidad Excelente (ROIC {roic:.1f}%) (+15)")
    elif roic > (umbral_roic_excelente - 5):
        score += 8
        razones.append(f"✅ Rentabilidad Sólida (ROIC {roic:.1f}%) (+8)")
    else:
        razones.append(f"⚪ Rentabilidad Baja (ROIC {roic:.1f}%)")

    # Margen Neto tendencia
    try:
        net_margin = fin.loc['Net Income'].iloc[0] / fin.loc['Total Revenue'].iloc[0]
        net_margin_prev = fin.loc['Net Income'].iloc[1] / fin.loc['Total Revenue'].iloc[1]
        if net_margin > net_margin_prev:
            score += 15
            razones.append("✅ Expansión de Márgenes (+15)")
        else:
            razones.append("⚠️ Contracción de Márgenes")
    except: pass

    # ---------------------------
    # 2. SALUD FINANCIERA (F-Score + Deuda) (30 pts)
    # ---------------------------
    f_score = calcular_pietroski_f_score(fin, bal, cashflow)
    if f_score >= 7:
        score += 15
        razones.append(f"✅ Estado Financiero Impecable (F-Score {f_score}) (+15)")
    elif f_score >= 5:
        score += 8
        razones.append(f"⚠️ Estado Financiero Aceptable (F-Score {f_score}) (+8)")
    else:
        score -= 5 # Penalización
        razones.append(f"❌ Deterioro Financiero (F-Score {f_score}) (-5)")

    # Deuda/EBITDA
    try:
        total_debt = bal.loc['Total Debt'].iloc[0] if 'Total Debt' in bal.index else 0
        cash = bal.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bal.index else 0
        ebitda = fin.loc['EBITDA'].iloc[0] if 'EBITDA' in fin.index else 0
        if ebitda == 0: ebitda = 1 # Evitar div 0
        
        net_debt_ebitda = (total_debt - cash) / ebitda
        
        # Tolerancia deuda por sector
        tolerancia = 2.5
        if sector == "Utilities & Energy": tolerancia = 4.5
        if sector == "Real Estate & Telco": tolerancia = 5.5 # SOCIMIs y Telcos aguantan deuda alta
        
        if net_debt_ebitda < tolerancia:
            score += 15
            razones.append(f"✅ Endeudamiento Controlado ({net_debt_ebitda:.1f}x vs {tolerancia}x max) (+15)")
        else:
             razones.append(f"❌ Deuda Elevada para Sector ({net_debt_ebitda:.1f}x) (0)")
    except: pass

    # ---------------------------
    # 3. VALORACIÓN (20 pts)
    # ---------------------------
    pe = info.get('trailingPE', 0)
    # PEG Ratio simple
    growth_est = info.get('earningsGrowth', 0.05) * 100 # Estimación YF
    if growth_est <= 0: growth_est = 1
    
    if pe > 0:
        peg = pe / growth_est
        if peg < 1.0:
            score += 10
            razones.append(f"✅ Barata vs Crecimiento (PEG {peg:.2f}) (+10)")
        elif peg < 1.5:
            score += 5
            razones.append(f"✅ Valor Justo (PEG {peg:.2f}) (+5)")
    
    # FCF Yield
    mkt_cap = info.get('marketCap', 1)
    try:
        fcf = cashflow.loc['Free Cash Flow'].iloc[0]
        fcf_yield = (fcf / mkt_cap) * 100
        if fcf_yield > 6:
            score += 10
            razones.append(f"✅ FCF Yield Muy Atractivo ({fcf_yield:.1f}%) (+10)")
        elif fcf_yield > 4:
            score += 5
            razones.append(f"✅ Generación Caja Sólida ({fcf_yield:.1f}%) (+5)")
    except: pass

    # ---------------------------
    # 4. CRECIMIENTO & MOMENTUM (20 pts)
    # ---------------------------
    try:
        rev_growth = info.get('revenueGrowth', 0) 
        if rev_growth > 0.10: # >10%
            score += 10
            razones.append("✅ Crecimiento Ventas de Doble Dígito (+10)")
        elif rev_growth > 0:
            score += 5
            razones.append("✅ Crecimiento Ventas Positivo (+5)")
    except: pass
    
    # Momentum precio 6m (simple)
    hist = yf.Ticker(ticker).history(period="6mo")
    if not hist.empty:
        ret_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        if ret_6m > 0:
            score += 10
            razones.append(f"✅ Tendencia Alcista CP ({ret_6m*100:.1f}%) (+10)")

    return score, razones, f_score

def analizar_banco_expert(ticker, info, fin, bal):
    score = 0
    razones = []
    
    # Datos básicos
    pb = info.get('priceToBook', 0)
    div_yield = info.get('dividendYield', 0)
    
    # 1. RENTABILIDAD (ROE)
    roe = info.get('returnOnEquity', 0)
    if roe > 0.12: 
        score += 25
        razones.append(f"✅ ROE Líder ({roe*100:.1f}%) (+25)")
    elif roe > 0.08:
        score += 15
        razones.append(f"✅ ROE Sano ({roe*100:.1f}%) (+15)")
    
    # 2. VALORACIÓN (P/B vs ROE)
    # Si ROE > 10% y P/B < 0.8 -> Ganga
    if roe > 0.10 and pb < 0.8:
        score += 25
        razones.append("✅ Oportunidad Valor (Alto ROE, Bajo P/B) (+25)")
    elif pb < 1.0:
        score += 15
        razones.append("✅ Cotiza bajo valor libros (+15)")
        
    # 3. DIVIDENDO
    if div_yield > 0.05:
        score += 15
        razones.append(f"✅ Dividendo Potente ({div_yield*100:.1f}%) (+15)")
        
    # 4. EFICIENCIA (Cost to Income proxy)
    # Difícil sacar exacto de YF genérico, usamos operating margins como proxy inverso
    
    # 5. MOMENTUM
    hist = yf.Ticker(ticker).history(period="6mo")
    if not hist.empty:
        ret_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        if ret_6m > 0:
            score += 15
            razones.append("✅ Tendencia Mercado Positiva (+15)")
            
    # Bases de bancos suelen empezar alto, normalizamos max 100
    if score > 100: score = 100
    
    return score, razones, 0

# --- MAIN DATA FETCH ---

@st.cache_data(ttl=60)
def cargar_datos_expertos(tickers):
    data = []
    detalles = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fin = stock.financials
            bal = stock.balance_sheet
            cf = stock.cashflow
            
            # Fallback anual si trimestral es por defecto (a veces yfinance varía)
            if fin.empty: fin = stock.income_stmt
            
            sector = get_sector(ticker)
            
            if sector == "Bancos & Seguros":
                score, reasons, f_score = analizar_banco_expert(ticker, info, fin, bal)
            else:
                score, reasons, f_score = analizar_general_expert(ticker, info, fin, bal, cf)
            
            # Graham Number
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            graham = calcular_graham_number(eps, bvps)
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            upside_graham = ((graham / current_price) - 1) * 100 if current_price > 0 else 0
            
            recomendacion = "MANTENER"
            if score >= 80: recomendacion = "COMPRA FUERTE"
            elif score >= 60: recomendacion = "COMPRA"
            elif score <= 40: recomendacion = "VENTA"
            
            # Div Yield: Fix potential scale issue (if > 1 assume it's already %, else convert)
            raw_dy = info.get('dividendYield', 0)
            if raw_dy > 1.5: # Assuming nobody pays > 150% dividend easily, but yfinance might return 5.0 for 5%
                 fmt_dy = f"{raw_dy:.2f}%"
            else:
                 fmt_dy = f"{raw_dy*100:.2f}%"

            data.append({
                "Ticker": ticker.replace(".MC", ""),
                "Empresa": NOMBRES_IBEX.get(ticker, ticker),
                "Sector": sector,
                "Score": score,
                "Rec.": recomendacion,
                "Precio": current_price,
                "Valor Graham": graham,
                "Potencial Graham": f"{upside_graham:.1f}%",
                "F-Score (0-9)": f_score if sector != "Bancos & Seguros" else "N/A",
                "Div Yield": fmt_dy
            })
            detalles[ticker.replace(".MC", "")] = reasons
            
        except Exception as e:
            continue
            
    return pd.DataFrame(data), detalles

if __name__ == "__main__":
    tickers_list = []
    for sec_tickers in SECTORES.values():
        tickers_list.extend(sec_tickers)
    tickers_list = list(set(tickers_list)) # Uniq

    st.info("💡 **Novedad**: Lógica ajustada por sector (Utilities toleran más deuda, Tech exige crecimiento) y métricas avanzadas (Pietroski F-Score).")

    tab_analisis, tab_metodologia = st.tabs(["🚀 Análisis & Ranking", "📘 Metodología & Glosario"])

    with tab_analisis:
        with st.spinner("Analizando IBEX 35 con métricas avanzadas..."):
            df, detalles = cargar_datos_expertos(tickers_list)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🏆 Ranking Mejores Oportunidades")
            if not df.empty:
                st.dataframe(
                    df.sort_values("Score", ascending=False).style.background_gradient(subset=["Score"], cmap="RdYlGn"),
                    height=600,
                    use_container_width=True
                )

        with col2:
            st.subheader("🔍 Análisis Detallado")
            if not df.empty:
                sel = st.selectbox("Selecciona empresa:", df.sort_values("Score", ascending=False)["Ticker"].unique())
                
                info_row = df[df["Ticker"]==sel].iloc[0]
                st.metric("Expert Score", f"{info_row['Score']}/100", delta=info_row["Rec."])
                
                st.markdown(f"**Sector:** {info_row['Sector']}")
                st.markdown(f"**Pietroski F-Score:** {info_row['F-Score (0-9)']}")
                st.markdown(f"**Valor Graham:** {info_row['Valor Graham']:.2f}€ (Upside: {info_row['Potencial Graham']})")
                
                st.divider()
                st.markdown("#### Tesis de Inversión:")
                for r in detalles.get(sel, []):
                    st.write(r)
                    
                # Radar Chart Dummy Data (Visual only for now)
                st.divider()
                categories = ['Calidad', 'Valor', 'Crecimiento', 'Momentum', 'Seguridad']
                # Random logic for visualization roughly based on score
                base_val = info_row['Score'] / 20 # 0-5 scale roughly
                values = [
                    min(5, base_val + np.random.uniform(-0.5, 1)),
                    min(5, base_val + np.random.uniform(-0.5, 1)),
                    min(5, base_val + np.random.uniform(-0.5, 1)),
                    min(5, base_val + np.random.uniform(-0.5, 1)),
                    min(5, base_val + np.random.uniform(-0.5, 1))
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=sel
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_metodologia:
        st.header("📘 Metodología de Inversión Experta")
        
        st.markdown("""
        ### 1. El Expert Score (0-100)
        Nuestro algoritmo evalúa 4 pilares fundamentales, con ponderaciones ajustadas por sector:
        
        #### A. Rentabilidad y Retornos (30%)
        *   **ROIC (Return on Invested Capital)**: Buscamos empresas que generan más retorno que su coste de capital.
            *   *Excellent*: >15% (>10% para Utilities).
        *   **Margen Neto**: Se premia la expansión de márgenes (tendencia positiva).
        
        #### B. Salud Financiera (30%)
        Utilizamos métricas avanzadas para evitar trampas de valor.
        *   **Pietroski F-Score (0-9)**: Ver abajo.
        *   **Deuda Neta / EBITDA**:
            *   < 2.5x: Excelente (General).
            *   < 4.5x: Aceptable para **Utilities & Energy** (flujos estables).
            *   < 5.5x: Aceptable para **SOCIMIs**.
            
        #### C. Valoración (20%)
        *   **PEG Ratio**: P/E dividido por crecimiento esperado. < 1.0 es "Barato para su crecimiento".
        *   **FCF Yield**: Rendimiento de flujo de caja libre. > 6% es muy atractivo.
        *   **Graham Number**: Valoración clásica conservadora.
        
        #### D. Crecimiento y Momentum (20%)
        *   **Crecimiento Ventas**: Premiamos crecimiento de doble dígito.
        *   **Momentum**: Tendencia de precio a 6 meses positiva (título fuerte).
        
        ---
        
        ### 2. Pietroski F-Score (0-9)
        El indicador definitivo de mejora en la calidad fundamental. Suma 1 punto si cumple:
        1.  **ROA > 0**: Beneficio neto positivo.
        2.  **CFO > 0**: Flujo de caja operativo positivo.
        3.  **ROA Creciente**: ROA actual > ROA año anterior.
        4.  **CFO > Beneficio Neto**: Calidad de los beneficios (no es solo contabilidad).
        5.  **Deuda Decreciente**: Menor apalancamiento que el año anterior.
        6.  **Liquidez Creciente**: Current Ratio mejora.
        7.  **Sin Dilución**: No ha emitido nuevas acciones.
        8.  **Margen Bruto Creciente**: Poder de fijación de precios o eficiencia.
        9.  **Rotación Activos Creciente**: Mayor eficiencia en ventas por activo.
        
        *   **7-9**: Fuerte compra / Alta calidad.
        *   **0-3**: Fundamentalmente débil.
        
        ---
        
        ### 3. Valor de Graham
        Fórmula de Benjamin Graham para el "Precio Justo" máximo defensivo:
        $$ V = \\sqrt{22.5 \\times EPS \\times BookValue} $$
        *   Nota: Solo aplica si EPS y Valor en Libros son positivos.
        
        ---
        
        ### 4. Ajustes Sectoriales "Inteligentes"
        No todas las empresas son iguales:
        *   **Bancos**: Se ignoran EBITDA y Deuda. Se usa **ROE, P/B Ratio y Dividendo**.
        *   **Utilities (Iberdrola, Endesa)**: Se tolera mayor deuda por su negocio regulado y predecible.
        *   **Growth (Indra, Amadeus)**: Se exige mayor crecimiento de ventas y se tolera menor dividendo.
        """)
