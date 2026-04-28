"""
CRUZBER — Demand Intelligence Platform v10
============================================
Streamlit dashboard — TFM ISDI (Equipo Troncal Hedy Lamarr)

Data sources:
  1. Prediccion_SnOP_NB39_S_.xlsx        → Validación S1-S27 (CatBoost)
  2. Forecast_S28_S39_2024_NB39.xlsx     → Predicción S28-S39
  3. pred_h12_por_provincia_sku_v3_ROI.csv → Stock & ROI por provincia (LightGBM)

Roles: Dirección · Supply Chain · Comercial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cohere

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRUZBER · Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.corp-header {
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 40%, #252525 100%);
    padding: 1.6rem 2.4rem;
    border-radius: 16px;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18), 0 2px 8px rgba(0,0,0,0.12);
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.corp-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #E8491D 0%, #F59E0B 50%, #E8491D 100%);
}
.corp-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,73,29,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.corp-header .header-left { display: flex; align-items: center; gap: 1.2rem; z-index: 1; }
.corp-header .header-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #E8491D, #F06529);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 12px rgba(232,73,29,0.3);
}
.corp-header .header-text h1 {
    color: #FFFFFF; margin: 0;
    font-size: 1.25rem; font-weight: 700;
    letter-spacing: 0.01em;
}
.corp-header .header-text p {
    color: rgba(255,255,255,0.45);
    margin: 0.2rem 0 0 0;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
}
.corp-header .header-right { display: flex; gap: 0.8rem; z-index: 1; }
.corp-header .h-badge {
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.corp-header .h-badge.live {
    background: rgba(16,185,129,0.12);
    color: #10B981;
    border: 1px solid rgba(16,185,129,0.25);
}
.corp-header .h-badge.version {
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] { background: #1A1A1A; }
[data-testid="stSidebar"] * { color: #D0D4DC !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #8B95A8 !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}

.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 140px; background: #FFF; border-radius: 12px;
    padding: 1rem 1.2rem; box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    border-top: 3px solid #E8491D;
}
.kpi .val { font-size: 1.7rem; font-weight: 800; color: #1A1F36; line-height: 1.1; }
.kpi .lbl { font-size: 0.7rem; color: #6B7B99; text-transform: uppercase;
            letter-spacing: 0.06em; margin-top: 0.2rem; }
.kpi.green { border-top-color: #10B981; }
.kpi.red   { border-top-color: #EF4444; }
.kpi.amber { border-top-color: #F59E0B; }
.kpi.blue  { border-top-color: #3B82F6; }

.sec-title { font-size: 1.05rem; font-weight: 700; color: #1A1F36;
             margin: 1.2rem 0 0.6rem 0; padding-bottom: 0.4rem;
             border-bottom: 2px solid #F0F2F6; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background: #F0F2F6; border-radius: 8px 8px 0 0;
                                padding: 10px 22px; font-weight: 600; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #1A1A1A !important; color: white !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos de validación…")
def load_validation():
    df = pd.read_excel("data/Prediccion_SnOP_NB39_S_.xlsx")
    df["familia"] = df["codigo_familia"]
    df["margen_unit"] = df["precio_unit"] - df["coste_escandallo"]
    df["margen_pct"] = np.where(df["precio_unit"] > 0,
                                df["margen_unit"] / df["precio_unit"] * 100, 0)
    return df


@st.cache_data(show_spinner="Cargando forecast predictivo…")
def load_forecast():
    df = pd.read_excel("data/Forecast_S28_S39_2024_NB39.xlsx")
    df["familia"] = df["codigo_familia"]
    return df


@st.cache_data(show_spinner="Cargando modelo OOS / ROI…")
def load_roi():
    df = pd.read_csv("data/pred_h12_por_provincia_sku_v3_ROI.csv", sep=";")
    comma_cols = ["pred_b3_prov", "pred_sop_prov", "stock_obj_final",
                  "upper_80_final", "upper_90_final", "upper_95_final",
                  "fr_final", "fr_polC_servido", "fr_v3_servido", "mejora_pp_servido",
                  "peso_prov", "csl_v2", "cv_v2",
                  "precio_coste_medio", "margen_medio_eur", "margen_pct",
                  "lost_sales_polC", "lost_sales_v3",
                  "inv_polC_eur", "inv_v3_eur",
                  "reduccion_lost_eur", "inv_extra_eur", "roi_pct_sku",
                  "servido_polC", "servido_v3", "no_servido_polC", "no_servido_v3"]
    for c in comma_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    df["familia"] = df["codigo_articulo"].str.split("-").str[0]
    df["semaforo"] = np.where(df["fr_v3_servido"] < 0.60, "rojo",
                     np.where(df["fr_v3_servido"] < 0.85, "amarillo", "verde"))
    return df


@st.cache_data(show_spinner="Cargando descripciones…")
def load_descriptions():
    try:
        df = pd.read_excel("data/Prediccion_SnOP_NB29_v2_PROD.xlsx",
                           usecols=["codigo_articulo", "descripcion"])
        return df.drop_duplicates("codigo_articulo").set_index("codigo_articulo")["descripcion"].to_dict()
    except Exception:
        return {}


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="corp-header">
    <div class="header-left">
        <div class="header-text">
            <h1>CRUZBER · Demand Intelligence Platform</h1>
            <p>Predicción de demanda & optimización de stock · Horizonte 12 semanas</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    df_val = load_validation()
    df_fcast = load_forecast()
    df_roi = load_roi()
    desc_map = load_descriptions()
except FileNotFoundError as e:
    st.error(f"⚠️ Fichero no encontrado: {e}")
    st.stop()

# Merge prices into forecast for monetary calcs
prices = df_val[["codigo_articulo", "precio_unit", "coste_escandallo"]].drop_duplicates("codigo_articulo")
df_fcast = df_fcast.merge(prices, on="codigo_articulo", how="left")
df_fcast["precio_unit"] = df_fcast["precio_unit"].fillna(0)
df_fcast["coste_escandallo"] = df_fcast["coste_escandallo"].fillna(0)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("assets/logo.png", width=150)
    except Exception:
        st.markdown("**CRUZBER**")
    st.markdown("---")

    # Role selector
    st.markdown("#### 👤 Perfil")
    role = st.selectbox("Selecciona tu rol", ["Dirección", "Supply Chain", "Comercial"],
                        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### 🎛️ Filtros")

    abc_sel = st.multiselect("Tipo ABC", ["A", "B", "C"], default=["A", "B", "C"])
    sb_options = sorted(df_val["sb_class"].unique())
    sb_sel = st.multiselect("Clase S-B", sb_options, default=sb_options)
    fam_sel = st.multiselect("Familia", sorted(df_val["familia"].dropna().unique()), default=[])
    conf_sel = st.multiselect("Confianza", sorted(df_val["confianza"].dropna().unique()), default=[])

    if role in ["Dirección", "Supply Chain"]:
        provs_disp = sorted(df_roi["Provincia"].dropna().unique())
        prov_sel = st.multiselect("Provincia", provs_disp, default=[])
    else:
        prov_sel = []

    st.markdown("---")
    st.caption("TFM ISDI · Equipo Hedy Lamarr\nCatBoost · LightGBM · Mondrian · Newsvendor")


# ── APPLY FILTERS ────────────────────────────────────────────────────────────
def filter_sop(df):
    m = df["tipo_abc"].isin(abc_sel) & df["sb_class"].isin(sb_sel)
    if fam_sel:
        m &= df["familia"].isin(fam_sel)
    if conf_sel and "confianza" in df.columns:
        m &= df["confianza"].isin(conf_sel)
    return df[m].copy()

def filter_fcast(df):
    m = df["tipo_abc"].isin(abc_sel) & df["sb_class"].isin(sb_sel)
    if fam_sel:
        m &= df["familia"].isin(fam_sel)
    return df[m].copy()

def filter_roi(df, skus_valid):
    m = df["codigo_articulo"].isin(skus_valid)
    if prov_sel:
        m &= df["Provincia"].isin(prov_sel)
    return df[m].copy()

dv = filter_sop(df_val)
dfp = filter_fcast(df_fcast)
skus_ok = set(dv["codigo_articulo"].unique())
dro = filter_roi(df_roi, skus_ok)

if dv.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()


# ── DEFINE TABS BY ROLE ──────────────────────────────────────────────────────
if role == "Dirección":
    tab_names = ["💰 P&L Simulator", "📈 Forecast", "🚨 Stock & ROI",
                 "📊 Rendimiento", "🔬 What-If", "🗺️ Mapa", "🤖 Asistente IA"]
elif role == "Supply Chain":
    tab_names = ["📈 Forecast", "🚨 Stock & ROI", "📊 Rendimiento",
                 "🔬 What-If", "🗺️ Mapa", "🤖 Asistente IA"]
else:  # Comercial
    tab_names = ["📈 Forecast", "📊 Rendimiento", "🗺️ Mapa", "🤖 Asistente IA"]

tabs = st.tabs(tab_names)
tab_map = {name: tab for name, tab in zip(tab_names, tabs)}


# ═══════════════════════════════════════════════════════════════════════════
# P&L SIMULATOR (Dirección only)
# ═══════════════════════════════════════════════════════════════════════════
if "💰 P&L Simulator" in tab_map:
    with tab_map["💰 P&L Simulator"]:
        st.markdown('<div class="sec-title">Simulador de P&L · Impacto financiero del modelo</div>',
                    unsafe_allow_html=True)

        # Base P&L data from PPT: 14.7M ventas, 59% margen bruto
        VENTAS_BASE = 14_700_000
        MARGEN_BRUTO_PCT = 0.593
        GASTOS_PERSONAL = 3_900_000
        OTROS_GASTOS = 3_400_000
        AMORTIZACION = 500_000
        EBIT_BASE = VENTAS_BASE * MARGEN_BRUTO_PCT - GASTOS_PERSONAL - OTROS_GASTOS - AMORTIZACION

        # Model data
        lost_polc = dro["lost_sales_polC"].sum() if not dro.empty else 5_969_429
        lost_v3 = dro["lost_sales_v3"].sum() if not dro.empty else 3_328_853
        red_lost = dro["reduccion_lost_eur"].sum() if not dro.empty else 2_640_576
        inv_extra = dro["inv_extra_eur"].sum() if not dro.empty else 1_204_624

        col_sl, col_pl = st.columns([1, 2])

        with col_sl:
            st.markdown("**Palancas de simulación**")

            pct_captura = st.slider("% captura de ventas perdidas", 0, 100, 75, 5,
                help="Qué % de las ventas perdidas recuperadas se convierten en venta real")
            ahorro_inv_pct = st.slider("% ahorro en costes de inventario", 0, 30, 10, 1,
                help="Reducción de costes de almacenamiento por mejor rotación")
            capex_extra = st.slider("Inversión extra en stock (K€)", 0, 3000,
                int(inv_extra / 1000), 50,
                help="Capital adicional necesario para la política v3") * 1000

            st.markdown("---")
            st.markdown("**Base P&L (2023)**")
            st.caption(f"Ventas: {VENTAS_BASE/1e6:.1f}M€ · Margen bruto: {MARGEN_BRUTO_PCT:.0%}")
            st.caption(f"EBIT base: {EBIT_BASE/1e6:.1f}M€")

        with col_pl:
            # Calculate scenario
            ventas_recuperadas = red_lost * (pct_captura / 100)
            coste_ventas_recuperadas = ventas_recuperadas * (1 - MARGEN_BRUTO_PCT)
            margen_incremental = ventas_recuperadas * MARGEN_BRUTO_PCT
            ahorro_inventario = dv["capital_inmovilizado_eur"].sum() * (ahorro_inv_pct / 100)
            coste_capital = capex_extra * 0.05  # 5% coste financiero

            delta_ebit = margen_incremental + ahorro_inventario - coste_capital
            ebit_nuevo = EBIT_BASE + delta_ebit
            ventas_nuevas = VENTAS_BASE + ventas_recuperadas
            roi_sim = delta_ebit / max(capex_extra, 1) * 100
            payback_sim = 12 / max(roi_sim / 100, 0.01)

            # KPIs
            kpi_roi = "green" if roi_sim > 100 else ("amber" if roi_sim > 50 else "red")
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi blue"><div class="val">{ventas_nuevas/1e6:.1f}M€</div>
                    <div class="lbl">Ventas simuladas</div></div>
                <div class="kpi green"><div class="val">{ebit_nuevo/1e6:.2f}M€</div>
                    <div class="lbl">EBIT simulado</div></div>
                <div class="kpi green"><div class="val">+{delta_ebit/1e3:,.0f}K€</div>
                    <div class="lbl">Δ EBIT</div></div>
                <div class="kpi {kpi_roi}"><div class="val">{roi_sim:.0f}%</div>
                    <div class="lbl">ROI</div></div>
                <div class="kpi blue"><div class="val">{min(payback_sim, 99):.1f}m</div>
                    <div class="lbl">Payback</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Waterfall chart
            steps = ["EBIT Base", "+ Margen ventas\nrecuperadas", "+ Ahorro\ninventario",
                     "- Coste\nfinanciero", "EBIT Simulado"]
            values = [EBIT_BASE, margen_incremental, ahorro_inventario, -coste_capital, ebit_nuevo]
            measures = ["absolute", "relative", "relative", "relative", "total"]
            colors = ["#3B82F6", "#10B981", "#10B981", "#EF4444", "#1A1A1A"]

            fig_wf = go.Figure(go.Waterfall(
                x=steps, y=values, measure=measures,
                connector=dict(line=dict(color="#D1D5DB", width=1)),
                increasing=dict(marker_color="#10B981"),
                decreasing=dict(marker_color="#EF4444"),
                totals=dict(marker_color="#1A1A1A"),
                textposition="outside",
                text=[f"{v/1e3:+,.0f}K€" if m != "absolute" and m != "total" else f"{v/1e3:,.0f}K€"
                      for v, m in zip(values, measures)],
            ))
            fig_wf.update_layout(
                title="Cascada de impacto en EBIT",
                template="plotly_white", height=420,
                margin=dict(l=40, r=20, t=60, b=80),
                yaxis_title="EUR",
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # Comparison table
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**P&L Base**")
                st.markdown(f"Ventas netas: **{VENTAS_BASE/1e6:.1f}M€**")
                st.markdown(f"Margen bruto: **{VENTAS_BASE*MARGEN_BRUTO_PCT/1e6:.1f}M€** ({MARGEN_BRUTO_PCT:.0%})")
                st.markdown(f"EBIT: **{EBIT_BASE/1e6:.2f}M€**")
            with col_b:
                st.markdown("**P&L Simulada**")
                st.markdown(f"Ventas netas: **{ventas_nuevas/1e6:.1f}M€** ({(ventas_recuperadas/VENTAS_BASE)*100:+.1f}%)")
                mb_new = (VENTAS_BASE*MARGEN_BRUTO_PCT + margen_incremental) / 1e6
                st.markdown(f"Margen bruto: **{mb_new:.1f}M€**")
                st.markdown(f"EBIT: **{ebit_nuevo/1e6:.2f}M€** ({delta_ebit/1e3:+,.0f}K€)")


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════════════════════════════
with tab_map["📈 Forecast"]:
    # Mode toggle
    modo = st.radio("Vista", ["📋 Validación (S1-S27)", "🔮 Predicción (S28-S39)"],
                    horizontal=True)

    if modo.startswith("📋"):
        # ── VALIDATION MODE ──
        t_real = dv["real"].sum()
        t_pred = dv["pred"].sum()
        wmape = dv["error_abs"].sum() / max(t_real, 1) * 100
        bias_pct = dv["sesgo"].sum() / max(t_real, 1) * 100
        n_skus = dv["codigo_articulo"].nunique()
        ventas_riesgo = dv["ventas_riesgo_eur"].sum()

        kw = "green" if wmape < 30 else ("amber" if wmape < 45 else "red")
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="val">{t_real:,.0f}</div><div class="lbl">Uds. reales H12</div></div>
            <div class="kpi blue"><div class="val">{t_pred:,.0f}</div><div class="lbl">Forecast H12</div></div>
            <div class="kpi {kw}"><div class="val">{wmape:.1f}%</div><div class="lbl">WMAPE</div></div>
            <div class="kpi red"><div class="val">{ventas_riesgo/1e6:.1f}M€</div><div class="lbl">Ventas en riesgo</div></div>
            <div class="kpi"><div class="val">{n_skus:,}</div><div class="lbl">SKUs activos</div></div>
        </div>
        """, unsafe_allow_html=True)

        # SKU explorer
        col_c, col_s = st.columns([3, 1])
        with col_s:
            sku_search = st.text_input("🔍 Buscar SKU", placeholder="Ej: 001-106", key="val_search")
            top_skus = dv.groupby("codigo_articulo")["real"].sum().nlargest(30).index.tolist()
            sku_opts = [s for s in dv["codigo_articulo"].unique() if sku_search.lower() in s.lower()][:40] if sku_search else top_skus
            sku_sel = st.selectbox("SKU", sku_opts, index=0 if sku_opts else None, key="val_sku")

        with col_c:
            if sku_sel:
                ds = dv[dv["codigo_articulo"] == sku_sel].sort_values("semana_anio")
                info = ds.iloc[0]
                desc_txt = desc_map.get(sku_sel, "")
                st.markdown(f"**{sku_sel}** {desc_txt} · ABC `{info['tipo_abc']}` · "
                            f"S-B `{info['sb_class']}` · Confianza `{info['confianza']}`")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(ds["semana_anio"]) + list(ds["semana_anio"][::-1]),
                    y=list(ds["pred_p90"]) + list(ds["pred_p10"].iloc[::-1]),
                    fill="toself", fillcolor="rgba(59,130,246,0.10)",
                    line=dict(width=0), name="Intervalo P10–P90"))
                fig.add_trace(go.Scatter(x=ds["semana_anio"], y=ds["real"],
                    mode="lines+markers", name="Real",
                    line=dict(color="#1A1A1A", width=2.5), marker=dict(size=5)))
                fig.add_trace(go.Scatter(x=ds["semana_anio"], y=ds["pred"],
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#E8491D", width=2, dash="dot"), marker=dict(size=5, symbol="diamond")))
                fig.update_layout(title=f"Validación H12 — {sku_sel}",
                    xaxis_title="Semana 2024", yaxis_title="Unidades (acum. 12 sem)",
                    template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

        # Aggregate
        st.markdown('<div class="sec-title">Demanda agregada semanal</div>', unsafe_allow_html=True)
        agg = dv.groupby("semana_anio")[["real", "pred"]].sum().reset_index()
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(x=agg["semana_anio"], y=agg["real"], name="Real", marker_color="#1A1A1A", opacity=0.8))
        fig_a.add_trace(go.Scatter(x=agg["semana_anio"], y=agg["pred"], name="Forecast",
                                   mode="lines+markers", line=dict(color="#E8491D", width=3)))
        fig_a.update_layout(template="plotly_white", height=330, barmode="overlay",
                            xaxis_title="Semana", yaxis_title="Unidades H12",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_a, use_container_width=True)

    else:
        # ── PREDICTION MODE ──
        total_pred_sem = dfp["pred_semana"].sum()
        total_h12 = dfp["pred_h12_rolling"].sum()
        fr_medio = dfp["fill_rate_fcast"].mean()
        n_rojo = (dfp["fr_riesgo"] == "Rojo").sum()
        n_skus_p = dfp["codigo_articulo"].nunique()
        # Monetary: pred × precio
        ventas_forecast = (dfp["pred_semana"] * dfp["precio_unit"]).sum()

        kfr = "green" if fr_medio > 0.90 else ("amber" if fr_medio > 0.70 else "red")
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi blue"><div class="val">{total_pred_sem:,.0f}</div><div class="lbl">Uds. forecast S28-S39</div></div>
            <div class="kpi"><div class="val">{total_h12:,.0f}</div><div class="lbl">H12 rolling total</div></div>
            <div class="kpi {kfr}"><div class="val">{fr_medio:.1%}</div><div class="lbl">Fill Rate forecast</div></div>
            <div class="kpi blue"><div class="val">{ventas_forecast/1e6:.1f}M€</div><div class="lbl">Ventas forecast (€)</div></div>
            <div class="kpi red"><div class="val">{n_rojo:,}</div><div class="lbl">🔴 SKUs en riesgo</div></div>
        </div>
        """, unsafe_allow_html=True)

        # SKU predicción
        col_c2, col_s2 = st.columns([3, 1])
        with col_s2:
            sku_search2 = st.text_input("🔍 Buscar SKU", placeholder="Ej: 001-106", key="pred_search")
            top_pred = dfp.groupby("codigo_articulo")["pred_semana"].sum().nlargest(30).index.tolist()
            sku_opts2 = [s for s in dfp["codigo_articulo"].unique() if sku_search2.lower() in s.lower()][:40] if sku_search2 else top_pred
            sku_sel2 = st.selectbox("SKU", sku_opts2, index=0 if sku_opts2 else None, key="pred_sku")

        with col_c2:
            if sku_sel2:
                # Historical (validation) + future (prediction)
                ds_hist = dv[dv["codigo_articulo"] == sku_sel2].sort_values("semana_anio")
                ds_fut = dfp[dfp["codigo_articulo"] == sku_sel2].sort_values("semana_anio")
                desc_txt = desc_map.get(sku_sel2, "")
                info_f = ds_fut.iloc[0] if not ds_fut.empty else ds_hist.iloc[0]
                st.markdown(f"**{sku_sel2}** {desc_txt} · ABC `{info_f['tipo_abc']}` · "
                            f"S-B `{info_f['sb_class']}` · FR Forecast `{ds_fut['fill_rate_fcast'].mean():.1%}`"
                            if not ds_fut.empty else f"**{sku_sel2}**")

                fig2 = go.Figure()
                # Historical real
                if not ds_hist.empty:
                    fig2.add_trace(go.Scatter(x=ds_hist["semana_anio"], y=ds_hist["real"],
                        mode="lines+markers", name="Real (histórico)",
                        line=dict(color="#1A1A1A", width=2), marker=dict(size=4)))
                    fig2.add_trace(go.Scatter(x=ds_hist["semana_anio"], y=ds_hist["pred"],
                        mode="lines", name="Forecast (validación)",
                        line=dict(color="#9CA3AF", width=1.5, dash="dot")))
                # Future prediction — use H12 rolling for comparable scale with validation
                if not ds_fut.empty:
                    fig2.add_trace(go.Scatter(x=ds_fut["semana_anio"], y=ds_fut["pred_h12_rolling"],
                        mode="lines+markers", name="Forecast H12 (predicción)",
                        line=dict(color="#E8491D", width=2.5), marker=dict(size=6, symbol="star")))
                    # Vertical line separating validation/prediction
                    fig2.add_vline(x=27.5, line_dash="dash", line_color="#D1D5DB",
                                  annotation_text="← Validación | Predicción →")

                fig2.update_layout(title=f"Histórico + Forecast — {sku_sel2}",
                    xaxis_title="Semana 2024", yaxis_title="Unidades",
                    template="plotly_white", height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig2, use_container_width=True)

        # Aggregate prediction
        st.markdown('<div class="sec-title">Forecast semanal agregado (S28-S39)</div>', unsafe_allow_html=True)
        agg_f = dfp.groupby("semana_anio").agg(
            uds=("pred_semana", "sum"),
            ventas=("pred_semana", lambda x: (x * dfp.loc[x.index, "precio_unit"]).sum()),
        ).reset_index()
        fig_af = go.Figure()
        fig_af.add_trace(go.Bar(x=agg_f["semana_anio"], y=agg_f["uds"], name="Uds forecast",
                                marker_color="#E8491D", opacity=0.85))
        fig_af.update_layout(template="plotly_white", height=300,
                             xaxis_title="Semana", yaxis_title="Unidades forecast")
        st.plotly_chart(fig_af, use_container_width=True)

        # Risk distribution
        st.markdown('<div class="sec-title">Distribución de riesgo FR</div>', unsafe_allow_html=True)
        risk_dist = dfp["fr_riesgo"].value_counts().reset_index()
        risk_dist.columns = ["Riesgo", "SKU×Semana"]
        fig_risk = px.pie(risk_dist, values="SKU×Semana", names="Riesgo",
                          color="Riesgo", color_discrete_map={"Verde": "#10B981", "Amarillo": "#F59E0B", "Rojo": "#EF4444"})
        fig_risk.update_layout(height=300)
        st.plotly_chart(fig_risk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# STOCK & ROI
# ═══════════════════════════════════════════════════════════════════════════
if "🚨 Stock & ROI" in tab_map:
    with tab_map["🚨 Stock & ROI"]:
        st.markdown('<div class="sec-title">Optimización de Stock & ROI — LightGBM + Mondrian + Newsvendor</div>',
                    unsafe_allow_html=True)

        if dro.empty:
            st.warning("No hay datos OOS para los filtros seleccionados.")
        else:
            dem_t = dro["dem_real_prov_acum"].sum()
            fr_a = dro["servido_polC"].sum() / max(dem_t, 1)
            fr_d = dro["servido_v3"].sum() / max(dem_t, 1)
            mpp = (fr_d - fr_a) * 100
            rl = dro["reduccion_lost_eur"].sum()
            ie = dro["inv_extra_eur"].sum()
            roi_g = rl / max(ie, 1) * 100
            pb = 12 / max(roi_g / 100, 0.01)
            la = dro["lost_sales_polC"].sum()
            ld = dro["lost_sales_v3"].sum()

            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi red"><div class="val">{fr_a:.1%}</div><div class="lbl">FR actual (PolC)</div></div>
                <div class="kpi green"><div class="val">{fr_d:.1%}</div><div class="lbl">FR optimizado (v3)</div></div>
                <div class="kpi green"><div class="val">{mpp:+.1f}pp</div><div class="lbl">Mejora</div></div>
                <div class="kpi green"><div class="val">{roi_g:.0f}%</div><div class="lbl">ROI</div></div>
                <div class="kpi blue"><div class="val">{pb:.1f}m</div><div class="lbl">Payback</div></div>
            </div>
            <div class="kpi-row">
                <div class="kpi red"><div class="val">{la/1e6:.1f}M€</div><div class="lbl">Ventas perdidas (PolC)</div></div>
                <div class="kpi amber"><div class="val">{ld/1e6:.1f}M€</div><div class="lbl">Ventas perdidas (v3)</div></div>
                <div class="kpi green"><div class="val">{rl/1e6:.1f}M€</div><div class="lbl">Reducción</div></div>
                <div class="kpi"><div class="val">{ie/1e6:.1f}M€</div><div class="lbl">Inversión extra</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Table
            df_alert = dro.nsmallest(60, "fr_v3_servido").copy()
            df_alert["🚦"] = df_alert["semaforo"].map({"rojo": "🔴", "amarillo": "🟡", "verde": "🟢"})
            cols = ["🚦", "codigo_articulo", "sb_class", "Provincia", "Autonomia",
                    "dem_real_prov_acum", "stock_obj_final", "fr_polC_servido", "fr_v3_servido",
                    "mejora_pp_servido", "lost_sales_polC", "lost_sales_v3", "roi_pct_sku"]
            rn = {"codigo_articulo": "SKU", "sb_class": "S-B", "Autonomia": "CCAA",
                  "dem_real_prov_acum": "Dem.Real", "stock_obj_final": "Stock Obj.",
                  "fr_polC_servido": "FR Antes", "fr_v3_servido": "FR Después",
                  "mejora_pp_servido": "Mejora pp", "lost_sales_polC": "Lost€ Antes",
                  "lost_sales_v3": "Lost€ Después", "roi_pct_sku": "ROI%"}
            st.dataframe(df_alert[cols].rename(columns=rn).style.format({
                "Dem.Real": "{:.0f}", "Stock Obj.": "{:.0f}",
                "FR Antes": "{:.1%}", "FR Después": "{:.1%}", "Mejora pp": "{:+.1f}",
                "Lost€ Antes": "{:,.0f}", "Lost€ Después": "{:,.0f}", "ROI%": "{:.0f}"}),
                height=600, use_container_width=True)

            # FR by CCAA
            st.markdown('<div class="sec-title">Fill Rate por CCAA: Antes vs Después</div>',
                        unsafe_allow_html=True)
            fr_cc = dro.groupby("Autonomia").agg(d=("dem_real_prov_acum", "sum"),
                sa=("servido_polC", "sum"), sd=("servido_v3", "sum")).reset_index()
            fr_cc["FR Antes"] = fr_cc["sa"] / fr_cc["d"].clip(lower=1)
            fr_cc["FR Después"] = fr_cc["sd"] / fr_cc["d"].clip(lower=1)
            fr_cc = fr_cc.sort_values("FR Después")
            fig_cc = go.Figure()
            fig_cc.add_trace(go.Bar(y=fr_cc["Autonomia"], x=fr_cc["FR Antes"],
                name="FR Antes", orientation="h", marker_color="#EF4444", opacity=0.6))
            fig_cc.add_trace(go.Bar(y=fr_cc["Autonomia"], x=fr_cc["FR Después"],
                name="FR Después", orientation="h", marker_color="#10B981", opacity=0.85))
            fig_cc.update_layout(template="plotly_white", height=450, barmode="overlay",
                xaxis=dict(tickformat=".0%", range=[0, 1]), margin=dict(l=140),
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_cc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════════════
with tab_map["📊 Rendimiento"]:
    st.markdown('<div class="sec-title">Rendimiento del modelo por segmento</div>', unsafe_allow_html=True)

    col_sb, col_abc = st.columns(2)
    with col_sb:
        wsb = dv.groupby("sb_class").apply(
            lambda g: g["error_abs"].sum() / max(g["real"].sum(), 1) * 100, include_groups=False
        ).reset_index(name="WMAPE")
        fig_sb = px.bar(wsb, x="sb_class", y="WMAPE", color="sb_class",
            color_discrete_map={"Smooth": "#10B981", "Erratic": "#F59E0B",
                "Intermittent": "#3B82F6", "Lumpy": "#EF4444", "Cold_Start": "#9CA3AF",
                "Dying": "#6B7280", "Ultra_Lumpy": "#DC2626"},
            text=wsb["WMAPE"].round(1).astype(str) + "%")
        fig_sb.update_layout(template="plotly_white", height=350, showlegend=False,
                             xaxis_title="", yaxis_title="WMAPE %")
        fig_sb.update_traces(textposition="outside")
        st.plotly_chart(fig_sb, use_container_width=True)

    with col_abc:
        wabc = dv.groupby("tipo_abc").apply(
            lambda g: g["error_abs"].sum() / max(g["real"].sum(), 1) * 100, include_groups=False
        ).reset_index(name="WMAPE")
        fig_abc = px.bar(wabc, x="tipo_abc", y="WMAPE", color="tipo_abc",
            color_discrete_map={"A": "#1A1A1A", "B": "#3B82F6", "C": "#9CA3AF"},
            text=wabc["WMAPE"].round(1).astype(str) + "%")
        fig_abc.update_layout(template="plotly_white", height=350, showlegend=False,
                              xaxis_title="", yaxis_title="WMAPE %")
        fig_abc.update_traces(textposition="outside")
        st.plotly_chart(fig_abc, use_container_width=True)

    # Scatter
    st.markdown('<div class="sec-title">Dispersión: Real vs Forecast</div>', unsafe_allow_html=True)
    dp = dv[dv["real"] > 0]
    sample = dp.sample(min(3000, len(dp)), random_state=42)
    fig_sc = px.scatter(sample, x="real", y="pred", color="sb_class", opacity=0.4,
        hover_data=["codigo_articulo", "tipo_abc", "semana_anio"],
        labels={"real": "Real (uds)", "pred": "Forecast (uds)"})
    mx = max(sample["real"].max(), sample["pred"].max()) * 1.05
    fig_sc.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
        line=dict(color="#D1D5DB", dash="dash", width=1), showlegend=False))
    fig_sc.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# WHAT-IF
# ═══════════════════════════════════════════════════════════════════════════
if "🔬 What-If" in tab_map:
    with tab_map["🔬 What-If"]:
        st.markdown('<div class="sec-title">Simulador de escenarios What-If</div>', unsafe_allow_html=True)
        st.caption("Elasticidades basadas en el ANCOVA (descuento) y correlación clima-ventas (temperatura).")

        col_p, col_g = st.columns([1, 2])
        with col_p:
            sim_sku = st.selectbox("SKU",
                dv[dv["real"] > 0]["codigo_articulo"].unique()[:100], key="wif_sku")
            df_sim = dv[dv["codigo_articulo"] == sim_sku].sort_values("semana_anio").copy()

            st.markdown("**Clima**")
            td = st.slider("Δ Temperatura (°C)", -10.0, 10.0, 0.0, 0.5)
            et = st.slider("Elasticidad temp", 0.0, 5.0, 2.0, 0.25)
            st.markdown("**Descuento**")
            dp_s = st.slider("Descuento adicional (%)", 0, 30, 0, 1)
            ed = st.slider("Elasticidad descuento", 0.0, 3.0, 0.8, 0.1)

        with col_g:
            if not df_sim.empty:
                factor = (1 + td * et / 100) * (1 + dp_s * ed / 100)
                df_sim["pred_wif"] = df_sim["pred"] * factor
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(x=df_sim["semana_anio"], y=df_sim["real"],
                    mode="lines+markers", name="Real", line=dict(color="#1A1A1A", width=2.5)))
                fig_w.add_trace(go.Scatter(x=df_sim["semana_anio"], y=df_sim["pred"],
                    mode="lines", name="Forecast original", line=dict(color="#9CA3AF", width=1.5, dash="dot")))
                fig_w.add_trace(go.Scatter(x=df_sim["semana_anio"], y=df_sim["pred_wif"],
                    mode="lines+markers", name="Forecast What-If",
                    line=dict(color="#E8491D", width=2.5), marker=dict(size=5, symbol="star")))
                fig_w.update_layout(title="Escenario What-If", template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_w, use_container_width=True)

                du = df_sim["pred_wif"].sum() - df_sim["pred"].sum()
                dpct = du / max(df_sim["pred"].sum(), 1) * 100
                ci1, ci2, ci3 = st.columns(3)
                with ci1: st.metric("Forecast original", f"{df_sim['pred'].sum():,.0f} uds")
                with ci2: st.metric("Forecast What-If", f"{df_sim['pred_wif'].sum():,.0f} uds", delta=f"{dpct:+.1f}%")
                with ci3: st.metric("Δ Unidades", f"{du:+,.0f}", delta=f"T:{td:+.1f}°C · D:{dp_s}%")


# ═══════════════════════════════════════════════════════════════════════════
# MAPA
# ═══════════════════════════════════════════════════════════════════════════
with tab_map["🗺️ Mapa"]:
    st.markdown('<div class="sec-title">Mapa geográfico</div>', unsafe_allow_html=True)

    if dro.empty:
        st.warning("No hay datos OOS para los filtros seleccionados.")
    else:
        COORDS = {
            "MADRID": (40.42, -3.70), "BARCELONA": (41.39, 2.17),
            "VALENCIA": (39.47, -0.38), "SEVILLA": (37.39, -5.99),
            "ZARAGOZA": (41.65, -0.88), "MALAGA": (36.72, -4.42),
            "MURCIA": (37.98, -1.13), "ALICANTE": (38.35, -0.49),
            "CADIZ": (36.53, -6.29), "VIZCAYA": (43.26, -2.92),
            "ASTURIAS": (43.36, -5.85), "CORDOBA": (37.88, -4.77),
            "GRANADA": (37.18, -3.60), "GUIPUZCOA": (43.32, -1.98),
            "NAVARRA": (42.82, -1.64), "PONTEVEDRA": (42.43, -8.64),
            "TOLEDO": (39.86, -4.02), "CANTABRIA": (43.46, -3.80),
            "BURGOS": (42.34, -3.70), "ALMERIA": (36.83, -2.46),
            "GIRONA": (41.98, 2.82), "TARRAGONA": (41.12, 1.25),
            "JAEN": (37.77, -3.79), "LEON": (42.60, -5.57),
            "CASTELLON": (39.98, -0.05), "HUELVA": (37.26, -6.95),
            "BADAJOZ": (38.88, -6.97), "CACERES": (39.47, -6.37),
            "LLEIDA": (41.62, 0.63), "A CORUÑA": (43.37, -8.40),
            "LUGO": (43.01, -7.56), "OURENSE": (42.34, -7.86),
            "SALAMANCA": (40.97, -5.66), "VALLADOLID": (41.65, -4.72),
            "PALENCIA": (42.01, -4.53), "AVILA": (40.66, -4.70),
            "SEGOVIA": (40.95, -4.12), "ZAMORA": (41.50, -5.74),
            "SORIA": (41.76, -2.47), "TERUEL": (40.35, -1.11),
            "HUESCA": (42.14, -0.41), "GUADALAJARA": (40.63, -3.17),
            "CUENCA": (40.07, -2.13), "CIUDAD REAL": (38.99, -3.93),
            "ALBACETE": (38.99, -1.86), "LA RIOJA": (42.29, -2.52),
            "ILLES BALEARS": (39.57, 2.65), "LAS PALMAS": (28.10, -15.42),
            "STA CRUZ DE TENERIFE": (28.47, -16.25),
            "CEUTA": (35.89, -5.32), "MELILLA": (35.29, -2.94),
            "ALAVA": (42.85, -2.67),
        }
        prov = dro.groupby("Provincia").agg(
            dem=("dem_real_prov_acum", "sum"), sa=("servido_polC", "sum"),
            sd=("servido_v3", "sum"), skus=("codigo_articulo", "nunique"),
            rl=("reduccion_lost_eur", "sum"), ie=("inv_extra_eur", "sum")).reset_index()
        prov["FR Antes"] = prov["sa"] / prov["dem"].clip(lower=1)
        prov["FR Después"] = prov["sd"] / prov["dem"].clip(lower=1)
        prov["ROI"] = prov["rl"] / prov["ie"].clip(lower=1) * 100
        prov["lat"] = prov["Provincia"].map(lambda p: COORDS.get(p, (40, -3))[0])
        prov["lon"] = prov["Provincia"].map(lambda p: COORDS.get(p, (40, -3))[1])

        metric = st.radio("Métrica", ["Demanda", "FR Después", "ROI %", "Reducción Lost €"], horizontal=True)
        cm = {"Demanda": "dem", "FR Después": "FR Después", "ROI %": "ROI", "Reducción Lost €": "rl"}[metric]
        cs = "RdYlGn" if "FR" in metric else ("Greens" if "ROI" in metric else ("YlOrRd" if "Lost" in metric else "Blues"))
        _sc = getattr(px, "scatter_map", None) or getattr(px, "scatter_mapbox", None)
        _sk = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"
        fig_m = _sc(prov, lat="lat", lon="lon", size="dem", color=cm, hover_name="Provincia",
            hover_data={"dem": ":,.0f", "skus": True, "FR Antes": ":.1%", "FR Después": ":.1%",
                        "rl": ":,.0f", "ROI": ":.0f", "lat": False, "lon": False},
            color_continuous_scale=cs, size_max=40, zoom=4.5, center={"lat": 40, "lon": -3.5})
        fig_m.update_layout(**{_sk: "carto-positron"}, height=530, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_m, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ASISTENTE IA
# ═══════════════════════════════════════════════════════════════════════════
with tab_map["🤖 Asistente IA"]:
    st.markdown('<div class="sec-title">Asistente IA — Consulta tus datos</div>', unsafe_allow_html=True)

    api_key = st.secrets.get("COHERE_API_KEY", "")
    if not api_key:
        st.warning('⚠️ Configura `COHERE_API_KEY` en Settings → Secrets.')
    else:
        # Build context
        t_r = dv["real"].sum()
        t_p = dv["pred"].sum()
        wm = dv["error_abs"].sum() / max(t_r, 1) * 100

        ctx = f"""Datos Cruzber (fabricante de portaequipajes, 15M€ facturación):

FORECAST VALIDACIÓN (S1-S27): {dv['codigo_articulo'].nunique()} SKUs, WMAPE={wm:.1f}%, Uds reales={t_r:,.0f}, Forecast={t_p:,.0f}, Ventas en riesgo={dv['ventas_riesgo_eur'].sum():,.0f}€

FORECAST PREDICTIVO (S28-S39): {dfp['codigo_articulo'].nunique()} SKUs, Uds forecast={dfp['pred_semana'].sum():,.0f}, FR medio={dfp['fill_rate_fcast'].mean():.1%}, SKUs riesgo rojo={(dfp['fr_riesgo']=='Rojo').sum()}"""

        if not dro.empty:
            dm = dro["dem_real_prov_acum"].sum()
            ctx += f"""

STOCK & ROI: FR antes={dro['servido_polC'].sum()/max(dm,1):.1%}, FR después={dro['servido_v3'].sum()/max(dm,1):.1%}, Lost sales antes={dro['lost_sales_polC'].sum():,.0f}€, Lost sales después={dro['lost_sales_v3'].sum():,.0f}€, Reducción={dro['reduccion_lost_eur'].sum():,.0f}€, Inv extra={dro['inv_extra_eur'].sum():,.0f}€, ROI={dro['reduccion_lost_eur'].sum()/max(dro['inv_extra_eur'].sum(),1)*100:.0f}%"""

        SYS = f"""Eres un analista de datos experto de Cruzber. Responde en español, conciso, con datos concretos. Solo usa los datos proporcionados.

{ctx}

Rol del usuario: {role}"""

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Pregunta sobre tus datos..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analizando…"):
                    try:
                        client = cohere.ClientV2(api_key=api_key)
                        # Try models in order of preference
                        model_options = ["command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024"]
                        answer = None
                        for model_name in model_options:
                            try:
                                response = client.chat(
                                    model=model_name,
                                    messages=[{"role": "system", "content": SYS},
                                        *[{"role": m["role"], "content": m["content"]}
                                          for m in st.session_state.messages]],
                                    temperature=0.3, max_tokens=1500)
                                answer = response.message.content[0].text
                                break
                            except Exception:
                                continue
                        if answer:
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.error("No se pudo conectar con ningún modelo de Cohere. Verifica tu API key.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if not st.session_state.messages:
            st.markdown("---")
            st.markdown("**Ejemplos:**")
            for ex in ["¿Cuál es el ROI del proyecto?",
                       "¿Qué SKUs tienen peor fill rate?",
                       "Resumen ejecutivo para dirección",
                       "¿Qué provincia tiene más ventas en riesgo?"]:
                st.markdown(f"  → *{ex}*")
