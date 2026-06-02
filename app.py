# -*- coding: utf-8 -*-
"""Dynamic Pricing Analyzer v2 — Streamlit App | OfficeMax México"""

import io, zipfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
OM_RED   = "#E31837"
OM_BLACK = "#1A1A1A"
OM_BLUE  = "#1565C0"
OM_GREEN = "#2E7D32"
OM_AMBER = "#F9A825"
OM_LGRAY = "#9E9E9E"

REC_COLORS = {
    "Subir precio":     OM_BLUE,
    "Mantener precio":  OM_AMBER,
    "Bajar / Promover": OM_GREEN,
    "No recomendable":  OM_LGRAY,
}

KEYWORDS_PREMIUM = [
    "PREMIUM","PRO ","PROFESIONAL","EXECUTIVE","EJECUTIV",
    "DELUXE","ELITE","PLUS","GOLD","CARBON","PIEL","CUERO","ERGON",
]

MONTH_NAMES = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dynamic Pricing | OfficeMax",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Roboto', Arial, sans-serif !important; }
.stApp { background-color: #F5F5F5 !important; }
.main .block-container {
    max-width: 1300px !important;
    padding-top: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
[data-testid="stSidebar"] { background-color: #1A1A1A !important; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { font-size: 13px !important; }
.kpi-card {
    background: white; border-radius: 12px; padding: 18px 12px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 4px solid #E31837;
}
.kpi-value { font-size: 26px; font-weight: 900; color: #1A1A1A; }
.kpi-label { font-size: 11px; color: #888; text-transform: uppercase;
             letter-spacing: 0.5px; margin-top: 4px; }
.section-header {
    background: linear-gradient(135deg, #E31837 0%, #C41430 100%);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-weight: 700; font-size: 15px; margin: 24px 0 10px 0;
}
.clean-step {
    padding: 8px 12px; border-left: 4px solid #E31837;
    margin: 5px 0; background: white; border-radius: 0 6px 6px 0; font-size: 13px;
}
.chip-green  { background:#2E7D32; color:white; padding:6px 18px; border-radius:20px;
               font-weight:700; font-size:14px; display:inline-block; }
.chip-yellow { background:#F9A825; color:white; padding:6px 18px; border-radius:20px;
               font-weight:700; font-size:14px; display:inline-block; }
.chip-red    { background:#E31837; color:white; padding:6px 18px; border-radius:20px;
               font-weight:700; font-size:14px; display:inline-block; }
.rec-card {
    border-radius: 12px; padding: 18px; text-align: center;
    background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.stTabs [data-baseweb="tab"] { padding: 10px 22px; font-weight: 600; }
.stTabs [aria-selected="true"] { border-bottom: 3px solid #E31837; color: #E31837; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k in ["df_main", "clean_report", "results"]:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _identify(dfs: dict) -> dict:
    out = {}
    for _, df in dfs.items():
        cols = set(df.columns)
        if "apparent_unit_cost" in cols:
            out["costos"] = df
        elif "Precio_Unitario" in cols:
            out["precios"] = df
        elif "tran_nbr" in cols:
            out["tickets"] = df
        elif "tran_date" in cols and "venta_con_iva" in cols:
            out["ventas"] = df
        elif "prod_nm" in cols and "tipo_marca" in cols:
            out["catalogo"] = df
    return out


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes):
    report = []

    raw = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    raw[Path(name).stem.lower()] = pd.read_csv(
                        f, encoding="latin-1", low_memory=False
                    )

    files = _identify(raw)
    missing = [k for k in ("ventas", "catalogo", "precios", "costos") if k not in files]
    if missing:
        raise ValueError(f"Archivos faltantes en el ZIP: {', '.join(missing)}")

    ventas   = files["ventas"].copy()
    catalogo = files["catalogo"].copy()
    precios  = files["precios"].copy()
    costos   = files["costos"].copy()

    for df in (ventas, catalogo, precios, costos):
        df["prod_nbr"] = df["prod_nbr"].astype(str).str.strip()

    n0 = len(ventas)
    report.append({"Paso": "📥 Ventas cargadas", "Eliminadas": "—",
                   "Detalle": f"{n0:,} filas originales"})

    ventas["tran_date"]     = pd.to_datetime(ventas["tran_date"],    errors="coerce")
    ventas["qty"]           = pd.to_numeric(ventas["qty"],           errors="coerce")
    ventas["venta_con_iva"] = pd.to_numeric(ventas["venta_con_iva"], errors="coerce")
    ventas["costo"]         = pd.to_numeric(ventas["costo"],         errors="coerce")
    ventas["margen"]        = pd.to_numeric(ventas["margen"],        errors="coerce")

    mask = ventas["tran_date"].isna() | ventas["qty"].isna() | ventas["venta_con_iva"].isna()
    n_rm = int(mask.sum())
    ventas = ventas[~mask].copy()
    if n_rm:
        report.append({"Paso": "🗑 Campos clave nulos", "Eliminadas": f"{n_rm:,}",
                       "Detalle": "tran_date, qty o venta_con_iva vacíos"})

    mask = (ventas["qty"] <= 0) | (ventas["venta_con_iva"] <= 0)
    n_rm = int(mask.sum())
    ventas = ventas[~mask].copy()
    if n_rm:
        report.append({"Paso": "🗑 Qty / Venta ≤ 0", "Eliminadas": f"{n_rm:,}",
                       "Detalle": "Devoluciones, ajustes o errores de captura"})

    n_rm = int(ventas.duplicated().sum())
    ventas = ventas.drop_duplicates()
    if n_rm:
        report.append({"Paso": "🗑 Duplicados exactos", "Eliminadas": f"{n_rm:,}",
                       "Detalle": "Filas 100% idénticas eliminadas"})

    ventas["precio_tx"] = ventas["venta_con_iva"] / ventas["qty"]
    p99 = ventas.groupby("dept_cd")["precio_tx"].transform(lambda x: x.quantile(0.99))
    mask = ventas["precio_tx"] > p99
    n_rm = int(mask.sum())
    ventas = ventas[~mask].copy()
    if n_rm:
        report.append({"Paso": "🗑 Precios outlier (>p99 por depto)", "Eliminadas": f"{n_rm:,}",
                       "Detalle": "Precio unitario > percentil 99 del departamento"})

    report.append({"Paso": "✅ Ventas limpias", "Eliminadas": "—",
                   "Detalle": f"{len(ventas):,} filas válidas quedan"})

    cat_slim = (catalogo[["prod_nbr","prod_nm","dept_nm","subdept_nm","class_nm",
                           "marca_fabricante","tipo_marca"]]
                .drop_duplicates("prod_nbr"))
    prec_slim = (precios[["prod_nbr","Precio_Unitario"]]
                 .rename(columns={"Precio_Unitario": "precio_catalogo"}))
    cost_slim = (costos[["prod_nbr","apparent_unit_cost"]]
                 .rename(columns={"apparent_unit_cost": "costo_unitario"}))

    df = (ventas
          .merge(cat_slim,  on="prod_nbr", how="left", suffixes=("", "_cat"))
          .merge(prec_slim, on="prod_nbr", how="left")
          .merge(cost_slim, on="prod_nbr", how="left"))

    df["mes_str"]        = df["tran_date"].dt.to_period("M").astype(str)
    df["año"]            = df["tran_date"].dt.year
    df["mes_calendario"] = df["tran_date"].dt.month
    df["es_premium"]     = df["prod_nm"].apply(
        lambda n: int(any(k in str(n).upper() for k in KEYWORDS_PREMIUM))
        if pd.notna(n) else 0
    )

    report.append({"Paso": "🔗 Merge final", "Eliminadas": "—",
                   "Detalle": (f"{len(df):,} filas · "
                               f"{df['prod_nbr'].nunique():,} SKUs · "
                               f"{df['store_nbr'].nunique()} tiendas · "
                               f"{df['mes_str'].nunique()} meses")})

    return df, pd.DataFrame(report)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ols_loglog(subset, min_obs, min_cv, min_r2, max_beta):
    n = len(subset)
    if n < min_obs:
        return None
    std_p  = subset["log_p"].std()
    if std_p < 1e-6:
        return None
    mean_p = abs(subset["log_p"].mean())
    if mean_p > 1e-9 and (std_p / mean_p) < min_cv:
        return None
    try:
        y   = subset["log_u1"].values
        X   = add_constant(subset["log_p"].values)
        res = OLS(y, X).fit()
        beta = float(res.params[1])
        r2   = float(res.rsquared)
        if r2 < min_r2 or abs(beta) > max_beta:
            return None
        rmse = float(np.sqrt(np.mean((y - res.fittedvalues) ** 2)))
        return {
            "alpha": round(float(res.params[0]), 4),
            "beta":  round(beta, 4),
            "r2":    round(r2, 4),
            "rmse":  round(rmse, 4),
            "n":     n,
            "pval":  round(float(res.pvalues[1]), 4),
        }
    except Exception:
        return None


def _build_monthly(df):
    df2 = df.copy()
    df2["mes"] = pd.PeriodIndex(df2["mes_str"], freq="M")
    mensual = (
        df2.groupby(["prod_nbr", "mes"])
        .agg(
            unidades   = ("qty",           "sum"),
            venta_tot  = ("venta_con_iva", "sum"),
            es_premium = ("es_premium",    "max"),
            prod_nm    = ("prod_nm",       "first"),
            subdept_nm = ("subdept_nm",    "first"),
            dept_nm    = ("dept_nm",       "first"),
            margen_avg = ("margen",        "mean"),
            costo_unit = ("costo_unitario","mean"),
        )
        .reset_index()
    )
    mensual["precio"] = mensual["venta_tot"] / mensual["unidades"]
    mensual = mensual[mensual["precio"] > 0].copy()
    mensual["log_u1"] = np.log1p(mensual["unidades"])
    mensual["log_p"]  = np.log(mensual["precio"])
    mensual["mes_dt"] = mensual["mes"].dt.to_timestamp()
    return mensual.sort_values(["prod_nbr", "mes"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_models(df_csv: bytes, min_obs: int, min_cv: float, min_r2: float,
               max_beta: float, pval_thresh: float, run_rolling: bool):
    df = pd.read_csv(io.BytesIO(df_csv), low_memory=False)
    df["tran_date"] = pd.to_datetime(df["tran_date"], errors="coerce")

    mensual = _build_monthly(df)
    todos_meses = sorted(mensual["mes"].unique())
    params = dict(min_obs=min_obs, min_cv=min_cv, min_r2=min_r2, max_beta=max_beta)

    # M1A — todos los meses por SKU
    rows_m1a = []
    for sku, grp in mensual.groupby("prod_nbr"):
        r = _ols_loglog(grp, **params)
        if r:
            rows_m1a.append({
                "prod_nbr":   sku,
                "prod_nm":    grp["prod_nm"].iloc[0],
                "subdept_nm": grp["subdept_nm"].iloc[0],
                "dept_nm":    grp["dept_nm"].iloc[0],
                "n_meses":    len(grp),
                **r,
            })
    df_m1a = pd.DataFrame(rows_m1a)

    # M1B — rolling 3 meses
    df_m1b = pd.DataFrame()
    df_m1c = pd.DataFrame()

    if run_rolling and len(todos_meses) >= 3:
        rows_m1b = []
        for sku, grp in mensual.groupby("prod_nbr"):
            gi = grp.set_index("mes")
            for i in range(len(todos_meses) - 2):
                vent = todos_meses[i: i + 3]
                sub  = gi[gi.index.isin(vent)]
                r    = _ols_loglog(sub, **params)
                if r:
                    rows_m1b.append({
                        "prod_nbr":   sku,
                        "prod_nm":    grp["prod_nm"].iloc[0],
                        "mes_inicio": str(vent[0]),
                        "mes_fin":    str(vent[-1]),
                        "mes_fin_dt": vent[-1].to_timestamp(),
                        **r,
                    })
        df_m1b = pd.DataFrame(rows_m1b)

    if run_rolling and len(todos_meses) >= 6:
        rows_m1c = []
        for sku, grp in mensual.groupby("prod_nbr"):
            gi = grp.set_index("mes")
            for i in range(len(todos_meses) - 5):
                vent = todos_meses[i: i + 6]
                sub  = gi[gi.index.isin(vent)]
                r    = _ols_loglog(sub, **params)
                if r:
                    rows_m1c.append({
                        "prod_nbr":   sku,
                        "prod_nm":    grp["prod_nm"].iloc[0],
                        "mes_inicio": str(vent[0]),
                        "mes_fin":    str(vent[-1]),
                        "mes_fin_dt": vent[-1].to_timestamp(),
                        **r,
                    })
        df_m1c = pd.DataFrame(rows_m1c)

    # M2 — extendido con dummies tienda + mes + premium + margen
    agg2 = (
        df.groupby(["prod_nbr", "store_nbr", "mes_str"])
        .agg(
            unidades   = ("qty",           "sum"),
            venta_tot  = ("venta_con_iva", "sum"),
            es_premium = ("es_premium",    "max"),
            margen     = ("margen",        "mean"),
        )
        .reset_index()
    )
    agg2["precio"] = agg2["venta_tot"] / agg2["unidades"]
    agg2 = agg2[agg2["precio"] > 0].copy()
    agg2["log_u1"] = np.log1p(agg2["unidades"])
    agg2["log_p"]  = np.log(agg2["precio"])

    ohe_s = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    s_enc = ohe_s.fit_transform(agg2[["store_nbr"]])
    s_cols = [f"s_{c}" for c in ohe_s.categories_[0][1:]]

    ohe_m = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    m_enc = ohe_m.fit_transform(agg2[["mes_str"]])
    m_cols = [f"m_{c}" for c in ohe_m.categories_[0][1:]]

    X2 = pd.concat([
        pd.DataFrame({
            "log_precio": agg2["log_p"].values,
            "es_premium": agg2["es_premium"].values,
            "margen":     agg2["margen"].fillna(0).values,
        }, index=agg2.index),
        pd.DataFrame(s_enc, columns=s_cols, index=agg2.index),
        pd.DataFrame(m_enc, columns=m_cols, index=agg2.index),
    ], axis=1)

    X2  = add_constant(X2)
    y2  = agg2["log_u1"].values
    msk = np.isfinite(X2.values).all(axis=1) & np.isfinite(y2)
    mod2 = OLS(y2[msk], X2[msk]).fit()
    rmse2 = float(np.sqrt(np.mean((y2[msk] - mod2.fittedvalues) ** 2)))

    m2 = {
        "n_obs":     int(mod2.nobs),
        "r2":        round(float(mod2.rsquared), 4),
        "r2_adj":    round(float(mod2.rsquared_adj), 4),
        "beta":      round(float(mod2.params["log_precio"]), 4),
        "beta_pval": round(float(mod2.pvalues["log_precio"]), 4),
        "premium":   round(float(mod2.params["es_premium"]), 4),
        "prem_pval": round(float(mod2.pvalues["es_premium"]), 4),
        "rmse":      round(rmse2, 4),
        "f_stat":    round(float(mod2.fvalue), 2),
    }

    # Classify SKUs
    def _classify(row):
        b, p, n = row["beta"], row["pval"], row["n_meses"]
        if p >= pval_thresh or n < min_obs or abs(b) > 3 or b > 0:
            return "No recomendable"
        if -1 < b < 0:
            return "Subir precio"
        if -1.5 <= b <= -1:
            return "Mantener precio"
        return "Bajar / Promover"

    if len(df_m1a) > 0:
        df_m1a["recomendacion"] = df_m1a.apply(_classify, axis=1)

    # Price simulation
    base_stats = mensual.groupby("prod_nbr").agg(
        precio_base   = ("precio",    "mean"),
        unidades_base = ("unidades",  "mean"),
        costo_unit    = ("costo_unit","mean"),
    ).reset_index()

    scenarios = [-0.10, -0.05, 0.00, 0.05, 0.10]
    labels    = ["-10%", "-5%", "Base 0%", "+5%", "+10%"]
    sim_rows  = []

    if len(df_m1a) > 0:
        sim_input = df_m1a.merge(base_stats, on="prod_nbr", how="left")
        for _, row in sim_input.iterrows():
            p0, u0, beta = row.get("precio_base"), row.get("unidades_base"), row["beta"]
            if pd.isna(p0) or pd.isna(u0):
                continue
            c0   = row.get("costo_unit", 0)
            c0   = 0 if pd.isna(c0) else float(c0)
            rev0  = p0 * u0
            marg0 = (p0 - c0) * u0 or 1e-9
            for chg, lbl in zip(scenarios, labels):
                p1    = p0 * (1 + chg)
                u1    = u0 * ((1 + chg) ** beta)
                rev1  = p1 * u1
                marg1 = (p1 - c0) * u1
                sim_rows.append({
                    "prod_nbr":          row["prod_nbr"],
                    "prod_nm":           row["prod_nm"],
                    "beta":              round(beta, 4),
                    "recomendacion":     row.get("recomendacion", ""),
                    "cambio":            lbl,
                    "precio_nuevo":      round(p1, 2),
                    "unidades_est":      round(u1, 1),
                    "ingreso_est":       round(rev1, 2),
                    "margen_est":        round(marg1, 2),
                    "delta_ingreso_pct": round((rev1 - rev0) / rev0 * 100, 1) if rev0 else 0,
                    "delta_margen_pct":  round((marg1 - marg0) / marg0 * 100, 1),
                })

    mensual_out = mensual.copy()
    mensual_out["mes"] = mensual_out["mes"].astype(str)

    return {
        "m1a":     df_m1a,
        "m1b":     df_m1b,
        "m1c":     df_m1c,
        "m2":      m2,
        "sim":     pd.DataFrame(sim_rows),
        "mensual": mensual_out,
        "n_total": mensual["prod_nbr"].nunique(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, color: str = OM_RED):
    col.markdown(
        f'<div class="kpi-card" style="border-top-color:{color};">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def traffic_light(r2, beta, pval, rmse):
    score = 0
    reasons = []
    if r2 >= 0.5:    score += 2; reasons.append(f"✅ R² = {r2:.3f} — bueno (≥ 0.5)")
    elif r2 >= 0.25: score += 1; reasons.append(f"⚠️ R² = {r2:.3f} — aceptable")
    else:                        reasons.append(f"❌ R² = {r2:.3f} — bajo")
    if pval < 0.05:  score += 2; reasons.append(f"✅ p-valor = {pval:.4f} — significativo")
    elif pval < 0.10:score += 1; reasons.append(f"⚠️ p-valor = {pval:.4f} — marginal")
    else:                        reasons.append(f"❌ p-valor = {pval:.4f} — no significativo")
    if beta < 0 and abs(beta) <= 3:
        score += 1; reasons.append(f"✅ Beta = {beta:.4f} — negativa e interpretable")
    else:            reasons.append(f"❌ Beta = {beta:.4f} — fuera de rango")
    if rmse < 0.5:   score += 1; reasons.append(f"✅ RMSE = {rmse:.4f} — error bajo")
    else:            reasons.append(f"⚠️ RMSE = {rmse:.4f} — error moderado")
    if score >= 5:   css, msg = "chip-green",  "🟢 Modelo confiable — apto para decisiones"
    elif score >= 3: css, msg = "chip-yellow", "🟡 Modelo aceptable — úsalo con precaución"
    else:            css, msg = "chip-red",    "🔴 Modelo débil — revisa parámetros o datos"
    return css, msg, reasons


def _layout(fig, h=320):
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=h, margin=dict(t=45, b=20, l=20, r=20),
        font=dict(family="Roboto, Arial"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 10px 0;">
        <span style="font-size:26px; font-weight:900; color:#E31837;">OFFICEMAX</span><br>
        <span style="font-size:10px; color:#aaa; letter-spacing:2px;">DYNAMIC PRICING ANALYZER</span>
    </div>
    <hr style="border-color:#333; margin:6px 0 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### 📦 Cargar datos")
    st.caption("Sube el ZIP con los 4 archivos CSV: ventas, catálogo, precios y costos.")
    uploaded = st.file_uploader("Archivo .ZIP", type=["zip"], label_visibility="collapsed")

    if uploaded:
        raw_bytes = uploaded.read()
        with st.spinner("Leyendo y limpiando..."):
            try:
                df_main, report_df = load_and_clean(raw_bytes)
                st.session_state["df_main"]      = df_main
                st.session_state["clean_report"] = report_df
                st.session_state["results"]      = None
                st.success("✅ Datos cargados")
                st.caption(report_df.iloc[-1]["Detalle"])
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("<hr style='border-color:#333; margin:16px 0;'>", unsafe_allow_html=True)
    st.caption(
        "**Flujo recomendado:**\n\n"
        "1. Sube el ZIP\n"
        "2. **Calculadora** → ajusta parámetros → ejecuta\n"
        "3. **Descriptivo** → explora los datos\n"
        "4. **Predictivo** → analiza resultados"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE (no data)
# ══════════════════════════════════════════════════════════════════════════════
df_main      = st.session_state["df_main"]
clean_report = st.session_state["clean_report"]

if df_main is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px 40px 20px;">
        <div style="font-size:60px;">📊</div>
        <h1 style="font-weight:900; color:#1A1A1A; margin:12px 0 6px 0;">Dynamic Pricing Analyzer</h1>
        <p style="font-size:17px; color:#666; max-width:580px; margin:0 auto 36px auto;">
            Analiza la elasticidad precio de tus productos y obtén recomendaciones de pricing
            basadas en datos reales de ventas OfficeMax.
        </p>
        <div style="background:#fff8f8; border:2px dashed #E31837; border-radius:16px;
                    padding:36px; max-width:460px; margin:0 auto;">
            <div style="font-size:44px;">📦</div>
            <p style="font-weight:700; color:#E31837; font-size:18px; margin:10px 0 6px 0;">
                Sube tu archivo ZIP en el panel izquierdo
            </p>
            <p style="color:#666; font-size:13px; margin:0;">
                El ZIP debe contener los archivos:<br>
                Ventas · Catálogo · Precios · Costos
            </p>
        </div>
        <div style="display:flex; justify-content:center; gap:40px; margin-top:48px; flex-wrap:wrap;">
            <div><div style="font-size:30px;">🧮</div><strong>Calculadora</strong>
                 <p style="font-size:12px; color:#888; margin:4px 0;">Modelo OLS log-log por SKU</p></div>
            <div><div style="font-size:30px;">📈</div><strong>Descriptivo</strong>
                 <p style="font-size:12px; color:#888; margin:4px 0;">KPIs y tendencias de ventas</p></div>
            <div><div style="font-size:30px;">🎯</div><strong>Predictivo</strong>
                 <p style="font-size:12px; color:#888; margin:4px 0;">Simulador y recomendaciones</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🧮  Calculadora",
    "📈  Dashboard Descriptivo",
    "🎯  Dashboard Predictivo",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — CALCULADORA
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    with st.expander("📋 Reporte de limpieza de datos", expanded=False):
        for _, row in clean_report.iterrows():
            c = OM_GREEN if "✅" in str(row["Paso"]) else (OM_RED if "🗑" in str(row["Paso"]) else OM_BLUE)
            st.markdown(
                f'<div class="clean-step" style="border-left-color:{c};">'
                f'<strong>{row["Paso"]}</strong>&nbsp;&nbsp;'
                f'<span style="color:{OM_RED}; font-weight:700;">{row["Eliminadas"]}</span>'
                f'&nbsp;&nbsp;<span style="color:#555;">{row["Detalle"]}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    section("⚙️ Parámetros del modelo de elasticidad")
    st.caption(
        "Ajusta estos valores antes de ejecutar. Determinan qué SKUs son suficientemente "
        "confiables para tomar decisiones de precio."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        min_obs = st.slider("Meses mínimos por SKU", 3, 18, 6,
            help="El modelo necesita al menos N meses de datos por SKU. Con muy pocos, los resultados son poco confiables.")
    with c2:
        min_r2 = st.slider("R² mínimo", 0.0, 0.7, 0.0, 0.05,
            help="R² mide qué tanto explica el modelo (0=nada, 1=perfecto). En 0 no filtras nada.")
    with c3:
        max_beta = st.slider("|Beta| máximo", 2.0, 15.0, 10.0, 0.5,
            help="Beta es la elasticidad. Valores mayores a 10 suelen ser errores estadísticos.")
    with c4:
        min_cv_pct = st.slider("CV mínimo de precio (%)", 0.0, 10.0, 2.0, 0.5,
            help="Si un producto siempre tuvo el mismo precio, no podemos estimar elasticidad.")
        min_cv = min_cv_pct / 100
    with c5:
        pval_thresh = st.slider("p-valor máximo", 0.05, 0.25, 0.10, 0.05,
            help="p-valor máximo para considerar que beta es estadísticamente válida. 0.10 = 90% de confianza.")

    run_rolling = st.checkbox(
        "Calcular análisis rolling trimestral y semestral (más lento ~1-3 min extra)",
        value=False,
        help="Activa para ver cómo cambia la elasticidad a lo largo del tiempo.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶️  Ejecutar análisis de elasticidad", type="primary")

    if run_btn:
        with st.spinner("Calculando modelos de elasticidad... puede tomar hasta 2 minutos."):
            try:
                csv_bytes = df_main.to_csv(index=False).encode("utf-8")
                res = run_models(
                    df_csv=csv_bytes,
                    min_obs=min_obs, min_cv=min_cv, min_r2=min_r2,
                    max_beta=max_beta, pval_thresh=pval_thresh,
                    run_rolling=run_rolling,
                )
                st.session_state["results"] = res
                st.success("✅ Análisis completado")
            except Exception as e:
                st.error(f"Error al ejecutar el modelo: {e}")
                st.exception(e)

    results = st.session_state["results"]

    if results is None:
        st.info("Configura los parámetros y presiona **▶️ Ejecutar análisis** para ver los resultados.")
    else:
        df_m1a = results["m1a"]
        m2     = results["m2"]

        section("🔬 Validación global del modelo — ¿vale la pena el análisis?")
        st.caption(
            "El Modelo 2 usa todos los SKUs y tiendas con controles estadísticos avanzados. "
            "Valida el análisis antes de ver resultados individuales por SKU."
        )

        css, msg, reasons = traffic_light(m2["r2"], m2["beta"], m2["beta_pval"], m2["rmse"])
        col_verd, col_metr = st.columns([1, 2])
        with col_verd:
            st.markdown(f'<div class="{css}">{msg}</div><br>', unsafe_allow_html=True)
            for r in reasons:
                st.caption(r)
        with col_metr:
            st.dataframe(pd.DataFrame({
                "Métrica": ["N observaciones","R²","R² ajustado","Beta precio (M2)",
                            "p-valor beta","Coef. premium","RMSE (log-escala)"],
                "Valor":   [f'{m2["n_obs"]:,}', f'{m2["r2"]:.4f}', f'{m2["r2_adj"]:.4f}',
                            f'{m2["beta"]:.4f}', f'{m2["beta_pval"]:.4f}',
                            f'{m2["premium"]:.4f}', f'{m2["rmse"]:.4f}'],
                "Interpretación": [
                    "Transacciones usadas en la regresión global",
                    "Proporción de varianza en ventas explicada por precio (0-1)",
                    "R² penalizado por número de variables",
                    f'Por cada +1% en precio → {m2["beta"]:.2f}% en unidades',
                    "Significativo si < 0.10",
                    "Premium venden " + ("más" if m2["premium"] > 0 else "menos") + " que no-premium",
                    "Error promedio en log-escala (menor = más preciso)",
                ],
            }), hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        n_valid = len(df_m1a[df_m1a["recomendacion"] != "No recomendable"]) if len(df_m1a) > 0 else 0
        section(
            f"📦 Resultados por SKU — {len(df_m1a):,} analizados de {results['n_total']:,} totales"
            f" · {n_valid} con recomendación válida"
        )

        if len(df_m1a) == 0:
            st.warning("Ningún SKU pasó los filtros. Prueba reducir el mínimo de meses o el CV mínimo.")
        else:
            rec_counts = df_m1a["recomendacion"].value_counts()
            c_rec = st.columns(4)
            for i, (rec, color) in enumerate(REC_COLORS.items()):
                cnt = rec_counts.get(rec, 0)
                pct = cnt / len(df_m1a) * 100
                c_rec[i].markdown(
                    f'<div class="rec-card" style="border-top:4px solid {color};">'
                    f'<div style="font-size:32px; font-weight:900; color:{color};">{cnt}</div>'
                    f'<div style="font-size:12px; font-weight:700; color:#333;">{rec.upper()}</div>'
                    f'<div style="font-size:11px; color:#999;">{pct:.1f}% del total</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                depts = sorted(df_m1a["dept_nm"].dropna().unique().tolist())
                dept_f = st.multiselect("Departamento", depts)
            with fc2:
                rec_f = st.multiselect("Recomendación", list(REC_COLORS.keys()))
            with fc3:
                srch = st.text_input("Buscar por nombre o SKU", placeholder="Ej: FOLDER, 50012983")

            show = df_m1a.copy()
            if dept_f: show = show[show["dept_nm"].isin(dept_f)]
            if rec_f:  show = show[show["recomendacion"].isin(rec_f)]
            if srch:
                show = show[
                    show["prod_nm"].str.contains(srch, case=False, na=False) |
                    show["prod_nbr"].str.contains(srch, case=False, na=False)
                ]

            disp = ["prod_nbr","prod_nm","dept_nm","beta","r2","rmse","pval","n_meses","recomendacion"]
            disp = [c for c in disp if c in show.columns]
            st.dataframe(
                show[disp].rename(columns={
                    "prod_nbr":"SKU","prod_nm":"Producto","dept_nm":"Departamento",
                    "beta":"Beta","r2":"R²","rmse":"RMSE","pval":"p-valor",
                    "n_meses":"Meses","recomendacion":"Recomendación",
                }).sort_values("Beta"),
                hide_index=True, use_container_width=True, height=380,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Descargar resultados por SKU (CSV)",
                    df_m1a.to_csv(index=False).encode("utf-8"),
                    "elasticidad_por_sku.csv", "text/csv",
                )
            with dl2:
                if len(results["sim"]) > 0:
                    st.download_button(
                        "⬇️ Descargar simulación de precios (CSV)",
                        results["sim"].to_csv(index=False).encode("utf-8"),
                        "simulacion_precios.csv", "text/csv",
                    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD DESCRIPTIVO
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    df = df_main.copy()

    section("📊 Indicadores clave de desempeño")
    total_rev   = df["venta_con_iva"].sum()
    total_units = df["qty"].sum()
    n_skus      = df["prod_nbr"].nunique()
    n_stores    = df["store_nbr"].nunique()
    n_months    = df["mes_str"].nunique()
    avg_mg      = df["margen"].mean() * 100 if "margen" in df.columns else 0

    kc = st.columns(5)
    kpi(kc[0], "Venta total",       f"${total_rev/1e6:.1f}M",  OM_RED)
    kpi(kc[1], "Unidades vendidas", f"{total_units/1e3:.0f}K", OM_BLUE)
    kpi(kc[2], "SKUs únicos",       f"{n_skus:,}",             OM_GREEN)
    kpi(kc[3], "Tiendas",           f"{n_stores}",             OM_AMBER)
    kpi(kc[4], "Margen promedio",   f"{avg_mg:.1f}%",          "#7B1FA2")

    st.markdown("<br>", unsafe_allow_html=True)

    # Time series
    section("📅 Evolución mensual de ventas")
    ts = (df.groupby("mes_str")
          .agg(venta=("venta_con_iva","sum"), unidades=("qty","sum"))
          .reset_index().sort_values("mes_str"))

    tc1, tc2 = st.columns(2)
    with tc1:
        fig = px.area(ts, x="mes_str", y="venta", title="Venta mensual total ($)",
                      labels={"mes_str":"Mes","venta":"Ventas ($)"},
                      color_discrete_sequence=[OM_RED])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)
    with tc2:
        fig = px.bar(ts, x="mes_str", y="unidades", title="Unidades vendidas por mes",
                     labels={"mes_str":"Mes","unidades":"Unidades"},
                     color_discrete_sequence=[OM_BLUE])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)

    # Department
    section("🏷️ Ventas por departamento")
    dept = (df.groupby("dept_nm")
            .agg(venta=("venta_con_iva","sum"), n_skus=("prod_nbr","nunique"),
                 margen=("margen","mean"))
            .reset_index().sort_values("venta", ascending=False))
    dept["dept_short"] = dept["dept_nm"].str[:28]

    dc1, dc2 = st.columns([3, 2])
    with dc1:
        fig = px.bar(dept, x="venta", y="dept_short", orientation="h",
                     title="Venta total por departamento",
                     labels={"venta":"Ventas ($)","dept_short":""},
                     color="venta", color_continuous_scale=[[0,"#FFCDD2"],[1,OM_RED]])
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(_layout(fig, h=380), use_container_width=True)
    with dc2:
        fig = px.pie(dept.head(8), values="venta", names="dept_short",
                     title="Participación (top 8)",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        fig.update_layout(showlegend=False, height=380, paper_bgcolor="white",
                          margin=dict(t=45,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Store performance
    section("🏪 Desempeño por tienda (Top 20)")
    stores = (df.groupby(["store_nbr","store_nm"])
              .agg(venta=("venta_con_iva","sum"), margen=("margen","mean"),
                   unidades=("qty","sum"))
              .reset_index().sort_values("venta", ascending=False).head(20))
    stores["label"] = stores["store_nbr"] + " " + stores["store_nm"].str.strip().str[:15]
    stores["margen_pct"] = stores["margen"] * 100

    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.bar(stores, x="label", y="venta", title="Top 20 tiendas por ventas",
                     labels={"label":"","venta":"Ventas ($)"},
                     color="margen_pct",
                     color_continuous_scale=[[0,OM_RED],[0.5,OM_AMBER],[1,OM_GREEN]])
        fig.update_xaxes(tickangle=55, tickfont=dict(size=9))
        fig.update_layout(coloraxis_colorbar=dict(title="Margen %"))
        st.plotly_chart(_layout(fig, h=380), use_container_width=True)
    with sc2:
        fig = px.scatter(stores, x="venta", y="margen_pct", size="unidades",
                         text="store_nbr", title="Ventas vs Margen por tienda",
                         labels={"venta":"Ventas ($)","margen_pct":"Margen (%)"},
                         color="margen_pct",
                         color_continuous_scale=[[0,OM_RED],[0.5,OM_AMBER],[1,OM_GREEN]])
        fig.update_traces(textposition="top center", textfont_size=8)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_layout(fig, h=380), use_container_width=True)

    # Top SKUs
    section("🏆 Top 15 productos")
    top_sku = (df.groupby(["prod_nbr","prod_nm"])
               .agg(venta=("venta_con_iva","sum"), unidades=("qty","sum"),
                    precio_prom=("precio_tx","mean"), margen=("margen","mean"))
               .reset_index().sort_values("venta", ascending=False).head(15))
    top_sku["label"] = top_sku["prod_nbr"] + " " + top_sku["prod_nm"].str[:22]

    sk1, sk2 = st.columns(2)
    with sk1:
        fig = px.bar(top_sku.sort_values("venta"), x="venta", y="label",
                     orientation="h", title="Por venta total ($)",
                     labels={"venta":"Ventas ($)","label":""},
                     color_discrete_sequence=[OM_RED])
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(_layout(fig, h=440), use_container_width=True)
    with sk2:
        fig = px.bar(top_sku.sort_values("unidades"), x="unidades", y="label",
                     orientation="h", title="Por unidades vendidas",
                     labels={"unidades":"Unidades","label":""},
                     color_discrete_sequence=[OM_BLUE])
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(_layout(fig, h=440), use_container_width=True)

    # Distributions
    section("📊 Distribuciones de precio, margen y unidades")
    p95_p = df["precio_tx"].quantile(0.95)
    p95_q = df["qty"].quantile(0.95)
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        fig = px.histogram(df[df["precio_tx"] <= p95_p], x="precio_tx", nbins=50,
                           title="Distribución de precios",
                           labels={"precio_tx":"Precio unitario ($)"},
                           color_discrete_sequence=[OM_RED])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig, h=280), use_container_width=True)
    with dc2:
        fig = px.histogram(df[df["margen"].between(-0.5, 1.0)], x="margen", nbins=40,
                           title="Distribución de márgenes",
                           labels={"margen":"Margen (ratio)"},
                           color_discrete_sequence=[OM_GREEN])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig, h=280), use_container_width=True)
    with dc3:
        fig = px.histogram(df[df["qty"] <= p95_q], x="qty", nbins=40,
                           title="Unidades por transacción",
                           labels={"qty":"Unidades"},
                           color_discrete_sequence=[OM_AMBER])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig, h=280), use_container_width=True)

    # Premium vs Non-Premium
    section("⭐ Premium vs No Premium")
    prem = (df.groupby("es_premium")
            .agg(venta=("venta_con_iva","sum"), unidades=("qty","sum"),
                 precio_prom=("precio_tx","mean"), margen_prom=("margen","mean"))
            .reset_index())
    prem["tipo"] = prem["es_premium"].map({0:"No Premium", 1:"Premium"})
    prem["margen_pct"] = prem["margen_prom"] * 100

    pm1, pm2, pm3 = st.columns(3)
    for col_w, metric, label in [
        (pm1, "venta",      "Venta total ($)"),
        (pm2, "precio_prom","Precio promedio ($)"),
        (pm3, "margen_pct", "Margen promedio (%)"),
    ]:
        fig = px.bar(prem, x="tipo", y=metric, title=label,
                     color="tipo",
                     color_discrete_map={"No Premium": OM_BLUE, "Premium": OM_RED},
                     text_auto=".2s")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        col_w.plotly_chart(_layout(fig, h=260), use_container_width=True)

    # Brand type
    if "tipo_marca" in df.columns:
        section("🏷️ Marca Propia vs Marca Externa")
        marca = (df.groupby("tipo_marca")
                 .agg(venta=("venta_con_iva","sum"), margen_prom=("margen","mean"))
                 .reset_index())
        marca["margen_pct"] = marca["margen_prom"] * 100
        mb1, mb2 = st.columns(2)
        with mb1:
            fig = px.bar(marca, x="tipo_marca", y="venta", title="Venta por tipo de marca",
                         color="tipo_marca", text_auto=".2s",
                         color_discrete_sequence=[OM_RED, OM_BLUE, OM_GREEN])
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Ventas ($)")
            st.plotly_chart(_layout(fig, h=280), use_container_width=True)
        with mb2:
            fig = px.bar(marca, x="tipo_marca", y="margen_pct", title="Margen por tipo de marca",
                         color="tipo_marca", text_auto=".1f",
                         color_discrete_sequence=[OM_GREEN, OM_AMBER, OM_BLUE])
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Margen (%)")
            st.plotly_chart(_layout(fig, h=280), use_container_width=True)

    # Seasonality
    section("📆 Estacionalidad — promedio por mes del año")
    seas = (df.groupby("mes_calendario")
            .agg(venta_prom=("venta_con_iva","mean"), unidades_prom=("qty","mean"))
            .reset_index())
    seas["mes_nombre"] = seas["mes_calendario"].map(MONTH_NAMES)
    se1, se2 = st.columns(2)
    with se1:
        fig = px.line(seas, x="mes_nombre", y="venta_prom",
                      title="Venta promedio por mes del año",
                      labels={"mes_nombre":"","venta_prom":"Venta promedio ($)"},
                      markers=True, color_discrete_sequence=[OM_RED])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig, h=280), use_container_width=True)
    with se2:
        fig = px.bar(seas, x="mes_nombre", y="unidades_prom",
                     title="Unidades promedio por mes del año",
                     labels={"mes_nombre":"","unidades_prom":"Unidades promedio"},
                     color_discrete_sequence=[OM_BLUE])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig, h=280), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — DASHBOARD PREDICTIVO
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    results = st.session_state["results"]

    if results is None:
        st.info("⏳ Primero ejecuta el análisis en la pestaña **🧮 Calculadora** para ver este dashboard.")
        st.stop()

    df_m1a = results["m1a"]
    df_m1b = results["m1b"]
    df_m1c = results["m1c"]
    df_sim  = results["sim"]
    m2      = results["m2"]

    if len(df_m1a) == 0:
        st.warning("No hay SKUs con modelo válido. Ajusta los parámetros en la Calculadora.")
        st.stop()

    rec_counts = df_m1a["recomendacion"].value_counts()

    # Recommendation summary
    section("🎯 Resumen de recomendaciones")
    rc = st.columns(4)
    for i, (rec, color) in enumerate(REC_COLORS.items()):
        cnt = rec_counts.get(rec, 0)
        pct = cnt / len(df_m1a) * 100
        rc[i].markdown(
            f'<div class="rec-card" style="border-left:6px solid {color};">'
            f'<div style="font-size:38px; font-weight:900; color:{color};">{cnt}</div>'
            f'<div style="font-weight:700; font-size:12px; color:#333;">{rec.upper()}</div>'
            f'<div style="font-size:11px; color:#999;">{pct:.1f}% del total</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Beta distribution + pie
    bc1, bc2 = st.columns([3, 2])
    with bc1:
        fig = px.histogram(df_m1a, x="beta", nbins=40,
                           color="recomendacion", color_discrete_map=REC_COLORS,
                           title="Distribución de elasticidad (β) por SKU",
                           labels={"beta":"Beta (elasticidad precio)"},
                           barmode="overlay", opacity=0.75)
        fig.add_vline(x=-1,   line_dash="dash",  line_color="gray", opacity=0.7,
                      annotation_text="β=-1",   annotation_position="top right")
        fig.add_vline(x=-1.5, line_dash="dot",   line_color="gray", opacity=0.5,
                      annotation_text="β=-1.5", annotation_position="top left")
        fig.add_vline(x=0,    line_color="black", line_width=0.5, opacity=0.4)
        st.plotly_chart(_layout(fig, h=360), use_container_width=True)
    with bc2:
        fig = px.pie(values=rec_counts.values, names=rec_counts.index,
                     title="SKUs por recomendación",
                     color=rec_counts.index, color_discrete_map=REC_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
        fig.update_layout(showlegend=False, height=360, paper_bgcolor="white",
                          margin=dict(t=45,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Beta vs R² quality scatter
    section("📐 Calidad del modelo por SKU")
    st.caption("Cada burbuja = un SKU. Grande = más meses de datos. Ideal: beta negativa y R² alto.")

    valid = df_m1a[df_m1a["recomendacion"] != "No recomendable"]
    if len(valid) > 0:
        fig = px.scatter(valid, x="beta", y="r2", size="n_meses",
                         color="recomendacion", color_discrete_map=REC_COLORS,
                         hover_data=["prod_nm","pval","rmse"],
                         title="Beta vs R² — SKUs con recomendación válida",
                         labels={"beta":"Beta","r2":"R²","n_meses":"Meses"}, size_max=30)
        fig.add_vline(x=-1,   line_dash="dash", line_color="gray", opacity=0.4)
        fig.add_vline(x=-1.5, line_dash="dot",  line_color="gray", opacity=0.35)
        fig.add_hline(y=0.3,  line_dash="dash", line_color=OM_AMBER, opacity=0.5,
                      annotation_text="R²=0.3 (umbral recomendado)")
        st.plotly_chart(_layout(fig, h=400), use_container_width=True)

    # Price simulator
    section("💹 Simulador de precios por SKU")
    st.caption("Selecciona un producto y ve cómo cambian sus ingresos y margen con distintos precios.")

    valid_skus = df_m1a[df_m1a["recomendacion"] != "No recomendable"]["prod_nm"].tolist()
    if valid_skus:
        sel_nm  = st.selectbox("Producto a simular", sorted(valid_skus))
        sku_row = df_m1a[df_m1a["prod_nm"] == sel_nm].iloc[0]
        sku_sim = df_sim[df_sim["prod_nm"] == sel_nm]

        if len(sku_sim) > 0:
            beta_v = sku_row["beta"]
            r2_v   = sku_row["r2"]
            pval_v = sku_row["pval"]
            n_v    = sku_row["n_meses"]
            rec_c  = REC_COLORS.get(sku_row["recomendacion"], OM_LGRAY)

            inf_col, chart_col = st.columns([1, 2])
            with inf_col:
                st.markdown(
                    f'<div style="background:white;border-radius:12px;padding:20px;'
                    f'border-left:5px solid {rec_c};">'
                    f'<div style="font-weight:700;font-size:14px;margin-bottom:12px;">{sel_nm[:55]}</div>'
                    f'<div style="font-size:13px;color:#555;line-height:2.2;">'
                    f'β: <strong>{beta_v:.4f}</strong><br>'
                    f'R²: <strong>{r2_v:.4f}</strong><br>'
                    f'p-valor: <strong>{pval_v:.4f}</strong><br>'
                    f'Meses: <strong>{n_v}</strong></div>'
                    f'<div style="background:{rec_c};color:white;padding:8px 14px;'
                    f'border-radius:20px;font-weight:700;text-align:center;'
                    f'font-size:12px;margin-top:14px;">{sku_row["recomendacion"].upper()}</div></div>',
                    unsafe_allow_html=True,
                )
                base_r = sku_sim[sku_sim["cambio"] == "Base 0%"]
                if len(base_r) > 0:
                    br = base_r.iloc[0]
                    st.markdown(
                        f'<div style="background:#F5F5F5;border-radius:8px;padding:14px;'
                        f'font-size:13px;margin-top:10px;">'
                        f'<strong>Precio base:</strong> ${br["precio_nuevo"]:,.2f}<br>'
                        f'<strong>Unidades/mes:</strong> {br["unidades_est"]:,.1f}<br>'
                        f'<strong>Ingreso est.:</strong> ${br["ingreso_est"]:,.0f}<br>'
                        f'<strong>Margen est.:</strong> ${br["margen_est"]:,.0f}</div>',
                        unsafe_allow_html=True,
                    )

            with chart_col:
                colors_sc = ["#1565C0","#90CAF9","#9E9E9E","#EF9A9A","#C62828"]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sku_sim["cambio"], y=sku_sim["delta_ingreso_pct"],
                                     name="Δ Ingreso", marker_color=colors_sc,
                                     text=[f"{v:+.1f}%" for v in sku_sim["delta_ingreso_pct"]],
                                     textposition="outside"))
                fig.add_trace(go.Bar(x=sku_sim["cambio"], y=sku_sim["delta_margen_pct"],
                                     name="Δ Margen", marker_color=colors_sc, opacity=0.5,
                                     text=[f"{v:+.1f}%" for v in sku_sim["delta_margen_pct"]],
                                     textposition="outside"))
                fig.add_hline(y=0, line_color="black", line_width=0.8)
                fig.update_layout(title="Cambio en Ingreso y Margen por escenario",
                                  barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                                  height=360, margin=dict(t=50,b=20,l=20,r=20),
                                  xaxis_title="Escenario de precio", yaxis_title="Cambio (%)",
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

            # Continuous curve
            base_r2 = sku_sim[sku_sim["cambio"] == "Base 0%"]
            if len(base_r2) > 0:
                br   = base_r2.iloc[0]
                p0   = br["precio_nuevo"]
                u0   = br["unidades_est"]
                m0   = br["margen_est"]
                rev0 = br["ingreso_est"]
                c0   = p0 - (m0 / u0 if u0 else 0)
                marg0_base = (p0 - c0) * u0 or 1

                rng = np.linspace(-0.20, 0.20, 200)
                pct_rev  = [(p0*(1+r)*u0*((1+r)**beta_v) - rev0)/rev0*100         for r in rng]
                pct_marg = [((p0*(1+r)-c0)*u0*((1+r)**beta_v) - marg0_base)/marg0_base*100 for r in rng]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=rng*100, y=pct_rev, name="Δ Ingreso",
                                          line=dict(color=OM_RED, width=2.5)))
                fig2.add_trace(go.Scatter(x=rng*100, y=pct_marg, name="Δ Margen",
                                          line=dict(color=OM_BLUE, width=2, dash="dash")))
                fig2.add_hline(y=0, line_color="gray", line_width=0.7)
                fig2.add_vline(x=0, line_color="gray", line_width=0.7)
                fig2.update_layout(title="Curva continua de sensibilidad precio ±20%",
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   height=300, margin=dict(t=50,b=20,l=20,r=20),
                                   xaxis_title="Cambio en precio (%)", yaxis_title="Cambio (%)",
                                   legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig2, use_container_width=True)

    # Rolling beta
    has_rolling = len(df_m1b) > 0 or len(df_m1c) > 0
    if has_rolling:
        section("📉 Evolución temporal de la elasticidad (rolling)")
        st.caption(
            "Muestra cómo cambia la beta de un SKU en el tiempo. "
            "Beta más cercana a 0 → menos sensible → buen momento para subir precio. "
            "Beta muy negativa → muy sensible → buen momento para promover."
        )
        roll_pool = set()
        if len(df_m1b) > 0: roll_pool |= set(df_m1b["prod_nm"].unique())
        if len(df_m1c) > 0: roll_pool |= set(df_m1c["prod_nm"].unique())

        roll_sel = st.selectbox("Producto (rolling)", sorted(roll_pool))
        fig3 = go.Figure()
        if len(df_m1b) > 0:
            d3 = df_m1b[df_m1b["prod_nm"] == roll_sel].sort_values("mes_fin_dt")
            if len(d3) > 0:
                fig3.add_trace(go.Scatter(x=d3["mes_fin_dt"], y=d3["beta"],
                                          mode="lines+markers", name="Trimestral (3 meses)",
                                          line=dict(color=OM_BLUE, width=2.2), marker=dict(size=6)))
        if len(df_m1c) > 0:
            d6 = df_m1c[df_m1c["prod_nm"] == roll_sel].sort_values("mes_fin_dt")
            if len(d6) > 0:
                fig3.add_trace(go.Scatter(x=d6["mes_fin_dt"], y=d6["beta"],
                                          mode="lines+markers", name="Semestral (6 meses)",
                                          line=dict(color=OM_RED, width=2.2),
                                          marker=dict(size=6, symbol="square")))
        b_glob = df_m1a.loc[df_m1a["prod_nm"] == roll_sel, "beta"]
        if len(b_glob) > 0:
            fig3.add_hline(y=b_glob.iloc[0], line_dash="dot", line_color=OM_GREEN,
                           annotation_text=f"Beta global = {b_glob.iloc[0]:.2f}")
        fig3.add_hline(y=-1,   line_dash="dash", line_color="gray", opacity=0.5,
                       annotation_text="β=-1")
        fig3.add_hline(y=-1.5, line_dash="dot",  line_color="gray", opacity=0.35,
                       annotation_text="β=-1.5")
        fig3.update_layout(title=f"Elasticidad rolling — {roll_sel[:50]}",
                           plot_bgcolor="white", paper_bgcolor="white",
                           height=370, margin=dict(t=55,b=20,l=20,r=20),
                           xaxis_title="Mes final de ventana", yaxis_title="Beta",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    # Top candidates table
    section("📋 Candidatos por tipo de acción")
    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown("**🔵 Subir precio** — inelásticos (β entre -1 y 0, significativo)")
        sub_up = df_m1a[df_m1a["recomendacion"] == "Subir precio"].sort_values("beta", ascending=False)
        if len(sub_up) > 0:
            st.dataframe(sub_up[["prod_nm","beta","r2","pval","n_meses"]]
                         .rename(columns={"prod_nm":"Producto","beta":"Beta",
                                           "r2":"R²","pval":"p-valor","n_meses":"Meses"})
                         .head(12), hide_index=True, use_container_width=True, height=340)
        else:
            st.info("Sin candidatos con los parámetros actuales.")
    with ct2:
        st.markdown("**🟢 Bajar / Promover** — elásticos (β < -1.5, significativo)")
        sub_dn = df_m1a[df_m1a["recomendacion"] == "Bajar / Promover"].sort_values("beta")
        if len(sub_dn) > 0:
            st.dataframe(sub_dn[["prod_nm","beta","r2","pval","n_meses"]]
                         .rename(columns={"prod_nm":"Producto","beta":"Beta",
                                           "r2":"R²","pval":"p-valor","n_meses":"Meses"})
                         .head(12), hide_index=True, use_container_width=True, height=340)
        else:
            st.info("Sin candidatos con los parámetros actuales.")

    # Model stats summary
    section("📊 Estadísticas del modelo M1A")
    stat_cols = st.columns(3)
    kpi(stat_cols[0], "Beta promedio (todos los SKUs)", f"{df_m1a['beta'].mean():.3f}", OM_BLUE)
    kpi(stat_cols[1], "R² promedio",                   f"{df_m1a['r2'].mean():.3f}",   OM_GREEN)
    kpi(stat_cols[2], "RMSE promedio (log-escala)",
        f"{df_m1a['rmse'].mean():.3f}" if 'rmse' in df_m1a.columns else "N/A", OM_AMBER)

    # Beta by department box
    if "dept_nm" in df_m1a.columns:
        valid_dept = df_m1a[df_m1a["recomendacion"] != "No recomendable"].dropna(subset=["dept_nm"])
        if len(valid_dept) > 3:
            st.markdown("<br>", unsafe_allow_html=True)
            fig4 = px.box(valid_dept, x="dept_nm", y="beta",
                          title="Distribución de Beta por departamento (SKUs válidos)",
                          labels={"dept_nm":"Departamento","beta":"Beta"},
                          color="dept_nm",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig4.add_hline(y=-1,   line_dash="dash", line_color="gray", opacity=0.5)
            fig4.add_hline(y=-1.5, line_dash="dot",  line_color="gray", opacity=0.35)
            fig4.update_xaxes(tickangle=40, tickfont=dict(size=10))
            fig4.update_layout(showlegend=False)
            st.plotly_chart(_layout(fig4, h=380), use_container_width=True)
