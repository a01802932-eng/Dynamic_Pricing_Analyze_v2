# -*- coding: utf-8 -*-
"""Dynamic Pricing Analyzer v3 — Streamlit App | OfficeMax México"""

import io, zipfile, warnings, os
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
OM_RED    = "#E31837"
OM_YELLOW = "#FFD100"
OM_BLACK  = "#1A1A1A"
OM_WHITE  = "#FFFFFF"
OM_GRAY   = "#F5F5F5"
OM_BLUE   = "#1565C0"
OM_GREEN  = "#2E7D32"
OM_AMBER  = "#F9A825"
OM_LGRAY  = "#9E9E9E"

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
MONTH_ORDER = [MONTH_NAMES[i] for i in range(1, 13)]

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
.main .block-container { max-width:1300px !important; padding-top:1rem !important;
    padding-left:2rem !important; padding-right:2rem !important; }
[data-testid="stSidebar"] { background-color: #1A1A1A !important; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { font-size:13px !important; }
[data-testid="stSidebar"] .stFileUploader label { color:#FFD100 !important; font-weight:700 !important; }
[data-testid="stSidebar"] .stExpander { border:1px solid #333 !important; border-radius:8px !important; }
.sidebar-logo-wrap { text-align:center; padding:12px 0 6px 0; }
.sidebar-divider { border:none; border-top:1px solid #333; margin:12px 0; }
.sidebar-section-label { font-size:13px; font-weight:700; color:#FFD100 !important;
    letter-spacing:0.5px; text-transform:uppercase; margin:10px 0 4px 0; }
.sidebar-hint { font-size:11px; color:#aaa !important; line-height:1.5; margin:0 0 8px 0; }
.sidebar-status-ok  { background:#1B5E20; color:#fff; border-radius:6px; padding:5px 10px;
    font-size:12px; font-weight:700; margin:4px 0; }
.sidebar-status-err { background:#B71C1C; color:#fff; border-radius:6px; padding:5px 10px;
    font-size:12px; font-weight:700; margin:4px 0; }
.kpi-card { background:white; border-radius:12px; padding:18px 12px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); border-top:4px solid #E31837; }
.kpi-value { font-size:26px; font-weight:900; color:#1A1A1A; }
.kpi-label { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px; }
.section-header { background:linear-gradient(135deg,#E31837 0%,#C41430 100%); color:white;
    padding:10px 18px; border-radius:8px; font-weight:700; font-size:15px; margin:24px 0 10px 0; }
.clean-step { padding:8px 12px; border-left:4px solid #E31837; margin:5px 0;
    background:white; border-radius:0 6px 6px 0; font-size:13px; }
.chip-green  { background:#2E7D32; color:white; padding:6px 18px; border-radius:20px;
    font-weight:700; font-size:14px; display:inline-block; }
.chip-yellow { background:#F9A825; color:white; padding:6px 18px; border-radius:20px;
    font-weight:700; font-size:14px; display:inline-block; }
.chip-red    { background:#E31837; color:white; padding:6px 18px; border-radius:20px;
    font-weight:700; font-size:14px; display:inline-block; }
.rec-card { border-radius:12px; padding:18px; text-align:center;
    background:white; box-shadow:0 2px 8px rgba(0,0,0,0.07); }
.narrative-box { background:white; border-radius:12px; padding:24px 28px;
    border-left:6px solid #E31837; box-shadow:0 2px 8px rgba(0,0,0,0.07);
    font-size:14px; line-height:1.8; color:#333; }
.action-badge-subir    { background:#1565C0; color:white; padding:3px 10px;
    border-radius:12px; font-size:11px; font-weight:700; }
.action-badge-promover { background:#2E7D32; color:white; padding:3px 10px;
    border-radius:12px; font-size:11px; font-weight:700; }
.action-badge-mantener { background:#F9A825; color:white; padding:3px 10px;
    border-radius:12px; font-size:11px; font-weight:700; }
.insight-box { background:#FFF8E1; border-left:4px solid #FFD100; border-radius:0 8px 8px 0;
    padding:8px 14px; font-size:13px; color:#333; margin:6px 0 16px 0; }
/* Nuclear uploader dark override */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #2C2C2C !important;
    border: 2px dashed #E31837 !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] * { color: #FFFFFF !important; }
div[data-testid="stFileUploaderDropzone"] svg { fill: #FFFFFF !important; }
div[data-testid="stFileUploaderDropzone"] button {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
    border: none !important;
}
div[data-testid="stFileUploaderDropzone"] > div { background-color: #2C2C2C !important; }
div[data-testid="stFileUploaderDropzone"] > div > div { background-color: #2C2C2C !important; }
div[data-testid="stFileUploaderFile"] {
    background-color: #3A3A3A !important;
    border: 1px solid #555 !important;
    border-radius: 6px !important;
}
div[data-testid="stFileUploaderFile"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
div[data-testid="stFileUploaderFile"] button {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
    border-radius: 50% !important;
}
/* Fix B — expander dark background */
[data-testid="stExpander"] > div:first-child {
    background-color: #2C2C2C !important;
    border: 1px solid #555555 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
details[data-testid="stExpander"] {
    background-color: #2C2C2C !important;
    border: 1px solid #444444 !important;
    border-radius: 8px !important;
}
details[data-testid="stExpander"] summary { color: #FFFFFF !important; }
details[data-testid="stExpander"] summary span { color: #FFFFFF !important; }
details[data-testid="stExpander"] > div { background-color: #2C2C2C !important; }
/* Fix B — ensure all sidebar text white */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
(function patchUploaders() {
    function applyDark() {
        var selectors = [
            '[data-testid="stFileUploaderDropzone"]',
            '[data-testid="stFileUploaderDropzone"] > div',
            '[data-testid="stFileUploaderDropzone"] > div > div',
            '[data-testid="stFileUploaderDropzone"] > div > div > div'
        ];
        selectors.forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.setProperty('background-color', '#2C2C2C', 'important');
                el.style.setProperty('color', '#FFFFFF', 'important');
            });
        });
        document.querySelectorAll('[data-testid="stFileUploaderDropzone"] svg path')
            .forEach(function(el) { el.style.setProperty('fill', '#FFFFFF', 'important'); });
        document.querySelectorAll('[data-testid="stFileUploaderFile"], [class*="uploadedFile"]')
            .forEach(function(el) {
                el.style.setProperty('background-color', '#3A3A3A', 'important');
                el.style.setProperty('color', '#FFFFFF', 'important');
            });
    }
    applyDark();
    var observer = new MutationObserver(function() { applyDark(); });
    observer.observe(document.body, { childList: true, subtree: true });
    setInterval(applyDark, 800);
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k in ["df_main", "clean_report", "results", "df_csv_bytes", "loaded_file_name",
           "ai_analysis", "ml_results", "promo_bytes", "promo_file_name"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if "show_email_form"   not in st.session_state: st.session_state["show_email_form"]   = False
if "pdf_bytes_to_send" not in st.session_state: st.session_state["pdf_bytes_to_send"] = None
if "email_from"        not in st.session_state: st.session_state["email_from"]        = ""
if "email_pass"        not in st.session_state: st.session_state["email_pass"]        = ""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _identify(dfs):
    out = {}
    for _, df in dfs.items():
        cols = set(df.columns)
        if "apparent_unit_cost" in cols:       out["costos"]   = df
        elif "Precio_Unitario" in cols:        out["precios"]  = df
        elif "tran_nbr" in cols:               out["tickets"]  = df
        elif "tran_date" in cols and "venta_con_iva" in cols: out["ventas"] = df
        elif "prod_nm" in cols and "tipo_marca" in cols:      out["catalogo"] = df
    return out


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes):
    report = []
    raw = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    raw[Path(name).stem.lower()] = pd.read_csv(f, encoding="latin-1", low_memory=False)

    files = _identify(raw)
    missing = [k for k in ("ventas","catalogo","precios","costos") if k not in files]
    if missing:
        raise ValueError(f"Archivos faltantes en el ZIP: {', '.join(missing)}")

    ventas   = files["ventas"].copy()
    catalogo = files["catalogo"].copy()
    precios  = files["precios"].copy()
    costos   = files["costos"].copy()

    for df in (ventas, catalogo, precios, costos):
        df["prod_nbr"] = df["prod_nbr"].astype(str).str.strip()

    n0 = len(ventas)
    report.append({"Paso":"📥 Ventas cargadas","Eliminadas":"—","Detalle":f"{n0:,} filas originales"})

    ventas["tran_date"]     = pd.to_datetime(ventas["tran_date"],    errors="coerce")
    ventas["qty"]           = pd.to_numeric(ventas["qty"],           errors="coerce")
    ventas["venta_con_iva"] = pd.to_numeric(ventas["venta_con_iva"], errors="coerce")
    ventas["costo"]         = pd.to_numeric(ventas["costo"],         errors="coerce")
    ventas["margen"]        = pd.to_numeric(ventas["margen"],        errors="coerce")

    mask = ventas["tran_date"].isna() | ventas["qty"].isna() | ventas["venta_con_iva"].isna()
    n_rm = int(mask.sum()); ventas = ventas[~mask].copy()
    if n_rm: report.append({"Paso":"🗑 Campos clave nulos","Eliminadas":f"{n_rm:,}","Detalle":"tran_date, qty o venta_con_iva vacíos"})

    mask = (ventas["qty"] <= 0) | (ventas["venta_con_iva"] <= 0)
    n_rm = int(mask.sum()); ventas = ventas[~mask].copy()
    if n_rm: report.append({"Paso":"🗑 Qty / Venta ≤ 0","Eliminadas":f"{n_rm:,}","Detalle":"Devoluciones, ajustes o errores de captura"})

    n_rm = int(ventas.duplicated().sum()); ventas = ventas.drop_duplicates()
    if n_rm: report.append({"Paso":"🗑 Duplicados exactos","Eliminadas":f"{n_rm:,}","Detalle":"Filas 100% idénticas eliminadas"})

    ventas["precio_tx"] = ventas["venta_con_iva"] / ventas["qty"]
    p99 = ventas.groupby("dept_cd")["precio_tx"].transform(lambda x: x.quantile(0.99))
    mask = ventas["precio_tx"] > p99; n_rm = int(mask.sum()); ventas = ventas[~mask].copy()
    if n_rm: report.append({"Paso":"🗑 Precios outlier (>p99)","Eliminadas":f"{n_rm:,}","Detalle":"Precio unitario > percentil 99 del departamento"})

    report.append({"Paso":"✅ Ventas limpias","Eliminadas":"—","Detalle":f"{len(ventas):,} filas válidas"})

    cat_slim  = catalogo[["prod_nbr","prod_nm","dept_nm","subdept_nm","class_nm","marca_fabricante","tipo_marca"]].drop_duplicates("prod_nbr")
    prec_slim = precios[["prod_nbr","Precio_Unitario"]].rename(columns={"Precio_Unitario":"precio_catalogo"})
    cost_slim = costos[["prod_nbr","apparent_unit_cost"]].rename(columns={"apparent_unit_cost":"costo_unitario"})

    df = (ventas
          .merge(cat_slim,  on="prod_nbr", how="left", suffixes=("","_cat"))
          .merge(prec_slim, on="prod_nbr", how="left")
          .merge(cost_slim, on="prod_nbr", how="left"))

    df["mes_str"]        = df["tran_date"].dt.to_period("M").astype(str)
    df["año"]            = df["tran_date"].dt.year
    df["mes_calendario"] = df["tran_date"].dt.month
    df["es_premium"]     = df["prod_nm"].apply(
        lambda n: int(any(k in str(n).upper() for k in KEYWORDS_PREMIUM)) if pd.notna(n) else 0)

    report.append({"Paso":"🔗 Merge final","Eliminadas":"—",
                   "Detalle":f"{len(df):,} filas · {df['prod_nbr'].nunique():,} SKUs · {df['store_nbr'].nunique()} tiendas · {df['mes_str'].nunique()} meses"})

    return df, pd.DataFrame(report)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ols_loglog(subset, min_obs, min_cv, min_r2, max_beta):
    n = len(subset)
    if n < min_obs: return None
    std_p = subset["log_p"].std()
    if std_p < 1e-6: return None
    mean_p = abs(subset["log_p"].mean())
    if mean_p > 1e-9 and (std_p / mean_p) < min_cv: return None
    try:
        y = subset["log_u1"].values
        X = add_constant(subset["log_p"].values)
        res = OLS(y, X).fit()
        beta = float(res.params[1]); r2 = float(res.rsquared)
        if r2 < min_r2 or abs(beta) > max_beta: return None
        return {"alpha":round(float(res.params[0]),4),"beta":round(beta,4),"r2":round(r2,4),
                "rmse":round(float(np.sqrt(np.mean((y-res.fittedvalues)**2))),4),
                "n":n,"pval":round(float(res.pvalues[1]),4)}
    except Exception: return None


def _build_monthly(df):
    df2 = df.copy()
    df2["mes"] = pd.PeriodIndex(df2["mes_str"], freq="M")
    mensual = (df2.groupby(["prod_nbr","mes"])
               .agg(unidades=("qty","sum"), venta_tot=("venta_con_iva","sum"),
                    es_premium=("es_premium","max"), prod_nm=("prod_nm","first"),
                    subdept_nm=("subdept_nm","first"), dept_nm=("dept_nm","first"),
                    margen_avg=("margen","mean"), costo_unit=("costo_unitario","mean"))
               .reset_index())
    mensual["precio"] = mensual["venta_tot"] / mensual["unidades"]
    mensual = mensual[mensual["precio"] > 0].copy()
    mensual["log_u1"] = np.log1p(mensual["unidades"])
    mensual["log_p"]  = np.log(mensual["precio"])
    mensual["mes_dt"] = mensual["mes"].dt.to_timestamp()
    mensual["mes_cal"]= mensual["mes_dt"].dt.month
    return mensual.sort_values(["prod_nbr","mes"]).reset_index(drop=True)


def _compute_monthly_calendar(mensual, df_m1a):
    """Seasonal action calendar: for each SKU x calendar month, recommend SUBIR/PROMOVER/MANTENER."""
    valid = df_m1a[df_m1a["recomendacion"].isin(["Subir precio","Mantener precio","Bajar / Promover"])]
    rows = []
    for _, sk in valid.iterrows():
        sku_data = mensual[mensual["prod_nbr"] == sk["prod_nbr"]]
        if len(sku_data) < 3: continue
        monthly = (sku_data.groupby("mes_cal")
                   .agg(u_prom=("unidades","mean"), p_prom=("precio","mean"))
                   .reset_index())
        u_mean = monthly["u_prom"].mean()
        if u_mean <= 0: continue
        monthly["u_idx"] = monthly["u_prom"] / u_mean
        for _, mrow in monthly.iterrows():
            mes = int(mrow["mes_cal"]); u_idx = mrow["u_idx"]
            if u_idx >= 1.10:
                accion = "SUBIR" if sk["recomendacion"] == "Subir precio" else "PROMOVER"
            elif u_idx <= 0.90:
                accion = "PROMOVER" if sk["recomendacion"] in ("Bajar / Promover","Mantener precio") else "MANTENER"
            else:
                accion = "MANTENER"
            rows.append({"prod_nbr":sk["prod_nbr"],"prod_nm":sk["prod_nm"],
                          "mes_cal":mes,"mes_nombre":MONTH_NAMES[mes],
                          "accion":accion,"u_idx":round(u_idx,2),
                          "recomendacion":sk["recomendacion"],"beta":sk["beta"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_models(df_csv: bytes, min_obs, min_cv, min_r2, max_beta, pval_thresh, run_rolling):
    df = pd.read_csv(io.BytesIO(df_csv), low_memory=False)
    df["tran_date"] = pd.to_datetime(df["tran_date"], errors="coerce")
    mensual = _build_monthly(df)
    todos_meses = sorted(mensual["mes"].unique())
    params = dict(min_obs=min_obs, min_cv=min_cv, min_r2=min_r2, max_beta=max_beta)

    # M1A
    rows_m1a = []
    for sku, grp in mensual.groupby("prod_nbr"):
        r = _ols_loglog(grp, **params)
        if r:
            rows_m1a.append({"prod_nbr":sku,"prod_nm":grp["prod_nm"].iloc[0],
                              "subdept_nm":grp["subdept_nm"].iloc[0],"dept_nm":grp["dept_nm"].iloc[0],
                              "n_meses":len(grp),**r})
    df_m1a = pd.DataFrame(rows_m1a)

    # M1B rolling 3m
    df_m1b = pd.DataFrame()
    df_m1c = pd.DataFrame()
    if run_rolling and len(todos_meses) >= 3:
        rows = []
        for sku, grp in mensual.groupby("prod_nbr"):
            gi = grp.set_index("mes")
            for i in range(len(todos_meses)-2):
                vent = todos_meses[i:i+3]; sub = gi[gi.index.isin(vent)]
                r = _ols_loglog(sub, **params)
                if r: rows.append({"prod_nbr":sku,"prod_nm":grp["prod_nm"].iloc[0],
                                    "mes_inicio":str(vent[0]),"mes_fin":str(vent[-1]),
                                    "mes_fin_dt":vent[-1].to_timestamp(),**r})
        df_m1b = pd.DataFrame(rows)

    if run_rolling and len(todos_meses) >= 6:
        rows = []
        for sku, grp in mensual.groupby("prod_nbr"):
            gi = grp.set_index("mes")
            for i in range(len(todos_meses)-5):
                vent = todos_meses[i:i+6]; sub = gi[gi.index.isin(vent)]
                r = _ols_loglog(sub, **params)
                if r: rows.append({"prod_nbr":sku,"prod_nm":grp["prod_nm"].iloc[0],
                                    "mes_inicio":str(vent[0]),"mes_fin":str(vent[-1]),
                                    "mes_fin_dt":vent[-1].to_timestamp(),**r})
        df_m1c = pd.DataFrame(rows)

    # M2 extendido
    agg2 = (df.groupby(["prod_nbr","store_nbr","mes_str"])
            .agg(unidades=("qty","sum"), venta_tot=("venta_con_iva","sum"),
                 es_premium=("es_premium","max"), margen=("margen","mean"))
            .reset_index())
    agg2["precio"] = agg2["venta_tot"] / agg2["unidades"]
    agg2 = agg2[agg2["precio"] > 0].copy()
    agg2["log_u1"] = np.log1p(agg2["unidades"]); agg2["log_p"] = np.log(agg2["precio"])

    ohe_s = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    s_enc = ohe_s.fit_transform(agg2[["store_nbr"]]); s_cols = [f"s_{c}" for c in ohe_s.categories_[0][1:]]
    ohe_m = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    m_enc = ohe_m.fit_transform(agg2[["mes_str"]]);  m_cols = [f"m_{c}" for c in ohe_m.categories_[0][1:]]

    X2 = pd.concat([pd.DataFrame({"log_precio":agg2["log_p"].values,
                                   "es_premium":agg2["es_premium"].values,
                                   "margen":agg2["margen"].fillna(0).values}, index=agg2.index),
                    pd.DataFrame(s_enc, columns=s_cols, index=agg2.index),
                    pd.DataFrame(m_enc, columns=m_cols, index=agg2.index)], axis=1)
    X2 = add_constant(X2); y2 = agg2["log_u1"].values
    msk = np.isfinite(X2.values).all(axis=1) & np.isfinite(y2)
    mod2 = OLS(y2[msk], X2[msk]).fit()
    # Muestra aleatoria para scatter predicho vs real (max 1500 puntos)
    y2_actual = y2[msk]
    y2_fitted = mod2.fittedvalues
    n_sample  = min(1500, len(y2_actual))
    idx_sample = np.random.choice(len(y2_actual), size=n_sample, replace=False)
    actual_sample = np.expm1(y2_actual[idx_sample])
    fitted_sample = np.expm1(y2_fitted[idx_sample])

    m2 = {"n_obs":int(mod2.nobs),"r2":round(float(mod2.rsquared),4),
          "r2_adj":round(float(mod2.rsquared_adj),4),
          "beta":round(float(mod2.params["log_precio"]),4),
          "beta_pval":round(float(mod2.pvalues["log_precio"]),4),
          "premium":round(float(mod2.params["es_premium"]),4),
          "prem_pval":round(float(mod2.pvalues["es_premium"]),4),
          "rmse":round(float(np.sqrt(np.mean((y2[msk]-mod2.fittedvalues)**2))),4),
          "f_stat":round(float(mod2.fvalue),2),
          "actual_sample": actual_sample.tolist(),
          "fitted_sample": fitted_sample.tolist()}

    # Classify — cap económico en |b|>3 (betas más extremas no son creíbles en retail)
    def _classify(row):
        b, p, n = row["beta"], row["pval"], row["n_meses"]
        if p >= pval_thresh or n < min_obs or b > 0 or abs(b) > 3: return "No recomendable"
        if -1 < b < 0:         return "Subir precio"
        if -1.5 <= b <= -1:    return "Mantener precio"
        return "Bajar / Promover"

    if len(df_m1a) > 0:
        df_m1a["recomendacion"] = df_m1a.apply(_classify, axis=1)

    # Simulation
    base_stats = mensual.groupby("prod_nbr").agg(
        precio_base=("precio","mean"), unidades_base=("unidades","mean"),
        costo_unit=("costo_unit","mean")).reset_index()
    scenarios = [-0.10,-0.05,0.00,0.05,0.10]; labels = ["-10%","-5%","Base 0%","+5%","+10%"]
    sim_rows = []
    if len(df_m1a) > 0:
        for _, row in df_m1a.merge(base_stats, on="prod_nbr", how="left").iterrows():
            p0, u0, beta = row.get("precio_base"), row.get("unidades_base"), row["beta"]
            if pd.isna(p0) or pd.isna(u0): continue
            c0 = float(row.get("costo_unit",0) or 0)
            rev0 = p0*u0; marg0 = (p0-c0)*u0 or 1e-9
            for chg, lbl in zip(scenarios, labels):
                p1=p0*(1+chg); u1=u0*((1+chg)**beta); rev1=p1*u1; marg1=(p1-c0)*u1
                sim_rows.append({"prod_nbr":row["prod_nbr"],"prod_nm":row["prod_nm"],
                                  "beta":round(beta,4),"recomendacion":row.get("recomendacion",""),
                                  "cambio":lbl,"precio_nuevo":round(p1,2),"unidades_est":round(u1,1),
                                  "ingreso_est":round(rev1,2),"margen_est":round(marg1,2),
                                  "delta_ingreso_pct":round((rev1-rev0)/rev0*100,1) if rev0 else 0,
                                  "delta_margen_pct":round((marg1-marg0)/marg0*100,1)})

    # Monthly calendar
    df_cal = _compute_monthly_calendar(mensual, df_m1a) if len(df_m1a) > 0 else pd.DataFrame()

    mensual_out = mensual.copy(); mensual_out["mes"] = mensual_out["mes"].astype(str)
    return {"m1a":df_m1a,"m1b":df_m1b,"m1c":df_m1c,"m2":m2,
            "sim":pd.DataFrame(sim_rows),"mensual":mensual_out,
            "cal":df_cal,"n_total":mensual["prod_nbr"].nunique()}


# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_narrative(df_m1a, m2, df_cal):
    n_tot   = len(df_m1a)
    n_subir = len(df_m1a[df_m1a["recomendacion"] == "Subir precio"])
    n_prom  = len(df_m1a[df_m1a["recomendacion"] == "Bajar / Promover"])
    n_mant  = len(df_m1a[df_m1a["recomendacion"] == "Mantener precio"])
    n_norec = len(df_m1a[df_m1a["recomendacion"] == "No recomendable"])
    n_valid = n_subir + n_prom + n_mant

    top_subir = df_m1a[df_m1a["recomendacion"]=="Subir precio"].sort_values("beta",ascending=False)
    top_prom  = df_m1a[df_m1a["recomendacion"]=="Bajar / Promover"].sort_values("beta")

    mejor_subir = top_subir["prod_nm"].iloc[0][:35] if len(top_subir) > 0 else "N/A"
    mejor_prom  = top_prom["prod_nm"].iloc[0][:35]  if len(top_prom)  > 0 else "N/A"

    r2_interp = "buena" if m2["r2"] >= 0.4 else ("aceptable" if m2["r2"] >= 0.15 else "baja")
    beta_interp = f"{m2['beta']:.2f}"
    sig = "sí es estadísticamente significativa" if m2["beta_pval"] < 0.10 else "no es estadísticamente significativa"

    meses_subir, meses_prom = [], []
    if len(df_cal) > 0:
        ms = df_cal[df_cal["accion"]=="SUBIR"]["mes_nombre"].value_counts().head(3).index.tolist()
        mp = df_cal[df_cal["accion"]=="PROMOVER"]["mes_nombre"].value_counts().head(3).index.tolist()
        meses_subir = ms; meses_prom = mp

    lines = [
        f"Se analizaron **{n_tot:,} SKUs** en total. "
        f"De estos, **{n_valid} tienen un modelo de elasticidad confiable** con los parámetros actuales "
        f"({n_norec} fueron descartados por poca variación de precio o baja significancia estadística).",
        "",
        f"🔵 **{n_subir} productos son inelásticos** (recomendación: *Subir precio*). "
        f"Sus consumidores no cambian mucho su comportamiento de compra ante subidas de precio. "
        f"Subir el precio generaría más ingresos sin perder muchas ventas. "
        + (f"El mejor candidato es **{mejor_subir}**." if n_subir > 0 else ""),
        "",
        f"🟢 **{n_prom} productos son elásticos** (recomendación: *Promover o bajar precio*). "
        f"Los clientes son muy sensibles al precio — hacer descuentos o promociones en estos "
        f"productos aumentaría el volumen de ventas de forma significativa. "
        + (f"El más elástico es **{mejor_prom}**." if n_prom > 0 else ""),
        "",
        f"🟡 **{n_mant} productos** se recomienda mantener el precio actual (elasticidad cercana a -1).",
        "",
        f"📊 El modelo global explica el **{m2['r2']*100:.1f}%** de la variación en ventas "
        f"(R² {r2_interp}). La relación precio-demanda {sig} "
        f"(β={beta_interp}, p={m2['beta_pval']:.3f}).",
    ]

    if meses_subir:
        lines += ["", f"📅 **Mejores meses para subir precios:** {', '.join(meses_subir)} "
                  f"(alta demanda histórica)."]
    if meses_prom:
        lines += [f"📅 **Mejores meses para promover:** {', '.join(meses_prom)} "
                  f"(demanda baja — los descuentos tienen más impacto)."]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(results, ml_results, ai_analysis, df_main) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.units import inch
    from datetime import datetime as _dt

    OM_RED_RL  = colors.HexColor("#E31837")
    OM_GRAY_RL = colors.HexColor("#666666")
    OM_LGRAY   = colors.HexColor("#F5F5F5")
    OM_DARK    = colors.HexColor("#1A1A1A")
    GREEN_RL   = colors.HexColor("#2E7D32")
    BLUE_RL    = colors.HexColor("#1565C0")
    MGRAY_RL   = colors.HexColor("#757575")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle("OmTitle", parent=styles["Heading1"],
                                  textColor=OM_RED_RL, fontSize=22, spaceAfter=4)
    sub_style   = ParagraphStyle("OmSub", parent=styles["Normal"],
                                  textColor=OM_GRAY_RL, fontSize=11, spaceAfter=2)
    date_style  = ParagraphStyle("OmDate", parent=styles["Normal"],
                                  textColor=OM_GRAY_RL, fontSize=9, spaceAfter=12)
    sec_style   = ParagraphStyle("OmSec", parent=styles["Heading2"],
                                  textColor=OM_DARK, fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("OmBody", parent=styles["Normal"],
                                  fontSize=9, leading=13, spaceAfter=4)
    foot_style  = ParagraphStyle("OmFoot", parent=styles["Normal"],
                                  textColor=OM_GRAY_RL, fontSize=8, alignment=1)

    # ── Page 1: Header ─────────────────────────────────────────────────────────
    story.append(Paragraph("Dynamic Pricing Analyzer", title_style))
    story.append(Paragraph("Reporte Ejecutivo de Optimización de Precios", sub_style))
    story.append(Paragraph(
        f"Generado el {_dt.now().strftime('%d de %B de %Y')}", date_style))
    story.append(HRFlowable(width="100%", thickness=2, color=OM_RED_RL, spaceAfter=12))

    # ── Executive summary ──────────────────────────────────────────────────────
    story.append(Paragraph("Resumen Ejecutivo", sec_style))
    summary_text = (ai_analysis or "Resumen no disponible — ejecuta el análisis primero.")
    for line in summary_text.split("\n"):
        clean = line.strip().replace("**","").replace("*","")
        if clean:
            story.append(Paragraph(clean, body_style))
    story.append(Spacer(1, 0.15*inch))

    # ── KPI summary table ──────────────────────────────────────────────────────
    story.append(Paragraph("Indicadores Clave", sec_style))
    df_m1a = results.get("m1a", pd.DataFrame())
    df_sim = results.get("sim", pd.DataFrame())
    rec_counts = df_m1a["recomendacion"].value_counts() if len(df_m1a) else {}
    n_subir = int(rec_counts.get("Subir precio", 0))
    n_prom  = int(rec_counts.get("Bajar / Promover", 0))
    n_mant  = int(rec_counts.get("Mantener precio", 0))
    # financial impact
    def _pdf_delta(recs, esc):
        if len(df_sim) == 0: return 0
        ns = df_m1a[df_m1a["recomendacion"].isin(recs)]["prod_nm"].tolist() if "prod_nm" in df_m1a.columns else []
        b = df_sim[(df_sim["prod_nm"].isin(ns)) & (df_sim["cambio"]=="Base 0%")]["ingreso_est"].sum() if ns else 0
        t = df_sim[(df_sim["prod_nm"].isin(ns)) & (df_sim["cambio"]==esc)]["ingreso_est"].sum() if ns else 0
        return t - b
    imp_mes = _pdf_delta(["Subir precio"], "+10%") + _pdf_delta(["Bajar / Promover"], "-10%")
    kpi_data = [
        ["Indicador", "Valor"],
        ["SKUs analizados",          f"{len(df_m1a):,}"],
        ["Subir precio",             f"{n_subir:,} productos"],
        ["Lanzar promoción",         f"{n_prom:,} productos"],
        ["Mantener precio",          f"{n_mant:,} productos"],
        ["Impacto mensual estimado", f"${imp_mes:,.0f} MXN/mes"],
        ["Impacto anualizado",       f"${imp_mes*12:,.0f} MXN/año"],
    ]
    kpi_table = Table(kpi_data, colWidths=[3.2*inch, 2.5*inch])
    kpi_ts = TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  OM_RED_RL),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, OM_LGRAY]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ])
    kpi_table.setStyle(kpi_ts)
    story.append(kpi_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Page 2: Top 20 recommendations ────────────────────────────────────────
    story.append(Paragraph("Top 20 Recomendaciones", sec_style))
    story.append(Paragraph(
        "Productos con mayor impacto financiero potencial", body_style))

    if len(df_m1a) > 0 and len(df_sim) > 0:
        _id = "prod_nm" if "prod_nm" in df_sim.columns else "prod_nbr"
        base_map = df_sim[df_sim["cambio"]=="Base 0%"].set_index(_id)["ingreso_est"].to_dict()
        best_map = {}
        for nm, grp in df_sim[df_sim["cambio"].isin(["+10%","-10%"])].groupby(_id):
            b = base_map.get(nm, 0)
            best_map[nm] = grp.loc[(grp["ingreso_est"] - b).abs().idxmax(), "ingreso_est"] - b
        _id_m = "prod_nm" if "prod_nm" in df_m1a.columns else "prod_nbr"
        top_df = df_m1a[df_m1a["recomendacion"] != "No recomendable"].copy()
        top_df["_imp"] = top_df[_id_m].map(best_map).fillna(0).abs()
        top_df = top_df.sort_values("_imp", ascending=False).head(20)

        rec_header = ["Producto", "Departamento", "Recomendación", "Beta (β)", "Impacto $/mes"]
        rec_rows   = [rec_header]
        rec_colors_map = {
            "Subir precio":     GREEN_RL,
            "Bajar / Promover": BLUE_RL,
            "Mantener precio":  MGRAY_RL,
        }
        color_cmds = []
        for row_i, (_, r) in enumerate(top_df.iterrows(), start=1):
            nm_col   = "prod_nm" if "prod_nm" in r.index else "prod_nbr"
            dept_col = "dept_nm" if "dept_nm" in r.index else ""
            rec_rows.append([
                str(r.get(nm_col, ""))[:35],
                str(r.get(dept_col, ""))[:20] if dept_col else "—",
                str(r.get("recomendacion", "")),
                f"{r.get('beta', 0):.3f}",
                f"${best_map.get(r.get(nm_col,''), 0):,.0f}",
            ])
            c = rec_colors_map.get(str(r.get("recomendacion","")), MGRAY_RL)
            color_cmds.append(("TEXTCOLOR", (2, row_i), (2, row_i), c))

        rec_widths = [2.5*inch, 1.4*inch, 1.3*inch, 0.7*inch, 0.9*inch]
        rec_table  = Table(rec_rows, colWidths=rec_widths)
        rec_ts_cmds = [
            ("BACKGROUND",    (0,0),  (-1,0),  OM_RED_RL),
            ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
            ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),  (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),  (-1,-1), [colors.white, OM_LGRAY]),
            ("GRID",          (0,0),  (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
            ("LEFTPADDING",   (0,0),  (-1,-1), 5),
            ("RIGHTPADDING",  (0,0),  (-1,-1), 5),
            ("TOPPADDING",    (0,0),  (-1,-1), 3),
            ("BOTTOMPADDING", (0,0),  (-1,-1), 3),
        ] + color_cmds
        rec_table.setStyle(TableStyle(rec_ts_cmds))
        story.append(rec_table)
    else:
        story.append(Paragraph("No hay recomendaciones disponibles.", body_style))

    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=OM_GRAY_RL))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Generado automáticamente por Dynamic Pricing Analyzer v3", foot_style))
    story.append(Paragraph("OfficeMax México — Análisis confidencial", foot_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL SENDER
# ══════════════════════════════════════════════════════════════════════════════

def send_report_email(pdf_bytes: bytes, recipients: list, subject: str,
                      body: str, from_email: str, app_password: str) -> tuple:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email.mime.text import MIMEText
    from email import encoders
    from datetime import datetime as _dt
    try:
        msg = MIMEMultipart()
        msg["From"]    = from_email
        msg["To"]      = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        fname = f"reporte_pricing_officemax_{_dt.now().strftime('%Y%m%d')}.pdf"
        part.add_header("Content-Disposition", f"attachment; filename={fname}")
        msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, recipients, msg.as_string())
        return True, f"✅ Reporte enviado a: {', '.join(recipients)}"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Error de autenticación. Verifica tu correo y contraseña de aplicación."
    except smtplib.SMTPException as e:
        return False, f"❌ Error al enviar: {str(e)}"
    except Exception as e:
        return False, f"❌ Error inesperado: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def kpi(col, label, value, color=OM_RED):
    col.markdown(
        f'<div style="background:white;border-radius:12px;border-top:4px solid {color};'
        f'padding:20px 16px;text-align:center;min-height:110px;display:flex;'
        f'flex-direction:column;justify-content:center;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:8px;">'
        f'<div style="font-size:clamp(1.2rem,3vw,1.8rem);font-weight:800;color:#1A1A1A;'
        f'word-break:break-word;overflow-wrap:anywhere;">{value}</div>'
        f'<div style="font-size:0.7rem;font-weight:600;color:#666;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-top:8px;white-space:nowrap;">{label}</div>'
        f'</div>', unsafe_allow_html=True)

def traffic_light(r2, beta, pval, rmse):
    score = 0; reasons = []
    if r2 >= 0.4:    score+=2; reasons.append(f"✅ R² = {r2:.3f} — bueno")
    elif r2 >= 0.15: score+=1; reasons.append(f"⚠️ R² = {r2:.3f} — aceptable (normal en retail)")
    else:                      reasons.append(f"❌ R² = {r2:.3f} — bajo")
    if pval < 0.05:  score+=2; reasons.append(f"✅ p-valor = {pval:.4f} — significativo")
    elif pval < 0.10:score+=1; reasons.append(f"⚠️ p-valor = {pval:.4f} — marginalmente significativo")
    else:                      reasons.append(f"❌ p-valor = {pval:.4f} — no significativo")
    if beta < 0 and abs(beta) <= 5:
        score+=1; reasons.append(f"✅ Beta = {beta:.4f} — negativa e interpretable")
    else:            reasons.append(f"❌ Beta = {beta:.4f} — fuera de rango económico")
    if rmse < 0.5:   score+=1; reasons.append(f"✅ RMSE = {rmse:.4f} — error bajo")
    else:            reasons.append(f"⚠️ RMSE = {rmse:.4f} — error moderado (normal con muchas tiendas)")
    if score >= 5:   css,msg = "chip-green",  "🟢 Modelo confiable — resultados interpretables"
    elif score >= 3: css,msg = "chip-yellow", "🟡 Modelo aceptable — úsalo con precaución"
    else:            css,msg = "chip-red",    "🔴 Modelo débil — reduce los parámetros mínimos"
    return css, msg, reasons

def _layout(fig, h=320):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      height=h, margin=dict(t=45,b=20,l=20,r=20),
                      font=dict(family="Roboto, Arial"))
    return fig

def _fmt_bar(fig, values, prefix="$", suffix="", decimals=0):
    """Add formatted text labels to a bar chart without text_auto."""
    fmt = f"{prefix}{{:,.{decimals}f}}{suffix}"
    fig.update_traces(text=[fmt.format(v) for v in values], textposition="outside",
                      textfont=dict(size=11))
    return fig


# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def run_ml_pipeline(df_csv: bytes, promo_bytes: bytes):
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(io.BytesIO(df_csv), low_memory=False)
    df["tran_date"]      = pd.to_datetime(df["tran_date"], errors="coerce")
    df["mes_str"]        = df["tran_date"].dt.to_period("M").astype(str)
    df["mes_cal"]        = df["tran_date"].dt.month
    df["venta_con_iva"]  = pd.to_numeric(df["venta_con_iva"], errors="coerce")
    df["qty"]            = pd.to_numeric(df["qty"],           errors="coerce")
    df["precio_catalogo"]= pd.to_numeric(df["precio_catalogo"], errors="coerce") \
                           if "precio_catalogo" in df.columns else pd.Series(np.nan, index=df.index)

    mensual = (df.groupby(["prod_nbr","mes_str"])
               .agg(unidades=("qty","sum"), venta_tot=("venta_con_iva","sum"),
                    mes_cal=("mes_cal","first"), es_premium=("es_premium","max"),
                    dept_nm=("dept_nm","first"), precio_catalogo=("precio_catalogo","mean"))
               .reset_index())
    mensual["precio"] = mensual["venta_tot"] / mensual["unidades"]
    mensual = mensual[mensual["precio"] > 0].copy()
    mensual["log_precio"]   = np.log(mensual["precio"])
    mensual["log_unidades"] = np.log1p(mensual["unidades"])
    mensual["log_venta"]    = np.log1p(mensual["venta_tot"])
    mensual["precio_vs_cat"] = (mensual["precio"] / mensual["precio_catalogo"].replace(0, np.nan)).fillna(1.0).clip(0.5, 1.5)
    mensual = mensual.sort_values(["prod_nbr","mes_str"]).reset_index(drop=True)

    # ── Detección promos jerarquía 3 niveles ──────────────────────────────────
    # N1: boletín oficial (si SKU coincide; si no, igual se usa N2 y N3)
    oficial_set = set()
    if promo_bytes and len(promo_bytes) > 0:
        try:
            pdf = pd.read_excel(io.BytesIO(promo_bytes))
            pdf["SKU"] = pdf["SKU"].astype(str).str.replace(".0","",regex=False).str.strip()
            pdf["Fecha_Inicio"] = pd.to_datetime(pdf["Fecha_Inicio"], dayfirst=True, errors="coerce")
            pdf["Fecha_Fin"]    = pd.to_datetime(pdf["Fecha_Fin"],    dayfirst=True, errors="coerce")
            pdf["_flag"] = pd.to_numeric(pdf["Promo_Flag"], errors="coerce")
            pdf = pdf[(pdf["_flag"]==1) & pdf["Fecha_Inicio"].notna() & pdf["Fecha_Fin"].notna()]
            for _, row in pdf.iterrows():
                try:
                    for m in pd.period_range(row["Fecha_Inicio"], row["Fecha_Fin"], freq="M"):
                        oficial_set.add((row["SKU"], str(m)))
                except: continue
        except: pass

    mensual["promo_official"] = mensual.apply(
        lambda r: 1 if (r["prod_nbr"], r["mes_str"]) in oficial_set else 0, axis=1)

    # N2: caída de precio ≥ 20% vs precio modal del SKU
    precio_modal = mensual.groupby("prod_nbr")["precio"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median())
    mensual["precio_modal"] = mensual["prod_nbr"].map(precio_modal)
    mensual["pct_vs_modal"] = mensual["precio"] / mensual["precio_modal"].replace(0, np.nan)
    mensual["promo_priceX"] = (mensual["pct_vs_modal"] <= 0.80).astype(int)
    mensual["pct_desc"]     = (1 - mensual["pct_vs_modal"].clip(upper=1.0)).clip(lower=0.0).fillna(0.0)

    # N3: spike de volumen ≥ p90 (solo SKUs sin N1 ni N2)
    p90 = mensual.groupby("prod_nbr")["unidades"].quantile(0.90)
    mensual["p90_units"] = mensual["prod_nbr"].map(p90)
    mensual["spike_raw"] = (mensual["unidades"] >= mensual["p90_units"]).astype(int)
    sku_l1l2 = (mensual.groupby("prod_nbr")
                .apply(lambda g: (g["promo_official"].sum() + g["promo_priceX"].sum()) > 0)
                .reset_index(name="tiene_l1_l2"))
    mensual = mensual.merge(sku_l1l2, on="prod_nbr", how="left")
    mensual["promo_spike"] = (mensual["spike_raw"].astype(bool) & ~mensual["tiene_l1_l2"]).astype(int)
    mensual["tiene_promo"] = ((mensual["promo_official"]==1) | (mensual["promo_priceX"]==1) | (mensual["promo_spike"]==1)).astype(int)
    mensual["nivel_promo"] = "Sin promo"
    mensual.loc[mensual["promo_spike"]==1,   "nivel_promo"] = "N3 - Spike volumen"
    mensual.loc[mensual["promo_priceX"]==1,  "nivel_promo"] = "N2 - Caida precio"
    mensual.loc[mensual["promo_official"]==1,"nivel_promo"] = "N1 - Boletin oficial"

    # ── Features de lag ───────────────────────────────────────────────────────
    grp = mensual.groupby("prod_nbr")
    mensual["uds_lag1"]       = grp["unidades"].shift(1)
    mensual["uds_roll3m"]     = grp["unidades"].shift(1).rolling(3, min_periods=1).mean().values
    mensual["precio_lag1"]    = grp["precio"].shift(1)
    mensual["precio_chg_pct"] = (mensual["precio"] / mensual["precio_lag1"].replace(0,np.nan) - 1).clip(-0.5,0.5)
    mensual["log_uds_lag1"]   = np.log1p(mensual["uds_lag1"].fillna(0))
    mensual["log_uds_roll3m"] = np.log1p(mensual["uds_roll3m"].fillna(0))
    mes_index = {m: i for i, m in enumerate(sorted(mensual["mes_str"].unique()))}
    mensual["mes_num"] = mensual["mes_str"].map(mes_index)

    skus_ok = mensual.groupby("prod_nbr")["mes_str"].count()
    skus_ok = skus_ok[skus_ok >= 6].index
    mensual = mensual[mensual["prod_nbr"].isin(skus_ok)].dropna(subset=["log_uds_lag1"]).reset_index(drop=True)

    le = LabelEncoder()
    mensual["dept_enc"] = le.fit_transform(mensual["dept_nm"].fillna("Unknown"))

    FEAT_COLS  = ["log_precio","mes_cal","mes_num","es_premium","dept_enc",
                  "pct_desc","precio_vs_cat","precio_chg_pct","log_uds_lag1","log_uds_roll3m"]
    FEAT_NAMES = ["Log Precio","Mes del año","Tendencia temporal","Es Premium","Departamento",
                  "% Descuento vs modal","Precio vs Catálogo","Cambio precio % (mes ant.)",
                  "Ventas mes anterior (log)","Media ventas 3m (log)"]

    X = mensual[FEAT_COLS].fillna(0)
    y_u, y_r = mensual["log_unidades"], mensual["log_venta"]
    all_months = sorted(mensual["mes_str"].unique())
    split_idx  = int(len(all_months) * 0.8)
    train_mask = mensual["mes_str"].isin(set(all_months[:split_idx]))
    test_mask  = mensual["mes_str"].isin(set(all_months[split_idx:]))
    X_train, X_test = X[train_mask], X[test_mask]

    # ── Entrenar RF y GB ──────────────────────────────────────────────────────
    comparacion = {}
    for nombre, (mu, mr) in {
        "Random Forest":      (RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
                               RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        "Gradient Boosting":  (HistGradientBoostingRegressor(max_iter=80, max_depth=5, learning_rate=0.05, random_state=42),
                               HistGradientBoostingRegressor(max_iter=80, max_depth=5, learning_rate=0.05, random_state=42)),
    }.items():
        mu.fit(X_train, y_u[train_mask]); mr.fit(X_train, y_r[train_mask])
        r2u = r2_score(y_u[test_mask], mu.predict(X_test))
        r2r = r2_score(y_r[test_mask], mr.predict(X_test))
        comparacion[nombre] = {"mod_u":mu,"mod_r":mr,"r2_u":round(r2u,4),"r2_r":round(r2r,4),
                                "r2_avg":round((r2u+r2r)/2,4),
                                "rmse_u":round(float(np.sqrt(mean_squared_error(y_u[test_mask],mu.predict(X_test)))),4),
                                "rmse_r":round(float(np.sqrt(mean_squared_error(y_r[test_mask],mr.predict(X_test)))),4)}

    ganador_nm = max(comparacion, key=lambda k: comparacion[k]["r2_avg"])
    gan = comparacion[ganador_nm]
    # HistGradientBoosting no expone feature_importances_ → usar permutation importance o RF fallback
    def _get_imp(model):
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        # Para HistGradientBoosting: usar el RF del comparacion como proxy
        rf_mod = comparacion.get("Random Forest",{}).get("mod_u")
        return rf_mod.feature_importances_ if rf_mod and hasattr(rf_mod,"feature_importances_") else np.ones(len(FEAT_COLS))/len(FEAT_COLS)
    imp = (_get_imp(gan["mod_u"]) + _get_imp(gan["mod_r"])) / 2
    feat_df = pd.DataFrame({"feature":FEAT_NAMES,"importancia":imp}).sort_values("importancia",ascending=False)

    pred_u = gan["mod_u"].predict(X_test); pred_r = gan["mod_r"].predict(X_test)
    n_s = min(800, len(pred_u)); idx_s = np.random.choice(len(pred_u), n_s, replace=False)

    sin = mensual[mensual["tiene_promo"]==0]["unidades"].mean()
    con = mensual[mensual["tiene_promo"]==1]["unidades"].mean() if mensual["tiene_promo"].sum()>0 else sin
    uplift = (con-sin)/sin*100 if sin>0 else 0
    promo_by_nivel = (mensual.groupby("nivel_promo").agg(n=("prod_nbr","count"),avg_u=("unidades","mean")).reset_index())
    promo_by_nivel["uplift"] = (promo_by_nivel["avg_u"]-sin)/sin*100

    mt = mensual[test_mask].copy()
    Xb = mt[FEAT_COLS].fillna(0).copy(); Xl = Xb.copy()
    Xl["log_precio"] += np.log(0.90); Xl["pct_desc"] = (Xb["pct_desc"]+0.10).clip(0,1)
    pb = np.expm1(gan["mod_u"].predict(Xb)); pl = np.expm1(gan["mod_u"].predict(Xl))
    mt["uplift_precio"] = (pl-pb)/(pb+1e-9)*100
    dept_map = mensual[["prod_nbr","dept_nm"]].drop_duplicates("prod_nbr").set_index("prod_nbr")["dept_nm"]
    top_sens = (mt.groupby("prod_nbr").agg(uplift=("uplift_precio","mean")).reset_index()
                .sort_values("uplift",ascending=False).head(10))
    top_sens["dept"] = top_sens["prod_nbr"].map(dept_map)

    # ── Heatmap timing: uplift promo por dept × mes y por SKU × mes ─────────
    mensual["mes_nombre"] = mensual["mes_cal"].map(MONTH_NAMES)

    # Base (sin promo) por dept × mes
    base_dm = (mensual[mensual["tiene_promo"]==0]
               .groupby(["dept_nm","mes_cal"])["unidades"].mean().rename("base"))
    promo_dm = (mensual[mensual["tiene_promo"]==1]
                .groupby(["dept_nm","mes_cal"])["unidades"].mean().rename("promo"))
    dept_mes = (pd.concat([base_dm, promo_dm], axis=1).reset_index())
    dept_mes["uplift_pct"] = ((dept_mes["promo"] - dept_mes["base"]) / dept_mes["base"] * 100).round(1)
    dept_mes["mes_nombre"] = dept_mes["mes_cal"].map(MONTH_NAMES)

    # Pivot para heatmap dept
    dept_pivot = (dept_mes.pivot_table(index="dept_nm", columns="mes_cal",
                                        values="uplift_pct", aggfunc="mean")
                  .reindex(columns=range(1,13)))
    dept_pivot.columns = [MONTH_NAMES[c] for c in dept_pivot.columns]

    # SKU × mes (solo SKUs con >= 5 meses de observaciones en promo)
    skus_con_promo = (mensual[mensual["tiene_promo"]==1]
                      .groupby("prod_nbr")["mes_cal"].count())
    skus_con_promo = skus_con_promo[skus_con_promo >= 3].index
    base_sm = (mensual[(mensual["tiene_promo"]==0) & mensual["prod_nbr"].isin(skus_con_promo)]
               .groupby(["prod_nbr","mes_cal"])["unidades"].mean().rename("base"))
    promo_sm = (mensual[(mensual["tiene_promo"]==1) & mensual["prod_nbr"].isin(skus_con_promo)]
                .groupby(["prod_nbr","mes_cal"])["unidades"].mean().rename("promo"))
    sku_mes = pd.concat([base_sm, promo_sm], axis=1).reset_index()
    sku_mes["uplift_pct"] = ((sku_mes["promo"] - sku_mes["base"]) / sku_mes["base"] * 100).round(1)
    sku_mes["dept_nm"] = sku_mes["prod_nbr"].map(dept_map)
    sku_mes["mes_nombre"] = sku_mes["mes_cal"].map(MONTH_NAMES)

    # ── Evaluación pre/durante/post por evento de promo (N2: caída >= 20%) ───
    VENTANA = 2
    all_months_ord = sorted(mensual["mes_str"].unique())
    mes_idx_ev = {m: i for i, m in enumerate(all_months_ord)}
    mensual["mes_num_ev"] = mensual["mes_str"].map(mes_idx_ev)
    modal_p = mensual.groupby("prod_nbr")["precio"].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else x.median())
    mensual["precio_modal_ev"] = mensual["prod_nbr"].map(modal_p)
    mensual["caida_pct_ev"]    = ((mensual["precio_modal_ev"] - mensual["precio"]) / mensual["precio_modal_ev"] * 100).fillna(0)

    # Detectar eventos: secuencias consecutivas de meses con caída >= 20%
    eventos_ev = []
    prod_nm_map = (mensual[["prod_nbr","prod_nm"]].drop_duplicates("prod_nbr")
                   .set_index("prod_nbr")["prod_nm"]) if "prod_nm" in mensual.columns else {}
    for sku, grp in mensual.groupby("prod_nbr"):
        grp = grp.sort_values("mes_num_ev")
        en_promo, meses_ev = False, []
        for _, row in grp.iterrows():
            if row["caida_pct_ev"] >= 20:
                en_promo = True; meses_ev.append(int(row["mes_num_ev"]))
            else:
                if en_promo and meses_ev:
                    eventos_ev.append({"prod_nbr":sku,"mes_i":min(meses_ev),"mes_f":max(meses_ev),
                                       "desc_pct":grp[grp["mes_num_ev"].isin(meses_ev)]["caida_pct_ev"].mean()})
                meses_ev = []; en_promo = False

    eval_rows = []
    for ev in eventos_ev:
        sku = ev["prod_nbr"]
        grp = mensual[mensual["prod_nbr"]==sku].set_index("mes_num_ev")
        def _avg(meses, col):
            vals = [grp.loc[m, col] for m in meses if m in grp.index]
            return float(np.mean(vals)) if vals else np.nan
        pre   = list(range(ev["mes_i"]-VENTANA, ev["mes_i"]))
        dur   = list(range(ev["mes_i"], ev["mes_f"]+1))
        post  = list(range(ev["mes_f"]+1, ev["mes_f"]+1+VENTANA))
        u_pre = _avg(pre,"unidades"); u_dur = _avg(dur,"unidades"); u_post = _avg(post,"unidades")
        r_pre = _avg(pre,"venta_tot"); r_dur = _avg(dur,"venta_tot")
        if np.isnan(u_pre) or u_pre==0: continue
        up_dur  = (u_dur  - u_pre) / u_pre * 100
        up_post = (u_post - u_pre) / u_pre * 100 if not np.isnan(u_post) else np.nan
        rv_up   = (r_dur  - r_pre) / r_pre * 100  if not np.isnan(r_dur)  else np.nan
        if   np.isnan(up_post):  retencion = "Sin datos"
        elif up_post >= 5:       retencion = "✅ Sostuvo"
        elif up_post >= -5:      retencion = "➡️ Volvió a normal"
        else:                    retencion = "❌ Cayó post-promo"
        eval_rows.append({
            "prod_nm":  str(prod_nm_map.get(sku, sku))[:38],
            "dept_nm":  str(dept_map.get(sku,"—"))[:22],
            "desc_pct": round(ev["desc_pct"],1),
            "uds_pre":  round(u_pre,1), "uds_dur": round(u_dur,1),
            "uds_post": round(u_post,1) if not np.isnan(u_post) else None,
            "up_dur":   round(up_dur,1),
            "up_post":  round(up_post,1) if not np.isnan(up_post) else None,
            "rv_up":    round(rv_up,1)   if not np.isnan(rv_up)   else None,
            "rentable": "✅ Sí" if (rv_up is not None and rv_up>0) else "❌ No",
            "retencion":retencion,
        })
    promo_eval = pd.DataFrame(eval_rows).sort_values("up_dur", ascending=False) if eval_rows else pd.DataFrame()

    comp_out = {k:{kk:vv for kk,vv in v.items() if kk not in ("mod_u","mod_r")} for k,v in comparacion.items()}
    return {"ganador":ganador_nm,"comparacion":comp_out,"feat_df":feat_df,
            "r2_u":gan["r2_u"],"r2_r":gan["r2_r"],
            "actual_u":np.expm1(y_u[test_mask].values[idx_s]).tolist(),
            "fitted_u":np.expm1(pred_u[idx_s]).tolist(),
            "actual_r":np.expm1(y_r[test_mask].values[idx_s]).tolist(),
            "fitted_r":np.expm1(pred_r[idx_s]).tolist(),
            "n_train":int(train_mask.sum()),"n_test":int(test_mask.sum()),
            "train_range":f"{all_months[0]} - {all_months[split_idx-1]}",
            "test_range":f"{all_months[split_idx]} - {all_months[-1]}",
            "n_skus":len(skus_ok),"promo_uplift":round(uplift,1),
            "promo_by_nivel":promo_by_nivel,
            "n_promo":int(mensual["tiene_promo"].sum()),"n_total":len(mensual),
            "top_sens":top_sens,
            "dept_pivot":dept_pivot,"sku_mes":sku_mes,"promo_eval":promo_eval}




# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ────────────────────────────────────────────────────────────────
    try:
        st.image("assets/logo-officemax-ball.png", width=220)
    except Exception:
        st.markdown("""
        <div class="sidebar-logo-wrap">
            <span style="font-size:26px;font-weight:900;color:#E31837;">OFFICEMAX</span><br>
            <span style="font-size:9px;color:#aaa;letter-spacing:2px;">DYNAMIC PRICING ANALYZER</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── Sección 1: Carga de datos ────────────────────────────────────────────
    st.markdown('<p class="sidebar-section-label">📦 Carga tus datos</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-hint">Tu ZIP debe contener 4 archivos: Ventas · Catálogo · Precios · Costos</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Archivo ZIP con datos", type=["zip"], label_visibility="collapsed")

    if uploaded:
        is_new_file = st.session_state.get("loaded_file_name") != uploaded.name
        if is_new_file:
            raw_bytes = uploaded.read()
            with st.spinner("Leyendo y limpiando datos..."):
                try:
                    df_main, report_df = load_and_clean(raw_bytes)
                    st.session_state["df_main"]          = df_main
                    st.session_state["clean_report"]     = report_df
                    st.session_state["results"]          = None
                    st.session_state["df_csv_bytes"]     = df_main.to_csv(index=False).encode("utf-8")
                    st.session_state["loaded_file_name"] = uploaded.name
                    _detail = report_df.iloc[-1]["Detalle"]
                    st.markdown(f'<div class="sidebar-status-ok">✅ Listo — {_detail}</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="sidebar-status-err">❌ Error: {e}</div>',
                                unsafe_allow_html=True)
        else:
            _rpt = st.session_state.get("clean_report")
            _det = _rpt.iloc[-1]["Detalle"] if _rpt is not None else "Datos en memoria"
            st.markdown(f'<div class="sidebar-status-ok">✅ Listo — {_det}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-status-err">⬆️ Sube tu archivo ZIP para comenzar</div>',
                    unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── Sección 2: Promociones ───────────────────────────────────────────────
    st.markdown('<p class="sidebar-section-label">🏷️ Promociones (opcional)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-hint">Sube tu calendario de promos para ver cuáles realmente funcionaron</p>',
                unsafe_allow_html=True)
    uploaded_promo = st.file_uploader("Excel de promociones", type=["xlsx","xls"],
                                       label_visibility="collapsed", key="promo_uploader")
    if uploaded_promo:
        if st.session_state.get("promo_file_name") != uploaded_promo.name:
            st.session_state["promo_bytes"]     = uploaded_promo.read()
            st.session_state["promo_file_name"] = uploaded_promo.name
            st.session_state["ml_results"]      = None
        st.markdown('<div class="sidebar-status-ok">✅ Promociones cargadas</div>', unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── Sección 3: Configuración avanzada (colapsada) ────────────────────────
    with st.expander("⚙️ Configuración avanzada", expanded=False):
        st.caption("Los valores predeterminados funcionan bien para datos de OfficeMax. Cámbialos solo si salen muy pocos productos.")
        min_obs = st.slider("Meses mínimos por producto", 3, 18, 6,
            help="Mínimo de meses con datos para analizar un producto. Baja a 4-5 si salen muy pocos.")
        min_r2 = st.slider("Precisión mínima del modelo (R²)", 0.0, 0.5, 0.0, 0.05,
            help="En retail, 0.10-0.20 ya es aceptable. No uses 0.5+.")
        max_beta = st.slider("Sensibilidad máxima al precio", 2.0, 15.0, 10.0, 0.5,
            help="Elasticidades mayores a 5 suelen ser errores estadísticos.")
        min_cv_pct = st.slider("Variación mínima de precio (%)", 0.0, 10.0, 1.0, 0.5,
            help="Si el precio nunca cambió, no se puede estimar la sensibilidad. 1% es permisivo.")
        min_cv = min_cv_pct / 100
        pval_thresh = st.slider("Nivel de confianza estadística", 0.05, 0.25, 0.10, 0.05,
            help="0.10 = 90% de confianza (estándar en análisis de negocio).")
        run_rolling = st.checkbox(
            "Calcular tendencia trimestral y semestral (~1-3 min extra)", value=False,
            help="Activa para ver cómo cambia la sensibilidad al precio a lo largo del tiempo.")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── Sección 4: Configuración de correo ───────────────────────────────────
    _default_email = os.environ.get("REPORT_EMAIL_FROM", "")
    _default_pass  = os.environ.get("REPORT_EMAIL_PASS",  "")
    with st.expander("📧 Configuración de correo", expanded=False):
        st.caption("Para enviar reportes por correo. Usa una cuenta Gmail con contraseña de aplicación.")
        _ef = st.text_input("Tu correo Gmail",
                             value=st.session_state.get("email_from") or _default_email,
                             placeholder="tucorreo@gmail.com", key="email_from_input")
        _ep = st.text_input("Contraseña de aplicación Gmail", type="password",
                             value=st.session_state.get("email_pass") or _default_pass,
                             placeholder="xxxx xxxx xxxx xxxx", key="email_pass_input")
        if _ef: st.session_state["email_from"] = _ef
        if _ep: st.session_state["email_pass"] = _ep
        st.caption("⚠️ Usa una contraseña de aplicación, no tu contraseña de Gmail. "
                   "Créala en: Cuenta Google → Seguridad → Verificación en 2 pasos → Contraseñas de aplicación")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── Botón principal CTA ──────────────────────────────────────────────────
    _datos_listos = st.session_state.get("df_csv_bytes") is not None
    run_btn = st.button(
        "🔍 Analizar mi catálogo",
        type="primary",
        use_container_width=True,
        disabled=not _datos_listos,
    )
    if not _datos_listos:
        st.markdown('<p class="sidebar-hint" style="text-align:center;">Sube primero tu archivo ZIP</p>',
                    unsafe_allow_html=True)

    if run_btn and _datos_listos:
        csv_bytes = st.session_state["df_csv_bytes"]
        with st.spinner("⚙️ Analizando tu catálogo… esto tarda unos 30 segundos"):
            try:
                res = run_models(df_csv=csv_bytes, min_obs=min_obs, min_cv=min_cv,
                                 min_r2=min_r2, max_beta=max_beta, pval_thresh=pval_thresh,
                                 run_rolling=run_rolling)
                st.session_state["results"] = res
            except Exception as e:
                st.error(f"Error en el análisis: {e}")
        with st.spinner("🌲 Entrenando modelos de predicción…"):
            try:
                promo_b = st.session_state.get("promo_bytes") or b""
                ml_res = run_ml_pipeline(csv_bytes, promo_b)
                st.session_state["ml_results"] = ml_res
            except Exception as e:
                st.warning(f"Análisis ML no completado: {e}")
        st.success("✅ Listo — revisa las pestañas")
        st.session_state["ai_analysis"] = None


# ══════════════════════════════════════════════════════════════════════════════
# LANDING
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
                El ZIP debe contener: Ventas · Catálogo · Precios · Costos
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ML PIPELINE — RF vs Gradient Boosting + Promo Detection
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["💡  Recomendaciones","📊  Resumen de Ventas","🎯  Inteligencia de Precios"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALCULADORA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    results = st.session_state["results"]
    if results is None:
        st.info("⬅️ Sube tu archivo ZIP y presiona **🔍 Analizar mi catálogo** en el panel izquierdo para ver las recomendaciones.")
        st.stop()

    df_m1a = results["m1a"]; m2 = results["m2"]
    n_valid = len(df_m1a[df_m1a["recomendacion"]!="No recomendable"]) if len(df_m1a)>0 else 0

    # ── 4 KPI cards arriba ───────────────────────────────────────────────────
    rec_counts_t1 = df_m1a["recomendacion"].value_counts() if len(df_m1a)>0 else {}
    n_sub_t1 = int(rec_counts_t1.get("Subir precio",0))
    n_pro_t1 = int(rec_counts_t1.get("Bajar / Promover",0))
    _imp_t1 = 0
    if len(results["sim"])>0:
        def _qdelta(recs, esc):
            ns = df_m1a[df_m1a["recomendacion"].isin(recs)]["prod_nm"].tolist()
            b = results["sim"][(results["sim"]["prod_nm"].isin(ns))&(results["sim"]["cambio"]=="Base 0%")]["ingreso_est"].sum()
            t = results["sim"][(results["sim"]["prod_nm"].isin(ns))&(results["sim"]["cambio"]==esc)]["ingreso_est"].sum()
            return t - b
        _imp_t1 = _qdelta(["Subir precio"],"+10%") + _qdelta(["Bajar / Promover"],"-10%")

    kc1,kc2,kc3,kc4 = st.columns(4)
    kpi(kc1, "Productos analizados",       f"{len(df_m1a):,}",            OM_RED)
    kpi(kc2, "✅ Sube el precio",           f"{n_sub_t1} productos",        OM_BLUE)
    kpi(kc3, "📢 Lanza una promoción",      f"{n_pro_t1} productos",        OM_GREEN)
    kpi(kc4, "Impacto mensual estimado",   f"+${_imp_t1:,.0f} MXN",        OM_AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Validación de calidad ────────────────────────────────────────────────
    with st.expander("📋 Detalle de calidad de datos", expanded=False):
        clean_rpt = st.session_state.get("clean_report")
        if clean_rpt is not None:
            for _, row in clean_rpt.iterrows():
                c = OM_GREEN if "✅" in str(row["Paso"]) else (OM_RED if "🗑" in str(row["Paso"]) else OM_BLUE)
                st.markdown(
                    f'<div class="clean-step" style="border-left-color:{c};">'
                    f'<strong>{row["Paso"]}</strong>&nbsp;&nbsp;'
                    f'<span style="color:{OM_RED};font-weight:700;">{row["Eliminadas"]}</span>'
                    f'&nbsp;&nbsp;<span style="color:#555;">{row["Detalle"]}</span></div>',
                    unsafe_allow_html=True)
        if len(df_m1a) > 0:
            validos_df = df_m1a[df_m1a["recomendacion"]!="No recomendable"]
            r2_validos = validos_df["r2"].mean() if len(validos_df)>0 else 0
            pct_sig    = (df_m1a["pval"] < 0.10).mean() * 100
            if r2_validos >= 0.35 and pct_sig >= 30:
                css_v, msg_v = "chip-green",  "🟢 Modelo estadísticamente confiable"
            elif r2_validos >= 0.15 and pct_sig >= 15:
                css_v, msg_v = "chip-yellow", "🟡 Modelo aceptable — úsalo con precaución"
            else:
                css_v, msg_v = "chip-red",    "🔴 Señal débil — reduce filtros en Configuración avanzada"
            st.markdown(f'<br><div class="{css_v}">{msg_v}</div>', unsafe_allow_html=True)
            st.caption(f"Precisión promedio del modelo: {r2_validos:.1%} · "
                       f"Productos estadísticamente confiables: {pct_sig:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    section(f"📦 Plan de acción — {n_valid} productos con recomendación de {len(df_m1a):,} analizados")
    st.caption("Filtra por departamento o recomendación para encontrar los productos que te interesan.")

    if len(df_m1a) == 0:
        st.warning("⚠️ Ningún producto pasó los filtros. Prueba bajando el mínimo de meses o la variación de precio en Configuración avanzada.")
    else:
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            depts = sorted(df_m1a["dept_nm"].dropna().unique().tolist())
            dept_f = st.multiselect("Departamento", depts, key="dept_f_cal")
        with fc2:
            _rec_labels = {
                "Subir precio":     "✅ Sube el precio",
                "Mantener precio":  "➡️ Mantén el precio",
                "Bajar / Promover": "📢 Lanza una promoción",
                "No recomendable":  "⚪ Sin recomendación",
            }
            rec_f = st.multiselect("Recomendación", list(_rec_labels.values()), key="rec_f_cal")
            _rec_reverse = {v:k for k,v in _rec_labels.items()}
            rec_f_orig = [_rec_reverse[r] for r in rec_f if r in _rec_reverse]
        with fc3:
            srch = st.text_input("Buscar producto o SKU", placeholder="Ej: FOLDER, 50012983")

        show = df_m1a.copy()
        if dept_f:    show = show[show["dept_nm"].isin(dept_f)]
        if rec_f_orig:show = show[show["recomendacion"].isin(rec_f_orig)]
        if srch:
            show = show[show["prod_nm"].str.contains(srch,case=False,na=False)|
                        show["prod_nbr"].str.contains(srch,case=False,na=False)]

        # Tabla principal — columnas de negocio
        show_disp = show[["prod_nbr","prod_nm","dept_nm","recomendacion","n_meses","pval"]].copy()
        show_disp["accion_sugerida"] = show_disp["recomendacion"].map({
            "Subir precio":     "✅ Sube el precio — los clientes no se van",
            "Mantener precio":  "➡️ Mantén el precio — estás en el punto óptimo",
            "Bajar / Promover": "📢 Lanza una promoción — bajar el precio genera volumen",
            "No recomendable":  "⚪ Datos insuficientes",
        })
        show_disp["confiabilidad"] = show_disp["pval"].apply(
            lambda p: "Alta ✅" if p < 0.05 else ("Media ⚠️" if p < 0.10 else "Baja ❌"))
        disp_cols = ["prod_nbr","prod_nm","dept_nm","accion_sugerida","n_meses","confiabilidad"]
        st.dataframe(
            show_disp[disp_cols].rename(columns={
                "prod_nbr":"SKU","prod_nm":"Producto","dept_nm":"Departamento",
                "accion_sugerida":"Acción Sugerida","n_meses":"Meses de datos",
                "confiabilidad":"Confiabilidad"}),
            hide_index=True, use_container_width=True, height=380)

        with st.expander("🔬 Ver detalles técnicos (elasticidades, R², p-valor)", expanded=False):
            st.caption("Índice de sensibilidad al precio (β): valor negativo = más precio → menos ventas. "
                       "Entre más negativo, más sensibles son los clientes.")
            tech_cols = [c for c in ["prod_nbr","prod_nm","dept_nm","beta","r2","rmse","pval","n_meses","recomendacion"] if c in show.columns]
            st.dataframe(show[tech_cols].rename(columns={
                "prod_nbr":"SKU","prod_nm":"Producto","dept_nm":"Departamento",
                "beta":"Índice de sensibilidad al precio",
                "r2":"Precisión del modelo (R²)","rmse":"Error medio (RMSE)",
                "pval":"Confianza estadística (p-valor)",
                "n_meses":"Meses de datos","recomendacion":"Recomendación"
            }).sort_values("Índice de sensibilidad al precio"),
            hide_index=True, use_container_width=True, height=300)

        st.markdown("<br>", unsafe_allow_html=True)
        dl1,dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Descargar plan de acción (CSV)",
                df_m1a.to_csv(index=False).encode("utf-8"),"plan_de_accion.csv","text/csv")
        with dl2:
            if len(results["sim"])>0:
                st.download_button("⬇️ Descargar simulación de precios (CSV)",
                    results["sim"].to_csv(index=False).encode("utf-8"),"simulacion_precios.csv","text/csv")

        # ── Validación visual del modelo (usa ML si está disponible) ────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _ml_scatter = st.session_state.get("ml_results")
        if _ml_scatter and _ml_scatter.get("actual_r"):
            section("🎯 ¿Qué tan preciso es el modelo de predicción?")
            st.caption("El modelo aprendió con datos históricos y se probó con ventas que nunca había visto. "
                       "Cada punto es un mes de ventas de un producto — entre más cerca de la línea diagonal, mejor predice.")
            actual = _ml_scatter["actual_r"]
            fitted = _ml_scatter["fitted_r"]
            r2_show = _ml_scatter["r2_r"]
            n_train = _ml_scatter["n_train"]
            n_test  = _ml_scatter["n_test"]
            model_label = "Modelo de predicción de ventas"
        else:
            section("🎯 ¿Qué tan preciso es el modelo de predicción?")
            st.caption("Comparación entre ventas reales y ventas estimadas por el modelo.")
            actual = m2.get("actual_sample", [])
            fitted = m2.get("fitted_sample", [])
            r2_show = m2["r2"]
            n_train = m2["n_obs"]
            n_test  = 0
            model_label = "Modelo de elasticidad"

        if actual and fitted:
            vv1, vv2 = st.columns([2, 1])
            with vv1:
                max_val = float(np.percentile([v for v in actual+fitted if v < 1e9], 95))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual, y=fitted, mode="markers",
                    marker=dict(size=4, color=OM_RED, opacity=0.35),
                    hovertemplate="Ventas reales: $%{x:,.0f}<br>Ventas estimadas: $%{y:,.0f}<extra></extra>",
                    name="Observaciones"))
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines", line=dict(color=OM_BLUE, dash="dash", width=2),
                    name="Predicción perfecta"))
                fig.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=380, margin=dict(t=30,b=40,l=40,r=20),
                    xaxis=dict(title="Ventas reales ($)", range=[0, max_val]),
                    yaxis=dict(title="Ventas estimadas por el modelo ($)", range=[0, max_val]),
                    legend=dict(orientation="h", y=1.05))
                st.plotly_chart(fig, use_container_width=True)

            with vv2:
                r2_pct  = r2_show * 100
                r2_color = OM_GREEN if r2_show >= 0.4 else (OM_AMBER if r2_show >= 0.15 else OM_RED)
                st.markdown(f"""
                <div style="background:white;border-radius:14px;padding:24px;
                            text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            border-top:5px solid {r2_color}; margin-bottom:16px;">
                    <div style="font-size:48px;font-weight:900;color:{r2_color};">{r2_pct:.1f}%</div>
                    <div style="font-size:13px;color:#555;margin-top:6px;">
                        Precisión del modelo de predicción de ventas
                    </div>
                    <div style="font-size:11px;color:#999;margin-top:4px;">{model_label}</div>
                </div>
                <div style="background:white;border-radius:14px;padding:18px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size:12px;color:#555;line-height:2.2;">
                    <b>Registros de entrenamiento:</b> {n_train:,}<br>
                    <b>Registros de prueba:</b> {n_test:,}<br>
                    <b>Productos con recomendación:</b> {n_valid}
                    </div>
                </div>
                """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# CACHED DESCRIPTIVO AGGREGATIONS  (evita recalcular al mover filtros en Tab 3)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_descriptivo(df_csv: bytes, dept_tuple, year_tuple, marca_tuple):
    df = pd.read_csv(io.BytesIO(df_csv), low_memory=False)
    df["tran_date"]     = pd.to_datetime(df["tran_date"], errors="coerce")
    df["mes_str"]       = df["tran_date"].dt.to_period("M").astype(str)
    df["año"]           = df["tran_date"].dt.year
    df["mes_calendario"]= df["tran_date"].dt.month
    df["precio_tx"]     = pd.to_numeric(df["venta_con_iva"], errors="coerce") / pd.to_numeric(df["qty"], errors="coerce")
    df["margen"]        = pd.to_numeric(df["margen"],        errors="coerce")
    df["venta_con_iva"] = pd.to_numeric(df["venta_con_iva"], errors="coerce")
    df["qty"]           = pd.to_numeric(df["qty"],           errors="coerce")

    if dept_tuple:  df = df[df["dept_nm"].isin(dept_tuple)]
    if year_tuple:  df = df[df["año"].isin(year_tuple)]
    if marca_tuple: df = df[df["tipo_marca"].isin(marca_tuple)]
    if len(df) == 0: return None

    ts = (df.groupby("mes_str").agg(venta=("venta_con_iva","sum"),unidades=("qty","sum"))
          .reset_index().sort_values("mes_str"))
    # dept with weighted margin
    dept_grp = df.groupby("dept_nm")
    dept = dept_grp.agg(venta=("venta_con_iva","sum"), n_skus=("prod_nbr","nunique")).reset_index()
    dept_util = dept_grp.apply(
        lambda g: (g["venta_con_iva"] * g["margen"]).sum() / g["venta_con_iva"].sum()
        if g["venta_con_iva"].sum() > 0 else 0).reset_index(name="margen_pond")
    dept = dept.merge(dept_util, on="dept_nm", how="left").sort_values("venta", ascending=False)
    dept["dept_short"] = dept["dept_nm"].str[:28]
    dept["margen_pct_pond"] = dept["margen_pond"] * 100

    stores = (df.groupby(["store_nbr","store_nm"])
              .agg(venta=("venta_con_iva","sum"), margen=("margen","mean"),
                   unidades=("qty","sum"), n_skus=("prod_nbr","nunique"))
              .reset_index().sort_values("venta",ascending=False).head(20))
    stores["label"] = stores["store_nbr"].astype(str)+" "+stores["store_nm"].astype(str).str.strip().str[:12]
    stores["margen_pct"] = stores["margen"]*100

    top_sku = (df.groupby(["prod_nbr","prod_nm"])
               .agg(venta=("venta_con_iva","sum"),unidades=("qty","sum"),
                    precio_prom=("precio_tx","mean"),margen=("margen","mean"))
               .reset_index().sort_values("venta",ascending=False).head(15))
    top_sku["label"] = top_sku["prod_nbr"].astype(str)+" "+top_sku["prod_nm"].str[:20]

    prem = (df.groupby("es_premium")
            .agg(venta=("venta_con_iva","sum"),unidades=("qty","sum"),
                 precio_prom=("precio_tx","mean"),margen_prom=("margen","mean"))
            .reset_index())
    prem["tipo"] = prem["es_premium"].map({0:"No Premium",1:"Premium"})
    prem["margen_pct"] = prem["margen_prom"]*100

    if "tipo_marca" in df.columns:
        marca = (df.groupby("tipo_marca")
                 .agg(venta=("venta_con_iva","sum"), margen_prom=("margen","mean"),
                      n_skus=("prod_nbr","nunique"), ticket_prom=("precio_tx","mean"))
                 .reset_index())
        marca["margen_pct"] = marca["margen_prom"]*100
    else:
        marca = pd.DataFrame()

    seas = (df.groupby("mes_calendario")
            .agg(venta_prom=("venta_con_iva","mean"),unidades_prom=("qty","mean"))
            .reset_index().sort_values("mes_calendario"))
    seas["mes_nombre"] = seas["mes_calendario"].map(MONTH_NAMES)

    # Monthly weighted margin series (Fix 4)
    df["_util_row"] = df["venta_con_iva"] * df["margen"]
    margen_ts = (df.groupby("mes_str")
                 .apply(lambda g: g["_util_row"].sum() / g["venta_con_iva"].sum()
                        if g["venta_con_iva"].sum() > 0 else 0)
                 .reset_index(name="margen_pond"))
    margen_ts["margen_pct"] = margen_ts["margen_pond"] * 100
    margen_ts = margen_ts.sort_values("mes_str")

    # margen_dinero: usa columna "utilidad" si existe, si no la estima desde venta*margen
    if "utilidad" in df.columns:
        utilidad_total = pd.to_numeric(df["utilidad"], errors="coerce").sum()
    elif "margen" in df.columns and "venta_con_iva" in df.columns:
        utilidad_total = df["_util_row"].sum()
    else:
        utilidad_total = 0.0
    venta_tot = df["venta_con_iva"].sum(); uds_tot = df["qty"].sum()
    kpis = {"venta":venta_tot,"unidades":uds_tot,
            "n_skus":df["prod_nbr"].nunique(),"n_stores":df["store_nbr"].nunique(),
            "margen":df["margen"].mean()*100,"margen_dinero":utilidad_total,
            "ticket_prom":df["venta_con_iva"].mean(),
            "precio_prom":venta_tot/uds_tot if uds_tot>0 else 0,
            "p95_p":df["precio_tx"].quantile(0.95),"p95_q":df["qty"].quantile(0.95)}

    dist_p  = df[df["precio_tx"]<=kpis["p95_p"]][["precio_tx"]].copy()
    dist_m  = df[df["margen"].between(-0.5,1.0)][["margen"]].copy()
    dist_q  = df[df["qty"]<=kpis["p95_q"]][["qty"]].copy()

    return {"ts":ts,"dept":dept,"stores":stores,"top_sku":top_sku,"prem":prem,
            "marca":marca,"seas":seas,"kpis":kpis,"margen_ts":margen_ts,
            "dist_p":dist_p,"dist_m":dist_m,"dist_q":dist_q}


# ══════════════════════════════════════════════════════════════════════════════
# SKU-SPECIFIC NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

def generate_sku_narrative(sku_row, sku_sim, sku_cal):
    """Genera un análisis completo en lenguaje natural para un SKU específico."""
    nm    = sku_row["prod_nm"]
    beta  = sku_row["beta"]
    r2    = sku_row["r2"]
    pval  = sku_row["pval"]
    n_m   = sku_row["n_meses"]
    rec   = sku_row["recomendacion"]
    color = REC_COLORS.get(rec, OM_LGRAY)

    # Interpretación de la elasticidad en palabras
    if rec == "Subir precio":
        elast_txt = (f"es **inelástico** (β={beta:.2f}): cuando sube el precio, "
                     f"los clientes casi no cambian su comportamiento de compra.")
        accion_txt = ("📈 **Sube el precio.** Puedes incrementarlo sin perder ventas significativas. "
                      "Cada 10% de aumento en precio genera más ingreso neto.")
    elif rec == "Mantener precio":
        elast_txt = (f"tiene elasticidad **unitaria** (β={beta:.2f}): cambios de precio "
                     f"afectan las ventas casi en la misma proporción.")
        accion_txt = ("⚖️ **Mantén el precio actual.** Subir o bajar el precio genera cambios "
                      "proporcionales en ventas — el ingreso total se mantiene similar.")
    elif rec == "Bajar / Promover":
        elast_txt = (f"es **elástico** (β={beta:.2f}): cuando baja el precio, "
                     f"las ventas aumentan más que proporcionalmente.")
        accion_txt = ("🏷️ **Activa promociones o reduce el precio.** Bajar 10% el precio puede "
                      "generar mucho más volumen de ventas y compensar el menor margen unitario.")
    else:
        # No recomendable — explica el motivo exacto
        razones = []
        if sku_row["pval"] >= 0.10:
            razones.append(f"el p-valor es {pval:.3f} (> 0.10): la relación precio-ventas **no es estadísticamente confiable** con estos datos")
        if sku_row["n_meses"] < 6:
            razones.append(f"solo hay **{n_m} meses de datos** (mínimo recomendado: 6)")
        if beta > 0:
            razones.append(f"la beta es **positiva ({beta:.2f})**, lo que implicaría que subir precio aumenta ventas — económicamente inusual, probablemente ruido")
        if abs(beta) > 3:
            razones.append(f"la beta es **extrema (β={beta:.2f})**: una elasticidad mayor a 3 en valor absoluto casi nunca es real en retail")
        if not razones:
            razones.append("los datos de este SKU no permiten estimar elasticidad de forma confiable")

        motivo = "; ".join(razones)
        return (f"### {nm[:60]}\n\n"
                f"⚠️ **Este producto no tiene recomendación de precio** porque {motivo}.\n\n"
                f"Esto no significa que el producto sea irrelevante — solo que con los datos disponibles "
                f"no podemos estimar con confianza cómo reaccionan las ventas ante un cambio de precio.\n\n"
                f"🔬 Datos del modelo: β={beta:.4f} · R²={r2:.4f} · p={pval:.4f} · {n_m} meses")

    # Simulación del mejor escenario
    sim_note = ""
    if len(sku_sim) > 0:
        base = sku_sim[sku_sim["cambio"]=="Base 0%"]
        if rec == "Subir precio":
            best = sku_sim[sku_sim["cambio"]=="+10%"]
        elif rec == "Bajar / Promover":
            best = sku_sim[sku_sim["cambio"]=="-10%"]
        else:
            best = base
        if len(best)>0 and len(base)>0:
            br = base.iloc[0]; be = best.iloc[0]
            chg_lbl = "+10%" if rec=="Subir precio" else ("-10%" if rec=="Bajar / Promover" else "sin cambio")
            sim_note = (f"\n\n💰 **Simulación ({chg_lbl} en precio):** "
                        f"precio ${be['precio_nuevo']:,.2f} → "
                        f"unidades estimadas {be['unidades_est']:,.0f} → "
                        f"ingreso {be['delta_ingreso_pct']:+.1f}% · margen {be['delta_margen_pct']:+.1f}%")

    # Calendario mensual
    cal_note = ""
    if len(sku_cal) > 0:
        meses_subir  = sku_cal[sku_cal["accion"]=="SUBIR"]["mes_nombre"].tolist()
        meses_prom   = sku_cal[sku_cal["accion"]=="PROMOVER"]["mes_nombre"].tolist()
        meses_mant   = sku_cal[sku_cal["accion"]=="MANTENER"]["mes_nombre"].tolist()
        cal_note = "\n\n📅 **¿Cuándo actuar?**\n"
        if meses_subir:
            cal_note += f"- 🔵 **Subir precio:** {', '.join(meses_subir)} (meses de alta demanda histórica)\n"
        if meses_prom:
            cal_note += f"- 🟢 **Promover / descuento:** {', '.join(meses_prom)} (demanda baja — las promos tienen más impacto)\n"
        if meses_mant:
            cal_note += f"- 🟡 **Mantener:** {', '.join(meses_mant)}\n"

    confianza = "alta" if (r2>=0.3 and pval<0.05) else ("media" if (r2>=0.1 and pval<0.10) else "baja")
    conf_nota = (f"\n\n🔬 **Confianza del modelo:** {confianza} "
                 f"(R²={r2:.3f}, p={pval:.3f}, {n_m} meses de datos)")

    return f"### {nm[:60]}\n\nEste producto {elast_txt}\n\n{accion_txt}{sim_note}{cal_note}{conf_nota}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD DESCRIPTIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Filtros globales al inicio
    section("🔍 Filtros")
    ff1, ff2, ff3, ff4 = st.columns(4)
    raw_df = df_main
    with ff1:
        all_depts = sorted(raw_df["dept_nm"].dropna().unique().tolist())
        dept_sel = st.multiselect("Departamento", all_depts, key="desc_dept")
    with ff2:
        all_years = sorted(raw_df["año"].dropna().unique().tolist())
        year_sel = st.multiselect("Año", all_years, key="desc_year")
    with ff3:
        if "tipo_marca" in raw_df.columns:
            all_marcas = sorted(raw_df["tipo_marca"].dropna().unique().tolist())
            marca_sel = st.multiselect("Tipo de marca", all_marcas, key="desc_marca")
        else:
            marca_sel = []
    with ff4:
        _all_prods_opts = ["Todos"] + sorted(raw_df["prod_nm"].dropna().unique().tolist())
        prod_sel = st.selectbox("Producto", _all_prods_opts, key="desc_prod")

    # Compute aggregations (cached — mismos bytes siempre → nunca re-corre)
    agg = compute_descriptivo(st.session_state["df_csv_bytes"],
                              tuple(dept_sel), tuple(year_sel), tuple(marca_sel))

    # Filtro de producto (se aplica sobre los agregados, no requiere re-cache)
    _prod_filter = "" if prod_sel == "Todos" else prod_sel
    if agg is None:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()
    kpis = agg["kpis"]

    # KPIs — 2 rows of 4 (Fix 3)
    section("📊 Indicadores clave del negocio")
    kc1r = st.columns(4)
    kpi(kc1r[0], "Ventas totales",             f"${kpis['venta']/1e6:.1f}M",        OM_RED)
    kpi(kc1r[1], "Unidades vendidas",          f"{kpis['unidades']/1e3:.0f}K",       OM_BLUE)
    kpi(kc1r[2], "Ticket promedio por compra", f"${kpis['ticket_prom']:,.0f}",       OM_AMBER)
    kpi(kc1r[3], "Precio promedio por unidad", f"${kpis['precio_prom']:,.2f}",       OM_BLUE)
    kc2r = st.columns(4)
    kpi(kc2r[0], "Productos distintos",        f"{kpis['n_skus']:,}",                OM_GREEN)
    kpi(kc2r[1], "Tiendas activas",            f"{kpis['n_stores']}",                OM_AMBER)
    kpi(kc2r[2], "Margen de utilidad",         f"{kpis['margen']:.1f}%",             "#7B1FA2")
    kpi(kc2r[3], "Utilidad total",             f"${kpis['margen_dinero']/1e6:.1f}M", OM_GREEN)
    st.markdown("<br>", unsafe_allow_html=True)

    # Time series
    section("📅 Evolución mensual de ventas")
    ts = agg["ts"]
    tc1,tc2 = st.columns(2)
    with tc1:
        fig = px.area(ts, x="mes_str", y="venta", title="Ventas mensuales totales ($)",
                      labels={"mes_str":"Mes","venta":"Ventas ($)"}, color_discrete_sequence=[OM_RED])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)
        # Insight: find best and worst month
        if len(ts) > 1:
            _best_m = ts.loc[ts["venta"].idxmax(), "mes_str"]
            _worst_m = ts.loc[ts["venta"].idxmin(), "mes_str"]
            st.markdown(f'<div class="insight-box">💡 Tu mejor mes fue <strong>{_best_m}</strong> — planea subidas de precio antes del pico de demanda.</div>', unsafe_allow_html=True)
    with tc2:
        fig = px.bar(ts, x="mes_str", y="unidades", title="Unidades vendidas por mes",
                     labels={"mes_str":"Mes","unidades":"Unidades"}, color_discrete_sequence=[OM_BLUE])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)

    # Department — Fix 6: dual-axis bars + margin line
    section("🏷️ Ventas y margen por departamento")
    dept = agg["dept"]
    fig_dept = go.Figure()
    dept_sorted = dept.sort_values("venta", ascending=True)
    fig_dept.add_trace(go.Bar(
        x=dept_sorted["venta"], y=dept_sorted["dept_short"],
        orientation="h", name="Ventas ($)",
        marker_color=OM_RED,
        text=[f"${v/1e6:.1f}M" for v in dept_sorted["venta"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>"))
    fig_dept.add_trace(go.Scatter(
        x=dept_sorted["margen_pct_pond"], y=dept_sorted["dept_short"],
        mode="markers+lines", name="Margen %",
        marker=dict(color=OM_YELLOW, size=10, symbol="diamond"),
        line=dict(color=OM_YELLOW, width=2, dash="dot"),
        xaxis="x2",
        hovertemplate="<b>%{y}</b><br>Margen: %{x:.1f}%<extra></extra>"))
    _mg_vals = dept_sorted["margen_pct_pond"].dropna()
    _mg_min  = float(_mg_vals.min()) if len(_mg_vals) else 0
    _mg_max  = float(_mg_vals.max()) if len(_mg_vals) else 1
    fig_dept.update_layout(
        title=dict(
            text="Ventas totales (barras) · Margen ponderado % (línea amarilla) por departamento",
            y=0.97, yanchor="top"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(350, len(dept_sorted) * 90),
        margin=dict(t=100, b=80, l=220, r=130),
        bargap=0.35,
        xaxis=dict(title="Ventas ($)", side="bottom"),
        xaxis2=dict(title=dict(text="Margen %", standoff=20),
                    overlaying="x", side="top",
                    showgrid=False, ticksuffix="%",
                    range=[_mg_min * 0.95, _mg_max * 1.05]),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(autorange=True))
    st.plotly_chart(fig_dept, use_container_width=True)

    # Store performance — Fix 7: geographic map
    section("🗺️ Distribución geográfica de tiendas")
    st.caption("Tamaño = margen · Color = ventas totales")
    stores = agg["stores"]
    import re as _re_map, unicodedata as _ud
    CITY_COORDS = {
        "ESTADIO":(19.4033,-99.1529),"REFORMA":(19.4269,-99.1675),
        "UNIVERSIDAD":(19.3264,-99.1771),"PERINORTE":(19.5667,-99.1333),
        "TOREO":(19.4500,-99.2600),"SATELITE":(19.5300,-99.2300),
        "LOMAS":(19.4350,-99.2450),"LOMASVERDES":(19.5100,-99.2400),
        "INTERLOMAS":(19.3900,-99.2800),"BOSQUES":(19.4100,-99.2700),
        "PERICOAPA":(19.3200,-99.1800),"COAPA":(19.3100,-99.1600),
        "COYOACAN":(19.3467,-99.1617),"PEDREGAL":(19.3200,-99.2200),
        "TEPEYAC":(19.4900,-99.1100),"LINDAVISTA":(19.4900,-99.1300),
        "VALLEJO":(19.4800,-99.1500),"TACUBAYA":(19.4017,-99.1758),
        "OBSERVATORIO":(19.3967,-99.1897),"MIXCOAC":(19.3800,-99.1900),
        "XOCHIMILCO":(19.2600,-99.1000),"TLALPAN":(19.2900,-99.1600),
        "TLAHUAC":(19.2900,-99.0100),"IZTAPALAPA":(19.3600,-99.0500),
        "IZTACALCO":(19.3900,-99.1000),"VENUSTIANO":(19.4300,-99.0900),
        "GUSTAVO":(19.4900,-99.1100),"AZCAPOTZALCO":(19.4900,-99.1900),
        "CUAJIMALPA":(19.3600,-99.2800),"MAGDALENA":(19.3400,-99.2100),
        "CONTRERAS":(19.3400,-99.2100),"ALVARO":(19.3600,-99.1600),
        "MILPA":(19.2200,-99.0300),"ALTA":(19.2200,-99.0300),
        "ECATEPEC":(19.6010,-99.0600),"NEZAHUALCOYOTL":(19.4000,-99.0167),
        "CHALCO":(19.2628,-98.8978),"TEXCOCO":(19.5158,-98.8783),
        "TOLUCA":(19.2826,-99.6557),"METEPEC":(19.2505,-99.5989),
        "NAUCALPAN":(19.4800,-99.2400),"TLALNEPANTLA":(19.5500,-99.2000),
        "CUAUTITLAN":(19.6800,-99.1800),"IZCALLI":(19.6800,-99.2200),
        "CUAUTITLANIZCALLI":(19.6800,-99.2200),"TULTITLAN":(19.6400,-99.1700),
        "COACALCO":(19.6200,-99.0900),"TECAMAC":(19.7100,-98.9700),
        "ATIZAPAN":(19.5600,-99.2600),"HUIXQUILUCAN":(19.3600,-99.2900),
        "NICOLASLOMA":(19.5300,-99.2000),"PACHUCA":(20.1011,-98.7591),
        "HIDALGO":(20.1011,-98.7591),"TULANCINGO":(20.0853,-98.3622),
        "QUERETARO":(20.5888,-100.3899),"JURIQUILLA":(20.7100,-100.4400),
        "CELAYA":(20.5232,-100.8149),"IRAPUATO":(20.6778,-101.3554),
        "LEON":(21.1221,-101.6820),"SILAO":(20.9361,-101.4325),
        "SALAMANCA":(20.5697,-101.1956),"GUANAJUATO":(21.0190,-101.2574),
        "MORELIA":(19.7060,-101.1950),"URUAPAN":(19.4150,-102.0630),
        "LAZARO":(17.9333,-102.2000),"LAZAROCARDENAS":(17.9333,-102.2000),
        "AGUASCALIENTES":(21.8818,-102.2918),
        "GUADALAJARA":(20.6597,-103.3496),"ZAPOPAN":(20.7207,-103.3921),
        "TLAQUEPAQUE":(20.6419,-103.3166),"TONALA":(20.6239,-103.2344),
        "TLAJOMULCO":(20.4733,-103.4420),
        "MONTERREY":(25.6866,-100.3161),"SANPEDRO":(25.6552,-100.4016),
        "GARZA":(25.8000,-100.2000),"GARCIA":(25.8090,-100.5861),
        "APODACA":(25.7847,-100.1886),"ESCOBEDO":(25.8000,-100.3200),
        "SALTILLO":(25.4232,-100.9963),"TORREON":(25.5428,-103.4068),
        "GOMEZ":(25.5666,-103.4892),"DURANGO":(24.0277,-104.6532),
        "CHIHUAHUA":(28.6353,-106.0889),"JUAREZ":(31.7381,-106.4870),
        "CIUDADJUAREZ":(31.7381,-106.4870),"HERMOSILLO":(29.0729,-110.9559),
        "OBREGON":(27.4863,-109.9307),"CULIACAN":(24.8091,-107.3940),
        "MAZATLAN":(23.2494,-106.4111),"TEPIC":(21.5044,-104.8953),
        "COLIMA":(19.2433,-103.7278),"PUEBLA":(19.0413,-98.2062),
        "CHOLULA":(19.0636,-98.3092),"TLAXCALA":(19.3139,-98.2392),
        "VERACRUZ":(19.1738,-96.1342),"XALAPA":(19.5438,-96.9102),
        "ORIZABA":(18.8503,-97.1003),"COATZACOALCOS":(18.1503,-94.4536),
        "OAXACA":(17.0654,-96.7236),"TUXTLA":(16.7500,-93.1167),
        "TAPACHULA":(14.9000,-92.2597),"VILLAHERMOSA":(17.9869,-92.9303),
        "CAMPECHE":(19.8458,-90.5258),"MERIDA":(20.9674,-89.5926),
        "CANCUN":(21.1619,-86.8515),"PLAYA":(20.6296,-87.0739),
        "PLAYADELCARMEN":(20.6296,-87.0739),"COZUMEL":(20.5083,-86.9458),
        "CHETUMAL":(18.5001,-88.2961),"ZACATECAS":(22.7709,-102.5832),
        "SANLUISPOTOSI":(22.1565,-100.9855),"POTOSI":(22.1565,-100.9855),
        "PERISUR":(19.2960,-99.1710),"INSURGENTES":(19.3700,-99.1800),
        "ALTAVISTA":(19.3500,-99.1900),"POLANCO":(19.4333,-99.1953),
        "SANTAFE":(19.3600,-99.2600),"SANTA":(19.3600,-99.2600),
        "CUERNAVACA":(18.9242,-99.2216),"TIJUANA":(32.5027,-117.0037),
        "TIJUANAESTECENTRO":(32.5027,-117.0037),
        # Fix 3 — 5 previously missing stores
        "GRAN SUR II":(19.3050,-99.1600),"GRAN SUR":(19.3050,-99.1600),
        "GRANSUR":(19.3050,-99.1600),"GRANSUR II":(19.3050,-99.1600),
        "MEXICALI":(32.6278,-115.4545),
        "SLP II":(22.1565,-100.9855),"SLP":(22.1565,-100.9855),
        "MARIANO OTERO":(20.6400,-103.4100),"MARIANO":(20.6400,-103.4100),
        "OTERO":(20.6400,-103.4100),
        "ENSENADA":(31.8667,-116.5960),
    }
    _ROMAN = {"I","II","III","IV","V","VI","VII","VIII","IX","X"}

    def _extract_city(store_nm):
        name = str(store_nm).upper().strip()
        name = ''.join(c for c in _ud.normalize('NFD', name) if _ud.category(c) != 'Mn')
        name = name.replace("-", " ").replace("_", " ")
        name = _re_map.sub(r'^\d+\s*', '', name)
        for pfx in ["NUM TIENDA ", "TIENDA ", "NUM ", "STORE "]:
            if name.startswith(pfx):
                name = name[len(pfx):]
                break
        name = _re_map.sub(r'\s+\d+$', '', name).strip()
        return name

    def _find_coords(city_key):
        if not city_key:
            return None
        # 1. exact match
        if city_key in CITY_COORDS:
            return CITY_COORDS[city_key]
        # 2. first word
        tokens = city_key.split()
        if tokens and tokens[0] in CITY_COORDS:
            return CITY_COORDS[tokens[0]]
        # 3. last word if not a Roman numeral
        if tokens and tokens[-1] not in _ROMAN and tokens[-1] in CITY_COORDS:
            return CITY_COORDS[tokens[-1]]
        # 4. partial / substring match
        for k, v in CITY_COORDS.items():
            if city_key in k or k in city_key:
                return v
        return None

    stores["ciudad"] = stores["store_nm"].apply(_extract_city)
    def _get_lat(c): r = _find_coords(c); return r[0] if r else None
    def _get_lon(c): r = _find_coords(c); return r[1] if r else None
    stores["lat"] = stores["ciudad"].map(_get_lat)
    stores["lon"] = stores["ciudad"].map(_get_lon)
    geo_df = stores.dropna(subset=["lat","lon"]).copy()
    _unmatched = stores[stores["lat"].isna()]["store_nm"].tolist()
    if _unmatched:
        with st.expander(f"🔍 {len(_unmatched)} tienda(s) sin coordenadas — haz clic para ver", expanded=False):
            st.caption("Estas tiendas no pudieron mapearse. Contacta al administrador para agregar sus coordenadas.")
            for _u in _unmatched:
                st.text(f"• {_u}")
    if len(geo_df) > 0:
        geo_df["size_val"] = (geo_df["margen_pct"].clip(lower=0) + 1) * 8
        fig_map = px.scatter_mapbox(
            geo_df, lat="lat", lon="lon",
            color="venta", size="size_val",
            hover_name="store_nm",
            hover_data={"venta":":$,.0f","margen_pct":":.1f","n_skus":True,"lat":False,"lon":False,"size_val":False},
            color_continuous_scale=["#FFCDD2", OM_RED],
            mapbox_style="carto-positron",
            zoom=4.5, center={"lat":23.6345,"lon":-102.5528},
            title="🗺️ Distribución geográfica de tiendas",
            labels={"venta":"Ventas ($)","margen_pct":"Margen %","n_skus":"SKUs"})
        fig_map.update_layout(height=480, margin=dict(t=45,b=10,l=10,r=10),
                              paper_bgcolor="white",
                              coloraxis_colorbar=dict(title="Ventas $"))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No se encontraron coordenadas para las tiendas en este filtro. Verifica que los nombres de tienda sigan el patrón NUM_TIENDA_CIUDAD.")
    # Scatter ventas vs margen (mantener como referencia tabular)
    with st.expander("📋 Ver tabla detallada de tiendas", expanded=False):
        st.dataframe(stores[["store_nm","venta","margen_pct","unidades","n_skus"]].rename(columns={
            "store_nm":"Tienda","venta":"Ventas ($)","margen_pct":"Margen %",
            "unidades":"Unidades","n_skus":"SKUs únicos"
        }).head(20), hide_index=True, use_container_width=True)

    # Top SKUs
    section("🏆 Top productos")
    top_sku = agg["top_sku"]
    if _prod_filter:
        top_sku = top_sku[top_sku["label"].str.contains(_prod_filter, case=False, na=False)]
        if len(top_sku) == 0:
            st.info(f"Sin productos que coincidan con '{_prod_filter}'")
            top_sku = agg["top_sku"]  # fallback
    sk1,sk2 = st.columns(2)
    with sk1:
        fig = px.bar(top_sku.sort_values("venta"), x="venta", y="label", orientation="h",
                     title="Por venta total ($)", labels={"venta":"Ventas ($)","label":""},
                     color_discrete_sequence=[OM_RED])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>")
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(_layout(fig,h=440), use_container_width=True)
    with sk2:
        fig = px.bar(top_sku.sort_values("unidades"), x="unidades", y="label", orientation="h",
                     title="Por unidades vendidas", labels={"unidades":"Unidades","label":""},
                     color_discrete_sequence=[OM_BLUE])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Unidades: %{x:,.0f}<extra></extra>")
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(_layout(fig,h=440), use_container_width=True)

    # Fix 4 — Weighted margin over time (replaces Premium section)
    section("📈 Evolución del margen promedio ponderado")
    st.caption("Margen bruto mensual como % de las ventas totales")
    margen_ts = agg["margen_ts"]
    if len(margen_ts) > 1:
        avg_margen = margen_ts["margen_pct"].mean()
        last_val   = margen_ts["margen_pct"].iloc[-1]
        last_mes   = margen_ts["mes_str"].iloc[-1]
        first_mes  = margen_ts["mes_str"].iloc[0]
        y_min      = float(margen_ts["margen_pct"].min())
        y_max      = float(margen_ts["margen_pct"].max())
        fig_mg = go.Figure()
        fig_mg.add_trace(go.Scatter(
            x=margen_ts["mes_str"], y=margen_ts["margen_pct"],
            mode="lines+markers", name="Margen %",
            line=dict(color=OM_RED, width=2.5),
            marker=dict(size=5, color=OM_RED),
            fill="tozeroy", fillcolor="rgba(227,24,55,0.10)",
            hovertemplate="<b>%{x}</b><br>Margen: %{y:.2f}%<extra></extra>"))
        fig_mg.add_hline(y=avg_margen, line_dash="dash", line_color="#555")
        fig_mg.add_annotation(
            x=last_mes, y=avg_margen,
            text=f"Promedio: {avg_margen:.1f}%",
            showarrow=False,
            xanchor="right", yanchor="bottom",
            yshift=8, xshift=-8,
            font=dict(size=12, color="#555555"),
            bgcolor="white", borderpad=3)
        fig_mg.add_annotation(x=last_mes, y=last_val,
                              text=f"{last_val:.1f}%", showarrow=True,
                              arrowhead=2, font=dict(color=OM_RED, size=12),
                              xanchor="right", yanchor="bottom", yshift=10)
        fig_mg.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(l=70, r=80, t=50, b=60),
            xaxis=dict(dtick="M6", tickformat="%b %Y", tickangle=45, title=""),
            yaxis=dict(title="Margen %", ticksuffix="%",
                       range=[y_min * 0.97, y_max * 1.03]),
            showlegend=False)
        st.plotly_chart(fig_mg, use_container_width=True)

    # Fix 5 — Marca: stat cards + grouped bar
    marca = agg["marca"]
    if len(marca) > 0:
        section("🏷️ Marca Propia vs Marca Externa")
        marca_colors = [OM_RED, "#555555", OM_AMBER, OM_BLUE]
        # Stat cards
        mc_cols = st.columns(len(marca))
        for i, (_, mrow) in enumerate(marca.iterrows()):
            border_c = OM_RED if i == 0 else "#555555"
            mc_cols[i].markdown(
                f'<div style="background:white;border-radius:12px;border-left:5px solid {border_c};'
                f'padding:18px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:12px;">'
                f'<div style="font-size:13px;font-weight:700;color:{border_c};margin-bottom:10px;">'
                f'{mrow["tipo_marca"]}</div>'
                f'<div style="font-size:12px;color:#555;line-height:2.2;">'
                f'<b>Ventas:</b> ${mrow["venta"]/1e6:.1f}M<br>'
                f'<b>Margen:</b> {mrow["margen_pct"]:.1f}%<br>'
                f'<b>SKUs:</b> {int(mrow["n_skus"]):,}<br>'
                f'<b>Ticket prom.:</b> ${mrow["ticket_prom"]:,.0f}'
                f'</div></div>', unsafe_allow_html=True)
        # Grouped bar chart
        fig_mb = go.Figure()
        metrics = [("venta","Ventas ($)","$"),("margen_pct","Margen (%)","")]
        for col_key, col_lbl, pfx in metrics:
            fmt_vals = [f"${v/1e6:.1f}M" if pfx=="$" else f"{v:.1f}%" for v in marca[col_key]]
            fig_mb.add_trace(go.Bar(
                x=marca["tipo_marca"], y=marca[col_key],
                name=col_lbl,
                text=fmt_vals, textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{col_lbl}: %{{y:,.1f}}<extra></extra>"))
        fig_mb.update_layout(barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                             height=280, margin=dict(t=30,b=20,l=20,r=20),
                             showlegend=True, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_mb, use_container_width=True)

    # Fix 8b — Seasonality: filled area chart + treemap
    section("📆 Estacionalidad — promedio por mes del año")
    seas = agg["seas"]
    se1, se2 = st.columns(2)
    with se1:
        # Filled area (Fix 8b-a)
        seas_ordered = seas.set_index("mes_nombre").reindex(
            [m for m in MONTH_ORDER if m in seas["mes_nombre"].values]).reset_index()
        fig_seas = go.Figure()
        fig_seas.add_trace(go.Scatter(
            x=seas_ordered["mes_nombre"], y=seas_ordered["venta_prom"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color=OM_RED, width=2.5),
            fillcolor=f"rgba(227,24,55,0.18)",
            marker=dict(size=7, color=OM_RED),
            hovertemplate="<b>%{x}</b><br>Venta promedio: $%{y:,.0f}<extra></extra>"))
        fig_seas.update_layout(
            title="Venta promedio por mes del año",
            plot_bgcolor="white", paper_bgcolor="white",
            height=280, margin=dict(t=40,b=20,l=40,r=20),
            xaxis=dict(title="", categoryorder="array", categoryarray=MONTH_ORDER),
            yaxis=dict(title="Venta promedio ($)"), showlegend=False)
        st.plotly_chart(fig_seas, use_container_width=True)
    with se2:
        # Fix 6 — Treemap: discrete OfficeMax palette, white bg, margen as label text
        dept_tree = agg["dept"].copy().sort_values("venta", ascending=False)
        _om_palette = [OM_RED, "#FFD100", "#1A1A1A", "#888888",
                       OM_BLUE, OM_GREEN, OM_AMBER, OM_LGRAY]
        _dept_colors = {
            row["dept_short"]: _om_palette[i % len(_om_palette)]
            for i, (_, row) in enumerate(dept_tree.iterrows())
        }
        dept_tree["color_cat"] = dept_tree["dept_short"].map(_dept_colors)
        dept_tree["margen_label"] = dept_tree["margen_pct_pond"].apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        fig_tree = go.Figure(go.Treemap(
            labels=dept_tree["dept_short"].tolist(),
            parents=[""] * len(dept_tree),
            values=dept_tree["venta"].tolist(),
            customdata=dept_tree[["margen_label","dept_nm"]].values,
            marker=dict(colors=dept_tree["color_cat"].tolist()),
            texttemplate="<b>%{label}</b><br>%{percentParent:.0%}<br>Margen: %{customdata[0]}",
            hovertemplate="<b>%{customdata[1]}</b><br>Ventas: $%{value:,.0f}<br>Margen: %{customdata[0]}<extra></extra>",
            textfont=dict(color="white", size=13)))
        fig_tree.update_layout(
            title="Participación y margen por departamento",
            height=280, margin=dict(t=40, b=0, l=0, r=0),
            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_tree, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD PREDICTIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    results = st.session_state["results"]
    if results is None:
        st.info("⬅️ Sube tus datos y presiona **🔍 Analizar mi catálogo** en el panel izquierdo para ver la inteligencia de precios.")
        st.stop()

    df_m1a = results["m1a"]; df_m1b = results["m1b"]; df_m1c = results["m1c"]
    df_sim = results["sim"]; m2 = results["m2"]; df_cal = results["cal"]

    if len(df_m1a) == 0:
        st.warning("No hay SKUs con modelo válido. Reduce el mínimo de meses o el CV en la Calculadora.")
        st.stop()

    rec_counts = df_m1a["recomendacion"].value_counts()

    # ── KPIs del dashboard predictivo ────────────────────────────────────────
    section("📊 Indicadores clave del análisis")
    n_validos  = int(rec_counts.get("Subir precio",0) + rec_counts.get("Mantener precio",0) + rec_counts.get("Bajar / Promover",0))
    n_subir    = int(rec_counts.get("Subir precio",0))
    n_promover = int(rec_counts.get("Bajar / Promover",0))
    n_mantener = int(rec_counts.get("Mantener precio",0))
    pct_valid  = n_validos / len(df_m1a) * 100 if len(df_m1a) > 0 else 0

    # Impacto financiero estimado
    _imp_sub = _imp_prm = 0
    if len(df_sim) > 0:
        def _quick_delta(recs, esc):
            ns = df_m1a[df_m1a["recomendacion"].isin(recs)]["prod_nm"].tolist()
            b = df_sim[(df_sim["prod_nm"].isin(ns)) & (df_sim["cambio"]=="Base 0%")]["ingreso_est"].sum()
            t = df_sim[(df_sim["prod_nm"].isin(ns)) & (df_sim["cambio"]==esc)]["ingreso_est"].sum()
            return t - b
        _imp_sub = _quick_delta(["Subir precio"], "+10%")
        _imp_prm = _quick_delta(["Bajar / Promover"], "-10%")
    _imp_total_anual = (_imp_sub + _imp_prm) * 12

    # R² y precio importance del ML
    _ml_kpi = st.session_state.get("ml_results")
    _r2_ml  = f"{_ml_kpi['r2_r']:.2f}" if _ml_kpi else "—"
    _precio_feats = {"Log Precio","% Descuento vs modal","Precio vs Catálogo","Cambio precio % (mes ant.)"}
    _precio_pct = f"{float(_ml_kpi['feat_df'][_ml_kpi['feat_df']['feature'].isin(_precio_feats)]['importancia'].sum())*100:.0f}%" if _ml_kpi else "—"

    # 2 rows of 3 (Fix 3)
    pk1 = st.columns(3)
    kpi(pk1[0], "Productos con recomendación", f"{n_validos} ({pct_valid:.0f}%)", OM_RED)
    kpi(pk1[1], "✅ Sube el precio",            f"{n_subir} productos",            OM_BLUE)
    kpi(pk1[2], "📢 Lanza una promoción",       f"{n_promover} productos",         OM_GREEN)
    pk2 = st.columns(2)
    kpi(pk2[0], "➡️ Mantén el precio",         f"{n_mantener} productos",         OM_AMBER)
    kpi(pk2[1], "Impacto anualizado estimado", f"+${_imp_total_anual:,.0f}",      OM_RED)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Resumen ejecutivo con IA — AL INICIO para ejecutivos ─────────────────
    ml_res = st.session_state.get("ml_results")
    # ── ¿Qué impulsa las ventas? — Pipeline ML ────────────────────────────────
    if ml_res:
        section("📊 ¿Qué impulsa tus ventas?")
        st.caption("Comparamos dos modelos para identificar los factores que más impactan la demanda. "
                   "El resultado clave: ¿qué variable de negocio puedes controlar para mover las ventas?")

        # Tabla comparativa
        ml_c1, ml_c2 = st.columns([1, 2])
        with ml_c1:
            comp = ml_res["comparacion"]
            ganador = ml_res["ganador"]
            comp_rows = []
            for nm, v in comp.items():
                comp_rows.append({"Modelo": f"{'★ ' if nm==ganador else ''}{nm}",
                                   "Precisión Unidades": f"{v['r2_u']*100:.1f}%",
                                   "Precisión Ventas $": f"{v['r2_r']*100:.1f}%",
                                   "Precisión Promedio": f"{v['r2_avg']*100:.1f}%"})
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
            st.caption(f"Entrenamiento: {ml_res['train_range']}  |  Prueba: {ml_res['test_range']}")
            st.markdown(
                f'<div style="background:{OM_GREEN};color:white;border-radius:8px;padding:10px 14px;'
                f'font-weight:700;font-size:13px;margin-top:8px;">'
                f'✅ Modelo seleccionado: Precisión {ml_res["comparacion"][ganador]["r2_avg"]*100:.0f}%</div>',
                unsafe_allow_html=True)

        with ml_c2:
            feat_df = ml_res["feat_df"]
            # Agrupar features por categoría para narrativa más clara
            grupos = {
                "🔴 Precio (controlable)":   {"Log Precio","% Descuento vs modal","Precio vs Catálogo","Cambio precio % (mes ant.)"},
                "🔵 Historial de demanda":    {"Ventas mes anterior (log)","Media ventas 3m (log)"},
                "⚫ Contexto de mercado":     {"Mes del año","Tendencia temporal","Departamento","Es Premium"},
            }
            # Fix 11 — normalize group sums to exactly 100%
            FEAT_GROUPS = {
                "🔴 Precio (controlable)": {"log_precio","pct_desc","precio_vs_cat","precio_chg_pct"},
                "🔵 Historial de demanda": {"log_uds_lag1","log_uds_roll3m"},
                "⚫ Contexto de mercado":  {"mes_cal","mes_num","es_premium","dept_enc"},
            }
            raw_feat_df = ml_res["feat_df"].copy()
            # map display names → internal names for lookup
            FEAT_NAME_MAP = {
                "Log Precio":"log_precio","% Descuento vs modal":"pct_desc",
                "Precio vs Catálogo":"precio_vs_cat","Cambio precio % (mes ant.)":"precio_chg_pct",
                "Ventas mes anterior (log)":"log_uds_lag1","Media ventas 3m (log)":"log_uds_roll3m",
                "Mes del año":"mes_cal","Tendencia temporal":"mes_num",
                "Es Premium":"es_premium","Departamento":"dept_enc",
            }
            raw_feat_df["feat_key"] = raw_feat_df["feature"].map(FEAT_NAME_MAP).fillna(raw_feat_df["feature"])
            grp_imp_raw = {}
            for grp_nm, feat_keys in FEAT_GROUPS.items():
                grp_imp_raw[grp_nm] = float(raw_feat_df[raw_feat_df["feat_key"].isin(feat_keys)]["importancia"].sum())
            total_imp = sum(grp_imp_raw.values()) or 1.0
            grp_imp_norm = {k: v/total_imp for k,v in grp_imp_raw.items()}
            grp_df = pd.DataFrame({"Grupo": list(grp_imp_norm.keys()),
                                    "Pct": list(grp_imp_norm.values())}).sort_values("Pct", ascending=True)
            colors_grp = [OM_LGRAY, OM_BLUE, OM_RED]
            grp_df_vals = [v*100 for v in grp_df["Pct"]]
            fig_fi = go.Figure(go.Bar(
                x=grp_df_vals,
                y=grp_df["Grupo"],
                orientation="h",
                marker_color=colors_grp,
                text=[f"{v:.1f}%" for v in grp_df_vals],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importancia: %{x:.1f}%<extra></extra>"))
            precio_total = grp_imp_norm.get("🔴 Precio (controlable)", 0) * 100
            fig_fi.update_layout(
                title=f"¿Qué impulsa la demanda? — Precio controlable: {precio_total:.0f}% de los factores accionables",
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(t=55,b=10,l=10,r=80),
                xaxis=dict(title="Importancia normalizada (%)", range=[0, max(grp_df_vals)*1.25]),
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        # Punchline que conecta ML con OLS
        precio_imp = ml_res["feat_df"][ml_res["feat_df"]["feature"]=="Log Precio"]["importancia"].values
        precio_pct = float(precio_imp[0]) * 100 if len(precio_imp) > 0 else 0
        precio_feats_set = {"Log Precio","% Descuento vs modal","Precio vs Catálogo","Cambio precio % (mes ant.)"}
        precio_total_pct = float(ml_res["feat_df"][ml_res["feat_df"]["feature"].isin(precio_feats_set)]["importancia"].sum()) * 100
        st.markdown(
            f'<div style="background:#E8F5E9;border-left:5px solid {OM_GREEN};border-radius:8px;'
            f'padding:14px 18px;margin:12px 0 4px 0;font-size:14px;">'
            f'<b>Respuesta del ML:</b> El <b>precio es el factor de negocio más importante que OfficeMax puede controlar</b> '
            f'({precio_total_pct:.0f}% de importancia entre factores accionables). '
            f'→ <b>Siguiente pregunta: ¿cuánto hay que cambiar el precio de cada producto?</b> '
            f'Eso lo responde el modelo OLS en el Paso 2: calcula la elasticidad β por SKU — '
            f'cuánto cambia la demanda ante un 1% de cambio en precio, producto por producto.'
            f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Paso 1b: Timing óptimo de promociones ─────────────────────────────
        if "dept_pivot" in ml_res and ml_res["dept_pivot"] is not None:
            section("📅 ¿Cuándo actuar? Timing óptimo por departamento")
            st.caption("Meses donde las promociones históricamente generaron más impacto en ventas por departamento. "
                       "Verde = mayor oportunidad · Rojo = menor efecto de las promos.")

            dept_pivot = ml_res["dept_pivot"]

            # Top 3 meses por departamento → bar chart claro
            dept_top = []
            for dept in dept_pivot.index:
                row = dept_pivot.loc[dept].dropna().sort_values(ascending=False)
                for mes, uplift in row.head(3).items():
                    dept_top.append({"Departamento": dept[:25], "Mes": mes,
                                     "Uplift %": round(uplift, 1)})
            dept_top_df = pd.DataFrame(dept_top)

            pb1, pb2 = st.columns([3, 2])
            with pb1:
                if len(dept_top_df) > 0:
                    fig_bar = px.bar(
                        dept_top_df.sort_values("Uplift %", ascending=False).head(20),
                        x="Uplift %", y="Departamento", color="Mes",
                        orientation="h", barmode="group",
                        title="Top meses de mayor uplift por departamento",
                        color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                          height=380, margin=dict(t=45,b=20,l=10,r=20),
                                          yaxis=dict(autorange="reversed"),
                                          legend=dict(orientation="h", y=1.08))
                    fig_bar.update_traces(
                        hovertemplate="<b>%{y}</b><br>%{x:.1f}% uplift en %{marker.color}<extra></extra>")
                    st.plotly_chart(fig_bar, use_container_width=True)

            with pb2:
                # Fix 12 — heatmap with high-contrast scale, bigger fonts
                fig_dh = go.Figure(data=go.Heatmap(
                    z=dept_pivot.values,
                    x=list(dept_pivot.columns),
                    y=[d[:22] for d in dept_pivot.index],
                    colorscale="RdYlGn",
                    zmin=0,
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}% uplift<extra></extra>",
                    showscale=True,
                    colorbar=dict(title=dict(text="Uplift %", font=dict(size=11)))))
                fig_dh.update_layout(
                    height=500, margin=dict(t=30,b=10,l=10,r=80),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(side="top", tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=12)))
                st.plotly_chart(fig_dh, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Paso 2: Resumen OLS ───────────────────────────────────────────────────
    section("🔍 ¿Qué productos deberías re-preciar?")
    rc = st.columns(4)
    for i,(rec,color) in enumerate(REC_COLORS.items()):
        cnt = rec_counts.get(rec,0)
        pct = cnt/len(df_m1a)*100 if len(df_m1a)>0 else 0
        rc[i].markdown(
            f'<div class="rec-card" style="border-left:5px solid {color};padding:14px;">'
            f'<div style="font-size:30px;font-weight:900;color:{color};">{cnt}</div>'
            f'<div style="font-size:11px;font-weight:700;color:#333;">{rec.upper()}</div>'
            f'<div style="font-size:10px;color:#999;">{pct:.1f}% del total</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fix 1/9 — SKU scatter wrapped in try/except; safe prod_nm fallback
    if len(df_sim) > 0:
        try:
            # Safe id column: prod_nm if available, else prod_nbr
            _id_col = "prod_nm" if "prod_nm" in df_sim.columns else "prod_nbr"
            _scat_base = df_sim[df_sim["cambio"]=="Base 0%"][[_id_col,"ingreso_est","precio_nuevo","unidades_est"]].copy()
            _best_esc = df_sim.copy()
            _best_esc["abs_delta"] = _best_esc["delta_ingreso_pct"].abs()
            _best_per_sku = _best_esc.sort_values("abs_delta", ascending=False).drop_duplicates(_id_col)
            _id_col_m1a = "prod_nm" if "prod_nm" in df_m1a.columns else "prod_nbr"
            _scat_df = df_m1a[[_id_col_m1a,"prod_nbr","beta","pval","recomendacion"]].copy()
            _scat_df = _scat_df.merge(
                _scat_base.rename(columns={_id_col:"_join_key","ingreso_est":"ingreso_base",
                                           "precio_nuevo":"precio_actual","unidades_est":"uds_base"}),
                left_on=_id_col_m1a, right_on="_join_key", how="left")
            _scat_df = _scat_df.merge(
                _best_per_sku[[_id_col,"delta_ingreso_pct"]].rename(
                    columns={_id_col:"_join_key2","delta_ingreso_pct":"impacto_pct"}),
                left_on=_id_col_m1a, right_on="_join_key2", how="left")
            _scat_df["impacto_mes"] = (_scat_df["ingreso_base"] * _scat_df["impacto_pct"].fillna(0) / 100).fillna(0)
            _hover_label = _scat_df[_id_col_m1a].astype(str)
            _rec_colors_scatter = {
                "Subir precio":     OM_RED,
                "Mantener precio":  "#FFA500",
                "Bajar / Promover": "#2196F3",
                "No recomendable":  OM_LGRAY,
            }
            section("🎯 Priorización de productos: sensibilidad al precio vs impacto financiero")
            st.caption("Los productos en la esquina inferior izquierda son muy sensibles al precio y de alto impacto — prioriza sus promociones.")
            fig_scat = go.Figure()
            for rec_s, grp_s in _scat_df.groupby("recomendacion"):
                _cd = np.column_stack([
                    grp_s[_id_col_m1a].astype(str).values,
                    grp_s["pval"].values,
                    grp_s["recomendacion"].values])
                fig_scat.add_trace(go.Scatter(
                    x=grp_s["beta"], y=grp_s["impacto_mes"],
                    mode="markers", name=rec_s,
                    marker=dict(size=10, color=_rec_colors_scatter.get(rec_s, OM_LGRAY),
                                opacity=0.8, line=dict(width=1, color="white")),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Índice sensibilidad (β): %{x:.3f}<br>"
                        "Impacto estimado: $%{y:,.0f}/mes<br>"
                        "p-valor: %{customdata[1]}<br>"
                        "Recomendación: %{customdata[2]}<extra></extra>"),
                    customdata=_cd))
            _y_max = _scat_df["impacto_mes"].max() if len(_scat_df) > 0 else 1
            fig_scat.add_vline(x=-1.0, line_dash="dash", line_color="#888", opacity=0.7)
            fig_scat.add_vline(x=-1.5, line_dash="dot",  line_color="#888", opacity=0.6)
            fig_scat.add_annotation(x=-0.5,  y=_y_max*0.92, text="Inelástico<br>→ Sube precio",
                                    showarrow=False, font=dict(size=11, color="#555"),
                                    bgcolor="rgba(255,255,255,0.8)")
            fig_scat.add_annotation(x=-1.25, y=_y_max*0.92, text="Unitario<br>→ Mantén",
                                    showarrow=False, font=dict(size=11, color="#555"),
                                    bgcolor="rgba(255,255,255,0.8)")
            fig_scat.add_annotation(x=-2.2,  y=_y_max*0.92, text="Elástico<br>→ Promociona",
                                    showarrow=False, font=dict(size=11, color="#555"),
                                    bgcolor="rgba(255,255,255,0.8)")
            fig_scat.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=30,b=40,l=50,r=20),
                xaxis=dict(title="Índice de sensibilidad al precio (β)",
                           gridcolor="#F0F0F0", zeroline=True, zerolinecolor="#CCC"),
                yaxis=dict(title="Impacto financiero estimado ($/mes)",
                           gridcolor="#F0F0F0", tickprefix="$"),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_scat, use_container_width=True)
        except Exception as _e_scat:
            st.warning(f"⚠️ No se pudo generar el gráfico de dispersión: {str(_e_scat)}")
        st.markdown("<br>", unsafe_allow_html=True)

    # Mini gráfica de distribución de betas + donut lado a lado
    mg1, mg2 = st.columns([3,2])
    with mg1:
        fig = px.histogram(df_m1a, x="beta", nbins=35,
                           color="recomendacion", color_discrete_map=REC_COLORS,
                           title="Distribución de elasticidad (β) de todos los productos analizados",
                           labels={"beta":"Beta","count":"SKUs"},
                           barmode="overlay", opacity=0.75)
        fig.add_vline(x=-1,   line_dash="dash", line_color="gray", opacity=0.6, annotation_text="β=-1")
        fig.add_vline(x=-1.5, line_dash="dot",  line_color="gray", opacity=0.5, annotation_text="β=-1.5")
        fig.add_vline(x=0, line_color="black", line_width=0.5, opacity=0.3)
        st.plotly_chart(_layout(fig, h=260), use_container_width=True)
    with mg2:
        fig = px.pie(values=rec_counts.values, names=rec_counts.index,
                     title="SKUs por recomendación",
                     color=rec_counts.index, color_discrete_map=REC_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10,
                          hovertemplate="<b>%{label}</b><br>%{value} SKUs<extra></extra>")
        fig.update_layout(showlegend=False, height=260, paper_bgcolor="white",
                          margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)


    # Fix 10 — Impact KPIs moved here (right after Paso 2 / OLS section)
    if len(df_sim) > 0:
        _sub_pool = df_m1a[df_m1a["recomendacion"]=="Subir precio"]["prod_nm"].tolist()
        _prm_pool = df_m1a[df_m1a["recomendacion"]=="Bajar / Promover"]["prod_nm"].tolist()

        def _delta_rev(nms, esc):
            base = df_sim[(df_sim["prod_nm"].isin(nms)) & (df_sim["cambio"]=="Base 0%")]["ingreso_est"].sum()
            best = df_sim[(df_sim["prod_nm"].isin(nms)) & (df_sim["cambio"]==esc)]["ingreso_est"].sum()
            return best - base

        def _delta_uds(nms, esc):
            base = df_sim[(df_sim["prod_nm"].isin(nms)) & (df_sim["cambio"]=="Base 0%")]["unidades_est"].sum()
            best = df_sim[(df_sim["prod_nm"].isin(nms)) & (df_sim["cambio"]==esc)]["unidades_est"].sum()
            return best - base

        _ing_sub = _delta_rev(_sub_pool, "+10%")
        _ing_prm = _delta_rev(_prm_pool, "-10%")
        _uds_prm = _delta_uds(_prm_pool, "-10%")
        _total   = _ing_sub + _ing_prm

        section("💰 Impacto potencial de negocio — si implementas las recomendaciones")
        st.caption("Estimación del modelo con los datos actuales. Con más historial, la precisión aumenta.")
        def _kpi_impact(col, label, value_str, color):
            parts = value_str.split("/")
            main_val = parts[0]
            suffix   = "/" + parts[1] if len(parts) > 1 else ""
            col.markdown(
                f'<div style="background:white;border-radius:12px;border-top:4px solid {color};'
                f'padding:20px 16px;text-align:center;min-height:130px;display:flex;'
                f'flex-direction:column;justify-content:center;'
                f'box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:8px;">'
                f'<div style="font-size:1.9rem;font-weight:800;color:#1A1A1A;line-height:1.1;">'
                f'{main_val}</div>'
                f'<div style="font-size:1rem;font-weight:500;color:#555;margin-bottom:4px;">'
                f'{suffix}</div>'
                f'<div style="font-size:0.7rem;font-weight:600;color:#666;text-transform:uppercase;'
                f'letter-spacing:0.05em;margin-top:6px;">{label}</div>'
                f'</div>', unsafe_allow_html=True)

        _ic1, _ic2, _ic3, _ic4 = st.columns(4)
        _kpi_impact(_ic1, f"Subir precio — {len(_sub_pool)} prod. (+10%)",
                    f"+${_ing_sub:,.0f}/mes", OM_BLUE)
        _kpi_impact(_ic2, f"Promover — {len(_prm_pool)} prod. (-10%)",
                    f"+${_ing_prm:,.0f}/mes" if _ing_prm >= 0 else f"${_ing_prm:,.0f}/mes", OM_GREEN)
        _kpi_impact(_ic3, "Unidades extra por promos", f"+{_uds_prm:,.0f}/mes", OM_AMBER)
        _kpi_impact(_ic4, "Impacto total anualizado",  f"+${_total*12:,.0f}/año", OM_RED)
        st.markdown("<br>", unsafe_allow_html=True)


    # ── ¿La promoción funcionó? (análisis pre/durante/post) ──────────────────
    _promo_eval = ml_res.get("promo_eval") if ml_res else None
    if _promo_eval is not None and len(_promo_eval) > 0:
        section("🏷️ ¿Tus promociones realmente funcionaron?")
        st.caption("Comparamos las ventas 2 meses antes, durante y 2 meses después de cada promoción detectada. "
                   "Así sabemos si la promo generó ventas reales o solo adelantó compras.")

        n_rent  = (_promo_eval["rentable"]=="✅ Sí").sum()
        n_sost  = (_promo_eval["retencion"]=="✅ Sostuvo").sum()
        n_cayo  = (_promo_eval["retencion"]=="❌ Cayó post-promo").sum()
        n_ev    = len(_promo_eval)

        # Resumen en lenguaje de negocio antes de la tabla
        _pct_rent = n_rent/n_ev*100 if n_ev else 0
        _pct_cayo = n_cayo/n_ev*100 if n_ev else 0
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>Resumen:</strong> De las {n_ev} promociones analizadas, '
            f'<strong>{n_rent} ({_pct_rent:.0f}%) generaron más ingresos</strong> durante la promo. '
            f'Sin embargo, en <strong>{n_cayo} ({_pct_cayo:.0f}%) la demanda cayó después</strong> — '
            f'señal de que esos clientes solo compraron por el descuento, no se fidelizaron.'
            f'</div>', unsafe_allow_html=True)

        pe1,pe2,pe3,pe4 = st.columns(4)
        kpi(pe1, "Promociones analizadas",          f"{n_ev}",                                    OM_BLUE)
        kpi(pe2, "Generaron más ingresos ✅",        f"{n_rent} ({n_rent/n_ev*100:.0f}%)" if n_ev else "—", OM_GREEN)
        kpi(pe3, "Demanda sostuvo después",          f"{n_sost} ({n_sost/n_ev*100:.0f}%)" if n_ev else "—", OM_AMBER)
        kpi(pe4, "Demanda cayó post-promo ❌",       f"{n_cayo} ({n_cayo/n_ev*100:.0f}%)" if n_ev else "—", OM_RED)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🟢 Verde = promo rentable y sostuvo la demanda  |  "
                   "🟡 Amarillo = subió durante pero volvió a normal  |  "
                   "🔴 Rojo = cayó después (el cliente solo compró por el descuento, no se fidelizó)")

        # Fix 13 — clean number formatting for display copy only
        show_eval = _promo_eval[[
            "prod_nm","dept_nm","desc_pct",
            "uds_pre","uds_dur","uds_post",
            "up_dur","up_post","rv_up",
            "rentable","retencion"
        ]].copy()
        # Round unit columns to int, pct to 1 decimal
        for _uc in ["uds_pre","uds_dur","uds_post"]:
            show_eval[_uc] = pd.to_numeric(show_eval[_uc], errors="coerce").round(0).astype("Int64")
        for _pc in ["up_dur","up_post","rv_up","desc_pct"]:
            show_eval[_pc] = pd.to_numeric(show_eval[_pc], errors="coerce").round(1)
        show_eval.columns = [
            "Producto","Departamento","Descuento %",
            "Uds ANTES","Uds DURANTE","Uds DESPUÉS",
            "Uplift uds %","Post-promo %","Uplift ventas %",
            "¿Rentable?","¿Sostuvo demanda?"
        ]

        def _color_promo(row):
            if row["¿Rentable?"]=="✅ Sí" and row["¿Sostuvo demanda?"]=="✅ Sostuvo":
                c = "#E8F5E9"
            elif row["¿Sostuvo demanda?"]=="❌ Cayó post-promo":
                c = "#FFEBEE"
            else:
                c = "#FFF9C4"
            return [f"background-color:{c}"]*len(row)

        fmt = {"Descuento %":"{:.1f}%","Uplift uds %":"{:+.1f}%",
               "Uplift ventas %":"{:+.1f}%","Post-promo %":"{:+.1f}%"}
        st.dataframe(
            show_eval.style.apply(_color_promo, axis=1).format(fmt, na_rep="—"),
            use_container_width=True, height=320, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # ── Selector: general o por producto ─────────────────────────────────────
    section("🔍 Análisis por producto")
    st.caption("Selecciona un producto para ver exactamente qué hacer con su precio y en qué meses. "
               "Puedes buscar cualquier producto — incluyendo los sin recomendación, que te dirán por qué.")

    # Construir opciones con etiqueta de recomendación
    rec_emoji = {"Subir precio":"🔵","Mantener precio":"🟡",
                 "Bajar / Promover":"🟢","No recomendable":"⚪"}
    sku_options = ["— Ver resumen general de todos los productos —"] + [
        f"{rec_emoji.get(r, '⚪')} {nm} (SKU: {sku})"
        for nm, r, sku in zip(df_m1a["prod_nm"], df_m1a["recomendacion"], df_m1a["prod_nbr"])
    ]
    sel_sku_pred = st.selectbox("Selecciona un producto para ver el análisis detallado:", sku_options, key="pred_sku_sel")

    st.markdown("<br>", unsafe_allow_html=True)

    if sel_sku_pred != "— Ver resumen general de todos los productos —":
        # ── MODO PRODUCTO ESPECÍFICO ──────────────────────────────────────────
        # strip emoji prefix and (SKU: XXXX) suffix
        import re as _re
        real_nm = _re.sub(r'\s*\(SKU:.*?\)\s*$', '', sel_sku_pred[2:].strip())
        sku_row = df_m1a[df_m1a["prod_nm"] == real_nm].iloc[0]
        sku_sim = df_sim[df_sim["prod_nm"] == real_nm]
        sku_cal = df_cal[df_cal["prod_nm"] == real_nm] if len(df_cal) > 0 else pd.DataFrame()
        rec     = sku_row["recomendacion"]
        rec_c   = REC_COLORS.get(rec, OM_LGRAY)
        beta_v  = sku_row["beta"]
        r2_v    = sku_row["r2"]
        pval_v  = sku_row["pval"]
        n_v     = sku_row["n_meses"]

        # ── Tarjeta de recomendación ──────────────────────────────────────────
        col_badge, col_meta = st.columns([1, 2])
        with col_badge:
            st.markdown(
                f'<div style="background:{rec_c};color:white;border-radius:16px;padding:24px;'
                f'text-align:center;font-size:20px;font-weight:900;">'
                f'{rec.upper()}</div>',
                unsafe_allow_html=True)
        with col_meta:
            st.markdown(
                f'<div style="background:white;border-radius:12px;padding:16px 20px;'
                f'border-left:4px solid {rec_c};font-size:13px;line-height:2;">'
                f'<b>Elasticidad (β):</b> {beta_v:.4f} &nbsp;|&nbsp; '
                f'<b>R²:</b> {r2_v:.4f} &nbsp;|&nbsp; '
                f'<b>p-valor:</b> {pval_v:.4f} &nbsp;|&nbsp; '
                f'<b>Meses de datos:</b> {n_v}</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── Narrativa principal (usa st.markdown nativo — maneja ** correctamente) ──
        narr = generate_sku_narrative(sku_row, sku_sim, sku_cal)
        if narr:
            with st.container():
                st.markdown(narr)

        st.markdown("---")

        # ── Simulación de precios ─────────────────────────────────────────────
        if len(sku_sim) > 0 and rec != "No recomendable":
            section("📊 Predicción — ¿cuánto gano si cambio el precio?")
            left, right = st.columns([1, 2])

            base = sku_sim[sku_sim["cambio"] == "Base 0%"]
            if len(base) > 0:
                br = base.iloc[0]
                with left:
                    st.markdown(
                        f'<div style="background:#F8F9FA;border-radius:10px;padding:16px;font-size:13px;line-height:2.2;">'
                        f'<b>Precio actual:</b> ${br["precio_nuevo"]:,.2f}<br>'
                        f'<b>Unidades/mes (est.):</b> {br["unidades_est"]:,.1f}<br>'
                        f'<b>Ingreso mensual:</b> ${br["ingreso_est"]:,.0f}<br>'
                        f'<b>Margen mensual:</b> ${br["margen_est"]:,.0f}'
                        f'</div>', unsafe_allow_html=True)

            with right:
                colors_sc = ["#1565C0","#90CAF9","#9E9E9E","#EF9A9A","#C62828"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sku_sim["cambio"], y=sku_sim["delta_ingreso_pct"],
                    name="Cambio en Ingreso", marker_color=colors_sc,
                    text=[f"{v:+.1f}%" for v in sku_sim["delta_ingreso_pct"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Ingreso: %{y:+.1f}%<extra></extra>"))
                fig.add_trace(go.Bar(
                    x=sku_sim["cambio"], y=sku_sim["delta_margen_pct"],
                    name="Cambio en Margen", marker_color=colors_sc, opacity=0.5,
                    text=[f"{v:+.1f}%" for v in sku_sim["delta_margen_pct"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Margen: %{y:+.1f}%<extra></extra>"))
                fig.add_hline(y=0, line_color="black", line_width=0.8)
                fig.update_layout(barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                                  height=320, margin=dict(t=30,b=20,l=20,r=20),
                                  xaxis_title="Escenario de precio", yaxis_title="Cambio (%)",
                                  legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, use_container_width=True)

        # ── Calendario mensual ────────────────────────────────────────────────
        if len(sku_cal) > 0 and rec != "No recomendable":
            section("📅 ¿En qué meses actuar? — basado en historial de demanda")
            st.caption("La barra muestra la demanda relativa de cada mes. "
                       "Meses con demanda alta (>1) son mejores para subir precio si el producto es inelástico, "
                       "o para promover si es elástico.")
            accion_colors = {"SUBIR": OM_BLUE, "PROMOVER": OM_GREEN, "MANTENER": OM_AMBER}
            fig = px.bar(
                sku_cal.sort_values("mes_cal"), x="mes_nombre", y="u_idx",
                color="accion",
                labels={"mes_nombre": "Mes", "u_idx": "Demanda relativa al promedio", "accion": "Acción"},
                color_discrete_map=accion_colors,
                category_orders={"mes_nombre": MONTH_ORDER, "accion": ["SUBIR","PROMOVER","MANTENER"]},
                text="accion")
            fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5,
                          annotation_text="Promedio anual")
            fig.update_traces(
                textposition="inside",
                hovertemplate="<b>%{x}</b><br>Demanda relativa: %{y:.2f}x<br>Acción: %{fullData.name}<extra></extra>")
            fig.update_layout(showlegend=True, legend_title="Acción recomendada")
            st.plotly_chart(_layout(fig, h=320), use_container_width=True)

        # st.stop() removed — simulador y claude siguen visibles
    section("💹 Simulador de precios — ¿cuánto gano si cambio el precio?")
    st.caption("Selecciona un producto y ve exactamente cuánto cambian los ingresos y el margen.")

    # Filtro por departamento para el simulador
    sim_depts = ["Todos"] + sorted(df_m1a["dept_nm"].dropna().unique().tolist())
    sd1, sd2 = st.columns([1,3])
    with sd1:
        sim_dept = st.selectbox("Departamento", sim_depts, key="sim_dept")
    valid_pool = df_m1a[df_m1a["recomendacion"]!="No recomendable"]
    if sim_dept != "Todos":
        valid_pool = valid_pool[valid_pool["dept_nm"]==sim_dept]

    if len(valid_pool) == 0:
        st.info("Sin productos válidos en este departamento.")
    else:
        with sd2:
            sel_nm = st.selectbox("Producto a simular", sorted(valid_pool["prod_nm"].tolist()), key="sim_sku")

        sku_row = df_m1a[df_m1a["prod_nm"]==sel_nm].iloc[0]
        sku_sim = df_sim[df_sim["prod_nm"]==sel_nm]

        if len(sku_sim) > 0:
            beta_v = sku_row["beta"]; r2_v = sku_row["r2"]; pval_v = sku_row["pval"]; n_v = sku_row["n_meses"]
            rec_c = REC_COLORS.get(sku_row["recomendacion"], OM_LGRAY)

            inf_col, chart_col = st.columns([1,2])
            with inf_col:
                st.markdown(
                    f'<div style="background:white;border-radius:12px;padding:20px;border-left:5px solid {rec_c};">'
                    f'<div style="font-weight:700;font-size:14px;margin-bottom:12px;">{sel_nm[:50]}</div>'
                    f'<div style="font-size:13px;color:#555;line-height:2.2;">'
                    f'Elasticidad (β): <strong>{beta_v:.4f}</strong><br>'
                    f'Ajuste del modelo (R²): <strong>{r2_v:.4f}</strong><br>'
                    f'Significancia (p): <strong>{pval_v:.4f}</strong><br>'
                    f'Meses de datos: <strong>{n_v}</strong></div>'
                    f'<div style="background:{rec_c};color:white;padding:8px 14px;border-radius:20px;'
                    f'font-weight:700;text-align:center;font-size:12px;margin-top:14px;">'
                    f'{sku_row["recomendacion"].upper()}</div></div>', unsafe_allow_html=True)

                base_r = sku_sim[sku_sim["cambio"]=="Base 0%"]
                if len(base_r)>0:
                    br = base_r.iloc[0]
                    st.markdown(
                        f'<div style="background:#F5F5F5;border-radius:8px;padding:14px;font-size:13px;margin-top:10px;">'
                        f'<strong>Precio actual:</strong> ${br["precio_nuevo"]:,.2f}<br>'
                        f'<strong>Unidades/mes:</strong> {br["unidades_est"]:,.1f}<br>'
                        f'<strong>Ingreso estimado:</strong> ${br["ingreso_est"]:,.0f}<br>'
                        f'<strong>Margen estimado:</strong> ${br["margen_est"]:,.0f}</div>',
                        unsafe_allow_html=True)

            with chart_col:
                colors_sc = ["#1565C0","#90CAF9","#9E9E9E","#EF9A9A","#C62828"]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sku_sim["cambio"], y=sku_sim["delta_ingreso_pct"],
                                     name="Δ Ingreso", marker_color=colors_sc,
                                     text=[f"{v:+.1f}%" for v in sku_sim["delta_ingreso_pct"]],
                                     textposition="outside",
                                     hovertemplate="<b>%{x}</b><br>Δ Ingreso: %{y:+.1f}%<extra></extra>"))
                fig.add_trace(go.Bar(x=sku_sim["cambio"], y=sku_sim["delta_margen_pct"],
                                     name="Δ Margen", marker_color=colors_sc, opacity=0.5,
                                     text=[f"{v:+.1f}%" for v in sku_sim["delta_margen_pct"]],
                                     textposition="outside",
                                     hovertemplate="<b>%{x}</b><br>Δ Margen: %{y:+.1f}%<extra></extra>"))
                fig.add_hline(y=0, line_color="black", line_width=0.8)
                fig.update_layout(title="Cambio en Ingreso y Margen por escenario de precio",
                                  barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                                  height=360, margin=dict(t=50,b=20,l=20,r=20),
                                  xaxis_title="Escenario de precio", yaxis_title="Cambio (%)",
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

            # Continuous curve
            base_r2 = sku_sim[sku_sim["cambio"]=="Base 0%"]
            if len(base_r2)>0:
                br=base_r2.iloc[0]; p0=br["precio_nuevo"]; u0=br["unidades_est"]
                m0=br["margen_est"]; rev0=br["ingreso_est"]
                c0 = p0 - (m0/u0 if u0 else 0); marg0_base=(p0-c0)*u0 or 1
                rng=np.linspace(-0.20,0.20,200)
                pct_rev  = [(p0*(1+r)*u0*((1+r)**beta_v)-rev0)/rev0*100 for r in rng]
                pct_marg = [((p0*(1+r)-c0)*u0*((1+r)**beta_v)-marg0_base)/marg0_base*100 for r in rng]
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(x=rng*100,y=pct_rev,name="Δ Ingreso",
                                          line=dict(color=OM_RED,width=2.5)))
                fig2.add_trace(go.Scatter(x=rng*100,y=pct_marg,name="Δ Margen",
                                          line=dict(color=OM_BLUE,width=2,dash="dash")))
                fig2.add_hline(y=0,line_color="gray",line_width=0.7)
                fig2.add_vline(x=0,line_color="gray",line_width=0.7)
                fig2.update_layout(title="Curva de sensibilidad continua (±20% en precio)",
                                   plot_bgcolor="white",paper_bgcolor="white",
                                   height=300,margin=dict(t=50,b=20,l=20,r=20),
                                   xaxis_title="Cambio en precio (%)",yaxis_title="Cambio en resultado (%)",
                                   legend=dict(orientation="h",y=1.1))
                st.plotly_chart(fig2, use_container_width=True)


    # ── Resumen ejecutivo con IA — al fondo ──
    section("🤖 Resumen ejecutivo con Inteligencia Artificial")
    st.caption("Claude genera un reporte integrado con los hallazgos clave, acciones inmediatas e impacto financiero. "
               "Ideal para presentar a dirección.")

    _api_key2 = os.environ.get("ANTHROPIC_API_KEY", "")
    if not _api_key2:
        st.warning("Configura ANTHROPIC_API_KEY en Railway para habilitar el reporte ejecutivo con IA.")
    else:
        _bc2, _ic2 = st.columns([1, 3])
        with _bc2:
            _gen_ai2 = st.button("🤖 Generar reporte ejecutivo", type="primary", key="claude_top")
        with _ic2:
            st.caption("~10 segundos · integra análisis de precios + impacto financiero + timing")

        if _gen_ai2:
            with st.spinner("Generando reporte ejecutivo… esto tarda unos 10 segundos"):
                try:
                    import anthropic as _anth2
                    _cli2 = _anth2.Anthropic(api_key=_api_key2)
                    _ns2  = len(df_m1a[df_m1a["recomendacion"]=="Subir precio"])
                    _np2  = len(df_m1a[df_m1a["recomendacion"]=="Bajar / Promover"])
                    _nm2  = len(df_m1a[df_m1a["recomendacion"]=="Mantener precio"])
                    _nn2  = len(df_m1a[df_m1a["recomendacion"]=="No recomendable"])
                    _vdf2 = df_m1a[df_m1a["recomendacion"]!="No recomendable"]
                    _r2v2 = float(_vdf2["r2"].mean()) if len(_vdf2)>0 else 0
                    _bmd2 = float(df_m1a["beta"].median())
                    _psg2 = float((df_m1a["pval"]<0.10).mean()*100)
                    _mlr2 = st.session_state["ml_results"]["r2_r"] if st.session_state.get("ml_results") else 0
                    _mlg2 = st.session_state["ml_results"]["ganador"] if st.session_state.get("ml_results") else "ML"

                    def _ct2(rec2, esc2, n2=6):
                        pool2 = df_m1a[df_m1a["recomendacion"]==rec2].nsmallest(n2,"pval")
                        rows2 = []
                        for _, r2 in pool2.iterrows():
                            sr2 = df_sim[(df_sim["prod_nm"]==r2["prod_nm"])&(df_sim["cambio"]==esc2)]
                            d2 = f" | delta={sr2.iloc[0]['delta_ingreso_pct']:+.1f}%" if len(sr2)>0 else ""
                            rows2.append(f"  * {r2['prod_nm'][:38]} | b={r2['beta']:.2f}{d2}")
                        return "\n".join(rows2) or "  (ninguno)"

                    _dep2 = ""
                    if "dept_nm" in df_m1a.columns:
                        for dn2, grp2 in df_m1a.groupby("dept_nm"):
                            vc2 = grp2["recomendacion"].value_counts()
                            _dep2 += f"  {str(dn2)[:22]}: " + " ".join(f"{r3}={c3}" for r3,c3 in vc2.items()) + "\n"

                    _cal2 = ""
                    if len(df_cal)>0:
                        ms2 = df_cal[df_cal["accion"]=="SUBIR"]["mes_nombre"].value_counts().head(3).index.tolist()
                        mp2 = df_cal[df_cal["accion"]=="PROMOVER"]["mes_nombre"].value_counts().head(3).index.tolist()
                        _cal2 = f"SUBIR en: {', '.join(ms2) or 'N/A'} | PROMOVER en: {', '.join(mp2) or 'N/A'}"

                    _fin2 = ""
                    if len(df_sim)>0:
                        def _tdf2(recs2, esc2b):
                            sk2 = df_m1a[df_m1a["recomendacion"].isin(recs2)]["prod_nm"].tolist()
                            b2  = df_sim[(df_sim["prod_nm"].isin(sk2))&(df_sim["cambio"]=="Base 0%")]["ingreso_est"].sum()
                            t2  = df_sim[(df_sim["prod_nm"].isin(sk2))&(df_sim["cambio"]==esc2b)]["ingreso_est"].sum()
                            return t2 - b2
                        _ds2 = _tdf2(["Subir precio"],"+10%")
                        _dp2 = _tdf2(["Bajar / Promover"],"-10%")
                        _fin2 = f"Subir: ${_ds2:+,.0f}/mes | Promos: ${_dp2:+,.0f}/mes | Total: ${_ds2+_dp2:+,.0f}/mes"

                    _prom2 = (
                        "Eres consultor senior de revenue management para OfficeMax Mexico.\n"
                        "Escribe un reporte ejecutivo integrado en espanol. Maximo 350 palabras.\n\n"
                        f"PIPELINE:\n"
                        f"- ML ({_mlg2}): R2={_mlr2:.3f} prediciendo ventas. Precio=driver controlable #1 (58% importancia).\n"
                        f"- OLS por SKU: {len(df_m1a)} analizados, {len(_vdf2)} validos. Beta mediana={_bmd2:.2f}. {_psg2:.0f}% significativos.\n\n"
                        f"DISTRIBUCION: Subir={_ns2} | Mantener={_nm2} | Promover={_np2} | Sin rec={_nn2}\n\n"
                        f"POR DEPARTAMENTO:\n{_dep2}\n"
                        f"TOP SUBIR PRECIO:\n{_ct2('Subir precio', '+10%')}\n\n"
                        f"TOP PROMOVER:\n{_ct2('Bajar / Promover', '-10%')}\n\n"
                        f"TIMING: {_cal2}\n"
                        f"IMPACTO: {_fin2}\n\n"
                        "SECCIONES (se especifico, no uses frases genericas):\n"
                        "1. Hallazgo clave: que dice el ML+OLS del catalogo\n"
                        "2. Acciones inmediatas: productos especificos, betas, impacto\n"
                        "3. Timing: cuando ejecutar segun estacionalidad\n"
                        "4. Proyeccion de impacto financiero\n"
                        "5. Proximos pasos: como escalar al catalogo completo"
                    )

                    _resp2 = _cli2.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=1000,
                        messages=[{"role":"user","content":_prom2}])
                    st.session_state["ai_analysis"] = _resp2.content[0].text
                except Exception as _e2:
                    st.error(f"Error al generar el reporte: {_e2}")

        if st.session_state.get("ai_analysis"):
            st.markdown(
                '<div class="narrative-box" style="border-left-color:#7B1FA2;">'
                '<div style="font-size:11px;color:#7B1FA2;font-weight:700;margin-bottom:8px;">'
                'REPORTE EJECUTIVO — CLAUDE (Anthropic)</div></div>',
                unsafe_allow_html=True)
            st.markdown(st.session_state["ai_analysis"])

    # ── PDF download + email buttons ──────────────────────────────────────────
    from datetime import datetime as _dt_pdf
    if st.session_state.get("results") and st.session_state.get("ml_results"):
        try:
            _pdf_bytes = generate_pdf_report(
                st.session_state["results"],
                st.session_state["ml_results"],
                st.session_state.get("ai_analysis", ""),
                st.session_state["df_main"])
            _col_dl, _col_mail, _ = st.columns([1, 1, 2])
            with _col_dl:
                st.download_button(
                    label="📄 Descargar reporte PDF",
                    data=_pdf_bytes,
                    file_name=f"reporte_pricing_officemax_{_dt_pdf.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True)
            with _col_mail:
                if st.button("📧 Enviar por correo", use_container_width=True, key="open_email_form"):
                    st.session_state["pdf_bytes_to_send"] = _pdf_bytes
                    st.session_state["show_email_form"]   = True
        except Exception as _epdf:
            st.warning(f"⚠️ No se pudo generar el PDF: {str(_epdf)}")

    # ── Email form ────────────────────────────────────────────────────────────
    if st.session_state.get("show_email_form") and st.session_state.get("pdf_bytes_to_send"):
        st.markdown("---")
        st.subheader("📧 Enviar reporte por correo")
        _recip_input = st.text_area(
            "Destinatarios (uno por línea o separados por coma)",
            placeholder="gerente@officemax.com\ndirector@officemax.com",
            height=100, key="recipients_input")
        _subj = st.text_input(
            "Asunto",
            value=f"Reporte de Optimización de Precios — {_dt_pdf.now().strftime('%B %Y')}",
            key="email_subject_input")
        _body = st.text_area(
            "Mensaje (el PDF irá adjunto)",
            value=("Estimado/a,\n\n"
                   "Adjunto encontrará el reporte ejecutivo de optimización de precios "
                   "generado por el Dynamic Pricing Analyzer de OfficeMax México.\n\n"
                   "El reporte incluye:\n"
                   "• Resumen ejecutivo con hallazgos clave\n"
                   "• Indicadores principales del análisis\n"
                   "• Top 20 recomendaciones de precio por producto\n\n"
                   "Saludos,\nEquipo de Pricing OfficeMax México"),
            height=180, key="email_body_input")
        _cs, _cc = st.columns(2)
        with _cs:
            if st.button("📤 Enviar ahora", type="primary", use_container_width=True, key="send_email_btn"):
                _from  = st.session_state.get("email_from", "")
                _pword = st.session_state.get("email_pass", "")
                if not _from or not _pword:
                    st.error("⚠️ Configura tu correo y contraseña en el panel izquierdo (📧 Configuración de correo)")
                elif not _recip_input.strip():
                    st.error("⚠️ Ingresa al menos un destinatario")
                else:
                    _recips = [r.strip() for r in _recip_input.replace(",","\n").splitlines() if r.strip()]
                    with st.spinner("Enviando reporte..."):
                        _ok, _msg = send_report_email(
                            st.session_state["pdf_bytes_to_send"],
                            _recips, _subj, _body, _from, _pword)
                    if _ok:
                        st.success(_msg)
                        st.session_state["show_email_form"]   = False
                        st.session_state["pdf_bytes_to_send"] = None
                    else:
                        st.error(_msg)
        with _cc:
            if st.button("Cancelar", use_container_width=True, key="cancel_email_btn"):
                st.session_state["show_email_form"]   = False
                st.session_state["pdf_bytes_to_send"] = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
