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
OM_RED   = "#E31837"
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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k in ["df_main", "clean_report", "results", "df_csv_bytes", "loaded_file_name",
           "ai_analysis", "ml_results", "promo_bytes", "promo_file_name"]:
    if _k not in st.session_state:
        st.session_state[_k] = None


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
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def kpi(col, label, value, color=OM_RED):
    col.markdown(f'<div class="kpi-card" style="border-top-color:{color};">'
                 f'<div class="kpi-value">{value}</div>'
                 f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

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
    st.caption("Sube el ZIP con los 4 archivos CSV.")
    uploaded = st.file_uploader("Archivo .ZIP", type=["zip"], label_visibility="collapsed")

    if uploaded:
        # Solo procesar si es un archivo NUEVO — el file_uploader retiene el archivo
        # en cada re-render, lo que ejecutaría este bloque (y borraría results) en cada clic.
        is_new_file = st.session_state.get("loaded_file_name") != uploaded.name
        if is_new_file:
            raw_bytes = uploaded.read()
            with st.spinner("Leyendo y limpiando..."):
                try:
                    df_main, report_df = load_and_clean(raw_bytes)
                    st.session_state["df_main"]          = df_main
                    st.session_state["clean_report"]     = report_df
                    st.session_state["results"]          = None
                    st.session_state["df_csv_bytes"]     = df_main.to_csv(index=False).encode("utf-8")
                    st.session_state["loaded_file_name"] = uploaded.name
                    st.success("✅ Datos cargados")
                    st.caption(report_df.iloc[-1]["Detalle"])
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.success("✅ Datos cargados")
            st.caption(st.session_state["clean_report"].iloc[-1]["Detalle"] if st.session_state["clean_report"] is not None else "")

    st.markdown("<hr style='border-color:#333; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🏷️ Promociones *(opcional)*")
    st.caption("Excel de promociones para el pipeline ML.")
    uploaded_promo = st.file_uploader("Excel de Promociones", type=["xlsx","xls"],
                                       label_visibility="collapsed", key="promo_uploader")
    if uploaded_promo:
        if st.session_state.get("promo_file_name") != uploaded_promo.name:
            st.session_state["promo_bytes"]     = uploaded_promo.read()
            st.session_state["promo_file_name"] = uploaded_promo.name
            st.session_state["ml_results"]      = None
            st.success("✅ Promociones cargadas")
        else:
            st.success("✅ Promociones cargadas")

    st.markdown("<hr style='border-color:#333; margin:16px 0;'>", unsafe_allow_html=True)
    st.caption("**Flujo recomendado:**\n\n1. Sube el ZIP\n2. **Calculadora** → ejecuta el modelo\n3. **Descriptivo** → explora los datos\n4. **Predictivo** → ve qué hacer")


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

    # ── Evaluación por SKU: ¿la promo funcionó? ──────────────────────────────
    skus_promo = mensual[mensual["tiene_promo"]==1]["prod_nbr"].unique()
    eval_rows = []
    prod_nm_map = mensual[["prod_nbr","prod_nm"]].drop_duplicates("prod_nbr").set_index("prod_nbr")["prod_nm"] if "prod_nm" in mensual.columns else {}
    for sku in skus_promo:
        grp = mensual[mensual["prod_nbr"]==sku]
        sin  = grp[grp["tiene_promo"]==0]
        con  = grp[grp["tiene_promo"]==1]
        if len(sin)==0 or len(con)==0: continue
        uds_base  = sin["unidades"].mean()
        uds_promo = con["unidades"].mean()
        rev_base  = sin["venta_tot"].mean()
        rev_promo = con["venta_tot"].mean()
        p_base    = sin["precio"].mean()
        p_promo   = con["precio"].mean()
        if uds_base<=0 or p_base<=0: continue
        uplift_uds = (uds_promo - uds_base) / uds_base * 100
        uplift_rev = (rev_promo - rev_base)  / rev_base  * 100
        pct_desc   = (p_base - p_promo) / p_base * 100
        rentable   = uplift_rev > 0
        eval_rows.append({
            "prod_nbr":   sku,
            "prod_nm":    str(prod_nm_map.get(sku, sku))[:40],
            "dept_nm":    str(dept_map.get(sku, "—"))[:25],
            "pct_desc":   round(pct_desc, 1),
            "uplift_uds": round(uplift_uds, 1),
            "uplift_rev": round(uplift_rev, 1),
            "rentable":   "✅ Sí" if rentable else "❌ No",
            "meses_promo":int(len(con)),
        })
    promo_eval = pd.DataFrame(eval_rows).sort_values("uplift_uds", ascending=False) if eval_rows else pd.DataFrame()

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


tab1, tab2, tab3 = st.tabs(["🧮  Calculadora","📈  Dashboard Descriptivo","🎯  Dashboard Predictivo"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALCULADORA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.expander("📋 Reporte de limpieza de datos", expanded=False):
        for _, row in clean_report.iterrows():
            c = OM_GREEN if "✅" in str(row["Paso"]) else (OM_RED if "🗑" in str(row["Paso"]) else OM_BLUE)
            st.markdown(
                f'<div class="clean-step" style="border-left-color:{c};">'
                f'<strong>{row["Paso"]}</strong>&nbsp;&nbsp;'
                f'<span style="color:{OM_RED}; font-weight:700;">{row["Eliminadas"]}</span>'
                f'&nbsp;&nbsp;<span style="color:#555;">{row["Detalle"]}</span></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("⚙️ Parámetros del modelo")
    st.caption("Ajusta estos valores antes de ejecutar. Si salen pocos productos, reduce el mínimo de meses o el CV.")

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        min_obs = st.slider("Meses mínimos por SKU", 3, 18, 6,
            help="Mínimo de meses con datos para analizar un SKU. Con menos de 6 los resultados son poco confiables. Baja a 4-5 solo si salen muy pocos productos.")
    with c2:
        min_r2 = st.slider("R² mínimo", 0.0, 0.5, 0.0, 0.05,
            help="Qué tan bien explica el precio las ventas. En retail 0.10-0.20 ya es aceptable. No uses 0.7+.")
    with c3:
        max_beta = st.slider("|Beta| máximo", 2.0, 15.0, 10.0, 0.5,
            help="Elasticidad máxima aceptable. Betas mayores a 5 suelen ser errores estadísticos.")
    with c4:
        min_cv_pct = st.slider("CV mínimo de precio (%)", 0.0, 10.0, 1.0, 0.5,
            help="Si el precio nunca cambió, no se puede estimar elasticidad. 1% es un umbral permisivo.")
        min_cv = min_cv_pct / 100
    with c5:
        pval_thresh = st.slider("p-valor máximo", 0.05, 0.25, 0.10, 0.05,
            help="Qué tan seguro eres de la beta. 0.10 = 90% de confianza (estándar en negocios).")

    run_rolling = st.checkbox(
        "Calcular Tendencia Trimestral y Semestral (~1-3 min extra)", value=False,
        help="Activa para ver cómo cambia la elasticidad a lo largo del tiempo (ventanas de 3 y 6 meses). Necesario para el calendario temporal detallado.")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶️  Ejecutar análisis de elasticidad", type="primary")

    if run_btn:
        csv_bytes = st.session_state["df_csv_bytes"]
        with st.spinner("⚙️ Paso 1/2 — Calculando elasticidad OLS..."):
            try:
                res = run_models(df_csv=csv_bytes, min_obs=min_obs, min_cv=min_cv,
                                 min_r2=min_r2, max_beta=max_beta, pval_thresh=pval_thresh,
                                 run_rolling=run_rolling)
                st.session_state["results"] = res
            except Exception as e:
                st.error(f"Error OLS: {e}"); st.exception(e)
        with st.spinner("🌲 Paso 2/2 — Ejecutando pipeline ML (RF vs Gradient Boosting)..."):
            try:
                promo_b = st.session_state.get("promo_bytes") or b""
                ml_res = run_ml_pipeline(csv_bytes, promo_b)
                st.session_state["ml_results"] = ml_res
            except Exception as e:
                st.warning(f"Pipeline ML no completado: {e}")
        st.success("✅ Análisis completado — ve al Dashboard Predictivo")

    results = st.session_state["results"]
    if results is None:
        st.info("Configura los parámetros y presiona **▶️ Ejecutar análisis**.")
    else:
        df_m1a = results["m1a"]; m2 = results["m2"]

        section("🔬 OLS por SKU — ¿cuánto cambia la demanda si cambia el precio?")
        st.caption("Regresión log-log individual por producto. No predice ventas totales — "
                   "estima la elasticidad β: cuánto cambia la demanda ante un 1% de cambio en precio.")

        n_valid = len(df_m1a[df_m1a["recomendacion"]!="No recomendable"]) if len(df_m1a)>0 else 0

        if len(df_m1a) > 0:
            validos_df  = df_m1a[df_m1a["recomendacion"]!="No recomendable"]
            r2_prom     = df_m1a["r2"].mean()
            r2_validos  = validos_df["r2"].mean() if len(validos_df)>0 else 0
            beta_median = df_m1a["beta"].median()
            pct_sig     = (df_m1a["pval"] < 0.10).mean() * 100

            # Semáforo basado en métricas agregadas M1A
            if r2_validos >= 0.35 and pct_sig >= 30:
                css_v, msg_v = "chip-green",  "🟢 Modelo confiable — resultados interpretables"
            elif r2_validos >= 0.15 and pct_sig >= 15:
                css_v, msg_v = "chip-yellow", "🟡 Modelo aceptable — úsalo con precaución"
            else:
                css_v, msg_v = "chip-red",    "🔴 Señal débil — reduce filtros o revisa datos"

            cv1, cv2 = st.columns([1, 2])
            with cv1:
                st.markdown(f'<div class="{css_v}">{msg_v}</div><br>', unsafe_allow_html=True)
                st.caption(f"✅ R² promedio (SKUs válidos): {r2_validos:.3f}")
                st.caption(f"✅ Beta mediana: {beta_median:.3f}")
                st.caption(f"✅ SKUs con p < 0.10: {pct_sig:.1f}%")
                st.caption(f"{'✅' if m2['beta_pval']<0.10 else '⚠️'} Señal global precio: p={m2['beta_pval']:.4f}")
            with cv2:
                st.dataframe(pd.DataFrame({
                    "Métrica": ["SKUs analizados","SKUs con rec. válida","R² promedio (todos SKUs)",
                                "R² promedio (SKUs válidos)","Beta mediana",
                                "% SKUs con p < 0.10","Observaciones en modelo global"],
                    "Valor":   [f"{len(df_m1a):,}", f"{n_valid:,}",
                                f"{r2_prom:.4f}", f"{r2_validos:.4f}",
                                f"{beta_median:.4f}", f"{pct_sig:.1f}%",
                                f"{m2['n_obs']:,}"],
                    "Interpretación": [
                        "Productos con suficientes datos para OLS",
                        "Pasan todos los filtros estadísticos",
                        "R² varía por SKU; productos con poca var. de precio tienen R² bajo",
                        "R² de los modelos que sí son estadísticamente válidos",
                        "Elasticidad típica: negativa = más precio → menos ventas",
                        "Porcentaje con relación precio-demanda estadísticamente significativa",
                        "Usado para validar que la señal existe a nivel agregado",
                    ]
                }), hide_index=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        section(f"📦 Elasticidad Total por producto — {len(df_m1a):,} analizados de {results['n_total']:,} · {n_valid} con recomendación válida")

        if len(df_m1a) == 0:
            st.warning("⚠️ Ningún SKU pasó los filtros. Prueba reduciendo el mínimo de meses a 3 o el CV a 0%.")
        else:
            rec_counts = df_m1a["recomendacion"].value_counts()
            c_rec = st.columns(4)
            for i,(rec,color) in enumerate(REC_COLORS.items()):
                cnt = rec_counts.get(rec,0); pct = cnt/len(df_m1a)*100
                c_rec[i].markdown(
                    f'<div class="rec-card" style="border-top:4px solid {color};">'
                    f'<div style="font-size:32px;font-weight:900;color:{color};">{cnt}</div>'
                    f'<div style="font-size:12px;font-weight:700;color:#333;">{rec.upper()}</div>'
                    f'<div style="font-size:11px;color:#999;">{pct:.1f}% del total</div></div>',
                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fc1,fc2,fc3 = st.columns(3)
            with fc1:
                depts = sorted(df_m1a["dept_nm"].dropna().unique().tolist())
                dept_f = st.multiselect("Filtrar departamento", depts, key="dept_f_cal")
            with fc2:
                rec_f = st.multiselect("Filtrar recomendación", list(REC_COLORS.keys()), key="rec_f_cal")
            with fc3:
                srch = st.text_input("Buscar nombre o SKU", placeholder="Ej: FOLDER, 50012983")

            show = df_m1a.copy()
            if dept_f: show = show[show["dept_nm"].isin(dept_f)]
            if rec_f:  show = show[show["recomendacion"].isin(rec_f)]
            if srch:
                show = show[show["prod_nm"].str.contains(srch,case=False,na=False)|
                            show["prod_nbr"].str.contains(srch,case=False,na=False)]

            disp = [c for c in ["prod_nbr","prod_nm","dept_nm","beta","r2","rmse","pval","n_meses","recomendacion"] if c in show.columns]
            st.dataframe(show[disp].rename(columns={
                "prod_nbr":"SKU","prod_nm":"Producto","dept_nm":"Departamento",
                "beta":"Beta","r2":"R²","rmse":"RMSE","pval":"p-valor",
                "n_meses":"Meses","recomendacion":"Recomendación"}).sort_values("Beta"),
                hide_index=True, use_container_width=True, height=380)

            st.markdown("<br>", unsafe_allow_html=True)
            dl1,dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Descargar resultados por SKU (CSV)",
                    df_m1a.to_csv(index=False).encode("utf-8"),"elasticidad_sku.csv","text/csv")
            with dl2:
                if len(results["sim"])>0:
                    st.download_button("⬇️ Descargar simulación de precios (CSV)",
                        results["sim"].to_csv(index=False).encode("utf-8"),"simulacion_precios.csv","text/csv")

        # ── Validación visual del modelo (usa ML si está disponible) ────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _ml_scatter = st.session_state.get("ml_results")
        if _ml_scatter and _ml_scatter.get("actual_r"):
            section("🎯 ¿Qué tan bien predice el modelo? — Gradient Boosting (Ventas en $)")
            st.caption("Validación del modelo ML en el conjunto de prueba (últimos 6 meses no vistos durante el entrenamiento). "
                       "Entre más cerca de la diagonal, mejor predice el modelo.")
            actual = _ml_scatter["actual_r"]
            fitted = _ml_scatter["fitted_r"]
            r2_show = _ml_scatter["r2_r"]
            n_train = _ml_scatter["n_train"]
            n_test  = _ml_scatter["n_test"]
            model_label = f"Gradient Boosting — {_ml_scatter.get('ganador','ML')}"
        else:
            section("🎯 ¿Qué tan bien predice el modelo? — Predicción vs Realidad")
            st.caption("Validación del modelo OLS global (M2) con controles de tienda y mes.")
            actual = m2.get("actual_sample", [])
            fitted = m2.get("fitted_sample", [])
            r2_show = m2["r2"]
            n_train = m2["n_obs"]
            n_test  = 0
            model_label = "OLS Global (M2)"

        if actual and fitted:
            vv1, vv2 = st.columns([2, 1])
            with vv1:
                max_val = float(np.percentile([v for v in actual+fitted if v < 1e9], 95))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual, y=fitted, mode="markers",
                    marker=dict(size=4, color=OM_RED, opacity=0.35),
                    hovertemplate="Real: $%{x:,.0f}<br>Predicho: $%{y:,.0f}<extra></extra>",
                    name="Observaciones"))
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines", line=dict(color=OM_BLUE, dash="dash", width=2),
                    name="Predicción perfecta"))
                fig.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=380, margin=dict(t=30,b=40,l=40,r=20),
                    xaxis=dict(title="Ventas reales ($)", range=[0, max_val]),
                    yaxis=dict(title="Ventas predichas ($)", range=[0, max_val]),
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
                        de la variación en ventas explicada por el modelo
                    </div>
                    <div style="font-size:11px;color:#999;margin-top:4px;">R² — {model_label}</div>
                </div>
                <div style="background:white;border-radius:14px;padding:18px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size:12px;color:#555;line-height:2.2;">
                    <b>Observaciones entrenamiento:</b> {n_train:,}<br>
                    <b>Observaciones prueba:</b> {n_test:,}<br>
                    <b>SKUs con rec. válida (OLS):</b> {n_valid}<br>
                    <b>Modelo ML:</b> {model_label}
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
    dept = (df.groupby("dept_nm").agg(venta=("venta_con_iva","sum"),n_skus=("prod_nbr","nunique"))
            .reset_index().sort_values("venta",ascending=False))
    dept["dept_short"] = dept["dept_nm"].str[:28]
    stores = (df.groupby(["store_nbr","store_nm"])
              .agg(venta=("venta_con_iva","sum"),margen=("margen","mean"),unidades=("qty","sum"))
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
    marca = (df.groupby("tipo_marca").agg(venta=("venta_con_iva","sum"),margen_prom=("margen","mean"))
             .reset_index()) if "tipo_marca" in df.columns else pd.DataFrame()
    if len(marca)>0: marca["margen_pct"] = marca["margen_prom"]*100
    seas = (df.groupby("mes_calendario")
            .agg(venta_prom=("venta_con_iva","mean"),unidades_prom=("qty","mean"))
            .reset_index().sort_values("mes_calendario"))
    seas["mes_nombre"] = seas["mes_calendario"].map(MONTH_NAMES)

    # margen_dinero: usa columna "utilidad" si existe, si no la estima desde venta*margen
    if "utilidad" in df.columns:
        utilidad_total = pd.to_numeric(df["utilidad"], errors="coerce").sum()
    elif "margen" in df.columns and "venta_con_iva" in df.columns:
        utilidad_total = (pd.to_numeric(df["venta_con_iva"], errors="coerce") *
                          pd.to_numeric(df["margen"], errors="coerce")).sum()
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
            "marca":marca,"seas":seas,"kpis":kpis,
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

    # KPIs
    section("📊 Indicadores clave")
    kc = st.columns(8)
    kpi(kc[0],"Venta total",         f"${kpis['venta']/1e6:.1f}M",              OM_RED)
    kpi(kc[1],"Unidades vendidas",   f"{kpis['unidades']/1e3:.0f}K",             OM_BLUE)
    kpi(kc[2],"Ticket promedio",     f"${kpis['ticket_prom']:,.0f}",             OM_AMBER)
    kpi(kc[3],"Precio prom. unit.",  f"${kpis['precio_prom']:,.2f}",             OM_BLUE)
    kpi(kc[4],"SKUs únicos",         f"{kpis['n_skus']:,}",                      OM_GREEN)
    kpi(kc[5],"Tiendas",             f"{kpis['n_stores']}",                      OM_AMBER)
    kpi(kc[6],"Margen promedio",     f"{kpis['margen']:.1f}%",                   "#7B1FA2")
    kpi(kc[7],"Margen en dinero",    f"${kpis['margen_dinero']/1e6:.1f}M",      OM_GREEN)
    st.markdown("<br>", unsafe_allow_html=True)

    # Time series
    section("📅 Evolución mensual de ventas")
    ts = agg["ts"]
    tc1,tc2 = st.columns(2)
    with tc1:
        fig = px.area(ts, x="mes_str", y="venta", title="Venta mensual total ($)",
                      labels={"mes_str":"Mes","venta":"Ventas ($)"}, color_discrete_sequence=[OM_RED])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)
    with tc2:
        fig = px.bar(ts, x="mes_str", y="unidades", title="Unidades vendidas por mes",
                     labels={"mes_str":"Mes","unidades":"Unidades"}, color_discrete_sequence=[OM_BLUE])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(_layout(fig), use_container_width=True)

    # Department
    section("🏷️ Ventas por departamento")
    dept = agg["dept"]
    dc1,dc2 = st.columns([3,2])
    with dc1:
        fig = px.bar(dept, x="venta", y="dept_short", orientation="h",
                     title="Venta total por departamento",
                     labels={"venta":"Ventas ($)","dept_short":"Departamento"},
                     color="venta", color_continuous_scale=[[0,"#FFCDD2"],[1,OM_RED]])
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"), showlegend=False)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>")
        st.plotly_chart(_layout(fig, h=380), use_container_width=True)
    with dc2:
        fig = px.pie(dept.head(8), values="venta", names="dept_short", title="Participación (top 8)",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=11,
                          hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>")
        fig.update_layout(showlegend=True, height=380, paper_bgcolor="white", margin=dict(t=45,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Store performance
    section("🏪 Desempeño por tienda (Top 20)")
    stores = agg["stores"]
    sc1,sc2 = st.columns(2)
    with sc1:
        fig = go.Figure(go.Bar(
            x=stores["label"], y=stores["venta"],
            marker=dict(color=stores["margen_pct"], colorscale=[[0,OM_RED],[0.5,OM_AMBER],[1,OM_GREEN]],
                        colorbar=dict(title="Margen %")),
            hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<br>Margen: %{marker.color:.1f}%<extra></extra>"))
        fig.update_layout(title="Top 20 tiendas por ventas (color = margen)",
                          xaxis=dict(tickangle=55, tickfont=dict(size=9)),
                          plot_bgcolor="white", paper_bgcolor="white",
                          height=380, margin=dict(t=45,b=80,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig = px.scatter(stores, x="venta", y="margen_pct", size="unidades", text="store_nbr",
                         title="Ventas vs Margen",
                         labels={"venta":"Ventas ($)","margen_pct":"Margen (%)","store_nbr":"Tienda"},
                         color="margen_pct", color_continuous_scale=[[0,OM_RED],[0.5,OM_AMBER],[1,OM_GREEN]])
        fig.update_traces(textposition="top center", textfont_size=8,
                          hovertemplate="<b>%{text}</b><br>Ventas: $%{x:,.0f}<br>Margen: %{y:.1f}%<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_layout(fig, h=380), use_container_width=True)

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

    # Distributions
    # Premium
    section("⭐ Premium vs No Premium")
    prem = agg["prem"]
    pm1,pm2,pm3 = st.columns(3)
    for col_w, metric, lbl, pfx in [
        (pm1,"venta","Venta total ($)","$"),
        (pm2,"precio_prom","Precio promedio ($)","$"),
        (pm3,"margen_pct","Margen promedio (%)",""),
    ]:
        vals = prem[metric].tolist()
        fig = px.bar(prem, x="tipo", y=metric, title=lbl,
                     color="tipo", color_discrete_map={"No Premium":OM_BLUE,"Premium":OM_RED})
        suf = "%" if pfx=="" else ""
        fig.update_traces(text=[f"{pfx}{v:,.1f}{suf}" for v in vals], textposition="outside",
                          hovertemplate="<b>%{x}</b><br>"+lbl+": %{y:,.2f}<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        col_w.plotly_chart(_layout(fig,h=260), use_container_width=True)

    # Brand type
    marca = agg["marca"]
    if len(marca) > 0:
        section("🏷️ Marca Propia vs Marca Externa")
        mb1,mb2 = st.columns(2)
        with mb1:
            fig = px.bar(marca, x="tipo_marca", y="venta", title="Venta por tipo de marca",
                         color="tipo_marca", color_discrete_sequence=[OM_RED,OM_BLUE,OM_GREEN])
            fig.update_traces(text=[f"${v/1e3:,.0f}K" for v in marca["venta"]], textposition="outside",
                              hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<extra></extra>")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Ventas ($)")
            st.plotly_chart(_layout(fig,h=280), use_container_width=True)
        with mb2:
            fig = px.bar(marca, x="tipo_marca", y="margen_pct", title="Margen por tipo de marca",
                         color="tipo_marca", color_discrete_sequence=[OM_GREEN,OM_AMBER,OM_BLUE])
            fig.update_traces(text=[f"{v:.1f}%" for v in marca["margen_pct"]], textposition="outside",
                              hovertemplate="<b>%{x}</b><br>Margen: %{y:.1f}%<extra></extra>")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Margen (%)")
            st.plotly_chart(_layout(fig,h=280), use_container_width=True)

    # Seasonality — ordered correctly
    section("📆 Estacionalidad — promedio por mes del año")
    seas = agg["seas"]
    se1,se2 = st.columns(2)
    with se1:
        fig = px.line(seas, x="mes_nombre", y="venta_prom", title="Venta promedio por mes del año",
                      labels={"mes_nombre":"","venta_prom":"Venta promedio ($)"},
                      markers=True, color_discrete_sequence=[OM_RED],
                      category_orders={"mes_nombre":MONTH_ORDER})
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Venta promedio: $%{y:,.0f}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig,h=280), use_container_width=True)
    with se2:
        fig = px.bar(seas, x="mes_nombre", y="unidades_prom", title="Unidades promedio por mes del año",
                     labels={"mes_nombre":"","unidades_prom":"Unidades promedio"},
                     color_discrete_sequence=[OM_BLUE],
                     category_orders={"mes_nombre":MONTH_ORDER})
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Unidades promedio: %{y:,.1f}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(_layout(fig,h=280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD PREDICTIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    results = st.session_state["results"]
    if results is None:
        st.info("⏳ Primero ejecuta el análisis en **🧮 Calculadora**.")
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

    pk = st.columns(6)
    kpi(pk[0], "SKUs con recomendación",  f"{n_validos} ({pct_valid:.0f}%)",   OM_RED)
    kpi(pk[1], "Subir precio",            f"{n_subir} SKUs",                    OM_BLUE)
    kpi(pk[2], "Promover / Bajar",        f"{n_promover} SKUs",                 OM_GREEN)
    kpi(pk[3], "Mantener precio",         f"{n_mantener} SKUs",                 OM_AMBER)
    kpi(pk[4], "Impacto anualizado est.", f"+${_imp_total_anual:,.0f}",         OM_RED)
    kpi(pk[5], "Precio explica demanda",  _precio_pct,                          OM_BLUE)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Paso 1: Pipeline ML — fundamento del modelo ───────────────────────────
    ml_res = st.session_state.get("ml_results")
    if ml_res:
        section("🔬 Paso 1 — ¿Qué impulsa las ventas? Machine Learning")
        st.caption("Random Forest vs Gradient Boosting para identificar los factores que más impactan la demanda. "
                   "El objetivo: saber qué variable de negocio podemos controlar para mover las ventas.")

        # Tabla comparativa
        ml_c1, ml_c2 = st.columns([1, 2])
        with ml_c1:
            comp = ml_res["comparacion"]
            ganador = ml_res["ganador"]
            comp_rows = []
            for nm, v in comp.items():
                comp_rows.append({"Modelo": f"{'★ ' if nm==ganador else ''}{nm}",
                                   "R² Unidades": f"{v['r2_u']:.3f}",
                                   "R² Ventas": f"{v['r2_r']:.3f}",
                                   "R² Promedio": f"{v['r2_avg']:.3f}"})
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
            st.caption(f"Train: {ml_res['train_range']}  |  Test: {ml_res['test_range']}")
            st.markdown(
                f'<div style="background:{OM_GREEN};color:white;border-radius:8px;padding:10px 14px;'
                f'font-weight:700;font-size:13px;margin-top:8px;">✅ Modelo ganador: {ganador}</div>',
                unsafe_allow_html=True)

        with ml_c2:
            feat_df = ml_res["feat_df"]
            # Agrupar features por categoría para narrativa más clara
            grupos = {
                "🔴 Precio (controlable)":   {"Log Precio","% Descuento vs modal","Precio vs Catálogo","Cambio precio % (mes ant.)"},
                "🔵 Historial de demanda":    {"Ventas mes anterior (log)","Media ventas 3m (log)"},
                "⚫ Contexto de mercado":     {"Mes del año","Tendencia temporal","Departamento","Es Premium"},
            }
            grp_imp = {}
            for grp_nm, feats in grupos.items():
                grp_imp[grp_nm] = float(feat_df[feat_df["feature"].isin(feats)]["importancia"].sum())
            grp_df = pd.DataFrame({"Grupo": list(grp_imp.keys()),
                                    "Importancia": list(grp_imp.values())}).sort_values("Importancia", ascending=True)
            colors_grp = [OM_LGRAY, OM_BLUE, OM_RED]
            fig_fi = go.Figure(go.Bar(
                x=grp_df["Importancia"] * 100,
                y=grp_df["Grupo"],
                orientation="h",
                marker_color=colors_grp,
                text=[f"{v*100:.1f}%" for v in grp_df["Importancia"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importancia: %{x:.1f}%<extra></extra>"))
            precio_total = grp_imp.get("🔴 Precio (controlable)", 0) * 100
            fig_fi.update_layout(
                title=f"Importancia de variables — {ganador} (🔴 = features de precio: {precio_total:.0f}%)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(t=45,b=10,l=10,r=60),
                xaxis=dict(title="Importancia (%)", range=[0, feat_df["importancia"].max()*115]),
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
            section("📅 Paso 1b — ¿Cuándo actuar? Timing óptimo por departamento")
            st.caption("Uplift promedio de ventas en meses con promoción vs sin promoción (datos históricos). "
                       "Verde = meses donde las promociones generan más impacto.")

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
                # Heatmap departamento × mes (compacto)
                fig_dh = go.Figure(data=go.Heatmap(
                    z=dept_pivot.values,
                    x=list(dept_pivot.columns),
                    y=[d[:22] for d in dept_pivot.index],
                    colorscale=[[0,"#EF5350"],[0.45,"#FFF9C4"],[1,"#43A047"]],
                    zmid=0,
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}% uplift<extra></extra>",
                    showscale=False))
                fig_dh.update_layout(
                    height=380, margin=dict(t=30,b=10,l=10,r=10),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(side="top", tickfont=dict(size=10)),
                    yaxis=dict(tickfont=dict(size=9)))
                st.plotly_chart(fig_dh, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Paso 2: Resumen OLS ───────────────────────────────────────────────────
    _paso2_label = "Paso 2 — " if ml_res else ""
    section(f"📊 {_paso2_label}¿Cuánto exactamente? OLS — Elasticidad precio por SKU")
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


    # ── Selector: general o por producto ─────────────────────────────────────
    section("🔍 Análisis por producto")
    st.caption("Selecciona un producto para ver exactamente qué hacer con su precio y en qué meses. "
               "Puedes buscar cualquier producto — incluyendo los sin recomendación, que te dirán por qué.")

    # Construir opciones con etiqueta de recomendación
    rec_emoji = {"Subir precio":"🔵","Mantener precio":"🟡",
                 "Bajar / Promover":"🟢","No recomendable":"⚪"}
    sku_options = ["— Ver resumen general de todos los productos —"] + [
        f"{rec_emoji.get(r, '⚪')} {nm}"
        for nm, r in zip(df_m1a["prod_nm"], df_m1a["recomendacion"])
    ]
    sel_sku_pred = st.selectbox("Producto:", sku_options, key="pred_sku_sel")

    st.markdown("<br>", unsafe_allow_html=True)

    if sel_sku_pred != "— Ver resumen general de todos los productos —":
        # ── MODO PRODUCTO ESPECÍFICO ──────────────────────────────────────────
        real_nm = sel_sku_pred[2:].strip()
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

        st.stop()

    # ── Impacto de negocio estimado ───────────────────────────────────────────
    section("💰 Impacto potencial de negocio — si implementas las recomendaciones")
    st.caption("Estimación del modelo con los datos actuales. Con más historial, la precisión aumenta.")

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

        _ing_sub  = _delta_rev(_sub_pool, "+10%")
        _ing_prm  = _delta_rev(_prm_pool, "-10%")
        _uds_prm  = _delta_uds(_prm_pool, "-10%")
        _total    = _ing_sub + _ing_prm

        _ic1, _ic2, _ic3, _ic4 = st.columns(4)
        kpi(_ic1, f"Subir precio — {len(_sub_pool)} SKUs (+10%)",
            f"+${_ing_sub:,.0f}/mes", OM_BLUE)
        kpi(_ic2, f"Promover — {len(_prm_pool)} SKUs (-10%)",
            f"+${_ing_prm:,.0f}/mes" if _ing_prm >= 0 else f"${_ing_prm:,.0f}/mes", OM_GREEN)
        kpi(_ic3, "Unidades extra por promos", f"+{_uds_prm:,.0f}/mes", OM_AMBER)
        kpi(_ic4, "Impacto total anualizado",  f"+${_total*12:,.0f}/año", OM_RED)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Calendario mensual de acciones ────────────────────────────────────────
    section("📅 ¿En qué meses actuar? — Calendario de acciones")
    st.caption(
        "Basado en patrones estacionales de demanda + elasticidad de cada producto. "
        "**SUBIR** = mes de alta demanda en producto inelástico. "
        "**PROMOVER** = mes ideal para descuentos o activaciones. "
        "**MANTENER** = sin cambios recomendados.")

    if len(df_cal) > 0:
        # Filtro de departamento para el calendario
        cal_depts = sorted(df_m1a["dept_nm"].dropna().unique().tolist())
        cal_dept_sel = st.selectbox("Filtrar por departamento (calendario)", ["Todos"] + cal_depts, key="cal_dept")

        # Aggregate: heatmap general (cuántos SKUs por mes x acción)
        skus_dept = []  # inicializar siempre para evitar NameError
        cal_agg = (df_cal.groupby(["mes_cal","mes_nombre","accion"])
                   .agg(n_skus=("prod_nbr","nunique")).reset_index())

        if cal_dept_sel != "Todos":
            skus_dept = df_m1a[df_m1a["dept_nm"]==cal_dept_sel]["prod_nbr"].tolist()
            cal_dept_df = df_cal[df_cal["prod_nbr"].isin(skus_dept)]
            cal_agg = (cal_dept_df.groupby(["mes_cal","mes_nombre","accion"])
                       .agg(n_skus=("prod_nbr","nunique")).reset_index())

        # Bar chart: mes x acción
        accion_colors = {"SUBIR":OM_BLUE,"PROMOVER":OM_GREEN,"MANTENER":OM_AMBER}
        fig = px.bar(
            cal_agg.sort_values("mes_cal"), x="mes_nombre", y="n_skus",
            color="accion", barmode="group",
            title="Número de SKUs por acción recomendada según el mes",
            labels={"mes_nombre":"Mes","n_skus":"Número de SKUs","accion":"Acción"},
            color_discrete_map=accion_colors,
            category_orders={"mes_nombre":MONTH_ORDER,
                             "accion":["SUBIR","PROMOVER","MANTENER"]},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Acción: %{fullData.name}<br>SKUs: %{y}<extra></extra>")
        st.plotly_chart(_layout(fig, h=340), use_container_width=True)

        # Heatmap por SKU (top 20 por volumen)
        if len(df_cal["prod_nbr"].unique()) > 1:
            st.caption("Heatmap por producto: qué hacer cada mes. Selecciona departamento arriba para filtrar.")
            top_skus_cal = (df_m1a[df_m1a["recomendacion"]!="No recomendable"]
                            .sort_values("n_meses", ascending=False)["prod_nbr"].head(20).tolist())
            if cal_dept_sel != "Todos":
                top_skus_cal = [s for s in top_skus_cal if s in skus_dept][:20]

            heat_data = df_cal[df_cal["prod_nbr"].isin(top_skus_cal)].copy()
            if len(heat_data) > 0:
                accion_num = {"SUBIR":2,"PROMOVER":1,"MANTENER":0}
                accion_txt = {"SUBIR":"S","PROMOVER":"P","MANTENER":"M"}
                pivot = (heat_data.pivot_table(index="prod_nm", columns="mes_cal",
                                               values="accion", aggfunc="first")
                         .reindex(columns=range(1,13)))
                pivot_num = pivot.map(lambda x: accion_num.get(x, -1) if pd.notna(x) else -1)

                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_num.values,
                    x=[MONTH_NAMES[i] for i in range(1,13)],
                    y=[nm[:35] for nm in pivot_num.index],
                    colorscale=[[0,"#F5F5F5"],[0.01,"#F5F5F5"],[0.34,OM_AMBER],[0.67,OM_GREEN],[1.0,OM_BLUE]],
                    zmin=-1, zmax=2, showscale=False,
                    text=[[accion_txt.get(pivot.iloc[i,j], "") if pd.notna(pivot.iloc[i,j]) else ""
                           for j in range(12)] for i in range(len(pivot))],
                    texttemplate="%{text}", textfont=dict(size=11, color="white"),
                    hovertemplate="<b>%{y}</b><br>Mes: %{x}<br>Acción: %{text}<extra></extra>",
                ))
                fig_heat.update_layout(
                    title="S = Subir precio · P = Promover · M = Mantener",
                    height=max(280, len(pivot)*28+60),
                    margin=dict(t=45,b=20,l=20,r=20),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(side="top"), yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.markdown(
                    '<span class="action-badge-subir">S = SUBIR PRECIO</span>&nbsp;&nbsp;'
                    '<span class="action-badge-promover">P = PROMOVER</span>&nbsp;&nbsp;'
                    '<span class="action-badge-mantener">M = MANTENER</span>',
                    unsafe_allow_html=True)
    else:
        st.info("No hay datos suficientes para el calendario. Asegúrate de que los SKUs tienen al menos 3 meses de datos.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Simulador de precios ──────────────────────────────────────────────────
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

    # ── ¿La promoción funcionó? ───────────────────────────────────────────────
    _promo_eval = ml_res.get("promo_eval") if ml_res else None
    if _promo_eval is not None and len(_promo_eval) > 0:
        section("🏷️ ¿Las promociones funcionaron?")
        st.caption("Comparación de ventas y unidades en meses con promoción detectada vs meses normales, por producto.")

        pe1, pe2, pe3 = st.columns(3)
        n_rent = (_promo_eval["rentable"]=="✅ Sí").sum()
        n_nort = (_promo_eval["rentable"]=="❌ No").sum()
        kpi(pe1, "Promos evaluadas",    f"{len(_promo_eval)} SKUs",         OM_BLUE)
        kpi(pe2, "Generaron más ventas",f"{n_rent} ({n_rent/len(_promo_eval)*100:.0f}%)", OM_GREEN)
        kpi(pe3, "No generaron más ventas", f"{n_nort}",                    OM_RED)

        st.markdown("<br>", unsafe_allow_html=True)

        show_eval = _promo_eval[["prod_nm","dept_nm","pct_desc","uplift_uds","uplift_rev","rentable","meses_promo"]].copy()
        show_eval.columns = ["Producto","Departamento","Descuento %","Uplift Unidades %","Uplift Ventas $%","¿Generó ventas?","Meses en promo"]

        def _color_row(row):
            color = "#E8F5E9" if row["¿Generó ventas?"]=="✅ Sí" else "#FFEBEE"
            return [f"background-color:{color}"]*len(row)

        st.dataframe(
            show_eval.style.apply(_color_row, axis=1)
                     .format({"Descuento %":"{:.1f}%","Uplift Unidades %":"{:+.1f}%","Uplift Ventas $%":"{:+.1f}%"}),
            use_container_width=True, height=300, hide_index=True)

        st.caption("Verde = la promo generó más ventas en $ (aunque sea poco). Rojo = las ventas bajaron o no cambiaron.")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Resumen ejecutivo con IA — sintetiza todo el analisis ────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("🤖 Resumen ejecutivo con IA")
    st.caption(
        "Claude genera un reporte integrado: pipeline ML, elasticidades OLS, "
        "impacto financiero y timing optimo. Presiona despues de revisar todo lo anterior.")

    _api_key2 = os.environ.get("ANTHROPIC_API_KEY", "")
    if not _api_key2:
        st.warning("Configura ANTHROPIC_API_KEY en Railway para habilitar esta funcion.")
    else:
        _bc2, _ic2 = st.columns([1, 3])
        with _bc2:
            _gen_ai2 = st.button("🤖 Generar resumen ejecutivo", type="primary", key="claude_bottom")
        with _ic2:
            st.caption("~10 segundos · integra ML + OLS + impacto + timing")

        if _gen_ai2:
            with st.spinner("Claude sintetizando el analisis completo..."):
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
                    st.error(f"Error al llamar a Claude: {_e2}")

        if st.session_state.get("ai_analysis"):
            st.markdown(
                '<div class="narrative-box" style="border-left-color:#7B1FA2;">' +
                '<div style="font-size:11px;color:#7B1FA2;font-weight:700;margin-bottom:8px;">' +
                'REPORTE EJECUTIVO — CLAUDE (Anthropic)</div></div>',
                unsafe_allow_html=True)
            st.markdown(st.session_state["ai_analysis"])
