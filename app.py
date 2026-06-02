# -*- coding: utf-8 -*-
"""Dynamic Pricing Analyzer v3 — Streamlit App | OfficeMax México"""

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
for _k in ["df_main", "clean_report", "results", "df_csv_bytes", "loaded_file_name"]:
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
    m2 = {"n_obs":int(mod2.nobs),"r2":round(float(mod2.rsquared),4),
          "r2_adj":round(float(mod2.rsquared_adj),4),
          "beta":round(float(mod2.params["log_precio"]),4),
          "beta_pval":round(float(mod2.pvalues["log_precio"]),4),
          "premium":round(float(mod2.params["es_premium"]),4),
          "prem_pval":round(float(mod2.pvalues["es_premium"]),4),
          "rmse":round(float(np.sqrt(np.mean((y2[msk]-mod2.fittedvalues)**2))),4),
          "f_stat":round(float(mod2.fvalue),2)}

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
        with st.spinner("Calculando modelos..."):
            try:
                csv_bytes = st.session_state["df_csv_bytes"]
                res = run_models(df_csv=csv_bytes, min_obs=min_obs, min_cv=min_cv,
                                 min_r2=min_r2, max_beta=max_beta, pval_thresh=pval_thresh,
                                 run_rolling=run_rolling)
                st.session_state["results"] = res
                st.success("✅ Análisis completado")
            except Exception as e:
                st.error(f"Error: {e}"); st.exception(e)

    results = st.session_state["results"]
    if results is None:
        st.info("Configura los parámetros y presiona **▶️ Ejecutar análisis**.")
    else:
        df_m1a = results["m1a"]; m2 = results["m2"]

        section("🔬 Validación Global del modelo — ¿los datos tienen señal de elasticidad?")
        st.caption("La Validación Global usa todos los SKUs con controles por tienda, mes y tipo de producto. "
                   "Si pasa esta prueba, tiene sentido analizar producto por producto.")
        css, msg, reasons = traffic_light(m2["r2"], m2["beta"], m2["beta_pval"], m2["rmse"])
        cv1, cv2 = st.columns([1,2])
        with cv1:
            st.markdown(f'<div class="{css}">{msg}</div><br>', unsafe_allow_html=True)
            for r in reasons: st.caption(r)
        with cv2:
            st.dataframe(pd.DataFrame({
                "Métrica":["N observaciones","R²","R² ajustado","Elasticidad precio global","p-valor",
                           "Coef. premium","RMSE (escala log)"],
                "Valor":[f'{m2["n_obs"]:,}',f'{m2["r2"]:.4f}',f'{m2["r2_adj"]:.4f}',
                         f'{m2["beta"]:.4f}',f'{m2["beta_pval"]:.4f}',
                         f'{m2["premium"]:.4f}',f'{m2["rmse"]:.4f}'],
                "Interpretación":[
                    "Transacciones usadas en el modelo global",
                    "% de varianza en ventas explicada por precio (0-1). En retail 0.15+ es aceptable.",
                    "R² penalizado por número de variables incluidas",
                    f'+1% en precio → {m2["beta"]:.2f}% en unidades vendidas',
                    "Significativo si < 0.10",
                    "Premium venden " + ("más" if m2["premium"]>0 else "menos") + " que no-premium",
                    "Error promedio en log-escala. < 0.5 es bueno.",
                ]}), hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        n_valid = len(df_m1a[df_m1a["recomendacion"]!="No recomendable"]) if len(df_m1a)>0 else 0
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

    kpis = {"venta":df["venta_con_iva"].sum(),"unidades":df["qty"].sum(),
            "n_skus":df["prod_nbr"].nunique(),"n_stores":df["store_nbr"].nunique(),
            "margen":df["margen"].mean()*100,
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
    ff1, ff2, ff3 = st.columns(3)
    raw_df = df_main  # reference only for filter options
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

    # Compute aggregations (cached — mismos bytes siempre → nunca re-corre)
    agg = compute_descriptivo(st.session_state["df_csv_bytes"],
                              tuple(dept_sel), tuple(year_sel), tuple(marca_sel))
    if agg is None:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()
    kpis = agg["kpis"]

    # KPIs
    section("📊 Indicadores clave")
    kc = st.columns(5)
    kpi(kc[0],"Venta total",      f"${kpis['venta']/1e6:.1f}M",   OM_RED)
    kpi(kc[1],"Unidades vendidas",f"{kpis['unidades']/1e3:.0f}K", OM_BLUE)
    kpi(kc[2],"SKUs únicos",      f"{kpis['n_skus']:,}",           OM_GREEN)
    kpi(kc[3],"Tiendas",          f"{kpis['n_stores']}",           OM_AMBER)
    kpi(kc[4],"Margen promedio",  f"{kpis['margen']:.1f}%",        "#7B1FA2")
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
    section("🏆 Top 15 productos")
    top_sku = agg["top_sku"]
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
    section("📊 Distribuciones de precio, margen y unidades")
    dc1,dc2,dc3 = st.columns(3)
    with dc1:
        fig = px.histogram(agg["dist_p"], x="precio_tx", nbins=50, title="Distribución de precios",
                           labels={"precio_tx":"Precio unitario ($)"}, color_discrete_sequence=[OM_RED])
        fig.update_layout(showlegend=False, bargap=0.05)
        st.plotly_chart(_layout(fig,h=280), use_container_width=True)
    with dc2:
        fig = px.histogram(agg["dist_m"], x="margen", nbins=40, title="Distribución de márgenes",
                           labels={"margen":"Margen (ratio)"}, color_discrete_sequence=[OM_GREEN])
        fig.update_layout(showlegend=False, bargap=0.05)
        st.plotly_chart(_layout(fig,h=280), use_container_width=True)
    with dc3:
        fig = px.histogram(agg["dist_q"], x="qty", nbins=40, title="Unidades por transacción",
                           labels={"qty":"Unidades"}, color_discrete_sequence=[OM_AMBER])
        fig.update_layout(showlegend=False, bargap=0.05)
        st.plotly_chart(_layout(fig,h=280), use_container_width=True)

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

    # ── MODO GENERAL ──────────────────────────────────────────────────────────
    section("📝 Resumen ejecutivo — ¿qué hacer?")
    narrative = generate_narrative(df_m1a, m2, df_cal)
    st.markdown(f'<div class="narrative-box">{narrative.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Conteo por recomendación ──────────────────────────────────────────────
    section("🎯 Resumen de recomendaciones")
    rc = st.columns(4)
    for i,(rec,color) in enumerate(REC_COLORS.items()):
        cnt = rec_counts.get(rec,0); pct = cnt/len(df_m1a)*100
        rc[i].markdown(
            f'<div class="rec-card" style="border-left:6px solid {color};">'
            f'<div style="font-size:38px;font-weight:900;color:{color};">{cnt}</div>'
            f'<div style="font-weight:700;font-size:12px;color:#333;">{rec.upper()}</div>'
            f'<div style="font-size:11px;color:#999;">{pct:.1f}% del total</div></div>',
            unsafe_allow_html=True)

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

    # ── Beta distribution ─────────────────────────────────────────────────────
    section("📊 Distribución de elasticidad — vista general")
    bc1,bc2 = st.columns([3,2])
    with bc1:
        fig=px.histogram(df_m1a,x="beta",nbins=40,color="recomendacion",
                         color_discrete_map=REC_COLORS,
                         title="Distribución de beta por SKU",
                         labels={"beta":"Beta (elasticidad precio)"},
                         barmode="overlay",opacity=0.75)
        fig.add_vline(x=-1,line_dash="dash",line_color="gray",opacity=0.7,
                      annotation_text="β=-1",annotation_position="top right")
        fig.add_vline(x=-1.5,line_dash="dot",line_color="gray",opacity=0.5,
                      annotation_text="β=-1.5",annotation_position="top left")
        fig.add_vline(x=0,line_color="black",line_width=0.5,opacity=0.4)
        st.plotly_chart(_layout(fig,h=340), use_container_width=True)
    with bc2:
        fig=px.pie(values=rec_counts.values,names=rec_counts.index,
                   title="SKUs por recomendación",
                   color=rec_counts.index,color_discrete_map=REC_COLORS)
        fig.update_traces(textposition="inside",textinfo="percent+label",textfont_size=11,
                          hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<extra></extra>")
        fig.update_layout(showlegend=False,height=340,paper_bgcolor="white",
                          margin=dict(t=45,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Rolling beta ──────────────────────────────────────────────────────────
    has_rolling = len(df_m1b)>0 or len(df_m1c)>0
    if has_rolling:
        section("📉 Tendencia de la elasticidad en el tiempo")
        roll_pool = set()
        if len(df_m1b)>0: roll_pool|=set(df_m1b["prod_nm"].unique())
        if len(df_m1c)>0: roll_pool|=set(df_m1c["prod_nm"].unique())
        roll_sel=st.selectbox("Producto",sorted(roll_pool),key="roll_sel")
        fig3=go.Figure()
        if len(df_m1b)>0:
            d3=df_m1b[df_m1b["prod_nm"]==roll_sel].sort_values("mes_fin_dt")
            if len(d3)>0:
                fig3.add_trace(go.Scatter(x=d3["mes_fin_dt"],y=d3["beta"],
                                          mode="lines+markers",name="Trimestral (3m)",
                                          line=dict(color=OM_BLUE,width=2.2),marker=dict(size=6)))
        if len(df_m1c)>0:
            d6=df_m1c[df_m1c["prod_nm"]==roll_sel].sort_values("mes_fin_dt")
            if len(d6)>0:
                fig3.add_trace(go.Scatter(x=d6["mes_fin_dt"],y=d6["beta"],
                                          mode="lines+markers",name="Semestral (6m)",
                                          line=dict(color=OM_RED,width=2.2),marker=dict(size=6,symbol="square")))
        b_glob=df_m1a.loc[df_m1a["prod_nm"]==roll_sel,"beta"]
        if len(b_glob)>0:
            fig3.add_hline(y=b_glob.iloc[0],line_dash="dot",line_color=OM_GREEN,
                           annotation_text=f"Beta global = {b_glob.iloc[0]:.2f}")
        fig3.add_hline(y=-1,line_dash="dash",line_color="gray",opacity=0.5,annotation_text="β=-1")
        fig3.add_hline(y=-1.5,line_dash="dot",line_color="gray",opacity=0.35,annotation_text="β=-1.5")
        fig3.update_layout(title=f"Elasticidad rolling — {roll_sel[:50]}",
                           plot_bgcolor="white",paper_bgcolor="white",
                           height=360,margin=dict(t=55,b=20,l=20,r=20),
                           xaxis_title="Mes",yaxis_title="Beta",
                           legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Candidate tables ──────────────────────────────────────────────────────
    section("📋 Lista de candidatos por acción")
    ct1,ct2 = st.columns(2)
    with ct1:
        st.markdown("**🔵 Subir precio** — beta entre 0 y -1 (inelásticos)")
        sub_up=df_m1a[df_m1a["recomendacion"]=="Subir precio"].sort_values("beta",ascending=False)
        if len(sub_up)>0:
            st.dataframe(sub_up[["prod_nm","dept_nm","beta","r2","pval","n_meses"]]
                         .rename(columns={"prod_nm":"Producto","dept_nm":"Depto",
                                           "beta":"Beta","r2":"R²","pval":"p-valor","n_meses":"Meses"})
                         .head(12),hide_index=True,use_container_width=True,height=320)
        else: st.info("Sin candidatos con los parámetros actuales.")
    with ct2:
        st.markdown("**🟢 Bajar / Promover** — beta < -1.5 (elásticos)")
        sub_dn=df_m1a[df_m1a["recomendacion"]=="Bajar / Promover"].sort_values("beta")
        if len(sub_dn)>0:
            st.dataframe(sub_dn[["prod_nm","dept_nm","beta","r2","pval","n_meses"]]
                         .rename(columns={"prod_nm":"Producto","dept_nm":"Depto",
                                           "beta":"Beta","r2":"R²","pval":"p-valor","n_meses":"Meses"})
                         .head(12),hide_index=True,use_container_width=True,height=320)
        else: st.info("Sin candidatos con los parámetros actuales.")

    # ── KPIs del modelo ───────────────────────────────────────────────────────
    section("📊 Métricas del análisis de Elasticidad Total")
    sk1,sk2,sk3 = st.columns(3)
    kpi(sk1,"Beta promedio",      f"{df_m1a['beta'].mean():.3f}", OM_BLUE)
    kpi(sk2,"R² promedio",        f"{df_m1a['r2'].mean():.3f}",  OM_GREEN)
    kpi(sk3,"RMSE promedio (log)",f"{df_m1a['rmse'].mean():.3f}" if 'rmse' in df_m1a.columns else "N/A", OM_AMBER)
