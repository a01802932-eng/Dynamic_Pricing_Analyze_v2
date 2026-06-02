# -*- coding: utf-8 -*-
"""Dynamic Pricing Analyzer — Streamlit App | OfficeMax México"""

import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# ── Colores OfficeMax ─────────────────────────────────────────────────────────
OM_RED    = "#E31837"
OM_RED_DK = "#C41430"
OM_BLACK  = "#1A1A1A"
OM_WHITE  = "#FFFFFF"
OM_GRAY   = "#F5F5F5"
OM_GRAY2  = "#E0E0E0"
OM_TEXT2  = "#666666"
OM_YELLOW = "#FFD700"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Analyzer | OfficeMax",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo base64 ───────────────────────────────────────────────────────────────
def _logo_b64() -> str:
    logo_path = Path(__file__).parent / "static" / "logo-officemax.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""

LOGO_B64 = _logo_b64()

# ── CSS OfficeMax ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Roboto', Arial, sans-serif !important;
}
.stApp {
    background-color: #F5F5F5 !important;
}
.main .block-container {
    max-width: 1200px !important;
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Sidebar — navbar oscuro ── */
[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #1A1A1A !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #FFFFFF !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #CCCCCC !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #FFFFFF !important;
    padding: 8px 12px !important;
    border-radius: 4px !important;
    transition: all 0.15s !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #E31837 !important;
    background: rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] [aria-checked="true"] + div {
    color: #E31837 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #333333 !important;
    margin: 12px 0 !important;
}

/* ── Títulos ── */
h1 {
    font-weight: 900 !important;
    font-size: 26px !important;
    color: #1A1A1A !important;
    border-bottom: 3px solid #E31837 !important;
    padding-bottom: 10px !important;
    margin-bottom: 24px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
h2 {
    font-weight: 700 !important;
    font-size: 20px !important;
    color: #1A1A1A !important;
    margin-top: 28px !important;
}
h3 {
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #1A1A1A !important;
}

/* ── Botón primario ── */
.stButton > button[kind="primary"],
button[kind="primary"] {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-family: 'Roboto', Arial, sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-size: 13px !important;
    transition: background 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #C41430 !important;
    border: none !important;
}

/* ── Botón secundario / download ── */
.stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    color: #E31837 !important;
    border: 2px solid #E31837 !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.stButton > button:not([kind="primary"]):hover {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
}
.stDownloadButton > button {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 13px !important;
}
.stDownloadButton > button:hover {
    background-color: #C41430 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    padding: 16px 20px !important;
    border-left: 4px solid #E31837 !important;
}
[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-weight: 700 !important;
    font-size: 22px !important;
}
[data-testid="stMetricLabel"] {
    color: #666666 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: #1A1A1A !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 4px 4px 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #CCCCCC !important;
    background: transparent !important;
    border-radius: 4px 4px 0 0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #E31837 !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF !important;
    border-radius: 0 0 6px 6px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1A1A1A !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #CCCCCC !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #E31837 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    border-color: #CCCCCC !important;
    border-radius: 4px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #E31837 !important;
    box-shadow: 0 0 0 2px rgba(227,24,55,0.15) !important;
}

/* ── Multiselect tags ── */
span[data-baseweb="tag"] {
    background: #E31837 !important;
    color: #FFFFFF !important;
    border-radius: 3px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #E31837 !important;
    border-color: #E31837 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    overflow: hidden !important;
}

/* ── Alertas ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 4px !important;
}

/* ── Radio ── */
.stRadio > label {
    font-weight: 700 !important;
    color: #1A1A1A !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Separadores ── */
hr {
    border-color: #E0E0E0 !important;
    margin: 24px 0 !important;
}

/* ── Info / success / warning / error banners ── */
div[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header OfficeMax ──────────────────────────────────────────────────────────
def render_header():
    logo_html = (
        f'<img src="data:image/png;base64,{LOGO_B64}" '
        f'style="height:38px;width:auto;object-fit:contain;" alt="OfficeMax México">'
        if LOGO_B64
        else '<span style="font-size:22px;font-weight:900;color:#E31837;letter-spacing:-1px;">OfficeMax</span>'
    )
    st.markdown(f"""
    <div style="
        background:#FFFFFF;
        border-bottom:3px solid #E31837;
        padding:12px 28px;
        display:flex;
        align-items:center;
        margin-bottom:28px;
        box-shadow:0 2px 6px rgba(0,0,0,0.08);
    ">
        {logo_html}
        <div style="margin-left:auto;display:flex;align-items:center;gap:12px;">
            <div style="
                background:#E31837;
                color:#FFFFFF;
                padding:7px 18px;
                border-radius:4px;
                font-size:11px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:1.5px;
                font-family:'Roboto',Arial,sans-serif;
            ">Dynamic Pricing Analyzer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Banner de paso ────────────────────────────────────────────────────────────
def step_banner(numero: int, titulo: str, descripcion: str = ""):
    st.markdown(f"""
    <div style="
        background:#1A1A1A;
        color:#FFFFFF;
        padding:14px 22px;
        border-radius:6px;
        margin-bottom:24px;
        display:flex;
        align-items:center;
        gap:16px;
    ">
        <div style="
            background:#E31837;
            color:#FFFFFF;
            font-weight:900;
            font-size:18px;
            min-width:40px;
            height:40px;
            border-radius:4px;
            display:flex;
            align-items:center;
            justify-content:center;
        ">{numero}</div>
        <div>
            <div style="font-weight:700;font-size:16px;text-transform:uppercase;letter-spacing:0.5px;">
                {titulo}
            </div>
            {f'<div style="font-size:13px;color:#CCCCCC;margin-top:2px;">{descripcion}</div>' if descripcion else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tarjeta de métrica custom ─────────────────────────────────────────────────
def metric_card(label: str, value: str, color: str = OM_RED):
    st.markdown(f"""
    <div style="
        background:#FFFFFF;
        border-radius:6px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
        padding:18px 20px;
        border-left:4px solid {color};
        margin-bottom:4px;
    ">
        <div style="font-size:11px;color:#666666;text-transform:uppercase;letter-spacing:0.5px;font-weight:700;margin-bottom:6px;">
            {label}
        </div>
        <div style="font-size:22px;font-weight:700;color:#1A1A1A;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "step": 1,
    "df_raw": None,
    "col_map": {},
    "df_clean": None,
    "quality": None,
    "df_base": None,
    "df_elasticity": None,
    "df_sim": None,
    "df_rec": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
STEPS = [
    "1 · Subir y Validar",
    "2 · Cálculos Base",
    "3 · Elasticidad",
    "4 · Simulación",
    "5 · Recomendaciones",
    "6 · Gráficas",
    "7 · Exportar",
]
STEP_DONE = {
    1: st.session_state.df_clean is not None,
    2: st.session_state.df_base is not None,
    3: st.session_state.df_elasticity is not None,
    4: st.session_state.df_sim is not None,
    5: st.session_state.df_rec is not None,
    6: True,
    7: True,
}

# Logo en sidebar
if LOGO_B64:
    st.sidebar.markdown(
        f'<div style="padding:16px 8px 4px;">'
        f'<img src="data:image/png;base64,{LOGO_B64}" style="width:100%;max-width:160px;opacity:0.9;">'
        f'</div>',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

# Progreso
progress_html = ""
for i, label in enumerate(STEPS, 1):
    done = STEP_DONE.get(i, False)
    icon = "✅" if done else "○"
    active = "font-weight:700;color:#E31837;" if st.session_state.step == i else "color:#CCCCCC;"
    progress_html += (
        f'<div style="padding:4px 0;font-size:13px;{active}">'
        f'{icon} {label}</div>'
    )
st.sidebar.markdown(
    f'<div style="padding:4px 8px;">{progress_html}</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

current = st.sidebar.radio(
    "Navegación", STEPS,
    index=st.session_state.step - 1,
    label_visibility="collapsed",
)
st.session_state.step = STEPS.index(current) + 1


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Subir y Validar
# ─────────────────────────────────────────────────────────────────────────────
def paso1():
    render_header()
    step_banner(1, "Subir y Validar Datos",
                "Carga tu archivo CSV o Excel y mapea las columnas al modelo")

    with st.expander("📄 ¿Cómo debe verse mi archivo? — Ver plantilla"):
        tmpl = pd.DataFrame({
            "sku":          ["SKU001", "SKU001", "SKU002", "SKU002"],
            "unidades":     [100, 120, 200, 180],
            "venta_neta":   [1500.0, 1680.0, 3000.0, 2880.0],
            "fecha":        ["2024-01", "2024-02", "2024-01", "2024-02"],
            "precio":       [15.0, 14.0, 15.0, 16.0],
            "costo":        [8.0, 8.0, 9.0, 9.0],
            "departamento": ["PAPELERIA", "PAPELERIA", "LIBRERIA", "LIBRERIA"],
            "tienda":       ["T001", "T001", "T002", "T002"],
        })
        st.dataframe(tmpl, use_container_width=True)
        st.download_button(
            "📥 Descargar plantilla_input.csv",
            tmpl.to_csv(index=False).encode("utf-8"),
            "plantilla_input.csv", "text/csv",
        )

    uploaded = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Sube un archivo CSV o Excel para comenzar.")
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded, encoding="latin-1", low_memory=False)
        else:
            df = pd.read_excel(uploaded)
        st.session_state.df_raw = df
        st.success(f"Archivo cargado: **{df.shape[0]:,} filas × {df.shape[1]} columnas**")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return

    df   = st.session_state.df_raw
    cols = ["(ninguna)"] + list(df.columns)

    st.subheader("Mapeo de columnas")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Obligatorias**")
        m_sku = st.selectbox("SKU / Producto",   cols, key="m_sku")
        m_uni = st.selectbox("Unidades",          cols, key="m_uni")
        m_ven = st.selectbox("Venta neta",         cols, key="m_ven")
        m_fec = st.selectbox("Fecha",              cols, key="m_fec")
    with c2:
        st.markdown("**Opcionales — precio / costo**")
        m_pre = st.selectbox("Precio unitario",    cols, key="m_pre")
        m_cos = st.selectbox("Costo unitario",     cols, key="m_cos")
    with c3:
        st.markdown("**Opcionales — segmentación**")
        m_dep = st.selectbox("Departamento",       cols, key="m_dep")
        m_tie = st.selectbox("Tienda",              cols, key="m_tie")
        m_ela = st.selectbox("Elasticidad pre-calculada", cols, key="m_ela")

    sin_mapear = [k for k, v in
                  {"sku": m_sku, "unidades": m_uni, "venta_neta": m_ven, "fecha": m_fec}.items()
                  if v == "(ninguna)"]
    if sin_mapear:
        st.warning(f"Columnas obligatorias sin mapear: {', '.join(sin_mapear)}")
        return

    col_map = {
        "sku": m_sku, "unidades": m_uni, "venta_neta": m_ven, "fecha": m_fec,
        "precio":       m_pre if m_pre != "(ninguna)" else None,
        "costo":        m_cos if m_cos != "(ninguna)" else None,
        "departamento": m_dep if m_dep != "(ninguna)" else None,
        "tienda":       m_tie if m_tie != "(ninguna)" else None,
        "elasticidad":  m_ela if m_ela != "(ninguna)" else None,
    }

    if not st.button("✅ Validar datos", type="primary"):
        return

    rename = {v: k for k, v in col_map.items() if v is not None}
    df_c   = df.rename(columns=rename).copy()

    df_c["unidades"]   = pd.to_numeric(df_c["unidades"],   errors="coerce")
    df_c["venta_neta"] = pd.to_numeric(df_c["venta_neta"], errors="coerce")
    df_c["sku"]        = df_c["sku"].astype(str).str.strip()
    if "precio"      in df_c.columns: df_c["precio"]      = pd.to_numeric(df_c["precio"],      errors="coerce")
    if "costo"       in df_c.columns: df_c["costo"]       = pd.to_numeric(df_c["costo"],       errors="coerce")
    if "elasticidad" in df_c.columns: df_c["elasticidad"] = pd.to_numeric(df_c["elasticidad"], errors="coerce")

    if "precio" not in df_c.columns or df_c["precio"].isna().all():
        df_c["precio"] = df_c["venta_neta"] / df_c["unidades"].replace(0, np.nan)

    errores, avisos = [], []
    if int((df_c["unidades"] <= 0).sum())  > 0: errores.append(f"{int((df_c['unidades']<=0).sum()):,} filas con unidades ≤ 0")
    if int((df_c["venta_neta"] <= 0).sum()) > 0: errores.append(f"{int((df_c['venta_neta']<=0).sum()):,} filas con venta_neta ≤ 0")
    if int(df_c[["sku","unidades","venta_neta"]].isna().any(axis=1).sum()) > 0:
        errores.append(f"{int(df_c[['sku','unidades','venta_neta']].isna().any(axis=1).sum()):,} filas con campos nulos")
    if int(df_c.duplicated().sum()) > 0:
        avisos.append(f"{int(df_c.duplicated().sum()):,} filas duplicadas detectadas")

    if "costo" in df_c.columns:
        if int((df_c["costo"] < 0).sum()) > 0:
            avisos.append(f"{int((df_c['costo']<0).sum()):,} filas con costo negativo")
        n_cgt = int(((df_c["costo"] > df_c["precio"]) & df_c["costo"].notna() & df_c["precio"].notna()).sum())
        if n_cgt > 0: avisos.append(f"{n_cgt:,} filas donde costo > precio")

    df_tmp = df_c[(df_c["unidades"] > 0) & (df_c["precio"] > 0)].copy()
    if len(df_tmp) > 0:
        cv = (df_tmp.groupby("sku")["precio"].std() /
              df_tmp.groupby("sku")["precio"].mean().replace(0, np.nan)).mean()
        if pd.notna(cv) and cv < 0.02:
            avisos.append(f"Varianza de precio muy baja (CV = {cv:.1%}) — elasticidad poco confiable")

    if errores:
        st.markdown(f"""
        <div style="background:#FFF0F2;border-left:4px solid #E31837;padding:16px 20px;
                    border-radius:4px;margin:12px 0;">
            <div style="font-weight:700;color:#E31837;margin-bottom:8px;">
                🔴 DATOS INSUFICIENTES — Requiere corrección
            </div>
            {''.join(f"<div style='color:#1A1A1A;font-size:13px;'>• {e}</div>" for e in errores)}
        </div>
        """, unsafe_allow_html=True)
        return
    elif avisos:
        st.markdown(f"""
        <div style="background:#FFFBEA;border-left:4px solid #FFD700;padding:16px 20px;
                    border-radius:4px;margin:12px 0;">
            <div style="font-weight:700;color:#1A1A1A;margin-bottom:8px;">
                🟡 DATOS PARCIALES — Puedes continuar con advertencias
            </div>
            {''.join(f"<div style='color:#1A1A1A;font-size:13px;'>• {w}</div>" for w in avisos)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#F0FFF4;border-left:4px solid #2E7D32;padding:14px 20px;
                    border-radius:4px;margin:12px 0;font-weight:700;color:#2E7D32;">
            🟢 DATOS LISTOS — Todo correcto para el análisis completo
        </div>
        """, unsafe_allow_html=True)

    df_c = df_c[(df_c["unidades"] > 0) & (df_c["venta_neta"] > 0) & (df_c["precio"] > 0)].copy()
    st.session_state.df_clean = df_c
    st.session_state.col_map  = col_map
    st.session_state.quality  = "yellow" if avisos else "green"
    for key in ("df_base", "df_elasticity", "df_sim", "df_rec"):
        st.session_state[key] = None

    st.subheader("Estadísticas del archivo")
    cols5 = st.columns(5)
    datos = [
        ("Filas válidas",   f"{len(df_c):,}"),
        ("SKUs únicos",      f"{df_c['sku'].nunique():,}"),
        ("Precio mínimo",   f"${df_c['precio'].min():,.2f}"),
        ("Precio máximo",   f"${df_c['precio'].max():,.2f}"),
        ("Departamentos",   f"{df_c['departamento'].nunique():,}" if "departamento" in df_c.columns else "—"),
    ]
    for col, (label, val) in zip(cols5, datos):
        with col:
            metric_card(label, val)

    with st.expander("Ver muestra de datos"):
        st.dataframe(df_c.head(30), use_container_width=True)

    st.markdown("""
    <div style="background:#E31837;color:#FFFFFF;padding:12px 20px;border-radius:4px;
                font-weight:700;font-size:13px;text-align:center;margin-top:16px;">
        ✅ DATOS VALIDADOS — Continúa al Paso 2 en el menú lateral
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Cálculos Base
# ─────────────────────────────────────────────────────────────────────────────
def paso2():
    render_header()
    step_banner(2, "Cálculos Base por SKU",
                "Precio, ingreso base y margen calculados a partir de los datos de venta")

    if st.session_state.df_clean is None:
        st.warning("Primero completa el **Paso 1**.")
        return

    df      = st.session_state.df_clean.copy()
    group_cols = ["departamento", "sku"] if "departamento" in df.columns else ["sku"]

    agg = (df.groupby(group_cols)
           .agg(unidades_tot=("unidades","sum"), venta_neta_tot=("venta_neta","sum"),
                precio_base=("precio","mean"), n_obs=("precio","count"))
           .reset_index())
    agg["ingreso_base"] = agg["precio_base"] * agg["unidades_tot"]

    if "costo" in df.columns and not df["costo"].isna().all():
        cp = df.groupby("sku")["costo"].mean().reset_index().rename(columns={"costo":"costo_unitario"})
        agg = agg.merge(cp, on="sku", how="left")
        agg["margen_unitario"] = agg["precio_base"] - agg["costo_unitario"]
        agg["margen_total"]    = agg["margen_unitario"] * agg["unidades_tot"]
    else:
        agg[["costo_unitario","margen_unitario","margen_total"]] = np.nan
        st.info("Sin columna de costo — solo se simulará ingreso, no margen.")

    st.session_state.df_base = agg

    cols4 = st.columns(4)
    kpis = [
        ("SKUs",            f"{agg['sku'].nunique():,}"),
        ("Ingreso total",   f"${agg['ingreso_base'].sum():,.0f}"),
        ("Margen total",    f"${agg['margen_total'].sum():,.0f}" if not agg["margen_total"].isna().all() else "—"),
        ("Precio promedio", f"${agg['precio_base'].mean():,.2f}"),
    ]
    for col, (label, val) in zip(cols4, kpis):
        with col:
            metric_card(label, val)

    st.markdown("<br>", unsafe_allow_html=True)
    display = [c for c in ["departamento","sku","n_obs","unidades_tot","precio_base",
                            "costo_unitario","ingreso_base","margen_unitario","margen_total"]
               if c in agg.columns]
    fmt = {"precio_base":"${:.2f}","costo_unitario":"${:.2f}","margen_unitario":"${:.2f}",
           "ingreso_base":"${:,.0f}","margen_total":"${:,.0f}","unidades_tot":"{:,.0f}"}
    st.dataframe(
        agg[display].sort_values("ingreso_base", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in display}),
        use_container_width=True,
    )
    st.markdown("""
    <div style="background:#E31837;color:#FFFFFF;padding:12px 20px;border-radius:4px;
                font-weight:700;font-size:13px;text-align:center;margin-top:16px;">
        ✅ LISTO — Continúa al Paso 3
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Elasticidad
# ─────────────────────────────────────────────────────────────────────────────
def paso3():
    render_header()
    step_banner(3, "Estimación de Elasticidad",
                "Regresión log-log OLS: log(unidades+1) = α + β·log(precio)")

    if st.session_state.df_clean is None:
        st.warning("Primero completa el **Paso 1**.")
        return
    if st.session_state.df_base is None:
        st.warning("Primero completa el **Paso 2**.")
        return

    df      = st.session_state.df_clean.copy()
    col_map = st.session_state.col_map

    fuente = st.radio(
        "Fuente de elasticidad",
        ["Estimar con regresión log-log OLS", "Usar columna pre-cargada del archivo"],
        horizontal=True,
    )
    c1, c2 = st.columns(2)
    MIN_OBS  = c1.slider("Mínimo de observaciones por SKU", 2, 12, 3)
    MAX_BETA = c2.slider("Límite |β| máximo aceptable",     3, 20, 10)

    if not st.button("▶ Calcular elasticidades", type="primary"):
        return

    if "Usar columna" in fuente:
        if not col_map.get("elasticidad") or "elasticidad" not in df.columns:
            st.error("No hay columna de elasticidad mapeada en el Paso 1.")
            return
        betas = (df.groupby("sku")["elasticidad"].mean().reset_index()
                 .rename(columns={"elasticidad":"beta"}))
        betas["r2"] = np.nan; betas["pval"] = np.nan
        betas["n_obs"] = df.groupby("sku")["elasticidad"].count().values
    else:
        df["periodo"] = (pd.to_datetime(df["fecha"], errors="coerce").dt.to_period("M").astype(str)
                         if "fecha" in df.columns else "ALL")
        mensual = (df.groupby(["sku","periodo"])
                   .agg(unidades=("unidades","sum"), venta_neta=("venta_neta","sum"))
                   .reset_index())
        mensual["precio"] = mensual["venta_neta"] / mensual["unidades"].replace(0, np.nan)
        mensual = mensual[mensual["precio"] > 0].copy()
        mensual["log_u"] = np.log1p(mensual["unidades"])
        mensual["log_p"] = np.log(mensual["precio"])

        def ols_beta(grp):
            grp = grp.dropna(subset=["log_u","log_p"])
            if len(grp) < MIN_OBS or grp["log_p"].std() < 1e-6: return None
            try:
                res = OLS(grp["log_u"].values, add_constant(grp["log_p"].values)).fit()
                b   = res.params[1]
                if abs(b) > MAX_BETA: return None
                return pd.Series({"beta":round(float(b),4),"r2":round(float(res.rsquared),4),
                                  "pval":round(float(res.pvalues[1]),4),"n_obs":int(len(grp))})
            except: return None

        with st.spinner("Estimando elasticidades..."):
            betas = mensual.groupby("sku").apply(ols_beta).dropna().reset_index()

    if len(betas) == 0:
        st.error("No se pudo calcular elasticidad para ningún SKU. Revisa variación de precio y observaciones.")
        return

    def clasificar(b):
        if pd.isna(b): return "Sin datos suficientes"
        if b < -1:     return "Elástico"
        if b < 0:      return "Inelástico"
        return "Sospechoso"

    betas["clasificacion"] = betas["beta"].apply(clasificar)
    st.session_state.df_elasticity = betas
    st.session_state.df_sim = None
    st.session_state.df_rec = None

    counts = betas["clasificacion"].value_counts()
    kpis = [
        ("Elásticos (β < −1)",      counts.get("Elástico",0),          OM_RED),
        ("Inelásticos (−1 < β < 0)", counts.get("Inelástico",0),        "#2E7D32"),
        ("Sospechosos (β ≥ 0)",     counts.get("Sospechoso",0),         OM_YELLOW),
        ("Sin datos suficientes",    counts.get("Sin datos suficientes",0), OM_TEXT2),
    ]
    cols4 = st.columns(4)
    for col, (label, val, color) in zip(cols4, kpis):
        with col:
            metric_card(label, str(val), color)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.histogram(
        betas.dropna(subset=["beta"]), x="beta", nbins=30,
        color="clasificacion",
        color_discrete_map={"Elástico":OM_RED,"Inelástico":"#2E7D32",
                            "Sospechoso":OM_YELLOW,"Sin datos suficientes":OM_TEXT2},
        title="Distribución de Elasticidad por SKU",
        labels={"beta":"Elasticidad (β)","count":"N de SKUs"},
    )
    fig.add_vline(x=-1, line_dash="dash", line_color=OM_RED,   annotation_text="β = −1")
    fig.add_vline(x=0,  line_dash="dash", line_color=OM_BLACK, annotation_text="β = 0")
    fig.update_layout(
        plot_bgcolor=OM_WHITE, paper_bgcolor=OM_WHITE,
        font_family="Roboto, Arial", font_color=OM_BLACK,
        title_font_size=15, title_font_color=OM_BLACK,
    )
    st.plotly_chart(fig, use_container_width=True)

    fmt = {"beta":"{:.4f}","r2":"{:.3f}","pval":"{:.4f}","n_obs":"{:.0f}"}
    st.dataframe(
        betas.sort_values("beta")
        .style.format({k: v for k, v in fmt.items() if k in betas.columns}),
        use_container_width=True,
    )
    st.markdown("""
    <div style="background:#E31837;color:#FFFFFF;padding:12px 20px;border-radius:4px;
                font-weight:700;font-size:13px;text-align:center;margin-top:16px;">
        ✅ LISTO — Continúa al Paso 4
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Simulación
# ─────────────────────────────────────────────────────────────────────────────
def paso4():
    render_header()
    step_banner(4, "Simulación de Escenarios de Precio",
                "Estándar (-10% a +10%) y promociones complejas (3x2, 2x1, 2do al 50%)")

    if st.session_state.df_base is None or st.session_state.df_elasticity is None:
        st.warning("Primero completa los pasos anteriores.")
        return

    df_base = st.session_state.df_base.copy()
    df_ela  = st.session_state.df_elasticity.copy()
    merged  = df_base.merge(df_ela[["sku","beta","clasificacion"]], on="sku", how="left")
    merged  = merged[merged["beta"].notna()].copy()

    if len(merged) == 0:
        st.error("No hay SKUs con elasticidad calculada.")
        return

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Escenarios estándar**")
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E0E0E0;border-radius:6px;
                    padding:14px 18px;font-size:13px;color:#1A1A1A;">
            −10% &nbsp;·&nbsp; −5% &nbsp;·&nbsp; Base &nbsp;·&nbsp; +5% &nbsp;·&nbsp; +10%
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("**Promociones complejas**")
        promos_sel = st.multiselect(
            "Selecciona promociones adicionales",
            ["3x2 (lleva 3, paga 2)", "2x1 (lleva 2, paga 1)", "2do al 50%"],
            default=[], label_visibility="collapsed",
        )

    CAMBIOS_STD   = [-0.10, -0.05, 0.00, 0.05, 0.10]
    ETIQUETAS_STD = ["-10%", "-5%", "Base", "+5%", "+10%"]
    PROMO_MULT    = {"3x2 (lleva 3, paga 2)": 2/3, "2x1 (lleva 2, paga 1)": 0.50, "2do al 50%": 0.75}

    CAMBIOS   = CAMBIOS_STD   + [PROMO_MULT[p] - 1 for p in promos_sel]
    ETIQUETAS = ETIQUETAS_STD + [p.split(" ")[0]   for p in promos_sel]

    if not st.button("▶ Simular escenarios", type="primary"):
        return

    rows = []
    for _, row in merged.iterrows():
        p0, u0, beta = row["precio_base"], row["unidades_tot"], row["beta"]
        c0   = row.get("costo_unitario", np.nan)
        rev0 = p0 * u0
        marg0 = (p0 - c0) * u0 if pd.notna(c0) else np.nan
        for cambio, etiq in zip(CAMBIOS, ETIQUETAS):
            p1   = p0 * (1 + cambio)
            u1   = u0 * ((1 + cambio) ** beta)
            ing1 = p1 * u1
            marg1 = (p1 - c0) * u1 if pd.notna(c0) else np.nan
            d_ing  = (ing1 - rev0) / rev0 * 100 if rev0 != 0 else 0
            d_marg = (marg1 - marg0) / abs(marg0) * 100 if pd.notna(marg0) and marg0 != 0 else np.nan
            entry = {
                "sku": row["sku"], "clasificacion": row["clasificacion"],
                "beta": round(float(beta),4), "precio_base": round(float(p0),2),
                "unidades_base": round(float(u0),1), "escenario": etiq,
                "precio_nuevo": round(float(p1),2), "unidades_simuladas": round(float(u1),1),
                "ingreso_simulado": round(float(ing1),2),
                "delta_ingreso_pct": round(float(d_ing),1),
                "margen_simulado": round(float(marg1),2) if pd.notna(marg1) else np.nan,
                "delta_margen_pct": round(float(d_marg),1) if pd.notna(d_marg) else np.nan,
            }
            if "departamento" in row.index:
                entry["departamento"] = row["departamento"]
            rows.append(entry)

    df_sim = pd.DataFrame(rows)
    st.session_state.df_sim = df_sim
    st.session_state.df_rec = None

    st.markdown(f"""
    <div style="background:#1A1A1A;color:#FFFFFF;padding:14px 20px;border-radius:6px;
                font-weight:700;font-size:14px;margin:16px 0;">
        ✅ Simulación completada:
        <span style="color:#E31837;">{merged['sku'].nunique()} SKUs</span>
        × {len(CAMBIOS)} escenarios
    </div>
    """, unsafe_allow_html=True)

    pivot = df_sim.pivot_table(index="sku", columns="escenario",
                               values="delta_ingreso_pct", aggfunc="first")
    orden = [e for e in ETIQUETAS if e in pivot.columns]
    pivot = pivot[orden]

    st.dataframe(
        pivot.style.format("{:+.1f}%")
                   .background_gradient(cmap="RdYlGn", axis=None, vmin=-20, vmax=20),
        use_container_width=True,
    )
    st.markdown("""
    <div style="background:#E31837;color:#FFFFFF;padding:12px 20px;border-radius:4px;
                font-weight:700;font-size:13px;text-align:center;margin-top:16px;">
        ✅ LISTO — Continúa al Paso 5
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Recomendaciones
# ─────────────────────────────────────────────────────────────────────────────
def paso5():
    render_header()
    step_banner(5, "Recomendaciones Automáticas por SKU",
                "Clasificación basada en elasticidad, significancia estadística y costo")

    if st.session_state.df_sim is None:
        st.warning("Primero completa el **Paso 4**.")
        return

    df_sim  = st.session_state.df_sim.copy()
    df_ela  = st.session_state.df_elasticity.copy()
    df_base = st.session_state.df_base.copy()

    c1, c2 = st.columns(2)
    MAX_PVAL = c1.slider("p-valor máximo para significancia", 0.05, 0.50, 0.10, 0.05)
    MIN_N    = c2.slider("Mínimo de observaciones",           2,    10,   3)

    std_esc = ["-10%","-5%","Base","+5%","+10%"]
    best = (df_sim[df_sim["escenario"].isin(std_esc)]
            .loc[lambda d: d.groupby("sku")["ingreso_simulado"].transform("max") == d["ingreso_simulado"]]
            [["sku","escenario","delta_ingreso_pct"]]
            .drop_duplicates("sku")
            .rename(columns={"escenario":"mejor_escenario","delta_ingreso_pct":"delta_mejor_pct"}))

    rec = df_ela.merge(best, on="sku", how="left")
    flag_skus = (set(df_base[df_base["costo_unitario"] > df_base["precio_base"]]["sku"])
                 if "costo_unitario" in df_base.columns else set())

    def recomendar(row):
        b, pval, n, sku = row.get("beta",np.nan), row.get("pval",np.nan), row.get("n_obs",np.nan), row["sku"]
        if sku in flag_skus:   return "Exploratoria",         "Costo > precio — revisar datos"
        if pd.isna(b):         return "No recomendar",        "Sin elasticidad calculada"
        if b > 0:              return "No recomendar",        f"Beta positiva (β={b:.2f}) — anomalía"
        if not (pd.isna(pval) or pval < MAX_PVAL):
            return "No recomendar", f"Beta no significativa (p={pval:.2f})"
        if not (pd.isna(n) or n >= MIN_N):
            return "No recomendar", f"Solo {int(n)} obs. (mínimo {MIN_N})"
        if -1 < b < 0:         return "Subir precio",         f"Inelástica (β={b:.2f})"
        if -1.5 <= b <= -1:    return "Mantener precio",      f"Cerca unitaria (β={b:.2f})"
        return "Bajar precio / promover", f"Elástica (β={b:.2f}): bajar mejora ingreso"

    rec[["recomendacion","razon"]] = rec.apply(lambda r: pd.Series(recomendar(r)), axis=1)
    st.session_state.df_rec = rec

    CATS = ["Subir precio","Mantener precio","Bajar precio / promover","No recomendar","Exploratoria"]
    COLORS_CAT = {
        "Subir precio":            ("#2E7D32", "🟢"),
        "Mantener precio":          ("#1565C0", "🔵"),
        "Bajar precio / promover":  (OM_YELLOW,  "🟡"),
        "No recomendar":            (OM_TEXT2,   "⚪"),
        "Exploratoria":             ("#9C27B0",  "🟣"),
    }
    cols5 = st.columns(5)
    for col, cat in zip(cols5, CATS):
        n = int((rec["recomendacion"] == cat).sum())
        color, icon = COLORS_CAT[cat]
        with col:
            metric_card(f"{icon} {cat}", str(n), color)

    st.markdown("<br>", unsafe_allow_html=True)
    filtro  = st.selectbox("Filtrar por categoría", ["Todas"] + CATS)
    df_show = rec if filtro == "Todas" else rec[rec["recomendacion"] == filtro]
    show_cols = [c for c in ["sku","beta","pval","n_obs","clasificacion",
                              "recomendacion","razon","mejor_escenario","delta_mejor_pct"]
                 if c in df_show.columns]
    st.dataframe(
        df_show[show_cols].sort_values("recomendacion")
        .style.format({"beta":"{:.4f}","pval":"{:.4f}","delta_mejor_pct":"{:+.1f}%"},
                      na_rep="—"),
        use_container_width=True,
    )
    st.markdown("""
    <div style="background:#E31837;color:#FFFFFF;padding:12px 20px;border-radius:4px;
                font-weight:700;font-size:13px;text-align:center;margin-top:16px;">
        ✅ LISTO — Continúa al Paso 6
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Gráficas
# ─────────────────────────────────────────────────────────────────────────────
def paso6():
    render_header()
    step_banner(6, "Gráficas de Análisis",
                "Visualizaciones interactivas de elasticidad, escenarios y recomendaciones")

    if st.session_state.df_sim is None:
        st.warning("Primero completa el **Paso 4**.")
        return

    df_sim  = st.session_state.df_sim.copy()
    df_ela  = st.session_state.df_elasticity
    df_rec  = st.session_state.df_rec
    df_base = st.session_state.df_base

    PLOTLY_LAYOUT = dict(
        plot_bgcolor=OM_WHITE, paper_bgcolor=OM_WHITE,
        font_family="Roboto, Arial", font_color=OM_BLACK,
        title_font_size=15, title_font_color=OM_BLACK,
        legend=dict(bgcolor=OM_WHITE, bordercolor=OM_GRAY2, borderwidth=1),
    )

    COLOR_ESC = {"-10%":"#1B5E20","-5%":"#81C784","Base":"#9E9E9E","+5%":"#EF9A9A","+10%":OM_RED}
    COLOR_REC = {
        "Subir precio":"#1565C0","Mantener precio":"#F9A825",
        "Bajar precio / promover":"#2E7D32","No recomendar":OM_TEXT2,"Exploratoria":"#9C27B0",
    }

    grafica = st.selectbox("Selecciona gráfica", [
        "Ingreso y margen por escenario — SKU individual",
        "Distribución de elasticidades",
        "Recomendaciones: β vs volumen",
        "Heatmap Δ ingreso por SKU y escenario",
        "Curva continua de revenue (−15% a +15%)",
    ])

    skus = sorted(df_sim["sku"].unique())

    if grafica == "Ingreso y margen por escenario — SKU individual":
        sku_sel  = st.selectbox("SKU", skus)
        sub      = df_sim[df_sim["sku"] == sku_sel].copy()
        beta_val = sub["beta"].iloc[0]
        has_m    = sub["delta_margen_pct"].notna().any()
        ncols    = 2 if has_m else 1
        fig      = make_subplots(rows=1, cols=ncols,
                                 subplot_titles=["Δ Ingreso (%)"] + (["Δ Margen (%)"] if has_m else []))
        colors   = [COLOR_ESC.get(e, "#7B1FA2") for e in sub["escenario"]]
        fig.add_trace(go.Bar(x=sub["escenario"], y=sub["delta_ingreso_pct"],
                             marker_color=colors, name="Ingreso",
                             text=sub["delta_ingreso_pct"].apply(lambda v: f"{v:+.1f}%"),
                             textposition="outside"), row=1, col=1)
        if has_m:
            fig.add_trace(go.Bar(x=sub["escenario"], y=sub["delta_margen_pct"],
                                 marker_color=colors, name="Margen",
                                 text=sub["delta_margen_pct"].apply(
                                     lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
                                 textposition="outside"), row=1, col=2)
        for ci in range(1, ncols + 1):
            fig.add_hline(y=0, line_dash="dash", line_color=OM_BLACK, row=1, col=ci)
        fig.update_layout(title=f"{sku_sel} — Escenarios de precio  (β = {beta_val:.2f})",
                          showlegend=False, height=420, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sub[["escenario","precio_nuevo","unidades_simuladas",
                           "ingreso_simulado","delta_ingreso_pct"]]
                     .style.format({"precio_nuevo":"${:.2f}","unidades_simuladas":"{:,.1f}",
                                    "ingreso_simulado":"${:,.2f}","delta_ingreso_pct":"{:+.1f}%"}),
                     use_container_width=True)

    elif grafica == "Distribución de elasticidades":
        if df_ela is None:
            st.warning("Completa el Paso 3.")
            return
        fig = px.histogram(df_ela.dropna(subset=["beta"]), x="beta", nbins=30,
                           color="clasificacion",
                           color_discrete_map={"Elástico":OM_RED,"Inelástico":"#2E7D32",
                                               "Sospechoso":OM_YELLOW,"Sin datos suficientes":OM_TEXT2},
                           title="Distribución de Elasticidades por SKU",
                           labels={"beta":"Elasticidad (β)","count":"N de SKUs"})
        fig.add_vline(x=-1, line_dash="dash", line_color=OM_RED,   annotation_text="β = −1")
        fig.add_vline(x=0,  line_dash="dash", line_color=OM_BLACK, annotation_text="β = 0")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif grafica == "Recomendaciones: β vs volumen":
        if df_rec is None or df_base is None:
            st.warning("Completa los Pasos 2 y 5.")
            return
        df_plot = df_rec.merge(df_base[["sku","unidades_tot"]], on="sku", how="left")
        fig = px.scatter(df_plot, x="beta", y="unidades_tot", color="recomendacion",
                         hover_name="sku", color_discrete_map=COLOR_REC, size_max=20,
                         title="Elasticidad vs Volumen por SKU",
                         labels={"beta":"Elasticidad (β)","unidades_tot":"Unidades totales"})
        fig.add_vline(x=-1, line_dash="dash", line_color=OM_TEXT2, annotation_text="β = −1")
        fig.add_vline(x=0,  line_dash="dash", line_color=OM_TEXT2, annotation_text="β = 0")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif grafica == "Heatmap Δ ingreso por SKU y escenario":
        pivot = df_sim.pivot_table(index="sku", columns="escenario",
                                   values="delta_ingreso_pct", aggfunc="first")
        fig = px.imshow(pivot, color_continuous_scale=[[0,"#E31837"],[0.5,"#FFFFFF"],[1,"#2E7D32"]],
                        color_continuous_midpoint=0, aspect="auto",
                        title="Δ Ingreso (%) por SKU y Escenario",
                        labels={"color":"Δ Ingreso (%)"})
        fig.update_layout(height=max(400, len(pivot) * 22 + 100), **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif grafica == "Curva continua de revenue (−15% a +15%)":
        sku_sel  = st.selectbox("SKU", skus)
        sub      = df_sim[df_sim["sku"] == sku_sel].iloc[0]
        p0, u0, beta_val = sub["precio_base"], sub["unidades_base"], sub["beta"]
        c0   = (df_base.set_index("sku").loc[sku_sel, "costo_unitario"]
                if "costo_unitario" in df_base.columns else np.nan)
        rev0 = p0 * u0
        marg0 = (p0 - c0) * u0 if pd.notna(c0) else None
        xs     = np.linspace(-0.15, 0.15, 300)
        rev_p  = [(p0*(1+x)*u0*(1+x)**beta_val - rev0)/rev0*100 for x in xs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs*100, y=rev_p, mode="lines", name="Δ Ingreso (%)",
                                 line=dict(color=OM_RED, width=2.5)))
        if marg0 and marg0 != 0:
            marg_p = [((p0*(1+x)-c0)*u0*(1+x)**beta_val - marg0)/abs(marg0)*100 for x in xs]
            fig.add_trace(go.Scatter(x=xs*100, y=marg_p, mode="lines", name="Δ Margen (%)",
                                     line=dict(color=OM_BLACK, width=1.8, dash="dash")))
        for cambio, etiq in [(-0.10,"-10%"),(-0.05,"-5%"),(0,"Base"),(0.05,"+5%"),(0.10,"+10%")]:
            y = (p0*(1+cambio)*u0*(1+cambio)**beta_val - rev0)/rev0*100
            fig.add_trace(go.Scatter(x=[cambio*100], y=[y], mode="markers+text", text=[etiq],
                                     textposition="top center",
                                     marker=dict(size=10, color=COLOR_ESC.get(etiq,"gray")),
                                     showlegend=False))
        fig.add_hline(y=0, line_dash="dash", line_color=OM_TEXT2)
        fig.add_vline(x=0, line_dash="dash", line_color=OM_TEXT2)
        fig.update_layout(title=f"{sku_sel} — Curva de Revenue  (β = {beta_val:.2f})",
                          xaxis_title="Cambio de precio (%)", yaxis_title="Δ (%)",
                          height=450, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — Exportar
# ─────────────────────────────────────────────────────────────────────────────
def paso7():
    render_header()
    step_banner(7, "Exportar Resultados",
                "Descarga todos los resultados en CSV")

    if st.session_state.df_sim is None:
        st.warning("Primero completa los pasos anteriores.")
        return

    df_sim  = st.session_state.df_sim
    df_rec  = st.session_state.df_rec
    df_ela  = st.session_state.df_elasticity
    df_base = st.session_state.df_base

    tab_sim, tab_rec, tab_ela, tab_base = st.tabs(
        ["Simulación completa", "Recomendaciones", "Elasticidades", "Cálculos Base"])

    def dl_btn(df, filename):
        if df is not None:
            st.download_button(
                f"📥 Descargar {filename}",
                df.to_csv(index=False).encode("utf-8"),
                filename, "text/csv",
            )
        else:
            st.info("Completa el paso correspondiente para habilitar esta exportación.")

    with tab_sim:
        if df_sim is not None: st.dataframe(df_sim, use_container_width=True)
        dl_btn(df_sim, "simulacion_completa.csv")
    with tab_rec:
        if df_rec is not None: st.dataframe(df_rec, use_container_width=True)
        dl_btn(df_rec, "recomendaciones.csv")
    with tab_ela:
        if df_ela is not None: st.dataframe(df_ela, use_container_width=True)
        dl_btn(df_ela, "elasticidades.csv")
    with tab_base:
        if df_base is not None: st.dataframe(df_base, use_container_width=True)
        dl_btn(df_base, "calculos_base.csv")

    st.markdown("---")
    st.markdown("""
    <div style="background:#1A1A1A;color:#FFFFFF;border-radius:6px;padding:24px 28px;margin-top:8px;">
        <div style="font-size:15px;font-weight:900;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:16px;border-bottom:2px solid #E31837;
                    padding-bottom:10px;">
            ⚠️ Limitaciones de esta herramienta
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:13px;">
            <div>
                <div style="color:#E31837;font-weight:700;margin-bottom:8px;">✅ PUEDE</div>
                <div style="color:#CCCCCC;line-height:1.8;">
                    • Estimar elasticidad exploratoria (OLS log-log)<br>
                    • Simular escenarios estándar (−10% a +10%)<br>
                    • Simular promociones complejas (3x2, 2x1, 2do al 50%)<br>
                    • Comparar ingreso y margen entre escenarios<br>
                    • Clasificar SKUs por acción sugerida<br>
                    • Exportar todos los resultados en CSV
                </div>
            </div>
            <div>
                <div style="color:#666666;font-weight:700;margin-bottom:8px;">❌ NO PUEDE</div>
                <div style="color:#CCCCCC;line-height:1.8;">
                    • Garantizar causalidad (solo correlación)<br>
                    • Fijar precios automáticamente<br>
                    • Reemplazar datos corregidos de costo<br>
                    • Modelar competencia ni sustitutos<br>
                    • Capturar inventario o stockouts<br>
                    • Controlar efectos de temporada complejos
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
ROUTER = {1: paso1, 2: paso2, 3: paso3, 4: paso4, 5: paso5, 6: paso6, 7: paso7}
ROUTER[st.session_state.step]()
