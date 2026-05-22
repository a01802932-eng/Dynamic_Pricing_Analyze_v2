# -*- coding: utf-8 -*-
"""Dynamic Pricing Analyzer — Streamlit App"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

STEP_STATUS = {
    1: st.session_state.df_clean is not None,
    2: st.session_state.df_base is not None,
    3: st.session_state.df_elasticity is not None,
    4: st.session_state.df_sim is not None,
    5: st.session_state.df_rec is not None,
    6: True,
    7: True,
}

st.sidebar.title("💰 Dynamic Pricing")
st.sidebar.markdown("---")
for i, label in enumerate(STEPS, 1):
    done = STEP_STATUS.get(i, False)
    st.sidebar.markdown(f"{'✅' if done else '⬜'} {label}")
st.sidebar.markdown("---")

current = st.sidebar.radio("Ir a paso", STEPS, index=st.session_state.step - 1,
                            label_visibility="collapsed")
st.session_state.step = STEPS.index(current) + 1


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Subir y Validar
# ─────────────────────────────────────────────────────────────────────────────
def paso1():
    st.title("Paso 1 — Subir archivo y validar datos")

    # Plantilla descargable
    with st.expander("📄 ¿Cómo debe verse mi archivo?"):
        tmpl = pd.DataFrame({
            "sku":         ["SKU001", "SKU001", "SKU002", "SKU002"],
            "unidades":    [100, 120, 200, 180],
            "venta_neta":  [1500.0, 1680.0, 3000.0, 2880.0],
            "fecha":       ["2024-01", "2024-02", "2024-01", "2024-02"],
            "precio":      [15.0, 14.0, 15.0, 16.0],
            "costo":       [8.0, 8.0, 9.0, 9.0],
            "departamento":["PAPELERIA", "PAPELERIA", "LIBRERIA", "LIBRERIA"],
            "tienda":      ["T001", "T001", "T002", "T002"],
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

    # Cargar
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

    df = st.session_state.df_raw
    cols = ["(ninguna)"] + list(df.columns)

    # Mapeo de columnas
    st.subheader("Mapeo de columnas")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Obligatorias**")
        m_sku  = st.selectbox("SKU / Producto",  cols, key="m_sku")
        m_uni  = st.selectbox("Unidades",         cols, key="m_uni")
        m_ven  = st.selectbox("Venta neta",        cols, key="m_ven")
        m_fec  = st.selectbox("Fecha",             cols, key="m_fec")
    with c2:
        st.markdown("**Opcionales — precio / costo**")
        m_pre  = st.selectbox("Precio unitario",   cols, key="m_pre")
        m_cos  = st.selectbox("Costo unitario",    cols, key="m_cos")
    with c3:
        st.markdown("**Opcionales — segmentación**")
        m_dep  = st.selectbox("Departamento",      cols, key="m_dep")
        m_tie  = st.selectbox("Tienda",             cols, key="m_tie")
        m_ela  = st.selectbox("Elasticidad pre-calculada", cols, key="m_ela")

    obligatorias = {"sku": m_sku, "unidades": m_uni, "venta_neta": m_ven, "fecha": m_fec}
    sin_mapear = [k for k, v in obligatorias.items() if v == "(ninguna)"]
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

    # Renombrar columnas
    rename = {v: k for k, v in col_map.items() if v is not None}
    df_c = df.rename(columns=rename).copy()

    # Coerciones numéricas
    df_c["unidades"]   = pd.to_numeric(df_c["unidades"],   errors="coerce")
    df_c["venta_neta"] = pd.to_numeric(df_c["venta_neta"], errors="coerce")
    df_c["sku"]        = df_c["sku"].astype(str).str.strip()
    if "precio" in df_c.columns:
        df_c["precio"] = pd.to_numeric(df_c["precio"], errors="coerce")
    if "costo" in df_c.columns:
        df_c["costo"]  = pd.to_numeric(df_c["costo"],  errors="coerce")
    if "elasticidad" in df_c.columns:
        df_c["elasticidad"] = pd.to_numeric(df_c["elasticidad"], errors="coerce")

    # Derivar precio si no existe
    if "precio" not in df_c.columns or df_c["precio"].isna().all():
        df_c["precio"] = df_c["venta_neta"] / df_c["unidades"].replace(0, np.nan)

    # Validaciones
    errores  = []
    avisos   = []

    n_neg_u  = int((df_c["unidades"] <= 0).sum())
    n_neg_v  = int((df_c["venta_neta"] <= 0).sum())
    n_null   = int(df_c[["sku", "unidades", "venta_neta"]].isna().any(axis=1).sum())
    n_dup    = int(df_c.duplicated().sum())

    if n_neg_u > 0:  errores.append(f"{n_neg_u:,} filas con unidades ≤ 0")
    if n_neg_v > 0:  errores.append(f"{n_neg_v:,} filas con venta_neta ≤ 0")
    if n_null > 0:   errores.append(f"{n_null:,} filas con campos obligatorios nulos")
    if n_dup > 0:    avisos.append(f"{n_dup:,} filas duplicadas detectadas")

    if "costo" in df_c.columns:
        n_neg_c  = int((df_c["costo"] < 0).sum())
        n_c_gt_p = int(((df_c["costo"] > df_c["precio"]) &
                        df_c["costo"].notna() & df_c["precio"].notna()).sum())
        if n_neg_c > 0:  avisos.append(f"{n_neg_c:,} filas con costo negativo")
        if n_c_gt_p > 0: avisos.append(f"{n_c_gt_p:,} filas donde costo > precio")

    # Varianza de precio (solo si hay suficientes datos limpios)
    df_tmp = df_c[(df_c["unidades"] > 0) & (df_c["precio"] > 0)].copy()
    if len(df_tmp) > 0:
        cv_mean = (df_tmp.groupby("sku")["precio"].std() /
                   df_tmp.groupby("sku")["precio"].mean().replace(0, np.nan)).mean()
        if pd.notna(cv_mean) and cv_mean < 0.02:
            avisos.append(f"Varianza de precio muy baja (CV promedio = {cv_mean:.1%}) — elasticidad será poco confiable")

    # Semáforo
    if errores:
        quality = "red"
        st.error("🔴 Datos insuficientes — requiere corrección antes de continuar")
        for e in errores:
            st.error(f"  • {e}")
        return
    elif avisos:
        quality = "yellow"
        st.warning("🟡 Datos parciales — puedes continuar con advertencias")
        for w in avisos:
            st.warning(f"  • {w}")
    else:
        quality = "green"
        st.success("🟢 Datos listos para análisis completo")

    # Limpiar filas inválidas
    df_c = df_c[(df_c["unidades"] > 0) & (df_c["venta_neta"] > 0) & (df_c["precio"] > 0)].copy()

    st.session_state.df_clean  = df_c
    st.session_state.col_map   = col_map
    st.session_state.quality   = quality
    # Resetear pasos siguientes al subir nuevo archivo
    for key in ("df_base", "df_elasticity", "df_sim", "df_rec"):
        st.session_state[key] = None

    # Estadísticas
    st.subheader("Estadísticas del archivo")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Filas válidas",  f"{len(df_c):,}")
    k2.metric("SKUs únicos",     f"{df_c['sku'].nunique():,}")
    k3.metric("Precio mínimo",  f"${df_c['precio'].min():,.2f}")
    k4.metric("Precio máximo",  f"${df_c['precio'].max():,.2f}")
    if "departamento" in df_c.columns:
        k5.metric("Departamentos", f"{df_c['departamento'].nunique():,}")

    with st.expander("Ver muestra de datos"):
        st.dataframe(df_c.head(30), use_container_width=True)

    st.info("✅ Listo — ve al **Paso 2** en el menú lateral")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Cálculos Base
# ─────────────────────────────────────────────────────────────────────────────
def paso2():
    st.title("Paso 2 — Cálculos Base por SKU")

    if st.session_state.df_clean is None:
        st.warning("Primero completa el **Paso 1**.")
        return

    df      = st.session_state.df_clean.copy()
    col_map = st.session_state.col_map

    group_cols = ["departamento", "sku"] if ("departamento" in df.columns) else ["sku"]

    agg = (df.groupby(group_cols)
           .agg(
               unidades_tot  = ("unidades",   "sum"),
               venta_neta_tot= ("venta_neta", "sum"),
               precio_base   = ("precio",     "mean"),
               n_obs         = ("precio",     "count"),
           )
           .reset_index())

    agg["ingreso_base"] = agg["precio_base"] * agg["unidades_tot"]

    if "costo" in df.columns and not df["costo"].isna().all():
        costo_p = (df.groupby("sku")["costo"].mean()
                   .reset_index().rename(columns={"costo": "costo_unitario"}))
        agg = agg.merge(costo_p, on="sku", how="left")
        agg["margen_unitario"] = agg["precio_base"] - agg["costo_unitario"]
        agg["margen_total"]    = agg["margen_unitario"] * agg["unidades_tot"]
    else:
        agg["costo_unitario"] = np.nan
        agg["margen_unitario"] = np.nan
        agg["margen_total"]   = np.nan
        st.info("Sin columna de costo — solo se simulará ingreso, no margen.")

    st.session_state.df_base = agg

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs",           f"{agg['sku'].nunique():,}")
    k2.metric("Ingreso total",  f"${agg['ingreso_base'].sum():,.0f}")
    if not agg["margen_total"].isna().all():
        k3.metric("Margen total", f"${agg['margen_total'].sum():,.0f}")
    k4.metric("Precio promedio", f"${agg['precio_base'].mean():,.2f}")

    display = [c for c in
               ["departamento", "sku", "n_obs", "unidades_tot", "precio_base",
                "costo_unitario", "ingreso_base", "margen_unitario", "margen_total"]
               if c in agg.columns]

    fmt = {
        "precio_base":     "${:.2f}",
        "costo_unitario":  "${:.2f}",
        "margen_unitario": "${:.2f}",
        "ingreso_base":    "${:,.0f}",
        "margen_total":    "${:,.0f}",
        "unidades_tot":    "{:,.0f}",
    }
    st.dataframe(
        agg[display].sort_values("ingreso_base", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in display}),
        use_container_width=True,
    )
    st.info("✅ Listo — ve al **Paso 3**")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Elasticidad
# ─────────────────────────────────────────────────────────────────────────────
def paso3():
    st.title("Paso 3 — Estimación de Elasticidad")

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
    MAX_BETA = c2.slider("Límite |beta| máximo aceptable",  3, 20, 10)

    if not st.button("▶ Calcular elasticidades", type="primary"):
        return

    if "Usar columna" in fuente:
        if not col_map.get("elasticidad") or "elasticidad" not in df.columns:
            st.error("No hay columna de elasticidad mapeada en el Paso 1.")
            return
        betas = (df.groupby("sku")["elasticidad"]
                 .mean().reset_index()
                 .rename(columns={"elasticidad": "beta"}))
        betas["r2"]    = np.nan
        betas["pval"]  = np.nan
        betas["n_obs"] = df.groupby("sku")["elasticidad"].count().values

    else:
        # Agregar por SKU x periodo mensual
        if "fecha" in df.columns:
            df["periodo"] = pd.to_datetime(df["fecha"], errors="coerce").dt.to_period("M").astype(str)
        else:
            df["periodo"] = "ALL"

        mensual = (df.groupby(["sku", "periodo"])
                   .agg(unidades=("unidades", "sum"), venta_neta=("venta_neta", "sum"))
                   .reset_index())
        mensual["precio"] = mensual["venta_neta"] / mensual["unidades"].replace(0, np.nan)
        mensual = mensual[mensual["precio"] > 0].copy()
        mensual["log_u"] = np.log1p(mensual["unidades"])
        mensual["log_p"] = np.log(mensual["precio"])

        def ols_beta(grp):
            grp = grp.dropna(subset=["log_u", "log_p"])
            if len(grp) < MIN_OBS:        return None
            if grp["log_p"].std() < 1e-6: return None
            try:
                res  = OLS(grp["log_u"].values,
                           add_constant(grp["log_p"].values)).fit()
                beta = res.params[1]
                if abs(beta) > MAX_BETA:   return None
                return pd.Series({
                    "beta":  round(float(beta), 4),
                    "r2":    round(float(res.rsquared), 4),
                    "pval":  round(float(res.pvalues[1]), 4),
                    "n_obs": int(len(grp)),
                })
            except:
                return None

        with st.spinner("Estimando elasticidades..."):
            resultados = mensual.groupby("sku").apply(ols_beta)
            betas = resultados.dropna().reset_index()

    if len(betas) == 0:
        st.error("No se pudo calcular elasticidad para ningún SKU. Revisa que haya variación de precio y suficientes observaciones.")
        return

    # Clasificación
    def clasificar(b):
        if pd.isna(b):  return "Sin datos suficientes"
        if b < -1:      return "Elástico"
        if b < 0:       return "Inelástico"
        return "Sospechoso"

    betas["clasificacion"] = betas["beta"].apply(clasificar)
    st.session_state.df_elasticity = betas
    # Resetear pasos dependientes
    st.session_state.df_sim = None
    st.session_state.df_rec = None

    # Métricas
    counts = betas["clasificacion"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Elásticos (β < −1)",     counts.get("Elástico", 0))
    k2.metric("🟢 Inelásticos (−1 < β < 0)", counts.get("Inelástico", 0))
    k3.metric("🟡 Sospechosos (β ≥ 0)",    counts.get("Sospechoso", 0))
    k4.metric("⚪ Sin datos",               counts.get("Sin datos suficientes", 0))

    # Histograma
    fig = px.histogram(
        betas.dropna(subset=["beta"]), x="beta", nbins=30,
        color="clasificacion",
        color_discrete_map={"Elástico": "#E53935", "Inelástico": "#43A047",
                            "Sospechoso": "#FF9800", "Sin datos suficientes": "#BDBDBD"},
        title="Distribución de Elasticidad por SKU",
        labels={"beta": "Elasticidad (β)", "count": "N de SKUs"},
    )
    fig.add_vline(x=-1, line_dash="dash", line_color="red",
                  annotation_text="β = −1", annotation_position="top right")
    fig.add_vline(x=0, line_dash="dash", line_color="orange",
                  annotation_text="β = 0", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    fmt = {"beta": "{:.4f}", "r2": "{:.3f}", "pval": "{:.4f}", "n_obs": "{:.0f}"}
    st.dataframe(
        betas.sort_values("beta")
        .style.format({k: v for k, v in fmt.items() if k in betas.columns}),
        use_container_width=True,
    )
    st.info("✅ Listo — ve al **Paso 4**")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Simulación
# ─────────────────────────────────────────────────────────────────────────────
def paso4():
    st.title("Paso 4 — Simulación de Escenarios de Precio")

    if st.session_state.df_base is None or st.session_state.df_elasticity is None:
        st.warning("Primero completa los pasos anteriores.")
        return

    df_base = st.session_state.df_base.copy()
    df_ela  = st.session_state.df_elasticity.copy()

    merged = df_base.merge(df_ela[["sku", "beta", "clasificacion"]], on="sku", how="left")
    merged = merged[merged["beta"].notna()].copy()

    if len(merged) == 0:
        st.error("No hay SKUs con elasticidad calculada.")
        return

    # Escenarios estándar
    st.subheader("Escenarios estándar")
    CAMBIOS_STD   = [-0.10, -0.05, 0.00, 0.05, 0.10]
    ETIQUETAS_STD = ["-10%", "-5%", "Base", "+5%", "+10%"]

    # Promociones complejas
    st.subheader("Promociones complejas")
    promos_sel = st.multiselect(
        "Selecciona promociones adicionales a simular",
        ["3x2 (lleva 3, paga 2)", "2x1 (lleva 2, paga 1)", "2do al 50%"],
        default=[],
    )
    PROMO_MULT = {
        "3x2 (lleva 3, paga 2)": 2/3,
        "2x1 (lleva 2, paga 1)": 0.50,
        "2do al 50%":            0.75,
    }

    CAMBIOS   = CAMBIOS_STD   + [PROMO_MULT[p] - 1 for p in promos_sel]
    ETIQUETAS = ETIQUETAS_STD + [p.split(" ")[0]   for p in promos_sel]

    if not st.button("▶ Simular escenarios", type="primary"):
        return

    rows = []
    for _, row in merged.iterrows():
        p0    = row["precio_base"]
        u0    = row["unidades_tot"]
        beta  = row["beta"]
        c0    = row.get("costo_unitario", np.nan)
        rev0  = p0 * u0
        marg0 = (p0 - c0) * u0 if pd.notna(c0) else np.nan

        for cambio, etiq in zip(CAMBIOS, ETIQUETAS):
            p1    = p0 * (1 + cambio)
            u1    = u0 * ((1 + cambio) ** beta)
            ing1  = p1 * u1
            marg1 = (p1 - c0) * u1 if pd.notna(c0) else np.nan

            d_ing  = (ing1  - rev0)  / rev0  * 100 if rev0  != 0 else 0
            d_marg = (marg1 - marg0) / abs(marg0) * 100 if pd.notna(marg0) and marg0 != 0 else np.nan

            entry = {
                "sku":                row["sku"],
                "clasificacion":      row["clasificacion"],
                "beta":               round(float(beta), 4),
                "precio_base":        round(float(p0), 2),
                "unidades_base":      round(float(u0), 1),
                "escenario":          etiq,
                "precio_nuevo":       round(float(p1), 2),
                "unidades_simuladas": round(float(u1), 1),
                "ingreso_simulado":   round(float(ing1), 2),
                "delta_ingreso_pct":  round(float(d_ing), 1),
                "margen_simulado":    round(float(marg1), 2) if pd.notna(marg1) else np.nan,
                "delta_margen_pct":   round(float(d_marg), 1) if pd.notna(d_marg) else np.nan,
            }
            if "departamento" in row.index:
                entry["departamento"] = row["departamento"]
            rows.append(entry)

    df_sim = pd.DataFrame(rows)
    st.session_state.df_sim = df_sim
    st.session_state.df_rec = None  # resetear recomendaciones

    st.success(f"Simulación completada: **{merged['sku'].nunique()} SKUs × {len(CAMBIOS)} escenarios**")

    # Pivot heatmap rápido
    pivot = df_sim.pivot_table(
        index="sku", columns="escenario",
        values="delta_ingreso_pct", aggfunc="first",
    )
    # Ordenar columnas en orden lógico
    orden = [e for e in ETIQUETAS if e in pivot.columns]
    pivot = pivot[orden]

    st.dataframe(
        pivot.style.format("{:+.1f}%")
                   .background_gradient(cmap="RdYlGn", axis=None, vmin=-20, vmax=20),
        use_container_width=True,
    )
    st.info("✅ Listo — ve al **Paso 5**")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Recomendaciones
# ─────────────────────────────────────────────────────────────────────────────
def paso5():
    st.title("Paso 5 — Recomendaciones Automáticas por SKU")

    if st.session_state.df_sim is None:
        st.warning("Primero completa el **Paso 4**.")
        return

    df_sim = st.session_state.df_sim.copy()
    df_ela = st.session_state.df_elasticity.copy()
    df_base = st.session_state.df_base.copy()

    c1, c2 = st.columns(2)
    MAX_PVAL = c1.slider("p-valor máximo para considerar significativa la beta", 0.05, 0.50, 0.10, 0.05)
    MIN_N    = c2.slider("Mínimo de observaciones para emitir recomendación", 2, 10, 3)

    # Mejor escenario estándar por SKU (maximizar ingreso en -10% a +10%)
    std_esc = ["-10%", "-5%", "Base", "+5%", "+10%"]
    std_df  = df_sim[df_sim["escenario"].isin(std_esc)].copy()
    best    = (std_df.loc[std_df.groupby("sku")["ingreso_simulado"].idxmax()]
               [["sku", "escenario", "delta_ingreso_pct"]]
               .rename(columns={"escenario": "mejor_escenario", "delta_ingreso_pct": "delta_mejor_pct"}))

    rec = df_ela.merge(best, on="sku", how="left")

    # Costo > precio flag
    if "costo_unitario" in df_base.columns:
        flag_skus = set(df_base[df_base["costo_unitario"] > df_base["precio_base"]]["sku"])
    else:
        flag_skus = set()

    def recomendar(row):
        b    = row.get("beta",  np.nan)
        pval = row.get("pval",  np.nan)
        n    = row.get("n_obs", np.nan)
        sku  = row["sku"]

        if sku in flag_skus:
            return "Exploratoria", "Costo > precio — revisar datos de costo"
        if pd.isna(b):
            return "No recomendar", "Sin elasticidad calculada"
        if b > 0:
            return "No recomendar", f"Beta positiva (β={b:.2f}) — anomalía estadística"

        sig     = pd.isna(pval) or (pval < MAX_PVAL)
        enough  = pd.isna(n)   or (n >= MIN_N)

        if not sig:
            return "No recomendar", f"Beta no significativa (p={pval:.2f} ≥ {MAX_PVAL})"
        if not enough:
            return "No recomendar", f"Solo {int(n)} observaciones (mínimo {MIN_N})"

        if -1 < b < 0:
            return "Subir precio",            f"Inelástica (β={b:.2f}): ingreso mejora al subir precio"
        if -1.5 <= b <= -1:
            return "Mantener precio",          f"Cerca de unitaria (β={b:.2f}): impacto neutro"
        return "Bajar precio / promover",       f"Elástica (β={b:.2f}): bajar precio mejora ingreso"

    rec[["recomendacion", "razon"]] = rec.apply(
        lambda r: pd.Series(recomendar(r)), axis=1)
    st.session_state.df_rec = rec

    # Tarjetas de resumen
    CATS = ["Subir precio", "Mantener precio", "Bajar precio / promover",
            "No recomendar", "Exploratoria"]
    ICONS = {"Subir precio":"🟢", "Mantener precio":"🔵",
             "Bajar precio / promover":"🟡", "No recomendar":"🔴", "Exploratoria":"⚪"}
    cols  = st.columns(len(CATS))
    for i, cat in enumerate(CATS):
        n = int((rec["recomendacion"] == cat).sum())
        cols[i].metric(f"{ICONS[cat]} {cat}", n)

    # Tabla filtrable
    st.subheader("Tabla de recomendaciones")
    filtro = st.selectbox("Filtrar por categoría", ["Todas"] + CATS)
    df_show = rec if filtro == "Todas" else rec[rec["recomendacion"] == filtro]

    show_cols = [c for c in ["sku", "beta", "pval", "n_obs", "clasificacion",
                              "recomendacion", "razon", "mejor_escenario", "delta_mejor_pct"]
                 if c in df_show.columns]
    fmt = {"beta": "{:.4f}", "pval": "{:.4f}", "delta_mejor_pct": "{:+.1f}%"}
    st.dataframe(
        df_show[show_cols].sort_values("recomendacion")
        .style.format({k: v for k, v in fmt.items() if k in df_show.columns}),
        use_container_width=True,
    )
    st.info("✅ Listo — ve al **Paso 6**")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Gráficas
# ─────────────────────────────────────────────────────────────────────────────
def paso6():
    st.title("Paso 6 — Gráficas")

    if st.session_state.df_sim is None:
        st.warning("Primero completa el **Paso 4**.")
        return

    df_sim  = st.session_state.df_sim.copy()
    df_ela  = st.session_state.df_elasticity
    df_rec  = st.session_state.df_rec
    df_base = st.session_state.df_base

    grafica = st.selectbox("Selecciona gráfica", [
        "Ingreso y margen por escenario — SKU individual",
        "Distribución de elasticidades",
        "Recomendaciones: β vs volumen",
        "Heatmap Δ ingreso por SKU y escenario",
        "Curva continua de revenue (-15% a +15%)",
    ])

    COLOR_ESCENARIO = {
        "-10%": "#1B5E20", "-5%": "#81C784", "Base": "#9E9E9E",
        "+5%": "#EF9A9A", "+10%": "#B71C1C",
    }
    COLOR_REC = {
        "Subir precio": "#1565C0", "Mantener precio": "#F9A825",
        "Bajar precio / promover": "#2E7D32",
        "No recomendar": "#BDBDBD", "Exploratoria": "#9C27B0",
    }

    skus = sorted(df_sim["sku"].unique())

    # ── Ingreso y margen por escenario ────────────────────────────────────────
    if grafica == "Ingreso y margen por escenario — SKU individual":
        sku_sel = st.selectbox("SKU", skus)
        sub = df_sim[df_sim["sku"] == sku_sel].copy()
        beta_val = sub["beta"].iloc[0]

        has_margin = sub["delta_margen_pct"].notna().any()
        ncols = 2 if has_margin else 1
        titles = ["Δ Ingreso (%)"] + (["Δ Margen (%)"] if has_margin else [])
        fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles)

        colors = [COLOR_ESCENARIO.get(e, "#7B1FA2") for e in sub["escenario"]]
        fig.add_trace(
            go.Bar(x=sub["escenario"], y=sub["delta_ingreso_pct"],
                   marker_color=colors, name="Ingreso",
                   text=sub["delta_ingreso_pct"].apply(lambda v: f"{v:+.1f}%"),
                   textposition="outside"),
            row=1, col=1,
        )
        if has_margin:
            fig.add_trace(
                go.Bar(x=sub["escenario"], y=sub["delta_margen_pct"],
                       marker_color=colors, name="Margen",
                       text=sub["delta_margen_pct"].apply(
                           lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
                       textposition="outside"),
                row=1, col=2,
            )

        for col_i in range(1, ncols + 1):
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=col_i)

        fig.update_layout(
            title=f"{sku_sel} — Escenarios de precio  (β = {beta_val:.2f})",
            showlegend=False, height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla resumen
        st.dataframe(
            sub[["escenario", "precio_nuevo", "unidades_simuladas",
                 "ingreso_simulado", "delta_ingreso_pct"]].style.format({
                "precio_nuevo":       "${:.2f}",
                "unidades_simuladas": "{:,.1f}",
                "ingreso_simulado":   "${:,.2f}",
                "delta_ingreso_pct":  "{:+.1f}%",
            }),
            use_container_width=True,
        )

    # ── Distribución de elasticidades ─────────────────────────────────────────
    elif grafica == "Distribución de elasticidades":
        if df_ela is None:
            st.warning("Completa el Paso 3.")
            return
        fig = px.histogram(
            df_ela.dropna(subset=["beta"]), x="beta", nbins=30,
            color="clasificacion",
            color_discrete_map={"Elástico": "#E53935", "Inelástico": "#43A047",
                                "Sospechoso": "#FF9800", "Sin datos suficientes": "#BDBDBD"},
            title="Distribución de Elasticidades por SKU",
            labels={"beta": "Elasticidad (β)", "count": "N de SKUs"},
        )
        fig.add_vline(x=-1, line_dash="dash", line_color="red",
                      annotation_text="β = −1")
        fig.add_vline(x=0,  line_dash="dash", line_color="orange",
                      annotation_text="β = 0")
        st.plotly_chart(fig, use_container_width=True)

    # ── β vs volumen ──────────────────────────────────────────────────────────
    elif grafica == "Recomendaciones: β vs volumen":
        if df_rec is None or df_base is None:
            st.warning("Completa los Pasos 2 y 5.")
            return
        df_plot = df_rec.merge(df_base[["sku", "unidades_tot"]], on="sku", how="left")
        fig = px.scatter(
            df_plot, x="beta", y="unidades_tot",
            color="recomendacion", hover_name="sku",
            color_discrete_map=COLOR_REC, size_max=20,
            title="Elasticidad vs Volumen por SKU",
            labels={"beta": "Elasticidad (β)", "unidades_tot": "Unidades totales"},
        )
        fig.add_vline(x=-1, line_dash="dash", line_color="gray", annotation_text="β = −1")
        fig.add_vline(x=0,  line_dash="dash", line_color="gray", annotation_text="β = 0")
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    elif grafica == "Heatmap Δ ingreso por SKU y escenario":
        pivot = df_sim.pivot_table(
            index="sku", columns="escenario",
            values="delta_ingreso_pct", aggfunc="first",
        )
        fig = px.imshow(
            pivot, color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0, aspect="auto",
            title="Δ Ingreso (%) por SKU y Escenario",
            labels={"color": "Δ Ingreso (%)"},
        )
        fig.update_layout(height=max(400, len(pivot) * 22 + 100))
        st.plotly_chart(fig, use_container_width=True)

    # ── Curva continua ────────────────────────────────────────────────────────
    elif grafica == "Curva continua de revenue (-15% a +15%)":
        sku_sel  = st.selectbox("SKU", skus)
        sub      = df_sim[df_sim["sku"] == sku_sel].iloc[0]
        p0, u0   = sub["precio_base"], sub["unidades_base"]
        beta_val = sub["beta"]
        c0       = st.session_state.df_base.set_index("sku").loc[sku_sel, "costo_unitario"] \
                   if "costo_unitario" in df_base.columns else np.nan
        rev0     = p0 * u0
        marg0    = (p0 - c0) * u0 if pd.notna(c0) else None

        xs = np.linspace(-0.15, 0.15, 300)
        rev_pct  = [(p0*(1+x)*u0*(1+x)**beta_val - rev0)/rev0*100 for x in xs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs*100, y=rev_pct, mode="lines",
                                 name="Δ Ingreso (%)", line=dict(color="#1976D2", width=2.5)))
        if marg0 and marg0 != 0:
            marg_pct = [((p0*(1+x)-c0)*u0*(1+x)**beta_val - marg0)/abs(marg0)*100 for x in xs]
            fig.add_trace(go.Scatter(x=xs*100, y=marg_pct, mode="lines",
                                     name="Δ Margen (%)", line=dict(color="#E53935", width=1.8, dash="dash")))

        # Marcar puntos estándar
        for cambio, etiq in [(-0.10,"-10%"),(-0.05,"-5%"),(0,"Base"),(0.05,"+5%"),(0.10,"+10%")]:
            y = (p0*(1+cambio)*u0*(1+cambio)**beta_val - rev0)/rev0*100
            fig.add_trace(go.Scatter(x=[cambio*100], y=[y], mode="markers+text",
                                     text=[etiq], textposition="top center",
                                     marker=dict(size=10, color=COLOR_ESCENARIO.get(etiq,"gray")),
                                     showlegend=False))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"{sku_sel} — Curva de Revenue  (β = {beta_val:.2f})",
            xaxis_title="Cambio de precio (%)",
            yaxis_title="Δ (%)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — Exportar
# ─────────────────────────────────────────────────────────────────────────────
def paso7():
    st.title("Paso 7 — Exportar Resultados")

    if st.session_state.df_sim is None:
        st.warning("Primero completa los pasos anteriores.")
        return

    df_sim  = st.session_state.df_sim
    df_rec  = st.session_state.df_rec
    df_ela  = st.session_state.df_elasticity
    df_base = st.session_state.df_base

    tab_sim, tab_rec, tab_ela, tab_base = st.tabs(
        ["Simulación", "Recomendaciones", "Elasticidades", "Cálculos Base"])

    def dl_btn(df, filename):
        if df is not None:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(f"📥 Descargar {filename}", csv, filename, "text/csv")
        else:
            st.info("Completa el paso correspondiente para habilitar esta exportación.")

    with tab_sim:
        if df_sim is not None:
            st.dataframe(df_sim, use_container_width=True)
        dl_btn(df_sim, "simulacion_completa.csv")

    with tab_rec:
        if df_rec is not None:
            st.dataframe(df_rec, use_container_width=True)
        dl_btn(df_rec, "recomendaciones.csv")

    with tab_ela:
        if df_ela is not None:
            st.dataframe(df_ela, use_container_width=True)
        dl_btn(df_ela, "elasticidades.csv")

    with tab_base:
        if df_base is not None:
            st.dataframe(df_base, use_container_width=True)
        dl_btn(df_base, "calculos_base.csv")

    # Slide de limitaciones (entregable obligatorio)
    st.markdown("---")
    st.subheader("⚠️ Limitaciones de esta herramienta")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Puede:**
- ✅ Estimar elasticidad exploratoria (OLS log-log)
- ✅ Simular escenarios estándar (−10% a +10%)
- ✅ Simular promociones complejas (3x2, 2x1, 2do al 50%)
- ✅ Comparar ingreso y margen entre escenarios
- ✅ Clasificar SKUs por acción sugerida
- ✅ Exportar todos los resultados en CSV
        """)
    with col_b:
        st.markdown("""
**No puede todavía:**
- ❌ Garantizar causalidad (solo correlación)
- ❌ Fijar precios automáticamente
- ❌ Reemplazar datos corregidos de costo
- ❌ Modelar competencia ni sustitutos
- ❌ Capturar inventario o stockouts
- ❌ Controlar efectos de temporada complejos
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Router principal
# ─────────────────────────────────────────────────────────────────────────────
ROUTER = {1: paso1, 2: paso2, 3: paso3, 4: paso4, 5: paso5, 6: paso6, 7: paso7}
ROUTER[st.session_state.step]()
