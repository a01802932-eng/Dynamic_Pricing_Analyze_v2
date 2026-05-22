# -*- coding: utf-8 -*-
"""
Elasticidad Precio - OfficeMax | Categoria: CARPETAS
======================================================
MODELO 1 (simple): log(unidades+1) = alpha + beta * log(precio_real)
  - Mensual    : una regresion por SKU usando todos los meses disponibles
  - Trimestral : ventana deslizante de 3 meses, avanzando 1 mes a la vez
  - Semestral  : ventana deslizante de 6 meses, avanzando 1 mes a la vez

MODELO 2 (extendido): agrega dummies de tienda, mes, flag premium, promocion
  log(unidades+1) = alpha + beta*log(precio) + delta*tienda
                  + omega*mes + gamma*SKU_premium + epsilon*promocion + ...
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from sklearn.preprocessing import OneHotEncoder
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings, os, glob

warnings.filterwarnings("ignore")

# ── Configuracion ─────────────────────────────────────────────────────────────
CATEGORIA    = "SUMINISTROS DE OFICINA"   # <-- cambia aqui para analizar otra categoria
SKU_OBJETIVO = None         # <-- prod_nbr especifico para deep dive (None = auto)

BASE    = r"C:\Users\ramir\OneDrive\Escritorio\datos_officmax"
DESKTOP = r"C:\Users\ramir\OneDrive\Escritorio"
OUT_XL  = os.path.join(DESKTOP, f"Elasticidad_{CATEGORIA}.xlsx")
OUT_PDF = os.path.join(DESKTOP, f"Elasticidad_{CATEGORIA}.pdf")
MIN_OBS  = 3
MIN_CV   = 0.02   # ~9% variacion de precio — lo minimo para que beta tenga sentido
MIN_R2   = 0.0    # sin filtro de R2 en ventanas cortas (3 puntos = 1 gdl)
MAX_BETA = 10     # cap: |beta|>10 no es interpretable economicamente

def find(base, pat):
    m = glob.glob(os.path.join(base, pat))
    if not m: raise FileNotFoundError(f"No encontrado: {pat}")
    return m[0]

# ── 1. Carga y merge ──────────────────────────────────────────────────────────
print("Cargando datos...")
ventas   = pd.read_csv(find(BASE,"Ventas_OfficeMax.csv"),  encoding="latin-1", low_memory=False)
catalogo = pd.read_csv(find(BASE,"Cat*.csv"),              encoding="latin-1", low_memory=False)
precios  = pd.read_csv(find(BASE,"Precios_de_Productos.csv"), encoding="latin-1")

for d in [ventas, catalogo, precios]:
    d["prod_nbr"] = d["prod_nbr"].astype(str).str.strip()

cat_slim  = catalogo[["prod_nbr","prod_nm","dept_nm","subdept_nm","class_nm",
                       "marca_fabricante","tipo_marca"]].drop_duplicates("prod_nbr")
prec_slim = precios[["prod_nbr","Precio_Unitario"]].rename(
            columns={"Precio_Unitario":"precio_catalogo"})

ventas["tran_date"]     = pd.to_datetime(ventas["tran_date"], errors="coerce")
ventas["qty"]           = pd.to_numeric(ventas["qty"],           errors="coerce")
ventas["venta_con_iva"] = pd.to_numeric(ventas["venta_con_iva"], errors="coerce")
ventas["margen"]        = pd.to_numeric(ventas["margen"],        errors="coerce")
ventas = ventas[(ventas["qty"] > 0) & (ventas["venta_con_iva"] > 0)].copy()
ventas["precio_tx"] = ventas["venta_con_iva"] / ventas["qty"]
ventas["mes"]       = ventas["tran_date"].dt.to_period("M")
ventas["mes_num"]   = ventas["mes"].apply(lambda x: x.ordinal)

df = (ventas
      .merge(cat_slim,  on="prod_nbr", how="left", suffixes=("","_cat"))
      .merge(prec_slim, on="prod_nbr", how="left"))

# ── 2. Filtrar categoria ──────────────────────────────────────────────────────
carp = df[df["dept_nm"].str.contains(CATEGORIA, na=False, case=False)].copy()
print(f"  {CATEGORIA}: {len(carp):,} filas | {carp['prod_nbr'].nunique()} SKUs | "
      f"{carp['store_nbr'].nunique()} tiendas | {carp['mes'].nunique()} meses")

# ── 3. Flags ──────────────────────────────────────────────────────────────────
# Premium: solo por keyword en nombre del producto (~7.5% de SKUs)
KEYWORDS = ["PREMIUM","PRO ","PROFESIONAL","EXECUTIVE","EJECUTIV",
            "DELUXE","ELITE","PLUS","GOLD","CARBON","PIEL","CUERO","ERGON"]
carp["es_premium"] = carp["prod_nm"].apply(
    lambda n: int(any(k in str(n).upper() for k in KEYWORDS)) if pd.notna(n) else 0)

print(f"  SKUs premium (keyword): {carp.groupby('prod_nbr')['es_premium'].max().sum()}")

# ── 4. Agregacion mensual por SKU (para Modelo 1) ─────────────────────────────
# Una obs por SKU x mes: suma de unidades, precio promedio ponderado
mensual = (
    carp.groupby(["prod_nbr","mes"])
    .agg(
        unidades  = ("qty",          "sum"),
        venta_tot = ("venta_con_iva","sum"),
        es_premium= ("es_premium",   "max"),
        prod_nm   = ("prod_nm",      "first"),
        subdept_nm= ("subdept_nm",   "first"),
    )
    .reset_index()
)
mensual["precio"]  = mensual["venta_tot"] / mensual["unidades"]
mensual = mensual[mensual["precio"] > 0].copy()
mensual["log_u1"]  = np.log1p(mensual["unidades"])   # log(units + 1)
mensual["log_p"]   = np.log(mensual["precio"])
mensual["mes_dt"]  = mensual["mes"].dt.to_timestamp()
mensual = mensual.sort_values(["prod_nbr","mes"]).reset_index(drop=True)

todos_meses = sorted(mensual["mes"].unique())
print(f"\n  Obs mensuales SKU-mes: {len(mensual):,}")

# ── 5. Funcion OLS log-log ────────────────────────────────────────────────────
def ols_loglog(subset, min_obs=MIN_OBS, min_cv=0, min_r2=0, max_beta=None):
    n = len(subset)
    std_p = subset["log_p"].std()
    if n < min_obs or std_p < 1e-6:
        return None
    mean_p = abs(subset["log_p"].mean())
    if mean_p > 1e-9 and (std_p / mean_p) < min_cv:
        return None
    try:
        res  = OLS(subset["log_u1"].values,
                   add_constant(subset["log_p"].values)).fit()
        beta = round(res.params[1], 4)
        r2   = round(res.rsquared, 4)
        if r2 < min_r2:
            return None
        if max_beta is not None and abs(beta) > max_beta:
            return None
        return {"alpha": round(res.params[0], 4),
                "beta":  beta,
                "r2":    r2,
                "n":     n,
                "pval":  round(res.pvalues[1], 4)}
    except Exception:
        return None

# ── 6. MODELO 1A: Mensual (un beta por SKU, todos los meses) ─────────────────
print("\n[MODELO 1A] Mensual — un beta por SKU usando todos los meses...")
betas_mensual = []
for sku, grp in mensual.groupby("prod_nbr"):
    r = ols_loglog(grp)
    if r:
        betas_mensual.append({"prod_nbr":sku, "prod_nm":grp["prod_nm"].iloc[0],
                               "subdept_nm":grp["subdept_nm"].iloc[0],
                               "n_meses":len(grp), **r})
df_mensual = pd.DataFrame(betas_mensual)
print(f"  SKUs con beta valida: {len(df_mensual)} de {mensual['prod_nbr'].nunique()}")

# ── 7. MODELO 1B: Trimestral rolling (ventana 3, paso 1) ─────────────────────
print("[MODELO 1B] Trimestral rolling (ventana 3 meses, paso 1 mes)...")
betas_trim = []
for sku, grp in mensual.groupby("prod_nbr"):
    grp_idx = grp.set_index("mes")
    for i in range(len(todos_meses) - 2):
        ventana = todos_meses[i: i+3]
        sub = grp_idx[grp_idx.index.isin(ventana)]
        r = ols_loglog(sub, min_cv=MIN_CV, min_r2=MIN_R2, max_beta=MAX_BETA)
        if r:
            betas_trim.append({
                "prod_nbr": sku,
                "prod_nm":  grp["prod_nm"].iloc[0],
                "mes_inicio": str(ventana[0]),
                "mes_fin":    str(ventana[-1]),
                "mes_fin_dt": ventana[-1].to_timestamp(),
                **r
            })
df_trim = pd.DataFrame(betas_trim)
print(f"  Betas trimestrales validas: {len(df_trim):,}")

# ── 8. MODELO 1C: Semestral rolling (ventana 6, paso 1) ──────────────────────
print("[MODELO 1C] Semestral rolling (ventana 6 meses, paso 1 mes)...")
betas_sem = []
for sku, grp in mensual.groupby("prod_nbr"):
    grp_idx = grp.set_index("mes")
    for i in range(len(todos_meses) - 5):
        ventana = todos_meses[i: i+6]
        sub = grp_idx[grp_idx.index.isin(ventana)]
        r = ols_loglog(sub, min_cv=MIN_CV, min_r2=MIN_R2, max_beta=MAX_BETA)
        if r:
            betas_sem.append({
                "prod_nbr": sku,
                "prod_nm":  grp["prod_nm"].iloc[0],
                "mes_inicio": str(ventana[0]),
                "mes_fin":    str(ventana[-1]),
                "mes_fin_dt": ventana[-1].to_timestamp(),
                **r
            })
df_sem = pd.DataFrame(betas_sem)
print(f"  Betas semestrales validas: {len(df_sem):,}")

# ── 9. MODELO 2: Extendido con dummies (nivel SKU x Tienda x Mes) ─────────────
print("\n[MODELO 2] Extendido con tienda, mes, premium, margen...")

# Agregar por SKU x tienda x mes
agg2 = (
    carp.groupby(["prod_nbr","store_nbr","mes"])
    .agg(
        unidades  = ("qty",          "sum"),
        venta_tot = ("venta_con_iva","sum"),
        es_premium= ("es_premium",   "max"),
        margen    = ("margen",       "mean"),
        prod_nm   = ("prod_nm",      "first"),
        store_nm  = ("store_nm",     "first"),
        subdept_nm= ("subdept_nm",   "first"),
    )
    .reset_index()
)
agg2["precio"]   = agg2["venta_tot"] / agg2["unidades"]
agg2             = agg2[agg2["precio"] > 0].copy()
agg2["log_u1"]   = np.log1p(agg2["unidades"])
agg2["log_p"]    = np.log(agg2["precio"])
agg2["mes_str"]  = agg2["mes"].astype(str)
agg2["mes_dt"]   = agg2["mes"].dt.to_timestamp()

# --- One-Hot Encoding de tiendas ---
ohe_store = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
store_enc  = ohe_store.fit_transform(agg2[["store_nbr"]])
store_cols = [f"store_{c}" for c in ohe_store.categories_[0][1:]]
df_store_enc = pd.DataFrame(store_enc, columns=store_cols, index=agg2.index)

# --- One-Hot Encoding de mes ---
ohe_mes = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
mes_enc  = ohe_mes.fit_transform(agg2[["mes_str"]])
mes_cols = [f"mes_{c}" for c in ohe_mes.categories_[0][1:]]
df_mes_enc = pd.DataFrame(mes_enc, columns=mes_cols, index=agg2.index)

# --- One-Hot Encoding de subcategoria ---
ohe_sub = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
sub_enc  = ohe_sub.fit_transform(agg2[["subdept_nm"]].fillna("OTRO"))
sub_cols = [f"sub_{i}" for i in range(sub_enc.shape[1])]
df_sub_enc = pd.DataFrame(sub_enc, columns=sub_cols, index=agg2.index)

# Construir X del modelo 2
X2 = pd.concat([
    pd.DataFrame({"log_precio":  agg2["log_p"].values,
                  "es_premium":  agg2["es_premium"].values,
                  "margen":      agg2["margen"].fillna(0).values},
                 index=agg2.index),
    df_store_enc,
    df_mes_enc,
    df_sub_enc,
], axis=1)

X2 = add_constant(X2)
y2 = agg2["log_u1"].values
mask2 = np.isfinite(X2.values).all(axis=1) & np.isfinite(y2)
mod2  = OLS(y2[mask2], X2[mask2]).fit()

print(f"  N obs: {int(mod2.nobs):,}")
print(f"  R2: {mod2.rsquared:.4f}  |  R2 adj: {mod2.rsquared_adj:.4f}")
print(f"  Beta log_precio: {mod2.params['log_precio']:.4f}  (p={mod2.pvalues['log_precio']:.4f})")
print(f"  Coef premium:    {mod2.params['es_premium']:.4f}  (p={mod2.pvalues['es_premium']:.4f})")

# Tabla de coeficientes de tienda para Modelo 2
store_coef2 = pd.DataFrame({
    "store_nbr": ohe_store.categories_[0][1:],
    "coef":      mod2.params[store_cols].values.round(4),
    "pval":      mod2.pvalues[store_cols].values.round(4),
}).sort_values("coef", ascending=False)
store_names = agg2[["store_nbr","store_nm"]].drop_duplicates()
store_coef2 = store_coef2.merge(store_names, on="store_nbr", how="left")
store_coef2["sig"] = (store_coef2["pval"] < 0.05).map({True:"*",False:""})

# Tabla completa de coeficientes Modelo 2 (solo variables clave, no todas las dummies)
vars_clave = ["const","log_precio","es_premium","margen"]
coef_tabla2 = pd.DataFrame({
    "variable":    mod2.params.index,
    "coeficiente": mod2.params.values.round(4),
    "std_error":   mod2.bse.values.round(4),
    "t_stat":      mod2.tvalues.values.round(3),
    "p_value":     mod2.pvalues.values.round(4),
    "significativo": (mod2.pvalues.values < 0.05),
})

# ── 10. Resumen por SKU (para exportar) ───────────────────────────────────────
sku_info = (
    carp.groupby(["prod_nbr","prod_nm","subdept_nm","tipo_marca"])
    .agg(unidades_tot=("qty","sum"), precio_prom=("precio_tx","mean"),
         margen_prom=("margen","mean"), n_tiendas=("store_nbr","nunique"),
         n_meses=("mes","nunique"), es_premium=("es_premium","max"))
    .reset_index()
    .sort_values("unidades_tot", ascending=False)
)

# Unir betas del modelo mensual al resumen
resumen_sku = sku_info.merge(
    df_mensual[["prod_nbr","beta","r2","pval","n_meses"]].rename(
        columns={"beta":"beta_mensual","r2":"r2_mensual",
                 "pval":"pval_mensual","n_meses":"n_meses_reg"}),
    on="prod_nbr", how="left")

# ── 11. Guardar Excel ─────────────────────────────────────────────────────────
print("\nGuardando Excel...")

# Metricas Modelo 2
metricas2 = pd.DataFrame({
    "metrica": ["N obs","R2","R2 adj","F-stat",
                "Beta log_precio","p-val precio",
                "Coef premium","p-val premium",
                "Coef margen","p-val margen"],
    "valor":   [int(mod2.nobs), round(mod2.rsquared,4), round(mod2.rsquared_adj,4),
                round(mod2.fvalue,2),
                round(mod2.params["log_precio"],4), round(mod2.pvalues["log_precio"],4),
                round(mod2.params["es_premium"],4), round(mod2.pvalues["es_premium"],4),
                round(mod2.params["margen"],4),      round(mod2.pvalues["margen"],4)]
})

df_trim_out = df_trim.copy()
df_sem_out  = df_sem.copy()

with pd.ExcelWriter(OUT_XL, engine="openpyxl") as w:
    resumen_sku.to_excel(    w, sheet_name="Resumen_SKU",          index=False)
    df_mensual.to_excel(     w, sheet_name="M1_Mensual",           index=False)
    df_trim_out.to_excel(    w, sheet_name="M1_Trimestral_Rolling",index=False)
    df_sem_out.to_excel(     w, sheet_name="M1_Semestral_Rolling", index=False)
    metricas2.to_excel(      w, sheet_name="M2_Metricas",          index=False)
    coef_tabla2.to_excel(    w, sheet_name="M2_Coeficientes",      index=False)
    store_coef2[["store_nbr","store_nm","coef","pval","sig"]].to_excel(
                             w, sheet_name="M2_Coef_Tiendas",      index=False)

print(f"  Guardado: {OUT_XL}")

# ── 12. GRAFICAS ──────────────────────────────────────────────────────────────
print("Generando graficas...")

with pdf_backend.PdfPages(OUT_PDF) as pdf:

    # ── PORTADA ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    txt = (
        "ANALISIS DE ELASTICIDAD PRECIO — OFFICEMAX\n"
        f"Categoria: {CATEGORIA}\n\n"
        "MODELO 1 (simple): log(unidades+1) = alpha + beta * log(precio_real)\n"
        f"  1A Mensual    : {len(df_mensual)} SKUs con beta valida\n"
        f"  1B Trimestral : {len(df_trim):,} betas (ventana 3m, paso 1m)\n"
        f"  1C Semestral  : {len(df_sem):,} betas (ventana 6m, paso 1m)\n\n"
        "MODELO 2 (extendido): + tienda + mes + premium + promo + margen\n"
        f"  N obs         : {int(mod2.nobs):,}\n"
        f"  R2            : {mod2.rsquared:.4f}\n"
        f"  Beta precio   : {mod2.params['log_precio']:.4f}  "
        f"(p={mod2.pvalues['log_precio']:.4f})"
    )
    ax.text(0.08, 0.88, txt, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))
    ax.set_title(f"Elasticidad Precio — OfficeMax {CATEGORIA}", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ── 1A: Distribucion de betas (mensual) ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Modelo 1A — Beta Mensual por SKU (todos los meses)", fontsize=13, fontweight="bold")

    betas_vals = df_mensual["beta"].dropna()
    axes[0].hist(betas_vals, bins=30, color="#5C6BC0", edgecolor="white", alpha=0.85)
    axes[0].axvline(-1, color="red",    linestyle="--", lw=1.5, label="beta=-1 (unitario)")
    axes[0].axvline( 0, color="orange", linestyle="--", lw=1.2, label="beta=0")
    axes[0].set_xlabel("Beta (elasticidad)")
    axes[0].set_ylabel("N de SKUs")
    axes[0].set_title(f"Distribucion de betas — {CATEGORIA}")
    axes[0].legend()

    top_sku = df_mensual.nlargest(15, "n_meses").sort_values("beta")
    colors  = ["#E53935" if b < -1 else "#43A047" if b < 0 else "#FF9800"
               for b in top_sku["beta"]]
    ax2 = axes[1]
    ax2.barh(range(len(top_sku)), top_sku["beta"], color=colors, edgecolor="white")
    ax2.set_yticks(range(len(top_sku)))
    ax2.set_yticklabels(top_sku["prod_nm"].str[:35], fontsize=7)
    ax2.axvline(-1, color="red",  linestyle="--", lw=1, alpha=0.5)
    ax2.axvline( 0, color="gray", linestyle="--", lw=0.8, alpha=0.4)
    ax2.set_xlabel("Beta")
    ax2.set_title("Top 15 SKUs (mas meses de datos)")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ── 1B y 1C: Evolucion temporal de beta por SKU ───────────────────────────
    # Elegir top SKUs por volumen que tengan betas en ambas ventanas
    top_skus_vol = resumen_sku.head(12)["prod_nbr"].tolist()
    skus_plot = [s for s in top_skus_vol
                 if s in df_trim["prod_nbr"].values or s in df_sem["prod_nbr"].values][:9]

    for page_start in range(0, len(skus_plot), 6):
        batch = skus_plot[page_start: page_start+6]
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle("Modelos 1B y 1C — Evolucion de Beta Trimestral y Semestral",
                     fontsize=13, fontweight="bold")
        axes_flat = axes.flatten()

        for idx, sku in enumerate(batch):
            ax = axes_flat[idx]
            d3 = df_trim[df_trim["prod_nbr"]==sku].sort_values("mes_fin_dt")
            d6 = df_sem[df_sem["prod_nbr"]==sku].sort_values("mes_fin_dt")
            nm = resumen_sku.loc[resumen_sku["prod_nbr"]==sku,"prod_nm"]
            nombre = str(nm.iloc[0])[:40] if len(nm)>0 else sku

            if len(d3) > 0:
                ax.plot(d3["mes_fin_dt"], d3["beta"], marker="o", ms=4,
                        lw=1.8, color="#1976D2", label="Trimestral (3m)")
            if len(d6) > 0:
                ax.plot(d6["mes_fin_dt"], d6["beta"], marker="s", ms=4,
                        lw=1.8, color="#E53935", label="Semestral (6m)")

            # Beta mensual como referencia horizontal
            b_men = df_mensual.loc[df_mensual["prod_nbr"]==sku,"beta"]
            if len(b_men) > 0:
                ax.axhline(b_men.iloc[0], color="green", linestyle=":", lw=1.5,
                           alpha=0.8, label=f"Mensual={b_men.iloc[0]:.2f}")

            ax.axhline(-1, color="gray", linestyle="--", lw=0.9, alpha=0.5)
            ax.axhline( 0, color="gray", linestyle="--", lw=0.6, alpha=0.4)
            ax.set_title(f"{sku} | {nombre}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Mes fin de ventana", fontsize=7)
            ax.set_ylabel("Beta", fontsize=8)
            ax.tick_params(axis="x", rotation=40, labelsize=6)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)

        for idx in range(len(batch), 6):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ── MODELO 2: Resultados ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    vars_show = ["const","log_precio","es_premium","margen"]
    filas = []
    for v in vars_show:
        if v in mod2.params.index:
            sig = "***" if mod2.pvalues[v]<0.01 else ("**" if mod2.pvalues[v]<0.05
                  else ("*" if mod2.pvalues[v]<0.10 else ""))
            filas.append([v, f"{mod2.params[v]:.4f}", f"{mod2.bse[v]:.4f}",
                          f"{mod2.tvalues[v]:.3f}", f"{mod2.pvalues[v]:.4f}", sig])
    tabla = ax.table(
        cellText=filas,
        colLabels=["Variable","Coef","Std Error","t-stat","p-value","Sig"],
        loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1, 2)
    ax.set_title(
        f"Modelo 2 — Coeficientes clave\n"
        f"N={int(mod2.nobs):,}  R2={mod2.rsquared:.4f}  R2adj={mod2.rsquared_adj:.4f}  "
        f"(+ {len(store_cols)} dummies tienda, {len(mes_cols)} dummies mes, {len(sub_cols)} dummies subcategoria)",
        fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ── MODELO 2: Coeficientes de tienda ─────────────────────────────────────
    n_show = min(40, len(store_coef2))
    top_h  = store_coef2.head(n_show//2)
    bot_h  = store_coef2.tail(n_show//2)
    show   = pd.concat([top_h, bot_h]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(16, 10))
    colors  = ["#43A047" if c >= 0 else "#E53935" for c in show["coef"]]
    labels  = show.apply(lambda r: f"{r['store_nbr']} {str(r['store_nm']).strip()[:20]}", axis=1)
    ax.barh(range(len(show)), show["coef"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente tienda (vs tienda base T001)")
    ax.set_title(f"Modelo 2 — Contribucion de Tienda a la Demanda de {CATEGORIA}\n"
                 "Verde = mas demanda que base | Rojo = menos demanda",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ── Comparativa Premium vs No Premium ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Premium vs No Premium — {CATEGORIA}", fontsize=13, fontweight="bold")

    prem    = resumen_sku[resumen_sku["es_premium"]==1]
    no_prem = resumen_sku[resumen_sku["es_premium"]==0]

    for i, (col, tit) in enumerate([("precio_prom","Precio Promedio"),
                                     ("margen_prom","Margen Promedio")]):
        axes[i].boxplot([no_prem[col].dropna(), prem[col].dropna()],
                        labels=["No Premium","Premium"], patch_artist=True,
                        boxprops=dict(facecolor="#90CAF9"),
                        medianprops=dict(color="red", lw=2))
        axes[i].set_title(tit, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()

print(f"  Guardado: {OUT_PDF}")

print("\n========== RESUMEN FINAL ==========")
print(f"Categoria: {CATEGORIA}")
print(f"\n[M1A] Mensual — beta por SKU (todos los meses):")
print(df_mensual["beta"].describe().round(3).to_string())
print(f"\n[M1B] Trimestral rolling — beta por SKU por ventana:")
print(df_trim["beta"].describe().round(3).to_string())
print(f"\n[M1C] Semestral rolling — beta por SKU por ventana:")
print(df_sem["beta"].describe().round(3).to_string())
print(f"\n[M2] Modelo extendido:")
print(f"  Beta precio   : {mod2.params['log_precio']:.4f}  (p={mod2.pvalues['log_precio']:.4f})")
print(f"  Coef premium  : {mod2.params['es_premium']:.4f}  (p={mod2.pvalues['es_premium']:.4f})")
print(f"  Coef margen   : {mod2.params['margen']:.4f}      (p={mod2.pvalues['margen']:.4f})")
print(f"  R2 / R2 adj   : {mod2.rsquared:.4f} / {mod2.rsquared_adj:.4f}")

# ── SIMULADOR DE ELASTICIDAD ──────────────────────────────────────────────────
print("\n========== SIMULADOR DE ELASTICIDAD ==========")

# Nombres exactos tal como aparecen en prod_nm
NOMBRES_SIM = [
    "IMANES STUK SUMMER 6PZ",
    "ENGRAPADORA BYARRILITO M/T ACME",
    "CINTA SCOTCH MAGICA 18X33",
    "PERFORADORA MAPED 12HJS 2/O",
]

# Sacar prod_nbr y beta directamente de df_mensual con match exacto
sim_base = df_mensual[df_mensual["prod_nm"].isin(NOMBRES_SIM)][
    ["prod_nbr","prod_nm","beta"]].drop_duplicates("prod_nm")

# Unir precio y unidades base: promedio de todos los meses del SKU
base_stats = (mensual.groupby("prod_nbr")
              .agg(precio_base=("precio","mean"), unidades_base=("unidades","mean"))
              .reset_index())
sim_base = sim_base.merge(base_stats, on="prod_nbr", how="left")

# Costo unitario desde ventas (promedio real por transaccion)
costos_raw = pd.read_csv(find(BASE, "Costos_de_Productos.csv"), encoding="latin-1")
costos_raw["prod_nbr"] = costos_raw["prod_nbr"].astype(str).str.strip()
carp_costos = carp.copy()
carp_costos["qty"]   = pd.to_numeric(carp_costos["qty"],   errors="coerce")
carp_costos["costo"] = pd.to_numeric(carp_costos["margen"] * carp_costos["venta_con_iva"] * -1 +
                                     carp_costos["venta_con_iva"], errors="coerce")
# Usar costo directo de ventas: costo_unit = costo / qty
ventas_costo = pd.read_csv(find(BASE, "Ventas_OfficeMax.csv"), encoding="latin-1", low_memory=False)
ventas_costo["prod_nbr"]   = ventas_costo["prod_nbr"].astype(str).str.strip()
ventas_costo["qty"]        = pd.to_numeric(ventas_costo["qty"],   errors="coerce")
ventas_costo["costo"]      = pd.to_numeric(ventas_costo["costo"], errors="coerce")
ventas_costo               = ventas_costo[(ventas_costo["qty"]>0)&(ventas_costo["costo"]>0)]
ventas_costo["costo_unit"] = ventas_costo["costo"] / ventas_costo["qty"]
costo_prom = ventas_costo.groupby("prod_nbr")["costo_unit"].mean().reset_index()
sim_base = sim_base.merge(costo_prom, on="prod_nbr", how="left")
print("SKUs en simulacion:")
print(sim_base[["prod_nm","beta","precio_base","unidades_base"]].to_string(index=False))

CAMBIOS   = [-0.10, -0.05, 0.00, 0.05, 0.10]
ETIQUETAS = ["-10%", "-5%", "0% (Base)", "+5%", "+10%"]
COLORES   = ["#2E7D32", "#81C784", "#9E9E9E", "#EF9A9A", "#C62828"]

sim_rows = []
for _, row in sim_base.iterrows():
    p0   = row["precio_base"]
    u0   = row["unidades_base"]
    beta = row["beta"]
    c0   = row["costo_unit"] if pd.notna(row["costo_unit"]) else 0
    rev0    = p0 * u0
    margen0 = (p0 - c0) * u0
    for cambio, etiq in zip(CAMBIOS, ETIQUETAS):
        p1      = p0 * (1 + cambio)
        u1      = u0 * (1 + cambio) ** beta
        ing1    = p1 * u1
        marg1   = (p1 - c0) * u1
        sim_rows.append({
            "sku":              row["prod_nm"],
            "beta":             round(beta, 4),
            "costo_unitario":   round(c0, 2),
            "precio_base":      round(p0, 2),
            "unidades_base":    round(u0, 1),
            "ingreso_base":     round(rev0, 2),
            "margen_base":      round(margen0, 2),
            "cambio_precio":    etiq,
            "precio_nuevo":     round(p1, 2),
            "unidades_sim":     round(u1, 1),
            "ingreso_simulado": round(ing1, 2),
            "margen_simulado":  round(marg1, 2),
            "delta_unidades":   round(u1 - u0, 1),
            "delta_ingreso":    round(ing1 - rev0, 2),
            "delta_ingreso_pct":round((ing1 - rev0) / rev0 * 100, 1),
            "delta_margen":     round(marg1 - margen0, 2),
            "delta_margen_pct": round((marg1 - margen0) / margen0 * 100, 1) if margen0 != 0 else 0,
        })

df_sim = pd.DataFrame(sim_rows)
print()
print(df_sim[["sku","cambio_precio","precio_nuevo","unidades_sim",
              "delta_unidades","ingreso_simulado","delta_ingreso_pct",
              "margen_simulado","delta_margen_pct"]].to_string(index=False))

# ── Guardar simulacion en Excel ───────────────────────────────────────────────
OUT_SIM = os.path.join(DESKTOP, f"Simulacion_Elasticidad_{CATEGORIA}.xlsx")
with pd.ExcelWriter(OUT_SIM, engine="openpyxl") as w:
    df_sim.to_excel(w, sheet_name="Simulacion", index=False)
print(f"\n  Guardado: {OUT_SIM}")

# ── Grafica 1: barras de Δ revenue por escenario ──────────────────────────────
nombres_sim = sim_base["prod_nm"].tolist()
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f"Simulador de Elasticidad — {CATEGORIA}\n"
             "Cambio en revenue ante variacion de precio (-10%, -5%, 0%, +5%, +10%)",
             fontsize=13, fontweight="bold")

for idx, nombre in enumerate(nombres_sim):
    ax  = axes[idx // 2][idx % 2]
    sub = df_sim[df_sim["sku"] == nombre]
    p0  = sub["precio_base"].iloc[0]
    u0  = sub["unidades_base"].iloc[0]
    beta = sub["beta"].iloc[0]

    x    = np.arange(len(CAMBIOS))
    x2    = np.arange(len(CAMBIOS)) * 1.6
    width = 0.65
    bars_ing  = ax.bar(x2 - width/2, sub["delta_ingreso_pct"].values,
                       width=width, color=COLORES, edgecolor="white", alpha=0.9, zorder=3, label="Ingreso")
    bars_marg = ax.bar(x2 + width/2, sub["delta_margen_pct"].values,
                       width=width, color=COLORES, edgecolor="white", alpha=0.5, zorder=3, label="Margen")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x2); ax.set_xticklabels(ETIQUETAS, fontsize=10)
    ax.set_ylabel("Delta (%)", fontsize=9)
    c0  = sub["costo_unitario"].iloc[0]
    ax.set_title(f"{nombre}\nbeta={beta:.2f}  |  precio=${p0:.0f}  costo=${c0:.0f}  ({u0:.1f} u/mes)",
                 fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=8)

    for bar, row in zip(bars_ing, sub.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (0.3 if h >= 0 else -1.0),
                f"{row.delta_ingreso_pct:+.1f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
OUT_SIM_PDF = os.path.join(DESKTOP, f"Simulacion_Elasticidad_{CATEGORIA}.pdf")
plt.savefig(OUT_SIM_PDF, bbox_inches="tight"); plt.show()
print(f"  Guardado: {OUT_SIM_PDF}")

# ── Grafica 2: curva continua precio vs revenue (-30% a +30%) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f"Curva de Revenue segun cambio de precio — {CATEGORIA}",
             fontsize=13, fontweight="bold")

cambios_cont = np.linspace(-0.15, 0.15, 300)

for idx, nombre in enumerate(nombres_sim):
    ax   = axes[idx // 2][idx % 2]
    sub  = df_sim[df_sim["sku"] == nombre]
    p0   = sub["precio_base"].iloc[0]
    u0   = sub["unidades_base"].iloc[0]
    beta = sub["beta"].iloc[0]
    rev0 = p0 * u0

    pct_curva = [(p0*(1+c) * u0*(1+c)**beta - rev0)/rev0*100 for c in cambios_cont]

    ax.plot(cambios_cont*100, pct_curva, color="#1976D2", lw=2.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.fill_between(cambios_cont*100, pct_curva, 0,
                    where=[p >= 0 for p in pct_curva], alpha=0.12, color="#43A047")
    ax.fill_between(cambios_cont*100, pct_curva, 0,
                    where=[p < 0 for p in pct_curva], alpha=0.12, color="#E53935")

    c0 = sim_base.loc[sim_base["prod_nm"]==nombre, "costo_unit"].iloc[0]
    marg_curva = [((p0*(1+c) - c0) * u0*(1+c)**beta - (p0-c0)*u0) / ((p0-c0)*u0)*100
                  for c in cambios_cont]
    ax.plot(cambios_cont*100, marg_curva, color="#E53935", lw=1.8, linestyle="--", label="Margen")

    for cambio, etiq, col in zip(CAMBIOS, ETIQUETAS, COLORES):
        dp  = (p0*(1+cambio) * u0*(1+cambio)**beta - rev0)/rev0*100
        dm  = ((p0*(1+cambio) - c0) * u0*(1+cambio)**beta - (p0-c0)*u0) / ((p0-c0)*u0)*100
        ax.scatter(cambio*100, dp,  color=col, s=90, zorder=5)
        ax.scatter(cambio*100, dm,  color=col, s=50, zorder=5, marker="D")
        ax.annotate(f"{etiq}\ning:{dp:+.1f}%\nmrg:{dm:+.1f}%", (cambio*100, dp),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=col, fontweight="bold")

    ax.set_xlabel("Cambio de precio (%)", fontsize=9)
    ax.set_ylabel("Delta (%)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(f"{nombre}  |  beta={beta:.2f}", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)

plt.tight_layout()
OUT_CURVAS = OUT_SIM_PDF.replace(".pdf", "_curvas.pdf")
plt.savefig(OUT_CURVAS, bbox_inches="tight"); plt.show()
print(f"  Guardado: {OUT_CURVAS}")

# ══════════════════════════════════════════════════════════════════════════════
# RECOMENDACIONES FINALES POR SKU
# ══════════════════════════════════════════════════════════════════════════════
print("\n========== RECOMENDACIONES FINALES ==========")

# Base: todos los SKUs con beta M1A
rec = df_mensual.merge(
    resumen_sku[["prod_nbr","unidades_tot","precio_prom","margen_prom","n_tiendas"]],
    on="prod_nbr", how="left")

def clasificar(row):
    b, p, n = row["beta"], row["pval"], row["n_meses"]
    if p >= 0.10 or n < 6 or abs(b) > 3 or b > 0:
        return "NO RECOMENDABLE"
    if -1 < b < 0:
        return "SUBIR PRECIO"
    if -1.5 <= b <= -1:
        return "MANTENER PRECIO"
    return "BAJAR / PROMOVER"

rec["recomendacion"] = rec.apply(clasificar, axis=1)

# Razon corta por SKU
def razon(row):
    b, p, n = row["beta"], row["pval"], row["n_meses"]
    cat = row["recomendacion"]
    if cat == "NO RECOMENDABLE":
        if p >= 0.10:    return f"Beta no significativa (p={p:.2f})"
        if n < 6:        return f"Solo {n} meses de datos"
        if b > 0:        return "Beta positiva (anomalia)"
        return f"|beta|={abs(b):.1f} fuera de rango creible"
    if cat == "SUBIR PRECIO":
        return f"Inelastica (beta={b:.2f}): subir 10% precio ~ +{abs(b+1)*10:.1f}% margen"
    if cat == "MANTENER PRECIO":
        return f"Cerca de unitaria (beta={b:.2f}): cambios de precio casi no mueven revenue"
    return f"Elastica (beta={b:.2f}): bajar precio 10% ~ +{abs(b)*10:.1f}% unidades"

rec["razon"] = rec.apply(razon, axis=1)

orden_cat = {"SUBIR PRECIO": 0, "MANTENER PRECIO": 1,
             "BAJAR / PROMOVER": 2, "NO RECOMENDABLE": 3}
rec["_ord"] = rec["recomendacion"].map(orden_cat)
rec = rec.sort_values(["_ord","beta"], ascending=[True, False]).drop("_ord", axis=1)

cols_rec = ["prod_nm","subdept_nm","beta","r2","pval","n_meses",
            "unidades_tot","precio_prom","recomendacion","razon"]

for cat in ["SUBIR PRECIO","MANTENER PRECIO","BAJAR / PROMOVER","NO RECOMENDABLE"]:
    sub = rec[rec["recomendacion"] == cat]
    print(f"\n--- {cat} ({len(sub)} SKUs) ---")
    if cat != "NO RECOMENDABLE":
        print(sub[cols_rec].to_string(index=False))
    else:
        print(sub[["prod_nm","beta","pval","n_meses","razon"]].to_string(index=False))

# ── Guardar en Excel (nueva hoja en el archivo principal) ─────────────────────
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    rec[cols_rec].to_excel(w, sheet_name="Recomendaciones", index=False)
print(f"\n  Hoja 'Recomendaciones' agregada a: {OUT_XL}")

# ── Grafica resumen de recomendaciones ────────────────────────────────────────
COLORES_CAT = {
    "SUBIR PRECIO":    "#1565C0",
    "MANTENER PRECIO": "#F9A825",
    "BAJAR / PROMOVER":"#2E7D32",
    "NO RECOMENDABLE": "#BDBDBD",
}

validos = rec[rec["recomendacion"] != "NO RECOMENDABLE"].copy()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f"Recomendaciones de Precio por SKU — {CATEGORIA}\n"
             "Basado en elasticidad M1A (β significativa, 6+ meses)",
             fontsize=13, fontweight="bold")

# Panel izq: scatter beta vs unidades, coloreado por recomendacion
ax = axes[0]
for cat, color in COLORES_CAT.items():
    sub = rec[rec["recomendacion"] == cat]
    ax.scatter(sub["beta"], sub["unidades_tot"],
               color=color, alpha=0.75, s=70, label=cat, edgecolors="white", zorder=3)
ax.axvline(-1, color="gray", lw=1, linestyle="--", alpha=0.6)
ax.axvline(-1.5, color="gray", lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Beta (elasticidad M1A)", fontsize=10)
ax.set_ylabel("Unidades totales vendidas", fontsize=10)
ax.set_title("Beta vs Volumen por SKU", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# Panel der: barras horizontales de los SKUs validos ordenados por beta
ax2 = axes[1]
validos_plot = validos.sort_values("beta")
colors_bar   = [COLORES_CAT[c] for c in validos_plot["recomendacion"]]
ax2.barh(range(len(validos_plot)), validos_plot["beta"],
         color=colors_bar, edgecolor="white", height=0.7)
ax2.axvline(-1,  color="gray", lw=1, linestyle="--", alpha=0.6, label="beta=-1")
ax2.axvline(-1.5,color="gray", lw=1, linestyle=":", alpha=0.5, label="beta=-1.5")
ax2.axvline(0,   color="black", lw=0.8)
ax2.set_yticks(range(len(validos_plot)))
ax2.set_yticklabels(validos_plot["prod_nm"].str[:40], fontsize=7)
ax2.set_xlabel("Beta (elasticidad)", fontsize=10)
ax2.set_title("SKUs validos ordenados por elasticidad", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
OUT_REC_PDF = os.path.join(DESKTOP, f"Recomendaciones_{CATEGORIA}.pdf")
plt.savefig(OUT_REC_PDF, bbox_inches="tight"); plt.show()
print(f"  Guardado: {OUT_REC_PDF}")

# ══════════════════════════════════════════════════════════════════════════════
# RECOMENDACIONES TEMPORALES (POR TRIMESTRE / SEMESTRE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n========== RECOMENDACIONES TEMPORALES ==========")
print("Logica: aunque todos los SKUs son elasticos (beta < -1),")
print("la elasticidad VARIA en el tiempo. Se identifica:")
print("  - Periodo MENOS elastico (beta max/menos negativo) -> subir precio")
print("  - Periodo MAS elastico  (beta min/mas negativo)    -> promover/descuento")
print()

# Solo SKUs del simulador (los mas relevantes con beta valida)
skus_temp = sim_base["prod_nbr"].tolist()

def clasificar_ventana(beta):
    if beta > -1.0:  return "Inelastica"
    if beta > -1.5:  return "Menos elastica"
    if beta > -2.5:  return "Elasticidad moderada"
    return "Muy elastica"

temp_rows = []
for _, sk_row in sim_base.iterrows():
    sku   = sk_row["prod_nbr"]
    nombre = sk_row["prod_nm"]
    b_base = sk_row["beta"]

    d3 = df_trim[df_trim["prod_nbr"] == sku].copy()
    d6 = df_sem[df_sem["prod_nbr"]  == sku].copy()

    for ventana_df, tipo in [(d3, "Trimestral"), (d6, "Semestral")]:
        if len(ventana_df) == 0:
            continue
        vdf = ventana_df.sort_values("beta")
        b_max = ventana_df["beta"].max()   # menos elastico (mas cerca de 0)
        b_min = ventana_df["beta"].min()   # mas elastico

        row_max = ventana_df.loc[ventana_df["beta"].idxmax()]
        row_min = ventana_df.loc[ventana_df["beta"].idxmin()]

        # Tendencia: comparar primera mitad vs segunda mitad
        mid = len(ventana_df) // 2
        vdf_sorted = ventana_df.sort_values("mes_fin_dt")
        tend_prim = vdf_sorted.iloc[:mid]["beta"].mean() if mid > 0 else b_base
        tend_seg  = vdf_sorted.iloc[mid:]["beta"].mean() if mid > 0 else b_base
        if tend_seg > tend_prim + 0.1:
            tendencia = "Menos elastico (mejora)"
        elif tend_seg < tend_prim - 0.1:
            tendencia = "Mas elastico (empeora)"
        else:
            tendencia = "Estable"

        recom_subir   = f"{row_max['mes_inicio']} a {row_max['mes_fin']}"
        recom_promover = f"{row_min['mes_inicio']} a {row_min['mes_fin']}"

        temp_rows.append({
            "SKU":              nombre,
            "ventana":          tipo,
            "n_ventanas":       len(ventana_df),
            "beta_promedio":    round(ventana_df["beta"].mean(), 3),
            "beta_max":         round(b_max, 3),
            "periodo_menos_elastico": recom_subir,
            "beta_min":         round(b_min, 3),
            "periodo_mas_elastico":   recom_promover,
            "tendencia":        tendencia,
            "rango_beta":       round(b_max - b_min, 3),
            "clasificacion_base": clasificar_ventana(b_base),
        })

df_temp = pd.DataFrame(temp_rows)
print(df_temp[["SKU","ventana","beta_promedio","beta_max","periodo_menos_elastico",
               "beta_min","periodo_mas_elastico","tendencia"]].to_string(index=False))

# Guardar hoja temporal en el Excel principal
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    df_temp.to_excel(w, sheet_name="Recom_Temporal", index=False)
print(f"\n  Hoja 'Recom_Temporal' agregada a: {OUT_XL}")

# ── Grafica temporal: evolucion de beta con periodos optimos marcados ──────────
fig, axes = plt.subplots(len(skus_temp), 1, figsize=(18, 5 * len(skus_temp)))
fig.suptitle(
    f"Recomendaciones Temporales — {CATEGORIA}\n"
    "Fondo azul = periodo menos elastico (subir precio)  |  Fondo verde = periodo mas elastico (promover/descuento)",
    fontsize=12, fontweight="bold")

if len(skus_temp) == 1:
    axes = [axes]

for idx, (_, sk_row) in enumerate(sim_base.iterrows()):
    sku    = sk_row["prod_nbr"]
    nombre = sk_row["prod_nm"]
    b_base = sk_row["beta"]
    ax     = axes[idx]

    d3 = df_trim[df_trim["prod_nbr"] == sku].sort_values("mes_fin_dt")
    d6 = df_sem[df_sem["prod_nbr"]  == sku].sort_values("mes_fin_dt")

    if len(d6) > 1:
        ax.plot(d6["mes_fin_dt"], d6["beta"], marker="s", ms=5,
                lw=2.2, color="#C62828", label="Semestral (6m)", zorder=4)
        # Sombrear ventana menos elastica (semestral)
        row_max6 = d6.loc[d6["beta"].idxmax()]
        ax.axvspan(pd.Period(row_max6["mes_inicio"]).to_timestamp(),
                   pd.Period(row_max6["mes_fin"]).to_timestamp(),
                   alpha=0.15, color="#1565C0", label=f"Menos elast.: {row_max6['mes_inicio']} a {row_max6['mes_fin']}")
        # Sombrear ventana mas elastica (semestral)
        row_min6 = d6.loc[d6["beta"].idxmin()]
        ax.axvspan(pd.Period(row_min6["mes_inicio"]).to_timestamp(),
                   pd.Period(row_min6["mes_fin"]).to_timestamp(),
                   alpha=0.13, color="#2E7D32", label=f"Mas elast.: {row_min6['mes_inicio']} a {row_min6['mes_fin']}")

    if len(d3) > 1:
        ax.plot(d3["mes_fin_dt"], d3["beta"], marker="o", ms=4,
                lw=1.6, color="#1976D2", alpha=0.7, label="Trimestral (3m)", zorder=3)

    # Beta mensual referencia
    ax.axhline(b_base, color="#43A047", linestyle=":", lw=1.8,
               label=f"Beta global={b_base:.2f}", alpha=0.9)
    ax.axhline(-1,  color="gray", linestyle="--", lw=0.9, alpha=0.5)
    ax.axhline(-1.5,color="gray", linestyle=":",  lw=0.7, alpha=0.4)
    ax.axhline(0,   color="black", linestyle="-", lw=0.5, alpha=0.3)

    ax.set_title(f"{nombre}  |  beta global={b_base:.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mes fin de ventana", fontsize=8)
    ax.set_ylabel("Beta", fontsize=9)
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.22)

plt.tight_layout()
OUT_TEMP_PDF = os.path.join(DESKTOP, f"Recom_Temporal_{CATEGORIA}.pdf")
plt.savefig(OUT_TEMP_PDF, bbox_inches="tight"); plt.show()
print(f"  Guardado: {OUT_TEMP_PDF}")

# ── CASOS INELASTICOS TEMPORALES: SKUs con beta en (-1,0) y p<0.10 ─────────────
print("\n--- CASOS INELASTICOS EN VENTANAS ROLLING (beta entre -1 y 0, p < 0.10) ---")

PVAL_CORTE = 0.10

inel_sem = df_sem[
    (df_sem["beta"] > -1) & (df_sem["beta"] < 0) & (df_sem["pval"] < PVAL_CORTE)
].copy()
inel_trim = df_trim[
    (df_trim["beta"] > -1) & (df_trim["beta"] < 0) & (df_trim["pval"] < PVAL_CORTE)
].copy()

if len(inel_sem) == 0 and len(inel_trim) == 0:
    print("  Ninguno encontrado con p < 0.10")
else:
    if len(inel_sem) > 0:
        print(f"\nSemestral — {len(inel_sem)} ventanas, {inel_sem['prod_nbr'].nunique()} SKUs:")
        print(inel_sem[["prod_nm","mes_inicio","mes_fin","beta","r2","pval"]].to_string(index=False))
    if len(inel_trim) > 0:
        print(f"\nTrimestral — {len(inel_trim)} ventanas, {inel_trim['prod_nbr'].nunique()} SKUs:")
        print(inel_trim[["prod_nm","mes_inicio","mes_fin","beta","r2","pval"]].to_string(index=False))

# Guardar tabla de casos inelasticos en Excel
inel_all = pd.concat([
    inel_sem.assign(ventana="Semestral"),
    inel_trim.assign(ventana="Trimestral"),
], ignore_index=True)

if len(inel_all) > 0:
    with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        inel_all[["prod_nm","ventana","mes_inicio","mes_fin",
                  "beta","r2","pval","n"]].to_excel(
            w, sheet_name="Casos_Inelasticos", index=False)
    print(f"\n  Hoja 'Casos_Inelasticos' agregada a: {OUT_XL}")

# ── Grafica detallada de cada SKU inelastico (semestral) ──────────────────────
skus_inel = inel_sem["prod_nbr"].unique().tolist()

if len(skus_inel) > 0:
    n_cols = min(2, len(skus_inel))
    n_rows = (len(skus_inel) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), squeeze=False)
    fig.suptitle(
        f"SKUs con Elasticidad INELASTICA en ventana semestral — {CATEGORIA}\n"
        "Sombreado naranja = periodos con beta en (-1, 0) y p < 0.10  (candidatos a subir precio)",
        fontsize=11, fontweight="bold")

    for idx, sku in enumerate(skus_inel):
        ax  = axes[idx // 2][idx % 2]
        d6  = df_sem[df_sem["prod_nbr"] == sku].sort_values("mes_fin_dt")
        d3  = df_trim[df_trim["prod_nbr"] == sku].sort_values("mes_fin_dt")
        nombre = d6["prod_nm"].iloc[0][:45] if len(d6) > 0 else sku

        # Beta mensual (global)
        b_global = df_mensual.loc[df_mensual["prod_nbr"] == sku, "beta"]
        b_ref    = b_global.iloc[0] if len(b_global) > 0 else np.nan

        if len(d6) > 0:
            ax.plot(d6["mes_fin_dt"], d6["beta"], marker="s", ms=5,
                    lw=2.2, color="#C62828", label="Semestral (6m)", zorder=4)

        if len(d3) > 1:
            ax.plot(d3["mes_fin_dt"], d3["beta"], marker="o", ms=3.5,
                    lw=1.5, color="#1976D2", alpha=0.65, label="Trimestral (3m)", zorder=3)

        # Sombrear ventanas inelasticas significativas
        inel_sku = inel_sem[inel_sem["prod_nbr"] == sku]
        for _, vrow in inel_sku.iterrows():
            ax.axvspan(pd.Period(vrow["mes_inicio"]).to_timestamp(),
                       pd.Period(vrow["mes_fin"]).to_timestamp(),
                       alpha=0.22, color="#FF6F00",
                       label=f"Inelastico p={vrow['pval']:.2f}")

        if not np.isnan(b_ref):
            ax.axhline(b_ref, color="#43A047", linestyle=":", lw=1.8,
                       label=f"Beta global={b_ref:.2f}", alpha=0.9)

        ax.axhline(-1, color="gray", linestyle="--", lw=1.0, alpha=0.6, label="beta=-1")
        ax.axhline(0,  color="black", linestyle="-",  lw=0.5, alpha=0.4)

        ax.set_title(f"{nombre}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mes fin de ventana", fontsize=8)
        ax.set_ylabel("Beta (elasticidad)", fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        # Deduplicar etiquetas de leyenda
        handles, labels = ax.get_legend_handles_labels()
        seen = set(); u_h = []; u_l = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); u_h.append(h); u_l.append(l)
        ax.legend(u_h, u_l, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.22)

    # Ocultar subplots vacios
    for idx in range(len(skus_inel), n_rows * n_cols):
        axes[idx // 2][idx % 2].set_visible(False)

    plt.tight_layout()
    OUT_INEL_PDF = os.path.join(DESKTOP, f"Casos_Inelasticos_{CATEGORIA}.pdf")
    plt.savefig(OUT_INEL_PDF, bbox_inches="tight"); plt.show()
    print(f"  Guardado: {OUT_INEL_PDF}")

# ══════════════════════════════════════════════════════════════════════════════
# CALENDARIO DE ACCION MENSUAL + HEATMAP
# Logica: cada ventana rolling "vota" los meses que cubre.
#   beta > -1.0            -> voto SUBIR    (inelastico)
#   -1.5 < beta <= -1.0    -> voto MANTENER
#   beta <= -1.5            -> voto PROMOVER
# Ventanas con p<0.10 pesan doble. Solo se acepta una accion si domina >50%.
# ══════════════════════════════════════════════════════════════════════════════
print("\n========== CALENDARIO DE ACCION MENSUAL POR SKU ==========")

MONTH_NAMES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
               7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
UMBRAL_SUBIR = -1.0
UMBRAL_MANT  = -1.5

def meses_ventana(inicio, fin):
    p = pd.Period(inicio, freq="M")
    e = pd.Period(fin,    freq="M")
    ms = []
    while p <= e:
        ms.append(p.month)
        p += 1
    return ms

# SKUs a analizar: todos con rolling valido (semestral o trimestral)
skus_con_rolling = sorted(
    set(df_sem["prod_nbr"].unique()) | set(df_trim["prod_nbr"].unique())
)

cal_rows = []
for sku in skus_con_rolling:
    nm = mensual.loc[mensual["prod_nbr"]==sku, "prod_nm"]
    nombre_sku = nm.iloc[0] if len(nm) > 0 else sku

    d_all = pd.concat([
        df_sem[df_sem["prod_nbr"]==sku].assign(tipo="sem"),
        df_trim[df_trim["prod_nbr"]==sku].assign(tipo="trim"),
    ], ignore_index=True)
    d_all = d_all[d_all["pval"] < 0.25]  # solo ventanas con algo de señal
    if len(d_all) == 0:
        continue

    mes_votos = {m: {"subir":0,"mantener":0,"promover":0,"n":0} for m in range(1,13)}
    for _, vrow in d_all.iterrows():
        b, p = vrow["beta"], vrow["pval"]
        peso = 2 if p < 0.10 else 1
        for m in meses_ventana(str(vrow["mes_inicio"]), str(vrow["mes_fin"])):
            mes_votos[m]["n"] += 1
            if b > UMBRAL_SUBIR:
                mes_votos[m]["subir"] += peso
            elif b > UMBRAL_MANT:
                mes_votos[m]["mantener"] += peso
            else:
                mes_votos[m]["promover"] += peso

    for mes in range(1, 13):
        v = mes_votos[mes]
        total = v["subir"] + v["mantener"] + v["promover"]
        if v["n"] == 0:
            accion = "Sin datos"
            score  = 0
        else:
            best = max(["subir","mantener","promover"], key=lambda k: v[k])
            dom  = v[best] / total if total > 0 else 0
            if dom < 0.50:
                accion = "Sin patron"
                score  = 0
            else:
                accion = {"subir":"SUBIR","mantener":"MANTENER","promover":"PROMOVER"}[best]
                score  = v[best] - max(v[k] for k in ["subir","mantener","promover"] if k != best)
        cal_rows.append({
            "prod_nbr":sku, "prod_nm":nombre_sku,
            "mes_num":mes, "mes_nm":MONTH_NAMES[mes],
            "votos_subir":v["subir"], "votos_mantener":v["mantener"],
            "votos_promover":v["promover"], "n_ventanas":v["n"],
            "accion":accion, "score":score,
        })

df_cal = pd.DataFrame(cal_rows)

# Guardar en Excel
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    df_cal.to_excel(w, sheet_name="Calendario_Accion", index=False)
print(f"  Hoja 'Calendario_Accion' agregada a: {OUT_XL}")

# Resumen texto por SKU (solo los que tienen algun patron)
skus_con_patron = df_cal[df_cal["accion"].isin(["SUBIR","MANTENER","PROMOVER"])]["prod_nbr"].unique()
for sku in skus_con_patron:
    rows = df_cal[df_cal["prod_nbr"]==sku].sort_values("mes_num")
    nombre_sku = rows["prod_nm"].iloc[0]
    meses_subir = rows[rows["accion"]=="SUBIR"]["mes_nm"].tolist()
    meses_prom  = rows[rows["accion"]=="PROMOVER"]["mes_nm"].tolist()
    meses_mant  = rows[rows["accion"]=="MANTENER"]["mes_nm"].tolist()
    print(f"\n{nombre_sku[:55]}")
    if meses_subir:
        print(f"  SUBIR PRECIO  : {', '.join(meses_subir)}")
    if meses_mant:
        print(f"  MANTENER      : {', '.join(meses_mant)}")
    if meses_prom:
        print(f"  PROMOVER      : {', '.join(meses_prom)}")

# ── HEATMAP: SKU x Mes, color = accion recomendada ───────────────────────────
ACCION_NUM = {"SUBIR":2, "MANTENER":1, "PROMOVER":-1, "Sin patron":0, "Sin datos":np.nan}
CMAP_CUSTOM = plt.cm.colors.ListedColormap(["#B3C6FF","#1565C0","#F9A825","#81C784","#2E7D32"])

# Solo SKUs con al menos 3 meses con patron definido
skus_heat = [s for s in skus_con_rolling
             if df_cal[(df_cal["prod_nbr"]==s) &
                        df_cal["accion"].isin(["SUBIR","MANTENER","PROMOVER"])].shape[0] >= 3]

if len(skus_heat) > 0:
    pivot = (df_cal[df_cal["prod_nbr"].isin(skus_heat)]
             .pivot(index="prod_nm", columns="mes_num", values="accion")
             .reindex(columns=range(1,13)))
    pivot_num = pivot.map(lambda x: ACCION_NUM.get(x, 0) if pd.notna(x) else 0)

    fig, ax = plt.subplots(figsize=(16, max(6, len(skus_heat)*0.55 + 2)))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "cal", ["#2E7D32","#81C784","#F9A825","#90CAF9","#1565C0"], N=256)
    im = ax.imshow(pivot_num.values.astype(float), aspect="auto",
                   cmap=cmap, vmin=-1, vmax=2, interpolation="nearest")

    ax.set_xticks(range(12))
    ax.set_xticklabels([MONTH_NAMES[m] for m in range(1,13)], fontsize=9)
    ax.set_yticks(range(len(pivot_num)))
    ax.set_yticklabels(pivot_num.index.str[:45], fontsize=7)

    # Texto en cada celda
    for i in range(len(pivot_num)):
        for j in range(12):
            accion = pivot.iloc[i, j] if pd.notna(pivot.iloc[i, j]) else ""
            etiq   = {"SUBIR":"S","MANTENER":"M","PROMOVER":"P"}.get(accion, "")
            if etiq:
                ax.text(j, i, etiq, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#1565C0", label="SUBIR PRECIO (inelastico)"),
        Patch(facecolor="#F9A825", label="MANTENER"),
        Patch(facecolor="#2E7D32", label="PROMOVER / DESCUENTO"),
        Patch(facecolor="#90CAF9", label="Sin patron claro"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(1.28, 1),
              fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Heatmap Calendario de Accion — {CATEGORIA}\n"
        "S=Subir precio | M=Mantener | P=Promover  (basado en ventanas rolling con p<0.25)",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Mes del año", fontsize=10)
    plt.tight_layout()
    OUT_HEAT = os.path.join(DESKTOP, f"Heatmap_Calendario_{CATEGORIA}.pdf")
    plt.savefig(OUT_HEAT, bbox_inches="tight"); plt.show()
    print(f"  Guardado: {OUT_HEAT}")

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSION EJECUTIVA (LLM via Ollama/Qwen o template)
# ══════════════════════════════════════════════════════════════════════════════
print("\n========== CONCLUSION EJECUTIVA ==========")

import requests as _http

n_skus_total   = mensual["prod_nbr"].nunique()
n_skus_validos = len(df_mensual)
n_subir_g = len(rec[rec["recomendacion"]=="SUBIR PRECIO"])
n_mant_g  = len(rec[rec["recomendacion"]=="MANTENER PRECIO"])
n_bajar_g = len(rec[rec["recomendacion"]=="BAJAR / PROMOVER"])
n_norec_g = len(rec[rec["recomendacion"]=="NO RECOMENDABLE"])
n_inel_skus = int(inel_sem["prod_nbr"].nunique()) if len(inel_sem) > 0 else 0

meses_data = carp["mes"].nunique()
cv_prom    = round(mensual.groupby("prod_nbr")["precio"].std().div(
                   mensual.groupby("prod_nbr")["precio"].mean()).mean() * 100, 1)

# Mejores candidatos
if len(inel_sem) > 0:
    cand_subir = inel_sem.sort_values("pval").iloc[0]["prod_nm"]
    cand_subir_mes = f"{inel_sem.sort_values('pval').iloc[0]['mes_inicio']} a {inel_sem.sort_values('pval').iloc[0]['mes_fin']}"
else:
    cand_subir     = "ninguno con beta inelastica estadisticamente significativa"
    cand_subir_mes = "N/A"

valid_promo = rec[(rec["recomendacion"]=="BAJAR / PROMOVER") & (rec["pval"]<0.10)]
if len(valid_promo) > 0:
    cand_promo      = valid_promo.sort_values("beta").iloc[0]["prod_nm"]
    cand_promo_beta = round(valid_promo.sort_values("beta").iloc[0]["beta"], 2)
else:
    cand_promo      = "ver hoja Recomendaciones"
    cand_promo_beta = np.nan

prompt_llm = f"""Eres un analista de revenue management para OfficeMax Mexico.
Escribe una conclusion ejecutiva CONCISA (maximo 180 palabras, 3 parrafos) sobre el analisis de elasticidad.

Datos del analisis:
- Categoria: {CATEGORIA}
- SKUs analizados: {n_skus_total} totales, {n_skus_validos} con modelo valido
- Periodo: {meses_data} meses de datos de ventas
- Escenarios simulados: -10%, -5%, 0% (Base), +5%, +10% de cambio de precio
- Recomendaciones globales: {n_subir_g + n_inel_skus} candidatos a subir precio, {n_bajar_g} a promover, {n_norec_g} no recomendables
- Mejor candidato para subir precio: {cand_subir} (inelastico en {cand_subir_mes})
- Mejor candidato para promover: {cand_promo} (beta={cand_promo_beta})
- Coeficiente de variacion promedio de precio: {cv_prom}%
- Limitaciones: datos posiblemente sinteticos, variacion de precio baja en muchos SKUs

Formato obligatorio (exactamente 3 parrafos, en espanol, sin titulos):
1. "En el departamento {CATEGORIA}, simulamos..." [contexto y escala del analisis]
2. "Encontramos que..." [hallazgos clave: SKUs para subir/promover, patron temporal si hay]
3. "Los resultados son exploratorios porque..." [2 limitaciones concretas del modelo]
"""

def llamar_ollama(prompt_text, timeout=90):
    for model in ["qwen2:latest","qwen2","qwen2.5","llama3.2","llama3","mistral"]:
        try:
            r = _http.post("http://localhost:11434/api/generate",
                           json={"model":model,"prompt":prompt_text,"stream":False},
                           timeout=timeout)
            if r.status_code == 200:
                txt = r.json().get("response","").strip()
                if len(txt) > 80:
                    print(f"  [modelo: Ollama/{model}]")
                    return txt
        except Exception:
            break
    return None

conclusion = llamar_ollama(prompt_llm)

if conclusion is None:
    print("  [Ollama no disponible - usando template]")
    cav1 = (f"la variacion de precio observada es limitada (CV promedio {cv_prom}% entre SKUs), "
            f"lo que reduce la precision de los coeficientes beta")
    cav2 = (f"el modelo asume una relacion log-lineal estable en el tiempo y no separa "
            f"efectos de temporada, tienda o competencia")
    subir_txt = (f"{cand_subir} muestra elasticidad inelastica en {cand_subir_mes}"
                 if n_inel_skus > 0 else "no se identificaron SKUs con beta inelastica global significativa")
    promo_txt = (f"{cand_promo} con beta={cand_promo_beta}"
                 if not np.isnan(cand_promo_beta) else "ver hoja Recomendaciones para candidatos a promover")
    conclusion = (
        f"En el departamento {CATEGORIA}, simulamos cambios de precio de -10% a +10% (5 escenarios) "
        f"en {len(sim_base)} SKUs seleccionados, usando regresion OLS log-log sobre "
        f"{n_skus_validos} productos validos de {n_skus_total} totales en {meses_data} meses de datos.\n\n"
        f"Encontramos que {n_subir_g + n_inel_skus} productos podrian tolerar una subida de precio "
        f"({subir_txt}), {n_bajar_g} productos son mas adecuados para promocion o descuento "
        f"({promo_txt}), y {n_norec_g} productos no deben recomendarse por baja significancia "
        f"estadistica o elasticidades fuera de rango creible.\n\n"
        f"Los resultados son exploratorios porque {cav1}; ademas, {cav2}. "
        f"Se recomienda validar con pruebas de precio controladas antes de implementar cambios."
    )

print(conclusion)

# Guardar conclusion en Excel y PDF
df_concl = pd.DataFrame({
    "seccion": ["Conclusion ejecutiva"],
    "texto":   [conclusion],
    "generado_por": ["Ollama/Qwen" if llamar_ollama else "template"],
    "fecha":   [pd.Timestamp.now().strftime("%Y-%m-%d")],
})
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    df_concl.to_excel(w, sheet_name="Conclusion_Ejecutiva", index=False)
print(f"\n  Hoja 'Conclusion_Ejecutiva' agregada a: {OUT_XL}")

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
ax.text(0.05, 0.92, "CONCLUSION EJECUTIVA", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")
ax.text(0.05, 0.80, conclusion, transform=ax.transAxes,
        fontsize=11, va="top", wrap=True,
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))
ax.set_title(f"Analisis de Elasticidad Precio — {CATEGORIA}", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
OUT_CONCL_PDF = os.path.join(DESKTOP, f"Conclusion_{CATEGORIA}.pdf")
plt.savefig(OUT_CONCL_PDF, bbox_inches="tight"); plt.show()
print(f"  Guardado: {OUT_CONCL_PDF}")
