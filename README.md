# Dynamic Pricing Analyzer

App web genérica de pricing dinámico — sube cualquier tabla de ventas, estima elasticidad,
simula cambios de precio y recibe recomendaciones automáticas por SKU.

Construida con **Streamlit** y desplegada en **Railway**.

---

## Demo rápida

1. Sube tu CSV/Excel (o usa `plantilla_input.csv`)
2. Mapea las columnas a las variables del modelo
3. Valida la calidad de los datos (semáforo verde/amarillo/rojo)
4. Estima elasticidad con regresión OLS log-log
5. Simula escenarios de precio (−10%, +10%, 3x2, 2x1, 2do al 50%)
6. Obtén recomendaciones por SKU: Subir / Bajar / Mantener / No recomendar
7. Descarga todos los resultados en CSV

---

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Railway

1. Conecta este repositorio en [railway.app](https://railway.app)
2. Railway detecta automáticamente `railway.toml`
3. El start command ya está configurado con el puerto dinámico `$PORT`

---

## Estructura del repo

```
├── app.py                  ← App Streamlit (7 pasos)
├── plantilla_input.csv     ← Datos de ejemplo
├── requirements.txt
├── railway.toml            ← Config de despliegue
├── .streamlit/
│   └── config.toml        ← Tema y configuración
│
└── (análisis OfficeMax — referencia)
    ├── Elasticidad_OfficeMax.ipynb
    ├── elasticidad_officmax.py
    ├── crear_notebook.py
    └── outputs/
```

---

## Columnas del archivo de entrada

| Columna | Tipo | Descripción |
|:---|:---:|:---|
| `sku` | Obligatoria | Identificador del producto |
| `unidades` | Obligatoria | Unidades vendidas |
| `venta_neta` | Obligatoria | Ingreso total de la venta |
| `fecha` | Obligatoria | Periodo (mes/año) |
| `precio` | Opcional | Precio unitario (se calcula si no existe) |
| `costo` | Opcional | Costo unitario (necesario para simular margen) |
| `departamento` | Opcional | Categoría / departamento |
| `tienda` | Opcional | Sucursal |
| `elasticidad` | Opcional | Beta pre-calculada (evita la estimación) |

---

## Tecnologías

- **Streamlit** — interfaz web
- **statsmodels** — regresión OLS log-log
- **Plotly** — gráficas interactivas
- **pandas / numpy** — procesamiento de datos

---

*Reto Tec · 2026*

---

## ¿Qué mide la elasticidad precio?

$$\log(\text{unidades}+1) = \alpha + \beta \cdot \log(\text{precio}_{\text{real}})$$

El coeficiente **β** indica cuánto cambian las unidades vendidas ante un cambio del 1% en el precio:

| β | Tipo | Interpretación |
|:---:|:---:|:---|
| β < −1 | **Elástico** | Subir precio 10% → ventas bajan >10% |
| −1 < β < 0 | **Inelástico** | Subir precio 10% → ventas bajan <10% |
| β > 0 | Atípico | Precio y ventas se mueven en la misma dirección |

---

## Estructura del análisis

### Modelo 1 — Simple (tres ventanas temporales)

```
log(unidades+1) = α + β · log(precio_real)
```

| Variante | Ventana | Paso |
|:---:|:---:|:---:|
| 1A Mensual | Todos los meses | — |
| 1B Trimestral | 3 meses | 1 mes |
| 1C Semestral | 6 meses | 1 mes |

### Modelo 2 — Extendido con efectos fijos

```
log(unidades+1) = α + β·log(precio) + δ·tienda + ω·mes + γ·premium + margen
```

Variables adicionales con **One-Hot Encoding**:
- `δ · tienda` — dummies de tienda (efecto fijo por sucursal)
- `ω · mes` — dummies de mes (estacionalidad)
- `γ · premium` — flag de producto premium (por keyword en nombre)

### Simulador de precios

Escenarios simulados para SKUs seleccionados:

| Escenario | Cambio |
|:---:|:---:|
| −10% | Reducción fuerte |
| −5% | Reducción moderada |
| 0% (Base) | Sin cambio |
| +5% | Incremento moderado |
| +10% | Incremento fuerte |

Métricas calculadas por escenario: precio nuevo, unidades estimadas, ingreso estimado, margen estimado.

---

## Archivos del repositorio

```
analisis-elasticidad/
│
├── Elasticidad_OfficeMax.ipynb   ← Notebook principal con gráficas y resultados
├── elasticidad_officmax.py       ← Script Python con el análisis completo
├── crear_notebook.py             ← Genera el .ipynb desde cero y lo ejecuta
├── requirements.txt              ← Dependencias
├── .gitignore
│
└── outputs/
    ├── Elasticidad_SUMINISTROS_DE_OFICINA.xlsx  ← Resultados en hojas
    ├── Recomendaciones_SUMINISTROS_DE_OFICINA.pdf
    └── Simulacion_Elasticidad_SUMINISTROS_DE_OFICINA.pdf
```

### Hojas del Excel

| Hoja | Contenido |
|:---|:---|
| `Resumen_SKU` | Todos los SKUs con volumen, precio, margen y β |
| `M1_Mensual` | Un β por SKU usando todos los meses |
| `M1_Trimestral_Rolling` | β por SKU × ventana de 3 meses |
| `M1_Semestral_Rolling` | β por SKU × ventana de 6 meses |
| `M2_Metricas` | R², F-stat, coeficientes clave del Modelo 2 |
| `M2_Coeficientes` | Tabla completa de coeficientes |
| `M2_Coef_Tiendas` | Coeficiente por tienda |
| `Recomendaciones` | SUBIR / MANTENER / BAJAR por SKU |
| `Recom_Temporal` | Periodos óptimos para subir o promover por SKU |
| `Casos_Inelasticos` | SKUs con beta inelástica significativa en ventanas rolling |
| `Calendario_Accion` | Votos mensuales (SUBIR / MANTENER / PROMOVER) por SKU |
| `Conclusion_Ejecutiva` | Texto de conclusión generado por LLM o template |

---

## Cómo usarlo

### 1. Clonar el repositorio

```bash
git clone https://github.com/EmilioRam1/analisis-elasticidad.git
cd analisis-elasticidad
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Agregar los datos

Los archivos CSV no están incluidos por confidencialidad. Colocarlos en `datos_officmax/`:

```
datos_officmax/
├── Ventas_OfficeMax.csv
├── Catálogo_de_Producto.csv
├── Precios_de_Productos.csv
└── Costos_de_Productos.csv
```

### 4. Ejecutar el análisis

```bash
python elasticidad_officmax.py
```

O abrir directamente el notebook (ya ejecutado con outputs):

```bash
jupyter notebook Elasticidad_OfficeMax.ipynb
```

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![pandas](https://img.shields.io/badge/pandas-2.x-lightblue)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS-green)
![scikit--learn](https://img.shields.io/badge/sklearn-OneHotEncoder-red)

- **pandas / numpy** — manipulación de datos
- **statsmodels** — regresiones OLS
- **scikit-learn** — One-Hot Encoding
- **matplotlib** — visualizaciones

---

## Autor

**Emilio Ramírez Álvarez**
[GitHub](https://github.com/EmilioRam1)

*Reto Tec — OfficeMax · 2026*
