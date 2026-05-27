# Predicción de Propinas — Regresión Lineal y KNN

Proyecto de machine learning que entrena y compara dos modelos de regresión (Lineal y KNN) para predecir una variable numérica continua.

> ⚠️ **El dataset de ejemplo (`tips`) debe ser reemplazado por sus datos de negocio reales.**

---

## Requisitos

- Python 3.8+

Instalar dependencias:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Ejecución

```bash
python proyecto_final.py
```

---

## Reemplazar el dataset

En la sección **Cargar Dataset**, sustituir:

```python
# Antes (datos de ejemplo)
df = sns.load_dataset('tips')

# Después (sus datos de negocio)
df = pd.read_csv('ruta/a/su_dataset.csv')
```

Asegúrese de ajustar los nombres de columnas en las secciones de **preparación** y **modelado** para que coincidan con su dataset.

---

## Salidas del modelo

Al finalizar la ejecución se obtienen las métricas de ambos modelos:

| Métrica | Regresión Lineal | KNN |
|---|---|---|
| R² | ✓ | ✓ |
| MAE | ✓ | ✓ |
| MSE | ✓ | ✓ |

Y visualizaciones comparativas de predicciones vs. valores reales.
