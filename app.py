# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard ML - Regresión", layout="wide")
st.title("📊 Dashboard de Predicción — Regresión Lineal vs KNN")
st.caption("Reemplaza el dataset de ejemplo con tus datos de negocio.")

# ── Cargar Dataset ────────────────────────────────────────────────────────────
# ⚠️  REEMPLAZA ESTA LÍNEA con tus datos de negocio:
#     df = pd.read_csv("ruta/a/tu_dataset.csv")
df = sns.load_dataset("tips")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Parámetros del modelo")
k_neighbors = st.sidebar.slider("Vecinos KNN (k)", min_value=1, max_value=20, value=5)
test_size    = st.sidebar.slider("% datos de prueba", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# ── Data Understanding ────────────────────────────────────────────────────────
st.header("1. Entendimiento de los datos")

col1, col2, col3 = st.columns(3)
col1.metric("Filas",    df.shape[0])
col2.metric("Columnas", df.shape[1])
col3.metric("Nulos",    int(df.isnull().sum().sum()))

with st.expander("Vista previa del dataset"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Estadísticas descriptivas"):
    st.dataframe(df.describe(include="all"), use_container_width=True)

# Distribución de variable objetivo
st.subheader("Distribución de la variable objetivo (tip)")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["tip"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Distribución de propinas")
sns.boxplot(x=df["tip"], ax=axes[1])
axes[1].set_title("Boxplot de propinas")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Boxplots categóricos
st.subheader("Propinas por variable categórica")
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, ["sex", "smoker", "day", "time"]):
    sns.boxplot(data=df, x=col, y="tip", ax=ax)
    ax.set_title(f"Propina por {col}")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Scatter total_bill vs tip
st.subheader("Total Cuenta vs Propina")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", ax=ax)
ax.set_title("Total Cuenta vs Propina")
st.pyplot(fig)
plt.close()

# Mapa de nulos
st.subheader("Mapa de valores nulos")
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(df.isnull(), cbar=False, cmap="Reds", ax=ax)
ax.set_title("Valores nulos por columna")
st.pyplot(fig)
plt.close()

# ── Preparación ───────────────────────────────────────────────────────────────
st.header("2. Preparación de los datos")

numericos   = ["total_bill", "size"]
categoricas = ["sex", "smoker", "day", "time"]

df_encoded = pd.get_dummies(df, columns=categoricas, drop_first=True)
scaler = StandardScaler()
df_encoded[numericos] = scaler.fit_transform(df_encoded[numericos])

X = df_encoded.drop("tip", axis=1)
y = df_encoded["tip"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=int(random_state)
)

col1, col2 = st.columns(2)
col1.metric("Muestras de entrenamiento", x_train.shape[0])
col2.metric("Muestras de prueba",        x_test.shape[0])

# ── Modelado ──────────────────────────────────────────────────────────────────
st.header("3. Entrenamiento y evaluación de modelos")

# Regresión Lineal
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# KNN
knn = KNeighborsRegressor(n_neighbors=k_neighbors)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

def metricas(y_true, y_pred):
    return {
        "R²":  round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mean_squared_error(y_true, y_pred), 4),
    }

metrics_lr  = metricas(y_test, y_pred_lr)
metrics_knn = metricas(y_test, y_pred_knn)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regresión Lineal")
    st.metric("R²",  metrics_lr["R²"])
    st.metric("MAE", metrics_lr["MAE"])
    st.metric("MSE", metrics_lr["MSE"])

with col2:
    st.subheader(f"KNN (k={k_neighbors})")
    st.metric("R²",  metrics_knn["R²"])
    st.metric("MAE", metrics_knn["MAE"])
    st.metric("MSE", metrics_knn["MSE"])

# Tabla comparativa
st.subheader("Comparación de modelos")
st.dataframe(
    pd.DataFrame({"Regresión Lineal": metrics_lr, f"KNN (k={k_neighbors})": metrics_knn}).T,
    use_container_width=True,
)

# ── Visualización de predicciones ─────────────────────────────────────────────
st.header("4. Predicciones vs Valores reales")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.regplot(x=x_test["total_bill"], y=y_test, ax=axes[0],
            marker="o", label="Datos Reales", color="steelblue")
axes[0].scatter(x_test["total_bill"], y_pred_lr, color="red",
                label="Predicción LR", alpha=0.5)
axes[0].set_title("Regresión Lineal")
axes[0].set_xlabel("Total Cuenta (normalizado)")
axes[0].set_ylabel("Propina")
axes[0].legend()

sns.regplot(x=x_test["total_bill"], y=y_test, ax=axes[1],
            marker="o", label="Datos Reales", color="steelblue",
            scatter_kws={"alpha": 0.5})
axes[1].scatter(x_test["total_bill"], y_pred_knn, color="green",
                label=f"Predicción KNN", alpha=0.5)
axes[1].set_title(f"KNN (k={k_neighbors})")
axes[1].set_xlabel("Total Cuenta (normalizado)")
axes[1].set_ylabel("Propina")
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Inferencia ────────────────────────────────────────────────────────────────
st.header("5. 🔮 Hacer una predicción")
st.write("Ingresa los valores y el modelo estimará la propina esperada.")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        inp_total_bill = st.number_input("Total de la cuenta (USD)", min_value=0.0, value=20.0, step=0.5)
        inp_size       = st.number_input("Tamaño de la mesa (personas)", min_value=1, max_value=10, value=2, step=1)
        inp_sex        = st.selectbox("Género del pagador", ["Male", "Female"])

    with col2:
        inp_smoker = st.selectbox("¿Fumador?", ["No", "Yes"])
        inp_day    = st.selectbox("Día", ["Thur", "Fri", "Sat", "Sun"])
        inp_time   = st.selectbox("Horario", ["Lunch", "Dinner"])

    modelo_elegido = st.radio("Modelo a usar", ["Regresión Lineal", f"KNN (k={k_neighbors})"], horizontal=True)
    submitted = st.form_submit_button("Predecir propina", use_container_width=True)

if submitted:
    input_dict = {
        "total_bill": inp_total_bill,
        "size":       inp_size,
        "sex":        inp_sex,
        "smoker":     inp_smoker,
        "day":        inp_day,
        "time":       inp_time,
    }
    input_df = pd.DataFrame([input_dict])

    # Codificación igual que en entrenamiento
    input_encoded = pd.get_dummies(input_df, columns=categoricas, drop_first=True)

    # Alinear columnas con las del modelo (agrega las que falten con 0)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    # Escalar numéricos
    input_encoded[numericos] = scaler.transform(input_encoded[numericos])

    # Predicción
    if modelo_elegido == "Regresión Lineal":
        pred = lr.predict(input_encoded)[0]
    else:
        pred = knn.predict(input_encoded)[0]

    st.success(f"💵 Propina estimada: **${pred:.2f} USD**")

    # Contexto visual: dónde cae vs la distribución real
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(df["tip"], bins=20, kde=True, ax=ax, color="steelblue", alpha=0.6)
    ax.axvline(pred, color="red", linewidth=2, linestyle="--", label=f"Tu predicción: ${pred:.2f}")
    ax.set_title("Predicción vs distribución real de propinas")
    ax.legend()
    st.pyplot(fig)
    plt.close()
