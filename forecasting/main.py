
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. CARGAR EL DATASET
# ==========================================

df = pd.read_csv("train.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

print("Dataset cargado correctamente")
print("Cantidad de registros originales:", len(df))

# ==========================================
# 2. PREPARAR LOS DATOS
# ==========================================

# Convertir la familia de productos a número
df["familia_id"] = df["family"].astype("category").cat.codes

# Crear variables con ventas anteriores
grupo = df.groupby(["store_nbr", "family"])["sales"]

df["venta_1_dia"] = grupo.shift(1)
df["venta_7_dias"] = grupo.shift(7)
df["venta_14_dias"] = grupo.shift(14)

# Variables de fecha
df["dia_semana"] = df["date"].dt.dayofweek
df["mes"] = df["date"].dt.month
df["dia_mes"] = df["date"].dt.day

# Eliminar registros sin ventas anteriores disponibles
df = df.dropna().reset_index(drop=True)

print("Registros disponibles para el modelo:", len(df))

# ==========================================
# 3. SELECCIONAR LAS VARIABLES
# ==========================================

variables = [
    "store_nbr",
    "familia_id",
    "onpromotion",
    "venta_1_dia",
    "venta_7_dias",
    "venta_14_dias",
    "dia_semana",
    "mes",
    "dia_mes"
]

X = df[variables]
y = df["sales"]

# ==========================================
# 4. DIVIDIR LOS DATOS POR FECHA
# ==========================================

fechas = sorted(df["date"].unique())

corte = fechas[int(len(fechas) * 0.80)]

entrenamiento = df["date"] <= corte
prueba = df["date"] > corte

X_train = X.loc[entrenamiento]
X_test = X.loc[prueba]

y_train = y.loc[entrenamiento]
y_test = y.loc[prueba]

print("\n========== DIVISIÓN DE DATOS ==========")
print("Fecha de corte:", pd.Timestamp(corte).date())
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# ==========================================
# 5. CREAR Y ENTRENAR EL MODELO
# ==========================================

modelo = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

print("\nEntrenando modelo...")

modelo.fit(X_train, y_train)

print("Modelo entrenado correctamente")

# ==========================================
# 6. REALIZAR LAS PREDICCIONES
# ==========================================

predicciones = modelo.predict(X_test)

# Evitar pronósticos negativos
predicciones = predicciones.clip(min=0)

# ==========================================
# 7. EVALUAR EL MODELO
# ==========================================

mse = mean_squared_error(y_test, predicciones)
rmse = mse ** 0.5
r2 = r2_score(y_test, predicciones)

print("\n========== RESULTADOS ==========")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ==========================================
# 8. COMPARAR VENTAS REALES Y PRONOSTICADAS
# ==========================================

resultados = pd.DataFrame({
    "Fecha": df.loc[prueba, "date"].values,
    "Tienda": df.loc[prueba, "store_nbr"].values,
    "Familia": df.loc[prueba, "family"].values,
    "Venta_real": y_test.values,
    "Venta_pronosticada": predicciones
})

resultados["Error_absoluto"] = (
    resultados["Venta_real"] -
    resultados["Venta_pronosticada"]
).abs()

print("\n========== PRIMERAS PREDICCIONES ==========")
print(resultados.head(10).round(2))

# Guardar todos los resultados
resultados.to_csv(
    "resultados_forecasting.csv",
    index=False
)

print("\nArchivo guardado: resultados_forecasting.csv")

# ==========================================
# COMPARAR CONTRA UN PRONÓSTICO SENCILLO
# ==========================================

# Pronóstico de referencia: ventas de hace 7 días
pronostico_base = df.loc[prueba, "venta_7_dias"].values

mse_base = mean_squared_error(y_test, pronostico_base)
rmse_base = mse_base ** 0.5
r2_base = r2_score(y_test, pronostico_base)

print("\n========== COMPARACIÓN DE MODELOS ==========")

print("\nModelo HistGradientBoosting:")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\nPronóstico sencillo (ventas de hace 7 días):")
print(f"MSE:  {mse_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²:   {r2_base:.4f}")

mejora = (1 - rmse / rmse_base) * 100

print(f"\nMejora del RMSE: {mejora:.2f}%")

# ==========================================
# 9. GRAFICAR RESULTADOS
# ==========================================

# Sumar las ventas de todas las tiendas y familias por fecha
grafica = resultados.groupby("Fecha").agg({
    "Venta_real": "sum",
    "Venta_pronosticada": "sum"
})

plt.figure(figsize=(12, 5))

plt.plot(
    grafica.index,
    grafica["Venta_real"],
    label="Ventas reales"
)

plt.plot(
    grafica.index,
    grafica["Venta_pronosticada"],
    label="Ventas pronosticadas"
)

plt.xlabel("Fecha")
plt.ylabel("Ventas totales")
plt.title("Forecasting de ventas con HistGradientBoostingRegressor")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

