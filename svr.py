import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from mealpy import Problem
from mealpy.swarm_based import SSA
from mealpy.utils.space import FloatVar, IntegerVar
import random, time

# asignamos un valor aleatorio a los pesos de cada neurona
# y nos aseguramos que estos sean siempre los mismo por cada ejecucion
np.random.seed(42)
random.seed(42)

# Carga de datos
excel_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\Mejores datos de chapinero.xlsx"
df = pd.read_excel(excel_path)

#remplazamos para que las columnas puedan ser leidas por pandas
df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "")
              .replace("°", "").replace("/", "_").replace(".", "") for c in df.columns]

# tratar el dato como tipo fecha
fecha_cols = [c for c in df.columns if "date" in c.lower() or "hora" in c.lower() or "datetime" in c.lower()]
if not fecha_cols:
    raise ValueError("No se encontró columna de fecha")
fecha_col = fecha_cols[0]

df[fecha_col] = pd.to_datetime(df[fecha_col])
df = df.sort_values(fecha_col).reset_index(drop=True)

# Selección de variable objetivo 
original_target = "PM2.5 (µg/m3)"
target_col = original_target.strip().replace(" ", "_").replace("(", "").replace(")", "") \
    .replace("°", "").replace("/", "_").replace(".", "")

columnas_excluir = [fecha_col]
variables = [c for c in df.columns 
             if c not in columnas_excluir 
             and pd.api.types.is_numeric_dtype(df[c])
             and not df[c].isna().all()]

if df[variables].isna().sum().sum() > 0:
    df[variables] = df[variables].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# usamos el 80% de los datos para el entrenamiento
train_split = 0.8
split_idx = int(train_split * len(df))

X_all = df[variables].values 
# convertimos la variable a predecir en un arreglo bidimentsion
y_all = df[target_col].values.reshape(-1, 1)

X_train_raw = X_all[:split_idx] # entrenamiento
y_train_raw = y_all[:split_idx] # prueba

X_test_raw = X_all[split_idx:] # entrenamiento
y_test_raw = y_all[split_idx:] # prueba

# Normalizacion
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)
y_train_scaled = scaler_y.transform(y_train_raw).ravel()
y_test_scaled = scaler_y.transform(y_test_raw).ravel()

# Crear ventanas temporales
def create_windows(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i - L:i].flatten())  # toma los valores previos a en las variables i el cual es el valor de la ventana del tiemp
        ys.append(y[i])  # toma el valor de i de la variable a predecir
    return np.array(Xs), np.array(ys)

sparrow_counter = 0 # cuantas veces se ha evaluado el modelo
sparrow_history = [] # resultados de cada ebvualuacion 

def evaluate_model(params):
    global sparrow_counter, sparrow_history
    sparrow_counter += 1 # incrementar cada vez que se evalua el modelo 

    L = int(params["lookback"]) # tamaño de la ventana del tiempo
    C = float(params["C"])
    epsilon = float(params["epsilon"])
    gamma = float(params["gamma"])

    X_train_local, y_train_local = create_windows(X_train_scaled, y_train_scaled, L)  # enviamos los valores de entrenamiento
    X_test_local, y_test_local = create_windows(X_test_scaled, y_test_scaled, L) # enviamos los valores de test

    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train_local, y_train_local)

    y_pred_scaled = model.predict(X_test_local).reshape(-1, 1)
    y_test_local = y_test_local.reshape(-1, 1)

# desnormalizacion
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_test_real = scaler_y.inverse_transform(y_test_local).reshape(-1)
# calcula erro cuadratico y coeficiente de determinacion
    mse = mean_squared_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

#guarda todos los parametro de cada horrion
    sparrow_history.append({
        "Gorrion": sparrow_counter,
        "Lookback": L,
        "C": C,
        "Epsilon": epsilon,
        "Gamma": gamma,
        "R2": r2,
        "MSE": mse
    })

    print(f"Iter={ssa.epoch if 'ssa' in globals() else '?'} | Gorrión={sparrow_counter:02d} | L={L} | C={C:.2f} | ε={epsilon:.4f} | γ={gamma:.5f} | R²={r2:.4f} | MSE={mse:.4f}")

    return mse, model

# Función objetivo SSA
def objective_function(solution):
    L = int(solution[0]) #  tamaño de la ventana
    C = float(solution[1])
    epsilon = float(solution[2])
    gamma = float(solution[3])

# hiperparametros del modelo lestim
    params = {
        "lookback": L,
        "C": C,
        "epsilon": epsilon,
        "gamma": gamma
    }

    mse, _ = evaluate_model(params)
    return mse

# Límites de hiperparámetros
bounds = [IntegerVar(lb=3, ub=50),     # L
FloatVar(lb=0.1, ub=100),    # C
FloatVar(lb=0.001, ub=0.05), # epsilon
FloatVar(lb=0.001, ub=0.2),  # gamma
] # gamma

# Optimización SSA
problem = Problem(
    obj_func=objective_function, # funcion que queremos minimizar
    bounds=bounds,  # limites para los hiper parametros
    minmax="min", # minimizar el valor devuelto por la funcion MSE
    name="SVR_SSA" # NOMBRE DESCRIPTIVO
)
# objeto de optimizacion 
ssa = SSA.OriginalSSA(epoch=5, pop_size=14) # iteraciones y gorriones
# inicia el proceso de optimizacion  y actualiza la poscion de los orriones seun el algortimo
def print_iter_info(optimizer=None):
    print(f"\n===== Iteración {optimizer.epoch + 1} de {optimizer.epoch_max} =====")

ssa.after_iteration = print_iter_info

best_solution = ssa.solve(problem)

# los mejores valores encontrados
best_pos = best_solution.solution
# diccionario con estos valores
best_params = {
    "lookback": int(best_pos[0]),
    "C": float(best_pos[1]),
    "epsilon": float(best_pos[2]),
    "gamma": float(best_pos[3])
}

# evalua el; modelo usando los mejores hiperparametros econtrados devuelve el mse y el mejor modelo
final_mse, best_model = evaluate_model(best_params)

# tamaño de la ventana optimo
L_opt = int(best_params["lookback"])

# enviamos los valores de entrenamiento y test
X_train_final, y_train_final = create_windows(X_train_scaled, y_train_scaled, L_opt)
X_test_final, y_test_final = create_windows(X_test_scaled, y_test_scaled, L_opt)

# evalua el mejor modelo
y_pred_scaled = best_model.predict(X_test_final).reshape(-1, 1)
# desnormalizar
y_pred_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
# desnormalizar
y_test_real = scaler_y.inverse_transform(y_test_final.reshape(-1, 1)).reshape(-1)
# guardamnos las fechas der cada prediccion justo despues de la ventana inicial

fechas_test = df[fecha_col].iloc[split_idx + L_opt:split_idx + L_opt + len(y_test_real)].reset_index(drop=True)

# creamos un data ser con las predicciones y el valor real

df_pred = pd.DataFrame({
    "Fecha": fechas_test,
    f"{target_col}_real": y_test_real,
    f"{target_col}_predicho": y_pred_real,
    "Error": y_test_real - y_pred_real
})

mse = mean_squared_error(y_test_real, y_pred_real) # Error cuadrático medio
mae = mean_absolute_error(y_test_real, y_pred_real)  # Error absoluto medio
r2 = r2_score(y_test_real, y_pred_real) # coeficiente de determinacion
rmse = np.sqrt(mse) # Promedio de los errores

#graifico
plt.figure(figsize=(15, 6))
plt.plot(df_pred["Fecha"], df_pred[f"{target_col}_real"], label="Real", color="blue", linewidth=1.5)
plt.plot(df_pred["Fecha"], df_pred[f"{target_col}_predicho"], label="Predicho", color="orange", linewidth=1.5, alpha=0.8)
plt.title(f"Predicción de {target_col} - Modelo SVR + SSA\nR² = {r2:.4f} | RMSE = {rmse:.4f}", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel(f"{target_col} (µg/m³)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\predicciones_PM25_SVR_SSA.xlsx"
df_pred.to_excel(output_path, index=False)

df_history = pd.DataFrame(sparrow_history)
history_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\historial_gorriones_PM25_SVR_SSA.xlsx"
df_history.to_excel(history_path, index=False)
