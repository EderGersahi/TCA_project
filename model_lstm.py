## Librerias

import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow
import pickle
import mlflow
import mlflow.tensorflow

from mlflow.models import infer_signature


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

#from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from scipy.signal import savgol_filter
from scipy import stats


#Definir el URI para el archivo local del sistema
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("lstm_tca")

# Funcion para crear la semilla   
seed=42
random.seed(seed)
np.random.seed(seed)
tensorflow.random.set_seed(seed)


# Cargar Dataset previamente limpio
df=pd.read_csv("reservaciones_clean.csv")

# Dropear una columna Cliente_Disp
df.drop(columns="Cliente_Disp",inplace=True)

# Convertir la variable de fecha de entrada a tipo fecha y ordenar el dataset por esta misma columna
df['h_fec_lld_ok']=pd.to_datetime(df['h_fec_lld_ok'])

df=df.sort_values('h_fec_lld_ok')

#Hacer copia del dataset y posteriormente seleccionar las unicas columnas que se utilizaran para el modelo
df2=df.copy()
df2=df2[['h_fec_lld_ok','h_tfa_total']]

# Agrupamos el dataset por fecha de llegada, tomando en cuenta la funcion de suma
df2=df2.groupby('h_fec_lld_ok').sum()

#La serie de tiempo obtenida se suaviza (3 técnicas) para poder tener un  mejor rendimiento del modelo de LSTM

df2['smoothed'] = df2['h_tfa_total'].rolling(window=7).mean() #media movil simple
df2['smoothed2'] = df2['h_tfa_total'].ewm(span=7, adjust=False).mean() # Suavizado exponencial
df2['smoothed3'] = savgol_filter(df2['h_tfa_total'], window_length=15, polyorder=2) # filtro de Savitzky-Golay


# Se selecciona una sola columna suavizada
df_lstm=df2[["smoothed2"]]

# Se hace una prueba de normalidad(shapiro-wilk) a la columna que se va a predecir

shapiro_test = stats.shapiro(df2['smoothed2'])
shapiro_test

# Selección de escalador basada en normalidad
if shapiro_test.pvalue < 0.05:
    print("Distribución NO normal → Usando MinMaxScaler")
    scaler = MinMaxScaler()
else:
    print("Distribución normal → Usando StandardScaler")
    scaler = StandardScaler()

# Escalado de datos
scaled_data = scaler.fit_transform(df_lstm)

## MODELO

#Funcion de sliding window

def create_sequences(data, window_size):
    '''
    Crea secuencias deslizantes a partir de la serie de tiempo.

    Parámetros:
    data : array-like
        Serie de tiempo escalada o normalizada (1D o 2D con una sola columna).
    window_size : int
        Número de pasos de tiempo que cada secuencia debe contener.

    Retorna:
    X : numpy.ndarray
        Arreglo de secuencias de entrada con forma (n_samples, window_size, 1).
    y : numpy.ndarray
        Arreglo de valores objetivo (el valor siguiente a cada secuencia).
    '''
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)



#Se define el tamaño de la ventana de salto y aplicamos a nuestro dataset escalado
window_size = 30
X, y = create_sequences(scaled_data, window_size)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Ajustar dimensiones para LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


def modelo_lstm(units=128, activation='relu', dropout_rate=0.2, learning_rate=0.001):
    '''
    Crea un modelo LSTM con tres capas LSTM y Dropout entre ellas para evitar sobreajuste.
    
    - Entrada: secuencias de longitud window_size con 1 feature.
    - Salida: capa densa para predicción continua (regresión).
    - Compilado con optimizador Adam y pérdida MSE.

    Retorna:
    - Modelo LSTM compilado listo para entrenar.
    '''
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))

    model.add(LSTM(units, activation=activation, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units, activation=activation, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    #model.compile(optimizer=Adam(learning_rate=learning_rate), loss=['mse', 'mape'])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape']
        )

    return model

# Wrap con KerasRegressor
model_lstm = KerasRegressor(model=modelo_lstm, verbose=0)


# -------TUNING----------

#se define el param grid

param_grid_lstm = {
    'model__units': [32, 64, 128],
    'model__activation': ['relu', 'tanh'],
    'model__dropout_rate': [0.01, 0.05],
    'model__learning_rate': [0.001, 0.0005],
    'epochs': [50, 100,150],
}

#Aplicar Grid Search
grid_lstm = GridSearchCV(estimator=model_lstm, param_grid=param_grid_lstm , cv=2)

grid_result_lstm = grid_lstm.fit(X_train, y_train)


## A partir de los resultados del tuning, crear un modelo final y entrenarlo
best_params = grid_result_lstm.best_params_
model_params = {

    'units': best_params['model__units'],
    'activation': best_params['model__activation'],
    'dropout_rate': best_params['model__dropout_rate'],
    'learning_rate': best_params['model__learning_rate'],
}
epochs = best_params['epochs']


final_model = modelo_lstm(**model_params)


# Guardar el scaler como archivo pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Predicción con el mejor modelo del GridSearchCV
y_pred = grid_result_lstm.best_estimator_.predict(X_test)

# Checar la forma para no tener problemas 
y_test = y_test.reshape(-1, 1)
y_pred = np.array(y_pred).reshape(-1, 1)

# Revertir escalado
y_test_original = scaler.inverse_transform(y_test)
y_pred_original = scaler.inverse_transform(y_pred)

# Calcular métricas
loss = final_model.evaluate(X_test, y_test)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse=np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original)

# 5. Mostrar resultados
print("Métricas del modelo LSTM:")
print(f"MSE: {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.4f}")


with mlflow.start_run() as run:
    mlflow.log_metric("MSE",mse)
    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("MAE",mae)
    mlflow.log_metric("MAPE",mape)
    mlflow.log_param("semilla", seed)
    mlflow.log_params(model_params)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("test_loss", loss[0])
    mlflow.log_artifact("scaler.pkl") #scaler
    mlflow.tensorflow.log_model(final_model,
                                signature=infer_signature(X_train, y_train),
                                registered_model_name="lstm_tca",
                                artifact_path="model")




# Definir por que utilizamos los modelos que utilizamos
# El objetivo, predecir cuántos días va redecir los modelos y por que
# cuantos dias predecir sin que el error se dispare
# 