from joblib import Parallel, delayed
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import random

# -------------------------------------------------Data Preprocessing------------------------------------------------------------------

# Load the dataset
url = 'Exercise7data.csv'
df = pd.read_csv(url)

# Drop specific columns
df = df.drop(['Arrival_time', 'Stop_id', 'Bus_id', 'Line_id'], axis=1)

# Features and target variable
x = df.drop(['Arrival_delay'], axis=1)
y = df['Arrival_delay']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# -------------------------------------------------Network Construction------------------------------------------------------------------

def build_model(layers, dropout_rate=0.2, optimizer_name='adam'):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(X_train.shape[1],)))  # Asegurarse de que coincida con el número de características

    # Hidden layers with Dropout
    for layer_size in layers:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))  # Add dropout

    # Output layer
    model.add(Dense(1))

    # Compile the model with a fresh instance of the optimizer for each iteration
    if optimizer_name == 'adam':
        optimizer = Adam()
    elif optimizer_name == 'sgd':
        optimizer = SGD()
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop()

    # Eliminar 'accuracy' de las métricas
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    return model


# -------------------------------------------------Callbacks and Monitoring------------------------------------------------------------------

# Setup callbacks
early_stop = EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=3)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_mae', save_best_only=True, mode='min')
tensorboard_callback = TensorBoard(log_dir=f'logs/{time.time()}')

# -------------------------------------------------Random Search Configuration------------------------------------------------------------------

# Definir el espacio de hiperparámetros
hyperparameter_space = {
    'optimizers': ['adam', 'sgd', 'rmsprop'],
    'layers_config': [
        [2, 2], [2, 4], [4, 4], [4, 8],
        [8, 8], [8, 16], [16, 16], [16, 32],
        [32, 32], [32, 64], [64, 64], [64, 128]
    ],
    'dropout_rates': [0.2, 0.3, 0.4, 0.5],
    'epochs_options': [50, 100, 200],
    'batch_sizes': [16, 32, 64]
}

# Definir el número de iteraciones para Random Search
n_iterations = 100  # Puedes ajustar este número según tus recursos computacionales

# Fijar semilla para reproducibilidad
random.seed(42)

# Generar combinaciones aleatorias
random_combinations = []
for _ in range(n_iterations):
    opt = random.choice(hyperparameter_space['optimizers'])
    layers = random.choice(hyperparameter_space['layers_config'])
    dropout = random.choice(hyperparameter_space['dropout_rates'])
    epochs = random.choice(hyperparameter_space['epochs_options'])
    batch_size = random.choice(hyperparameter_space['batch_sizes'])
    random_combinations.append((opt, layers, dropout, epochs, batch_size))


# -------------------------------------------------Training and Evaluation------------------------------------------------------------------

def train_and_evaluate(opt_name, layers, dropout_rate, epochs, batch_size):
    # Build and train the model
    model = build_model(layers, dropout_rate, opt_name)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr, checkpoint, tensorboard_callback],
        verbose=0
    )

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return the result for this combination
    return {
        'Optimizer': opt_name,
        'Layers': str(layers),
        'Dropout Rate': dropout_rate,
        'Epochs': epochs,
        'Batch Size': batch_size,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }


# -------------------------------------------------Execute in Parallel------------------------------------------------------------------

# Usar joblib para paralelizar el proceso de entrenamiento
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate)(opt_name, layers, dropout_rate, epochs, batch_size)
    for opt_name, layers, dropout_rate, epochs, batch_size in random_combinations
)

# -------------------------------------------------Compilation and Saving Results------------------------------------------------------------------

# Convert results list to a DataFrame
results_df = pd.DataFrame(results)

# Opcional: Limpiar caracteres especiales en el DataFrame antes de guardar
# Esto reemplaza caracteres no ASCII en las columnas de tipo string
results_df = results_df.applymap(
    lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x)

# Guardar los resultados en un archivo CSV
try:
    results_df.to_csv('random_search_results.csv', index=False, encoding='utf-8-sig')
    print(f"Entrenamiento completado. Se probaron {n_iterations} combinaciones aleatorias.")
    print("Resultados guardados en 'random_search_results.csv'.")
except UnicodeEncodeError as e:
    print("Error al guardar el archivo CSV:", e)
    print("Intentando con una codificación diferente...")
    results_df.to_csv('random_search_results.csv', index=False, encoding='latin1', errors='replace')
    print("Resultados guardados en 'random_search_results.csv' con codificación 'latin1'.")

# Mostrar el DataFrame en el entorno (por ejemplo, Jupyter Notebook)
# results_df