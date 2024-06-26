from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "http://localhost:4200"}})

crypto_ids = [
    'dogecoin', 'shiba-inu', 'the-graph', 'axie-infinity',
    'the-sandbox', 'akash-network', 'pendle', 'singularitynet',
    'aioz-network'
]

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps, 0])  # Solo predecir el precio
    return np.array(X), np.array(y)

@app.route('/predict/<crypto_id>/<periodo_param>', methods=['GET'])
def predict(crypto_id, periodo_param):
    try:
        # Obtener datos históricos de la criptomoneda desde CoinCap API
        response = requests.get(f'https://api.coincap.io/v2/assets/{crypto_id}/history?interval=d1')
        data = response.json()['data']
        df = pd.DataFrame(data)

        # Obtener datos adicionales de CoinCap API
        additional_response = requests.get(f'https://api.coincap.io/v2/assets/{crypto_id}')
        additional_data = additional_response.json()['data']
        additional_df = pd.DataFrame([additional_data])

        # Convertir la columna 'date' a formato datetime y establecerla como índice
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Convertir las columnas relevantes a tipo float
        df['priceUsd'] = df['priceUsd'].astype(float)
        additional_df['supply'] = additional_df['supply'].astype(float)
        additional_df['marketCapUsd'] = additional_df['marketCapUsd'].astype(float)
        additional_df['volumeUsd24Hr'] = additional_df['volumeUsd24Hr'].astype(float)
        additional_df['changePercent24Hr'] = additional_df['changePercent24Hr'].astype(float)
        additional_df['vwap24Hr'] = additional_df['vwap24Hr'].astype(float)

        # Añadir columnas adicionales a df
        for col in ['supply', 'marketCapUsd', 'volumeUsd24Hr', 'changePercent24Hr', 'vwap24Hr']:
            df[col] = additional_df[col].values[0]

        # Dividir los datos en conjunto de entrenamiento y prueba
        train_size = int(len(df) * 0.8)
        train_data, test_data = df[:train_size], df[train_size:]

        # Escalar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        time_steps = 30
        X_train, y_train = prepare_data(train_scaled, time_steps)
        X_test, y_test = prepare_data(test_scaled, time_steps)

        # Crear y entrenar el modelo GRU
        model = Sequential([
            GRU(units=512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.4),
            GRU(units=256, return_sequences=True),
            Dropout(0.4),
            GRU(units=128),
            Dropout(0.4),
            Dense(units=1)
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping],
                  verbose=1)

        # Realizar predicciones
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Desescalar los datos para mostrar los precios reales y predichos
        train_predictions = scaler.inverse_transform(
            np.hstack((train_predictions, np.zeros((train_predictions.shape[0], X_train.shape[2] - 1))))
        )
        test_predictions = scaler.inverse_transform(
            np.hstack((test_predictions, np.zeros((test_predictions.shape[0], X_test.shape[2] - 1))))
        )
        train_data_inverse = scaler.inverse_transform(train_scaled)
        test_data_inverse = scaler.inverse_transform(test_scaled)

        # Calcular la precisión del modelo en los datos de prueba
        test_rmse = np.sqrt(mean_squared_error(test_data_inverse[time_steps:, 0], test_predictions[:, 0]))

        # Predicción de precios para el próximo mes
        future_days = int(periodo_param)
        last_sequence = test_scaled[-time_steps:]  # Utilizamos la última secuencia de los datos de prueba
        future_prices = []

        for _ in range(future_days):
            X_future = last_sequence.reshape(1, time_steps, last_sequence.shape[1])  # Reshape to include all features
            future_price = model.predict(X_future)

            # Agregar ruido aleatorio para simular la fluctuación del precio
            noise = np.random.normal(0, 0.01)
            future_price = future_price + noise

            # Append the predicted price along with zeros for other features
            future_prices.append(np.hstack([future_price[0, 0], np.zeros(last_sequence.shape[1]-1)]))

            # Update last_sequence, keeping all features
            last_sequence = np.vstack([last_sequence[1:], future_prices[-1]])

        # Desescalar los precios futuros para mostrarlos en el gráfico
        future_prices = np.array(future_prices)  # No need to reshape here
        future_prices = scaler.inverse_transform(future_prices)

        future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1)[1:]

        last_real_price = float(df['priceUsd'].iloc[-1])
        first_predicted_price = float(future_prices[0, 0])
        last_predicted_price = float(future_prices[-1, 0])
        percentage_change_first_day = ((first_predicted_price - last_real_price) / last_real_price) * 100
        percentage_change_last_day = ((last_predicted_price - last_real_price) / last_real_price) * 100

        max_value = future_prices[:, 0].max()
        max_date = future_dates[future_prices[:, 0].argmax()]
        min_value = future_prices[:, 0].min()
        min_date = future_dates[future_prices[:, 0].argmin()]

        # Crear respuesta JSON con los datos necesarios
        return {
            "real_dates": df.index.tolist(),
            "real_prices": df['priceUsd'].tolist(),
            "future_prices": future_prices[:, 0].tolist(),
            "future_dates": future_dates.tolist(),
            "last_real_price": last_real_price,
            "first_predicted_price": first_predicted_price,
            "last_predicted_price": last_predicted_price,
            "percentage_change_first_day": percentage_change_first_day,
            "percentage_change_last_day": percentage_change_last_day,
            "max_value": max_value,
            "max_date": max_date.date().isoformat(),
            "min_value": min_value,
            "min_date": min_date.date().isoformat(),
            "test_rmse": test_rmse
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_change/<crypto_id>/<periodo_param>', methods=['GET'])
def predict_change(crypto_id, periodo_param):
    try:
        # Obtener datos históricos de la criptomoneda desde CoinCap API
        response = requests.get(f'https://api.coincap.io/v2/assets/{crypto_id}/history?interval=d1')
        data = response.json()['data']
        df = pd.DataFrame(data)

        # Obtener datos adicionales de CoinCap API
        additional_response = requests.get(f'https://api.coincap.io/v2/assets/{crypto_id}')
        additional_data = additional_response.json()['data']
        additional_df = pd.DataFrame([additional_data])

        # Convertir la columna 'date' a formato datetime y establecerla como índice
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Convertir las columnas relevantes a tipo float
        df['priceUsd'] = df['priceUsd'].astype(float)
        additional_df['supply'] = additional_df['supply'].astype(float)
        additional_df['marketCapUsd'] = additional_df['marketCapUsd'].astype(float)
        additional_df['volumeUsd24Hr'] = additional_df['volumeUsd24Hr'].astype(float)
        additional_df['changePercent24Hr'] = additional_df['changePercent24Hr'].astype(float)
        additional_df['vwap24Hr'] = additional_df['vwap24Hr'].astype(float)

        # Añadir columnas adicionales a df
        for col in ['supply', 'marketCapUsd', 'volumeUsd24Hr', 'changePercent24Hr', 'vwap24Hr']:
            df[col] = additional_df[col].values[0]

        # Dividir los datos en conjunto de entrenamiento y prueba
        train_size = int(len(df) * 0.8)
        train_data, test_data = df[:train_size], df[train_size:]

        # Escalar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        time_steps = 30
        X_train, y_train = prepare_data(train_scaled, time_steps)
        X_test, y_test = prepare_data(test_scaled, time_steps)

        # Crear y entrenar el modelo GRU
        model = Sequential([
            GRU(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(units=128),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

        # Predicción de precios para el próximo mes
        future_days = int(periodo_param)
        last_sequence = test_scaled[-time_steps:]  # Utilizamos la última secuencia de los datos de prueba
        future_prices = []

        for _ in range(future_days):
            X_future = last_sequence.reshape(1, time_steps, last_sequence.shape[1])  # Reshape to include all features
            future_price = model.predict(X_future)

            # Agregar ruido aleatorio para simular la fluctuación del precio
            noise = np.random.normal(0, 0.01)
            future_price = future_price + noise

            # Append the predicted price along with zeros for other features
            future_prices.append(np.hstack([future_price[0, 0], np.zeros(last_sequence.shape[1]-1)]))

            # Update last_sequence, keeping all features
            last_sequence = np.vstack([last_sequence[1:], future_prices[-1]])

        # Desescalar los precios futuros para mostrarlos en el gráfico
        future_prices = np.array(future_prices)  # No need to reshape here
        future_prices = scaler.inverse_transform(future_prices)

        # Obtener los últimos precios reales y la primera predicción
        last_real_price = df['priceUsd'].iloc[-1]
        first_predicted_price = future_prices[0, 0]

        # Calcular el porcentaje de cambio
        percentage_change = ((first_predicted_price - last_real_price) / last_real_price) * 100

        # Crear respuesta JSON con el porcentaje de cambio
        response_data = {
            'crypto_id': crypto_id,
            'last_real_price': last_real_price,
            'first_predicted_price': first_predicted_price,
            'percentage_change': percentage_change
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
