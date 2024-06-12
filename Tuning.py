import optuna
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import os

class MedicalPrognosisModel:
    def __init__(self, data_filepath, feedback_filepath, model_filepath='Hannah_updated.h5', tuning_threshold=50):
        self.data_filepath = os.path.expanduser(data_filepath)
        self.feedback_filepath = os.path.expanduser(feedback_filepath)
        self.model_filepath = model_filepath
        self.tuning_threshold = tuning_threshold
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.model = None
        self.scaler = StandardScaler()
        self.load_data()
        self.prepare_model()

    def load_data(self):
        data = pd.read_csv(self.data_filepath)
        if data.empty:
            raise ValueError("No data loaded. Check the file path and contents.")
        data.dropna(axis=1, how='all', inplace=True)
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        if 'severity_score' in data.columns:
            data.drop(columns=['severity_score'], inplace=True)
        X = data.drop(columns=['prognosis'])
        y = data['prognosis']
        self.label_encoder.fit(y)
        y_encoded = self.onehot_encoder.fit_transform(self.label_encoder.transform(y).reshape(-1, 1))
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    def prepare_model(self):
        best_params = self.tune_hyperparameters(self.X_train, self.y_train)
        self.model = self.build_model(best_params)

    def tune_hyperparameters(self, X, y):
        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            units = trial.suggest_categorical('units', [32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units, activation='relu', input_dim=X.shape[1], kernel_initializer='he_normal'),
                tf.keras.layers.Dropout(dropout_rate)] +
                [layer for _ in range(n_layers - 1) for layer in (tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_normal'), tf.keras.layers.Dropout(dropout_rate))] +
                [tf.keras.layers.Dense(y.shape[1], activation='softmax')])
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=0)
            _, accuracy = model.evaluate(self.X_val, self.y_val, verbose=0)
            return accuracy
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        return study.best_trial.params

    def build_model(self, params):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(params['units'], activation='relu', input_dim=self.X_train.shape[1], kernel_initializer='he_normal'),
            tf.keras.layers.Dropout(params['dropout_rate'])] +
            [layer for _ in range(params['n_layers'] - 1) for layer in (tf.keras.layers.Dense(params['units'], activation='relu', kernel_initializer='he_normal'), tf.keras.layers.Dropout(params['dropout_rate']))] +
            [tf.keras.layers.Dense(self.y_train.shape[1], activation='softmax')])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params['lr'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def update_model_with_feedback(self):
        feedback_data = pd.read_csv(self.feedback_filepath)
        feedback_data.dropna(subset=['prognosis'], inplace=True)
        feedback_X = feedback_data.drop(columns=['prognosis'])
        feedback_y = feedback_data['prognosis']
        feedback_y_encoded = self.onehot_encoder.transform(self.label_encoder.transform(feedback_y).reshape(-1, 1))
        feedback_X_scaled = self.scaler.transform(feedback_X)
        self.X_train = np.concatenate([self.X_train, feedback_X_scaled])
        self.y_train = np.concatenate([self.y_train, feedback_y_encoded])
        if feedback_data.shape[0] >= self.tuning_threshold:
            best_params = self.tune_hyperparameters(self.X_train, self.y_train)
            self.model = self.build_model(best_params)
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)
        self.model.save(self.model_filepath)

if __name__ == "__main__":
    data_filepath = '~/desktop/HannahP1_Dataset/Training-Training.csv'
    feedback_filepath = '~/desktop/HannahP1_Dataset/Feedback.csv'
    model = MedicalPrognosisModel(data_filepath, feedback_filepath)
    model.update_model_with_feedback()
