import optuna
import tensorflow as tf
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



def load_and_preprocess_data(filepath, label_encoder=None, onehot_encoder=None, fit_encoders=False):
    data = pd.read_csv(filepath)

    # Drop 'severity_score' column if it exists
    if 'severity_score' in data.columns:
        data.drop(columns=['severity_score'], inplace=True)

    X = data.drop(columns=['prognosis'])
    y = data['prognosis']

    if fit_encoders:
        # Fit the encoders on the combined data
        label_encoder.fit(y)
        onehot_encoder.fit(label_encoder.transform(y).reshape(-1, 1))

    # Encoding the target
    y_encoded = onehot_encoder.transform(label_encoder.transform(y).reshape(-1, 1)).toarray()

    return X, y_encoded


def load_feedback_data(feedback_file_path):
    feedback_data = pd.read_csv(feedback_file_path)
    # Additional preprocessing steps as needed
    return feedback_data

class HyperparameterOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        units = trial.suggest_categorical('units', [32, 64, 128, 256])

        # Model building
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=units, activation='relu', input_dim=self.X_train.shape[1]))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(units=units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(self.y_train.shape[1], activation='softmax'))

        # Compile the model with legacy optimizer for M1/M2 Macs
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=50, batch_size=32,
                  verbose=0, callbacks=[early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(self.X_val, self.y_val, verbose=0)
        return accuracy

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        print("Best hyperparameters: {}".format(study.best_trial.params))
        with open('best_hyperparams.json', 'w') as f:
            json.dump(study.best_trial.params, f)




if __name__ == "__main__":
    # Create encoders
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()

    # Load and combine target variables to fit encoders
    original_target = pd.read_csv('~/desktop/HannahP1_Dataset/Training-Training.csv')['prognosis']
    feedback_target = pd.read_csv('~/desktop/HannahP1_Dataset/Feedback.csv')['prognosis']
    combined_target = pd.concat([original_target, feedback_target])

    # Fit encoders on combined target
    label_encoder.fit(combined_target)
    onehot_encoder.fit(label_encoder.transform(combined_target).reshape(-1, 1))

    # Preprocess original and feedback data
    original_X, original_y = load_and_preprocess_data('~/desktop/HannahP1_Dataset/Training-Training.csv', label_encoder, onehot_encoder)
    feedback_X, feedback_y = load_and_preprocess_data('~/desktop/HannahP1_Dataset/Feedback.csv', label_encoder, onehot_encoder)

    # Split original data
    X_train, X_val, y_train, y_val = train_test_split(original_X, original_y, test_size=0.2, random_state=42)

    # Combine feedback data with training data
    X_train = pd.concat([X_train, feedback_X])
    y_train = np.concatenate([y_train, feedback_y])
    # Hyperparameter optimization
    optimizer = HyperparameterOptimizer(X_train, y_train, X_val, y_val)
    number_of_trials = 100  # Define the number of trials
    optimizer.optimize(n_trials=number_of_trials)
