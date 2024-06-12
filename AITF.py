import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import csv
import Levenshtein
from keras.models import load_model
import logging
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential


data = pd.read_csv('~/desktop/HannahP1_Dataset/Training-Training.csv')
data.drop(columns=['severity_score'], inplace=True)


X = data.drop(columns=['prognosis'])
y = data['prognosis']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_encoded = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)


def collect_feedback(symptoms, correct_prognosis):
    filepath = '~/desktop/HannahP1_Dataset/Feedback.csv'
    filepath = os.path.expanduser(filepath)

    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        feedback_entry = symptoms + [correct_prognosis]
        writer.writerow(feedback_entry)

    global label_encoder, onehot_encoder, model
    if correct_prognosis not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, correct_prognosis)
        new_class_index = len(label_encoder.classes_) - 1
        new_onehot_vector = np.zeros((1, new_class_index + 1))
        new_onehot_vector[0, new_class_index] = 1
        onehot_encoder.categories_[0] = np.append(onehot_encoder.categories_[0], new_onehot_vector)

        new_output_neurons = len(label_encoder.classes_)
        model.layers[-1] = tf.keras.layers.Dense(new_output_neurons, activation='softmax')

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)



symptom_names = list(X.columns)
symptom_to_index = {symptom: index for index, symptom in enumerate(symptom_names)}


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'))  # First layer
    model.add(tf.keras.layers.Dropout(0.4089889802056872))  # Dropout from best hyperparameters

    # Additional layers based on best hyperparameters
    for _ in range(1, 2):  # 'n_layers' - 1 because the first layer is already defined
        model.add(tf.keras.layers.Dense(256, activation='relu'))  # 'units' from best hyperparameters
        model.add(tf.keras.layers.Dropout(0.4089889802056872))  # 'dropout_rate'

    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))  # Output layer

    # Compile model with optimal learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001947653910886403)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


class ModelUpdater:
    def __init__(self, model, X_train, y_train, label_encoder, onehot_encoder, feedback_threshold=2):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder
        self.feedback_counter = 0  # Initialize feedback counter
        self.feedback_threshold = feedback_threshold  # Threshold to trigger re-tuning

    def preprocess_feedback_X(self, feedback_X):
        num_samples_before = len(feedback_X)

        # Add missing dummy variables
        missing_cols = set(self.X_train.columns) - set(feedback_X.columns)
        missing_data = pd.DataFrame(0, index=feedback_X.index, columns=list(missing_cols))
        feedback_X = pd.concat([feedback_X, missing_data], axis=1)

        # Reorder columns to match X_train
        feedback_X = feedback_X[self.X_train.columns]

        if len(feedback_X) != num_samples_before:
            raise ValueError(f"preprocess_feedback_X altered the number of samples: Before({num_samples_before}), After({len(feedback_X)})")
        feedback_X = scaler.transform(feedback_X)

        return feedback_X

    def load_feedback_data(self, feedback_filepath):
        try:
            feedback_data = pd.read_csv(feedback_filepath)
            feedback_data.dropna(subset=[feedback_data.columns[-1]],
                                 inplace=True)  # Drop rows where target (last column) is NaN
            feedback_X = feedback_data.iloc[:, :-1]
            feedback_y = feedback_data.iloc[:, -1].astype(str)

            # Preprocess feedback_X to match training data format
            feedback_X_processed = self.preprocess_feedback_X(feedback_X)

            # Encode feedback_y
            feedback_y_encoded = self.onehot_encoder.transform(self.label_encoder.transform(feedback_y).reshape(-1, 1))

            return feedback_X_processed, feedback_y_encoded
        except Exception as e:
            logging.error(f"Error in load_feedback_data: {e}")
            raise

    def update_model_with_feedback(self, feedback_filepath):
        try:
            feedback_X, feedback_y = self.load_feedback_data(feedback_filepath)

            # Concatenate feedback data with existing training data
            self.X_train = np.concatenate([self.X_train, feedback_X])
            self.y_train = np.concatenate([self.y_train, feedback_y])

            # Retrain the model
            self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)

            # Save the updated model
            self.model.save('Hannah_updated.h5')
        except Exception as e:
            logging.error(f"Error in update_model_with_feedback: {e}")
            raise
# Usage
# model_updater = ModelUpdater(model, X_train, y_train, label_encoder, onehot_encoder)
# model_updater.update_model_with_feedback('~/desktop/HannahP1_Dataset/Feedback.csv')

def symptoms_to_vector(symptom_list):
    vector = [0] * len(symptom_names)
    for symptom in symptom_list:
        if symptom in symptom_to_index:
            vector[symptom_to_index[symptom]] = 1
    return vector


def correct_spelling(symptom_list):
    corrected_symptoms = []
    for symptom in symptom_list:
        symptom = symptom.strip().lower()
        if symptom in symptom_names:
            corrected_symptoms.append(symptom)
        else:
            distances = [Levenshtein.distance(symptom, s) for s in symptom_names]
            closest_idx = np.argmin(distances)
            if distances[closest_idx] <= 3:  # Maximum allowed distance
                corrected_symptoms.append(symptom_names[closest_idx])
            else:
                corrected_symptoms.append(symptom)
    return corrected_symptoms


def suggest_similar_symptoms(symptom):
    distances = [Levenshtein.distance(symptom, s) for s in symptom_names]
    sorted_indices = np.argsort(distances)
    similar_symptoms = [symptom_names[i] for i in sorted_indices if distances[i] <= 5 and distances[i] > 0]
    print(f"Similar symptoms for {symptom}: {similar_symptoms}")  # Debug print statement
    return similar_symptoms


def predict(symptoms, model, label_encoder):
    prediction = model.predict(np.array([symptoms]))
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]



def cli_interface(model_updater):
    symptom_names = list(X.columns)
    symptom_to_index = {symptom: index for index, symptom in enumerate(symptom_names)}

    print(f"Enter the present symptoms (separated by commas) from the following list:\n{', '.join(symptom_names)}\n")
    symptoms_input_text = input().split(',')
    corrected_symptoms = correct_spelling(symptoms_input_text)
    print("Corrected Symptoms:", ', '.join(corrected_symptoms))

    for i in range(len(corrected_symptoms)):
        if corrected_symptoms[i] not in symptom_names:
            user_input = input(f"Did you mean {corrected_symptoms[i]}? Please enter the correct symptom: ")
            corrected_symptoms[i] = user_input.strip().lower()

    symptoms_input = symptoms_to_vector(corrected_symptoms)

    predicted_prognosis = predict(symptoms_input, model_updater.model, model_updater.label_encoder)
    print("Predicted Prognosis:", predicted_prognosis)

    for symptom in corrected_symptoms:
        suggest_similar_symptoms(symptom)

    while True:
        feedback = input("Is the prediction correct? (yes/no/unconfirmed): ").lower()
        if feedback == 'no':
            correct_prognosis = input("Please enter the correct prognosis: ")
            symptoms = symptoms_to_vector(corrected_symptoms)
            collect_feedback(symptoms, correct_prognosis, model_updater.label_encoder)
            model_updater.update_model_with_feedback('~/desktop/HannahP1_Dataset/Feedback.csv')
            print("Feedback collected and model updated. Thank you!")
            break
        elif feedback == 'yes':
            print("Thank you for confirming!")
            break
        elif feedback == 'unconfirmed':
            print("No data collected")
            break
        else:
            print("Invalid response. Please respond with 'yes' or 'no'.")



def collect_feedback(symptoms, correct_prognosis, label_encoder):
    filepath = '~/desktop/HannahP1_Dataset/Feedback.csv'
    filepath = os.path.expanduser(filepath)

    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        feedback_entry = symptoms + [correct_prognosis]
        writer.writerow(feedback_entry)


if __name__ == "__main__":
    # Load or initialize model
    try:
        model = load_model('Hannah_updated.h5')
    except OSError:
        # Create a new model if the file does not exist
        model = create_model()  # This now uses the tuned hyperparameters
        print("New model created as 'Hannah_updated.h5' was not found.")

    model_updater = ModelUpdater(model, X_train, y_train, label_encoder, onehot_encoder)
    cli_interface(model_updater)




#checklist:
#Memory///X
#Feedback///X
#tune perameters optuna//X
#symtomChecker///X
#sevarityScore
#probabilityScore
##outputCalculator
#ActiveLearning!?
#GUI
#dataPrivacy
#realWorldTesting