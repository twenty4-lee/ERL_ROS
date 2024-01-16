#!/usr/bin/env python3
import rospy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features_from_spectrum(file_path):
    spectrum_data = pd.read_csv(file_path)
    peaks, _ = find_peaks(spectrum_data['Count'], height=5, width=3)
    features = [(spectrum_data['Energy [keV]'].iloc[peak], spectrum_data['Count'].iloc[peak]) for peak in peaks]
    return features

def prepare_dataset(features_dict, max_peaks=50):
    X = []
    y = []
    for label, feature_list in features_dict.items():
        for feature_set in feature_list:
            if feature_set['features']:
                sorted_features = sorted(feature_set['features'], key=lambda x: x[1], reverse=True)[:max_peaks]
                flat_features = [val for peak in sorted_features for val in peak]
                flat_features.extend([0] * (max_peaks * 2 - len(flat_features)))
                X.append(flat_features)
                y.append(label)
    return np.array(X), np.array(y)

def train_model():
    rospy.loginfo("Model training started")

    # Define isotopes and distances
    isotopes = ['eu', 'cs', 'ba', 'am', 'co', 'na', 'ti']
    distances = [1, 3, 5, 7, 9]
    measurements = range(1, 6)

    # Create file paths and extract features for all isotopes
    features_dict = {isotope: [] for isotope in isotopes}
    for isotope in isotopes:
        for cm in distances:
            for i in measurements:
                file_name = f'../Downloads/datasets/{isotope}/{isotope}{cm}cm-{i}.csv'
                features = extract_features_from_spectrum(file_name)
                features_dict[isotope].append({'file_path': file_name, 'features': features})

    # Prepare the features and labels from the features_dict
    X, y = prepare_dataset(features_dict)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    rf_y_pred = model.predict(X_test)
    rospy.loginfo("\n" + classification_report(y_test, rf_y_pred))

    # Save the model
    model_path = './src/my_package/trained_random_forest_model.pkl'
    joblib.dump(model, model_path)
    rospy.loginfo("Model trained and saved at: " + model_path)

if __name__ == '__main__':
    try:
        rospy.init_node('model_training_node', anonymous=True)
        train_model()
    except rospy.ROSInterruptException:
        pass