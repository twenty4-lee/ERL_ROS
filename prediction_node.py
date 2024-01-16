#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize the ROS node
rospy.init_node('prediction_node', anonymous=True)

# Set up ROS Publisher
prediction_publisher = rospy.Publisher('spectrum_prediction', String, queue_size=10)

# Modified feature extraction function
def extract_features(file_path):
    spectrum_data = pd.read_csv(file_path)
    peaks, _ = find_peaks(spectrum_data['Count'], height=5, width=3)
    features = []
    for peak in peaks:
        peak_energy = spectrum_data['Energy [keV]'].iloc[peak]
        peak_count = spectrum_data['Count'].iloc[peak]
        features.append((peak_energy, peak_count))
    return features

def predict_spectrum(file_input):
    try:
        # Load data and extract features
        features = extract_features(file_input)

        # Sort peaks and select top 50 peaks
        sorted_features = sorted(features, key=lambda x: x[1], reverse=True)[:50] 
        flat_features = [val for peak in sorted_features for val in peak]

        # Ensure all spectra have 100 features
        if len(flat_features) < 100:
            flat_features.extend([0] * (100 - len(flat_features)))

        # Reshape for model compatibility
        X_new = np.array(flat_features).reshape(1, -1)

        # Load the model
        model = joblib.load('./src/my_package/trained_random_forest_model.pkl')

        # Perform prediction using the model
        predicted_isotope = model.predict(X_new)[0]

        # Convert the result to a message
        result_message = f"The spectrum is identified as: {predicted_isotope}"

        # Publish the result to the ROS topic
        prediction_publisher.publish(result_message)
        rospy.loginfo(result_message)

    except Exception as e:
        # Handle exceptions
        rospy.logerr(f"Error: {str(e)}")

if __name__ == '__main__':
    # Get user input for the file path
    file_input = input('Add excel file here: ')
    
    # Perform prediction
    predict_spectrum(file_input)
