# Idris Nemsia
# This file loads a trained Siamese Neural Network model and evaluates it using
# one-shot learning on the Olivetti Faces dataset (Dataset used during training, unseen classes). 
# It tests the model’s ability to recognize 10 unseen individuals using just one random  
# example per identity each trial.

# Imports
import numpy as np
from keras.models import load_model
from one_shot_functions import one_shot_trials, visualize_one_shot_trials, write_results
from sklearn.datasets import fetch_olivetti_faces

# Get data information
faces = fetch_olivetti_faces()
X = faces.images
X = np.expand_dims(X, axis=-1) 
y = faces.target
unique_y_test = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

# Set number of trials
num_trials = 10

# Directory containing trained models
model_name = "snn_ts80000.h5"
model_path = f'trained_model/snn_ts80000.h5'

# Number of trials for one-shot testing
print(f"Loading model: {model_path}")
model = load_model(model_path)  # Load the model

# Perform one-shot trials and get metrics
print(f"Performing one-shot trials for {model_path}")
accuracy = one_shot_trials(X, y, num_trials, unique_y_test, model)
# Write in file
write_results("Results.txt", 
              f"Results from {num_trials} one-shot trials of {model_name} on original dataset", 
              accuracy)

# Uncomment here to visualize an example of these one-shot trials
#visualize_one_shot_trials(X, y, unique_y_test, model)