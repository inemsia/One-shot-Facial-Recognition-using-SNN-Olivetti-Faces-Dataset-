# Idris Nemsia
# In this file, we load a trained Siamese Neural Network model and test it using 
# one-shot learning trials on the Yale face dataset. We evaluate how well the model 
# can recognize faces it has never seen before by comparing just one example per class
import numpy as np
from keras.models import load_model
from one_shot_functions import one_shot_trials, load_yale, write_results, visualize_one_shot_trials

# Import data
X, y, unique_y = load_yale()
unique_y_test = np.random.choice(unique_y, size=10, replace=False)
# Set the number of trials
num_trials = 3
# Directory containing trained models
model_name = "snn_ts80000.h5"
model_path = f'trained_model/{model_name}'

# Number of trials for one-shot testing

print(f"Loading model: {model_path}")
model = load_model(model_path)  # Load the model

# Perform one-shot trials and get metrics
print(f"Performing one-shot trials for {model_path}")
accuracy = one_shot_trials(X, y, num_trials, unique_y_test, model)
print(f'unique_y_test = {unique_y_test}')

# Write in file
write_results("Results.txt", 
              f"Results from {num_trials} one-shot trials of {model_name} on Yale dataset: {unique_y_test}", 
              accuracy)

# Uncomment here to visualize an example of these one-shot trials
visualize_one_shot_trials(X, y, unique_y_test, model)
