# Idris Nemsia
# This file containes helper functions used to organize the code for evaluations
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 

# This function runs a single classification task for one-shot learning.
# It compares a random input image against one image from each image from the support set
# and returns the true and predicted labels.
def classification_test(X, y, unique_y_test, model):
    # Initialize input arrays
    X1, X2 = [], []
    # Get all images of testing dataset individuals
    indices_test = np.where(np.isin(y, unique_y_test))[0]
    index_input = np.random.choice(indices_test)
    input_label = y[index_input]
    input_image = X[index_input]

    # Create an array containing a random image for each person from the testing dataset
    for unique_label in unique_y_test:
        indices_person = np.where(y == unique_label)[0]
        index_test_image = np.random.choice(indices_person)
        test_image = X[index_test_image]
        X2.append(test_image)
        X1.append(input_image)

    X1, X2 = np.array(X1), np.array(X2)

    # Use the model to make predictions on the similarity of the pairs
    predictions = model.predict([X1, X2])
    max_index = np.argmax(predictions)
    predicted_label = unique_y_test[max_index]
    
    return input_label, predicted_label


# This function repeats one-shot classification tests multiple times to evaluate accuracy.
def one_shot_trials(X, y, num_trials, unique_y_test , model):
    # Initialize counts
    true_positives = {label: 0 for label in unique_y_test}
    false_positives = {label: 0 for label in unique_y_test}
    false_negatives = {label: 0 for label in unique_y_test}
    # Used to calculate the accuracy
    correct_count = 0

    for trial_num in range(num_trials):
        print(f'Trial number = {trial_num + 1}')
        
        input_label, predicted_label = classification_test(X, y, unique_y_test, model)
        
        if input_label == predicted_label:
            correct_count += 1
            true_positives[input_label] += 1  # Correctly predicted
        else:
            false_positives[predicted_label] += 1  # Predicted incorrectly
            false_negatives[input_label] += 1  # True label was not predicted


    # Calculate overall accuracy
    accuracy = correct_count / num_trials
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# This function writes our results to a file
def write_results(file, title, accuracy):
    with open("file", "a") as f:  # Use "a" to append to the file
        f.write(f"------------------- {title} ------------------- ")
        f.write(f"Accuracy = {accuracy * 100:.2f}%\n")

# This function loads the Yale dataset information and returns it
def load_yale():
    # Path to the Yale Face Database A
    dataset_path = "yale_dataset_for_second_evaluation"

    # Initialize lists to store images and labels
    X = []
    y = []

    # Loop through all subject directories
    for subject_dir in os.listdir(dataset_path):
        if subject_dir.startswith("subject"):  # Yale files are named "subjectXX"
            subject_path = os.path.join(dataset_path, subject_dir)
            for file_name in os.listdir(subject_path):
                if file_name.endswith(".pgm"):
                    # Load the image in grayscale
                    img_path = os.path.join(subject_path, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Check if the image was loaded successfully
                    if img is not None:
                        # Resize the image to a fixed size (64x64)
                        img = cv2.resize(img, (64, 64))

                        # Append image and label
                        X.append(img)
                        y.append(subject_dir)
                    else:
                        print(f"Error loading image: {img_path}")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    unique_y = np.unique(y)

    print(f"Loaded {len(X)} images with {len(np.unique(y))} unique labels.")
    X = X / 255
    return X, y, unique_y

# This function visualizes a one-shot trial, showcasing if the prediction was accurate or not
def visualize_one_shot_trials(X, y, unique_y_test, model):
    # Perform a classification test to get predictions and the input image
    X1, X2 = [], []
    indices_test = np.where(np.isin(y, unique_y_test))[0]
    input_index = np.random.choice(indices_test)
    input_image = X[input_index]
    input_label = y[input_index]

    # Create a support set and compute predictions
    for unique_label in unique_y_test:
        indices_person = np.where(y == unique_label)[0]
        index_test_image = np.random.choice(indices_person)
        test_image = X[index_test_image]
        X1.append(input_image)
        X2.append(test_image)

    X1, X2 = np.array(X1), np.array(X2)
    predictions = model.predict([X1, X2])
    max_index = np.argmax(predictions)
    predicted_label = unique_y_test[max_index]
    correctTitle = "Correct prediction" if predicted_label == input_label else "Incorrect prediction"
    # Set up the layout: 2 rows with 6 slots each
    plt.figure(figsize=(15, 8))

    # First row: input image + 5 support set images
    plt.subplot(2, 6, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title(f"Input Label:\n{input_label}")
    plt.axis("off")
    
    for i, (test_image, score, label) in enumerate(zip(X2[:5], predictions.flatten()[:5], unique_y_test[:5])):
        plt.subplot(2, 6, i + 2)
        plt.imshow(test_image, cmap="gray")
        color = "green" if label == predicted_label else "red"
        plt.title(f"Label: {label}\nScore: {score:.2f}", color=color)
        plt.axis("off")
    
    # Second row: Indented - reserve first slot with an empty plot
    plt.subplot(2, 6, 7)  # Empty slot at the beginning of the second row to create indentation
    plt.axis("off")
    # Second row: Indented - skip 1 slot before starting from 7th position
    for i, (test_image, score, label) in enumerate(zip(X2[5:], predictions.flatten()[5:], unique_y_test[5:])):
        plt.subplot(2, 6, i + 8)  # Start at position 7 for the second row
        plt.imshow(test_image, cmap="gray")
        color = "green" if label == predicted_label else "red"
        plt.title(f"Label: {label}\n Similarity Score: {score:.2f}", color=color)
        plt.axis("off")

    # Display the predicted label
    plt.tight_layout()
    plt.suptitle(f"Predicted Label: {predicted_label} ({correctTitle})", fontsize=16)
    plt.show()
