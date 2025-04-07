# One-Shot Facial Recognition Using Siamese Neural Networks

This project focuses on training and evaluating a Siamese Neural Network (SNN) for one-shot facial recognition. Using the Olivetti Faces dataset for training, the model is tested on both seen and unseen data (including the Yale face dataset) to evaluate its generalization capabilities in recognizing individuals from just a single example.

---

## Project Structure

### `snn_training.ipynb`
- Jupyter Notebook used to train and save the Siamese Neural Network.
- Uses 80,000 pairs of images from the Olivetti Faces dataset.
- Sections include:
  - Data loading and preprocessing
  - Organizing data into pairs
  - Building the CNN-based feature extractor (`cnn_part`)
  - Building the full Siamese architecture
  - Model training and saving

### `trained_model/`
- Directory that stores the trained SNN model after training is complete.

### `evaluation_one-shot_trials/`
Contains all files related to model evaluation:

- **`evaluation.py`**  
  Loads the trained SNN and performs one-shot learning evaluations on the Olivetti Faces dataset using 10 unseen individuals.  
  Each trial uses only one image per person for recognition.

- **`evaluation_unseen_dataset.py`**  
  Evaluates the same trained model on the Yale face dataset to test the model's ability to generalize to completely unseen classes.

- **`one_shot_functions.py`**  
  Contains helper functions used in both evaluation scripts for organizing and running one-shot learning trials.

---

## Dataset Information

- **Training Dataset:** Olivetti Faces (used to create training pairs)
- **Evaluation Datasets:**
  - Training dataset: Olivetti Faces (with identities excluded during training)
  - Second evaluation dataset (Not used during training): Yale Face Dataset

---
