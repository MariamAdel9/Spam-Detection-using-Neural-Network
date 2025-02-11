
# Spam Detection Neural Network

This project implements a spam detection system using a neural network. It combines two datasets containing spam and real messages, preprocesses the text data, and trains a neural network model to classify messages as either "spam" or "real". The trained model and the text vectorizer are then saved for later use.



## Overview

The project performs the following tasks:
- **Data Loading:** Reads two CSV files (one for fraud emails and one for spam messages) and cleans the data.
- **Preprocessing:** Cleans and preprocesses the text data by removing non-alphabet characters, converting text to lowercase, removing stopwords, and applying stemming.
- **Feature Extraction:** Converts the preprocessed text into numerical features using `CountVectorizer`.
- **Model Training:** Builds and trains a neural network using TensorFlow Keras for binary classification (spam vs. ham).
- **Saving Artifacts:** Saves the trained neural network model and the CountVectorizer for future predictions.


## Prerequisites

Ensure you have Python 3 installed. The project requires the following Python packages:

- **pandas**
- **nltk**
- **scikit-learn**
- **tensorflow**

## Datasets

- **fraud_email_.csv**: Contains fraud email messages with a `Class` column (where 1 indicates spam and 0 indicates ham) and a `Text` column containing the message.
- **spam.csv**: Contains spam messages with `v1` as the label and `v2` as the message content.

Ensure these CSV files are available at the paths specified in the code, or update the file paths accordingly.


## Code Workflow

1. **Import Libraries:**  
   The script imports libraries such as `pandas`, `re`, `nltk`, `sklearn`, and `tensorflow`.

2. **Data Loading and Cleaning:**  
   - Reads the two CSV datasets.
   - Cleans column names by stripping extra spaces.
   - Maps the `Class` column in the fraud email dataset to spam/ham labels.
   - Concatenates both datasets into one DataFrame.
   - Handles missing values in the message column.

3. **Text Preprocessing:**  
   A function `preprocess_text` is defined to:
   - Remove non-alphabet characters.
   - Convert text to lowercase.
   - Remove stopwords using NLTK's English stopwords list.
   - Apply stemming using NLTK's `PorterStemmer`.

4. **Feature Extraction:**  
   The preprocessed text is vectorized using `CountVectorizer` (with a maximum of 4000 features).

5. **Model Building and Training:**  
   - A neural network is built using TensorFlow's Keras API.
   - The model architecture includes:
     - A Dense layer with 512 units and ReLU activation.
     - A Dropout layer to prevent overfitting.
     - A second Dense layer with 256 units.
     - An output Dense layer with 1 unit and sigmoid activation for binary classification.
   - The model is compiled with the Adam optimizer and binary cross-entropy loss.
   - The model is trained for 10 epochs on 80% of the data, with 20% reserved for validation.

6. **Saving the Model and Vectorizer:**  
   - The trained model is saved as `spam_detection_nn_model.h5`.
   - The `CountVectorizer` is saved as `count_vectorizer.pkl` using the `pickle` module.

## example of running
![example](image.png)
The model may not be very accurate due to the limited data the model was trained on, for more accuracy train the model with more datasets.

## Authors
Mariam Adel https://github.com/MariamAdel9 
