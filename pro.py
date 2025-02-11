import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the fraud email dataset
fraud_file_path = r'C:\Users\mariam\aipro\fraud_email_.csv'
df_fraud = pd.read_csv(fraud_file_path)

# Clean up column names (remove leading/trailing spaces)
df_fraud.columns = df_fraud.columns.str.strip()

# Convert the 'Class' column to spam/ham labels (1 -> spam, 0 -> ham)
df_fraud['label'] = df_fraud['Class'].apply(lambda x: 'spam' if x == 1 else 'ham')

# Assuming the text column is 'Text'
df_fraud = df_fraud[['Text', 'label']]  # Keep only text and label columns
df_fraud.columns = ['message', 'label']  # Rename for consistency

# Load the spam dataset
spam_file_path = r'C:\Users\mariam\aipro\spam.csv'
df_spam = pd.read_csv(spam_file_path)

# Clean up column names in the spam dataset
df_spam.columns = df_spam.columns.str.strip()

# Assuming the class column is 'v1' for label and 'v2' for message in the spam dataset
df_spam = df_spam[['v1', 'v2']]
df_spam.columns = ['label', 'message']  # Rename for consistency

# Combine both datasets into one
df = pd.concat([df_fraud, df_spam], ignore_index=True)

# Handle missing or NaN values in the 'message' column
df['message'] = df['message'].fillna('')

# Preprocessing the text data
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure the text is a string
        return ""
    ps = nltk.PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # Remove stopwords and stem
    return ' '.join(review)

# Apply text preprocessing
df['processed_message'] = df['message'].apply(preprocess_text)

# Vectorize the text data using CountVectorizer
cv = CountVectorizer(max_features=4000)  # Adjust max_features based on your dataset size
X = cv.fit_transform(df['processed_message']).toarray()
Y = pd.get_dummies(df['label'])
Y = Y.iloc[:, 1].values  # Only keep 'spam' column as target variable

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=X_train.shape[1]))  # Adjust units based on dataset size
model.add(Dropout(0.5))  # Dropout layer to avoid overfitting
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))
# Save the trained model
model_filename = "spam_detection_nn_model.h5"
model.save(model_filename)


# Save the vectorizer as well for future use
vectorizer_filename = "count_vectorizer.pkl"
pickle.dump(cv, open(vectorizer_filename, 'wb'))
