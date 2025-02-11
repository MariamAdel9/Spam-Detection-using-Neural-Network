import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the saved model and vectorizer
model_filename = "spam_detection_nn_model.h5"
vectorizer_filename = "count_vectorizer.pkl"

model = load_model(model_filename)
cv = pickle.load(open(vectorizer_filename, 'rb'))

# Preprocessing the text data (same as during training)
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure the text is a string
        return ""
        
    # Clean the text by removing non-alphabet characters
    review = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase and split into words
    review = review.lower().split()
    # Remove stopwords and apply stemming
    ps = nltk.PorterStemmer()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    # Join words back together
    processed_text = ' '.join(review)

    return processed_text

# Function to predict spam or ham
def predict_spam_or_ham(message):
    # Preprocess and vectorize the input message
    processed_message = preprocess_text(message)
    vectorized_message = cv.transform([processed_message]).toarray()
    
    # Predict using the trained model
    prediction = model.predict(vectorized_message)
    return 'spam' if prediction >= 0.5 else 'real'

# Example usage:
test_message = "Congratulations! You've won a $1000 gift card. Call now to claim."
prediction = predict_spam_or_ham(test_message)
print(f"Predicted label for the input message: {prediction}")

test_message1 = "Call me later please!"
prediction1 = predict_spam_or_ham(test_message1)
print(f"Predicted label for the input message: {prediction1}")


test_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim.",
    "Call me later please!",
    "Limited-time offer! Click the link to win exciting prizes.",
    "Don't forget about the meeting at 3 PM tomorrow.",
    "URGENT! Your account will be deactivated unless you verify your details immediately.",
    "Hey, want to grab lunch today?",
    "You've been pre-approved for a personal loan. Apply now!",
    "I'll be home by 7 PM. See you then.",
    "This is a friendly reminder to pay your utility bill by the due date.",
    "Win a free vacation! Just reply with your email address.",
    "Are you free this weekend? Let's plan something fun!",
    "Get 50% off on all products. Shop now before the sale ends!",
    "Your OTP is 123456. Do not share it with anyone.",
    "Hi! Just checking in to see how you're doing.",
    "Claim your free Bitcoin now! Visit our website to learn more.",
    "Please review the attached document and provide feedback.",
    "Congrats on your promotion! Let's celebrate soon.",
    "Your subscription has been successfully renewed.",
    "Important: Update your bank details to avoid account suspension.",
    "See you at the party tonight. Don't forget to bring your favorite dish!",
    "Reminder: Your doctor's appointment is scheduled for tomorrow at 10 AM.",
    "Earn money working from home! No prior experience needed.",
    "Can you send me the latest version of the report?",
    "Exclusive deal just for you! Use code SAVE50 at checkout.",
    "Hope you had a great weekend! Let's catch up soon.",
    "You’re one step away from activating your account. Click here!",
    "Let's reschedule our meeting to a more convenient time.",

]


for i, message in enumerate(test_messages, 1):
    prediction = predict_spam_or_ham(message)  
    print(f"Message {i}: {message}")
    print(f"Predicted label: {prediction}")
    print("--------------------------------------------------")
