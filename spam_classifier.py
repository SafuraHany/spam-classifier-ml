from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
messages = [
    "Win a free lottery now",
    "Call this number to claim prize",
    "Hello how are you",
    "Let's meet tomorrow",
    "Congratulations you won money"
]

labels = ["spam", "spam", "ham", "ham", "spam"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test message
test_message = ["Free money waiting for you"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("Message:", test_message[0])
print("Prediction:", prediction[0])