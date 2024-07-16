import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import json

# Load NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load intents JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Function to preprocess sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(stemmer.stem(word.lower())) for word in sentence_words if word.lower() not in stop_words]
    return ' '.join(sentence_words)

# Prepare training data
sentences = []
labels = []

for intent in intents['intents']:
    for example in intent['examples']:
        sentences.append(clean_up_sentence(example))
        labels.append(intent['intent'])

# Vectorize sentences and train the model using a pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Define hyperparameters for GridSearch
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10]
}

# Perform GridSearch to find the best parameters
skf = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(pipeline, parameters, cv=skf)
grid_search.fit(sentences, labels)

# Save the model and vectorizer for future use
joblib.dump(grid_search.best_estimator_, 'chatbot_model.pkl')

# Function to classify user input and generate response
def classify(sentence):
    sentence = clean_up_sentence(sentence)
    model = joblib.load('chatbot_model.pkl')
    X_test = model.named_steps['vectorizer'].transform([sentence])
    intent_tag = model.named_steps['classifier'].predict(X_test)[0]
    return intent_tag, model.named_steps['classifier'].predict_proba(X_test).max()

# Function to get response for classified intent
def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['intent'] == intent_tag:
            return np.random.choice(intent['response'])

# Interactive chat session with the user
print("Start chatting with the chatbot (type 'quit' to exit).")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print('see you again :)')
        break

    intent_tag, confidence = classify(user_input)
    if confidence < 0.5:
        response = "I'm not sure I understand. Can you please rephrase?"
    else:
        response = get_response(intent_tag)
    print(f"Bot: {response}")
