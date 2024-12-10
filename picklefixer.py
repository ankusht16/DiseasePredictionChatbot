import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
import nltk

stemmer = LancasterStemmer()

# Step 1.1: Load the latest intents JSON
with open('intents2.json', 'r') as f:
    intents = json.load(f)

words = []

# Step 1.2: Extract patterns from the intents and tokenize them
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(stemmer.stem(word.lower()) for word in word_list)

# Step 1.3: Remove duplicates and sort the vocabulary
words = sorted(set(words))

# Step 1.4: Save the new words.pkl file
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

print(f"New vocabulary size: {len(words)}")
