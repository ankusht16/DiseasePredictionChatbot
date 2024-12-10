import pickle

words = pickle.load(open('words.pkl', 'rb'))
print(f"Vocabulary size in words.pkl: {len(words)}")  # Should print 1305
