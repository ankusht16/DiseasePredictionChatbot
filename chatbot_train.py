import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_model_optimization as tfmot
import shap
import json
import pickle

# Load the dataset
with open("intents2.json") as file:
    data = json.load(file)

# Preprocess data
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Sorting labels
labels = sorted(labels)

# Saving labels into a pickle file
pickle.dump(labels, open('labels.pkl', 'wb'))

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
training = tfidf.fit_transform(docs_x).toarray()  # Sparse matrix for optimization
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))  # Save TF-IDF model

# Create output labels
output = []
for doc in docs_y:
    output_row = [0] * len(labels)
    output_row[labels.index(doc)] = 1
    output.append(output_row)

output = np.array(output)

# Compact neural network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(Dense(64, input_shape=(training.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(len(labels), activation='softmax'))

# Apply pruning with reduced final sparsity
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,  # Start with 20% sparsity
        final_sparsity=0.6,    # Reduce to 60% sparsity
        begin_step=2000,       # Start pruning after 2000 steps
        end_step=10000         # Finish pruning at 10000 steps
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the pruned model
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs')
]

history = pruned_model.fit(training, output, epochs=50, batch_size=32, verbose=1, callbacks=callbacks)

# Strip pruning wrappers for deployment
pruned_model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Save the pruned model
pruned_model_stripped.save("pruned_model.h5")

# Quantize the pruned model for efficiency
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model_stripped)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('pruned_model.tflite', 'wb') as f:
    f.write(tflite_model)

# SHAP explainability
explainer = shap.KernelExplainer(pruned_model_stripped.predict, training[:100])  # Use a small subset for explanation
shap_values = explainer.shap_values(training[:5])  # Explain the first 5 samples
shap.summary_plot(shap_values, training[:5], feature_names=tfidf.get_feature_names_out())

print("Model training, pruning, and quantization complete!")
