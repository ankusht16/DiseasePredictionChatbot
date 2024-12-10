import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
import tensorflow as tf
import json

# Load Data
with open("intents2.json") as file:
    data = json.load(file)

texts = []
labels = []
label_map = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        if intent["tag"] not in label_map:
            label_map[intent["tag"]] = len(label_map)
        labels.append(label_map[intent["tag"]])

# Convert to DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Split Data
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Tokenize and Encode Data
def encode_data(texts, labels, max_length=128):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    labels = np.array(labels)
    
    print(f"padded_sequences shape: {padded_sequences.shape}")
    print(f"labels shape: {labels.shape}")

    return padded_sequences, labels

train_sequences, train_labels = encode_data(X_train, y_train)
val_sequences, val_labels = encode_data(X_val, y_val)

# Model
def create_model(vocab_size, embedding_dim=128, max_length=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SimpleRNN(128, return_sequences=True),  # First RNN layer with return_sequences=True
        Dropout(0.3),  # Dropout to prevent overfitting
        SimpleRNN(64),  # Second RNN layer
        Dense(64, activation='relu'),  # Dense layer to capture learned features
        Dropout(0.3),  # Dropout to prevent overfitting
        Dense(len(label_map), activation='softmax')  # Output layer for classification
    ])
    return model

vocab_size = len(tokenizer.word_index) + 1
model = create_model(vocab_size)

# Compile Model with adjusted learning rate
optimizer = Adam(learning_rate=0.0001)  # Learning rate that worked well in previous experiment
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Convert data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_labels)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.batch(32)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Increased epochs to allow more learning
    callbacks=[early_stopping]
)

# Evaluate Model
loss, accuracy = model.evaluate(val_dataset)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
