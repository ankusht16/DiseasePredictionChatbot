import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Custom Callback for Metrics
class MetricsCallback(Callback):
    def __init__(self, X_val, y_val):
        super(MetricsCallback, self).__init__()
        self.validation_data = (X_val, y_val)

    def on_epoch_end(self, epoch, logs=None):
        val_predictions = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        val_true = np.argmax(self.validation_data[1], axis=1)
        precision = precision_score(val_true, val_predictions, average='weighted', zero_division=0)
        recall = recall_score(val_true, val_predictions, average='weighted', zero_division=0)
        f1 = f1_score(val_true, val_predictions, average='weighted', zero_division=0)
        print(f"Epoch {epoch + 1} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1-score: {f1:.4f}")

# Load Data
with open("intents2.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

# Prepare Training Data
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1 if w in wrds else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

X_train, X_val, y_train, y_val = train_test_split(training, output, test_size=0.2, random_state=42)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * 0.95

# Optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5),
    'Adam': Adam(clipvalue=0.5),
    'RMSprop': RMSprop(clipvalue=0.5)
}

# Models
models = {
    'Single Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3)],
    'Two Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3)],
    'Three Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3),
                    Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                    Dense(128), BatchNormalization(), Activation('relu')],
    'Four Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu')],
    'Five Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                   Dense(128), BatchNormalization(), Activation('relu')],
    'Six Layer': [Dense(128, input_shape=(len(training[0]),)), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
                  Dense(128), BatchNormalization(), Activation('relu')],
}

# Train and Evaluate Models
results = []

def train_and_evaluate_model(optimizer_name, optimizer, model_name, layers):
    model = Sequential(layers)
    model.add(Dense(len(output[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    metrics_callback = MetricsCallback(X_val, y_val)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(scheduler)

    print(f"Training model with {model_name} and optimizer: {optimizer_name}")
    hist = model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=1000, 
                     batch_size=16, 
                     verbose=1,
                     callbacks=[metrics_callback, early_stopping, lr_scheduler])

    stopped_epoch = early_stopping.stopped_epoch
    final_accuracy = hist.history['accuracy'][stopped_epoch]  # Corrected to training accuracy
    final_loss = hist.history['loss'][stopped_epoch]
    final_val_loss = hist.history['val_loss'][stopped_epoch]
    final_val_accuracy = hist.history['val_accuracy'][stopped_epoch]
    
    results.append((optimizer_name, model_name, stopped_epoch, final_accuracy, final_loss, final_val_loss, final_val_accuracy))

for optimizer_name, optimizer in optimizers.items():
    for model_name, layers in models.items():
        train_and_evaluate_model(optimizer_name, optimizer, model_name, layers)

# Save results to a file
with open('model_training_results.txt', 'w') as f:
    for result in results:
        f.write(f"Optimizer: {result[0]}, Model: {result[1]}, Stopped Epoch: {result[2]}, "
                f"Final Accuracy: {result[3]:.4f}, Final Loss: {result[4]:.4f}, "
                f"Final Validation Loss: {result[5]:.4f}, Final Validation Accuracy: {result[6]:.4f}\n")

# Display results
for result in results:
    print(f"Optimizer: {result[0]}, Model: {result[1]}, Stopped Epoch: {result[2]}, "
          f"Final Accuracy: {result[3]:.4f}, Final Loss: {result[4]:.4f}, "
          f"Final Validation Loss: {result[5]:.4f}, Final Validation Accuracy: {result[6]:.4f}")

print('Results saved to model_training_results.txt')
