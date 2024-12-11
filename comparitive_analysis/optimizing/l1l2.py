import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import random
import json
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1_l2
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import cProfile

class MetricsCallback(Callback):
    def __init__(self, X_val, y_val):
        super(MetricsCallback, self).__init__()
        self.validation_data = (X_val, y_val)

    def on_epoch_end(self, epoch, logs=None):
        val_predictions = numpy.argmax(self.model.predict(self.validation_data[0]), axis=1)
        val_true = numpy.argmax(self.validation_data[1], axis=1)
        precision = precision_score(val_true, val_predictions, average='weighted', zero_division=0)
        recall = recall_score(val_true, val_predictions, average='weighted', zero_division=0)
        f1 = f1_score(val_true, val_predictions, average='weighted', zero_division=0)
        print(f"Epoch {epoch + 1} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1-score: {f1:.4f}")

class ProfilerCallback(Callback):
    def __init__(self):
        super(ProfilerCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.profile = cProfile.Profile()
        self.profile.enable()

    def on_epoch_end(self, epoch, logs=None):
        self.profile.disable()
        profile_path = f'./logs/plugins/profile/epoch_{epoch}.prof'
        self.profile.dump_stats(profile_path)
        print(f"Profiler log saved to: {profile_path}")

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

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

X_train, X_val, y_train, y_val = train_test_split(training, output, test_size=0.2, random_state=42)

# Define a list of optimizers to experiment with
optimizers = {
    'SGD': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    'Adam': Adam(),
    'RMSprop': RMSprop()
}

# Regularization strength
regularization_strength = 0.0001

for optimizer_name, optimizer in optimizers.items():
    # Create a new model for each optimizer
    model = Sequential()
    model.add(Dense(128, input_shape=(len(training[0]),), kernel_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength)))
    model.add(Activation('relu'))

    model.add(Dense(256, kernel_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength)))
    model.add(Activation('relu'))

    model.add(Dense(len(output[0]), activation='softmax'))

    # Compile the model with the current optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    metrics_callback = MetricsCallback(X_val, y_val)

    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

    profiler_callback = ProfilerCallback()

    # Train the model
    print(f"Training model with optimizer: {optimizer_name}")
    hist = model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=1000, 
                     batch_size=8, 
                     verbose=1,
                     callbacks=[metrics_callback, early_stopping, profiler_callback])

    stopped_epoch = early_stopping.stopped_epoch

    print(f"Training stopped at epoch: {stopped_epoch}")
