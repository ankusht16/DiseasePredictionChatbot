import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and Encode Data
def encode_data(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_masks shape: {attention_masks.shape}")
    print(f"labels shape: {labels.shape}")

    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = encode_data(X_train, y_train)
val_input_ids, val_attention_masks, val_labels = encode_data(X_val, y_val)

# Model
def create_model():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
    return model

model = create_model()

# Explicitly define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile Model
optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Convert data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_input_ids, 'attention_mask': train_attention_masks}, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_labels)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_input_ids, 'attention_mask': val_attention_masks}, val_labels))
val_dataset = val_dataset.batch(32)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Evaluate Model
loss, accuracy = model.evaluate(val_dataset)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
