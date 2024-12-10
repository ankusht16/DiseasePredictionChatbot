import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import os
from tkinter import *

# Initialize the stemmer
stemmer = LancasterStemmer()

# Load the trained model
model = load_model('chatbot_model.h5')

# Load the intents JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
intents_json = json.loads(open(os.path.join(BASE_DIR, 'intents2.json')).read())

# Load words and labels
words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
labels = pickle.load(open(os.path.join(BASE_DIR, 'labels.pkl'), 'rb'))

# Download NLTK resources if not already present
nltk.download('punkt')

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = set(clean_up_sentence(sentence))
    bag = [1 if word in sentence_words else 0 for word in words]
    if show_details:
        print("Words found:", sentence_words)
    return np.array(bag)

def predict_class(sentence, model):
    # Generate the bag of words for the input sentence
    p = bow(sentence, words, show_details=False)
    
    # Predict probabilities for all classes (intents)
    res = model.predict(np.array([p]))[0]

    # Print the entire list of probabilities with corresponding labels
    print("\nComplete Probability List:")
    for i, prob in enumerate(res):
        print(f"{labels[i]}: {prob:.4f}")

    # Filter out predictions below a threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort results by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Prepare the final result list with intents and their probabilities
    return_list = [{"intent": labels[r[0]], "probability": str(r[1])} for r in results]

    # Print the filtered list in the command line for quick debugging
    print("\nFiltered Probabilities Above Threshold:")
    print(return_list)

    return return_list

def getResponse(intents, intents_json):
    if not intents:
        default_response = "I'm sorry, I didn't understand that. Can you please rephrase?"
        return {"response": default_response, "precautions": [], "treatments": []}
    
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = {
                "response": random.choice(i['responses']),
                "precautions": i.get('precautions', []),
                "treatments": i.get('treatments', []),
            }
            break
    return result

def chatbot_response(text):
    # Predict intent
    intents = predict_class(text, model)
    
    # Get response information
    response_info = getResponse(intents, intents_json)
    
    # Display the main response in the chat log
    main_response = "Bot: " + response_info['response'] + '\n'
    ChatLog.insert(END, main_response)
    
    if response_info['response'] == "I'm sorry, I didn't understand that. Can you please rephrase?":
        ChatLog.insert(END, "Bot: You can try asking a different question.\n\n")
    else:
        precautions_text = " " + ', '.join(response_info.get('precautions', [])) + '\n'
        treatments_text = " " + ', '.join(response_info.get('treatments', [])) + '\n\n'
        ChatLog.insert(END, precautions_text)
        ChatLog.insert(END, treatments_text)

    return response_info

# Create the GUI with Tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        # Get chatbot response
        res = chatbot_response(msg)

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
