import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import json
import random
import pickle
import tkinter as tk

stemmer = LancasterStemmer()

# Load model and data
try:
    model = tf.keras.models.load_model('chatbot_model.h5')
    intents_json = json.loads(open('intents2.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    labels = pickle.load(open('labels.pkl', 'rb'))
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

print(f"Model input shape: {model.input_shape}")
print(f"Vocabulary size: {len(words)}")

# State management
session_state = {
    "current_intents": [],
    "all_intents_detected": [],  # Store all detected intents initially
    "asked_questions": [],
    "current_question": None,
    "awaiting_followup": False,
    "completed_intents": set()  # Track which intents have had their follow-ups
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]

def bow(sentence, words):
    bag = [0] * len(words)
    for s in clean_up_sentence(sentence):
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    input_length = model.input_shape[1]
    if len(bag) < input_length:
        bag.extend([0] * (input_length - len(bag)))
    elif len(bag) > input_length:
        bag = bag[:input_length]

    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.10

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    detected_intents = [labels[r[0]] for r in results]
    session_state["current_intents"] = detected_intents
    session_state["all_intents_detected"] = detected_intents[:]  # Store all initially detected intents
    return detected_intents

def get_follow_up_question():
    for intent in session_state["current_intents"]:
        if intent not in session_state["completed_intents"]:
            intent_data = next((i for i in intents_json["intents"] if i["tag"] == intent), None)
            if intent_data and "follow_up_questions" in intent_data:
                for question in intent_data["follow_up_questions"]:
                    if question["key"] not in session_state["asked_questions"]:
                        session_state["asked_questions"].append(question["key"])
                        session_state["current_question"] = question
                        return question
            session_state["completed_intents"].add(intent)  # Mark intent as completed if no more questions
    return None

def handle_follow_up_response(response):
    response = response.lower()

    # Validate input is only 'yes' or 'no'
    if response not in ["yes", "no"]:
        return {"response": "Please respond with 'yes' or 'no'."}

    question_data = session_state["current_question"]
    effect_key = "yes_effect" if response == "yes" else "no_effect"
    effect = question_data.get(effect_key)

    if effect:
        intent = effect.get("increase_probability") or effect.get("decrease_probability")

        if intent in session_state["current_intents"]:
            # Remove the intent from the list first to avoid duplication
            session_state["current_intents"].remove(intent)

            # Move intent to the correct position based on effect
            if effect_key == "yes_effect" and "increase_probability" in effect:
                session_state["current_intents"].insert(0, intent)  # Move to top
                print(f"Moved {intent} to the top of the list.")
            elif effect_key == "no_effect" and "decrease_probability" in effect:
                session_state["current_intents"].append(intent)  # Move to bottom
                print(f"Moved {intent} to the bottom of the list.")

        print(f"Updated intent order: {session_state['current_intents']}")

    # Proceed to next follow-up question or finalize
    next_question = get_follow_up_question()
    if next_question:
        return {"response": next_question["question"]}
    else:
        return finalize_response()

def finalize_response():
    if session_state["current_intents"]:
        final_intent = session_state["current_intents"][0]
        intent_data = next(i for i in intents_json["intents"] if i["tag"] == final_intent)
        response = {
            "response": random.choice(intent_data["responses"]),
            "precautions": ", ".join(intent_data.get("precautions", [])),
            "treatments": ", ".join(intent_data.get("treatments", []))
        }
        reset_session()
        return response
    else:
        reset_session()
        return {"response": "I'm sorry, I couldn't determine the condition. Please consult a doctor."}

def reset_session():
    session_state.update({
        "current_intents": [],
        "all_intents_detected": [],
        "asked_questions": [],
        "current_question": None,
        "awaiting_followup": False,
        "completed_intents": set()
    })

def chatbot_response(text):
    if session_state["awaiting_followup"]:
        return handle_follow_up_response(text)
    else:
        intents = predict_class(text)

        if len(intents) > 1:
            first_question = get_follow_up_question()
            if first_question:
                session_state["awaiting_followup"] = True
                return {"response": first_question["question"]}
            else:
                return finalize_response()
        else:
            return finalize_response()

# GUI Setup
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, f"You: {msg}\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        response = chatbot_response(msg)
        ChatLog.insert(tk.END, f"Bot: {response['response']}\n\n")
        if "precautions" in response:
            ChatLog.insert(tk.END, f"Precautions: {response['precautions']}\n")
        if "treatments" in response:
            ChatLog.insert(tk.END, f"Treatments: {response['treatments']}\n")

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

base = tk.Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=False, height=False)

ChatLog = tk.Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=tk.DISABLED)
scrollbar = tk.Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = tk.Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                       bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send)

EntryBox = tk.Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
