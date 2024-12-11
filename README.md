# Disease Prediction Chatbot Optimized for Low-End Devices
This project is a Disease Prediction Chatbot designed specifically to address the challenges faced in rural and remote areas, where users often rely on low-end devices and have limited or no internet connectivity. The chatbot provides an offline solution, enabling individuals to identify potential diseases based on their symptoms, even in resource-constrained environments.

Key Features:
Optimized for Low-End Devices:
The model architecture avoids complex transformers like BERT and DistilBERT, instead leveraging simpler architectures such as Feedforward Neural Networks (FNN). This ensures a smaller memory and processing footprint without compromising accuracy.

Follow-Up Questions for Diagnosis:
The chatbot mimics the process of a real doctor by asking follow-up questions. This approach enhances the accuracy of disease prediction and improves the user experience.

Offline Accessibility:
Unlike many modern models that rely on an active internet connection, this chatbot works completely offline, making it ideal for areas with poor network connectivity.

Accurate Yet Lightweight:
The project demonstrates how simpler models, when trained on carefully designed datasets, can achieve accuracy comparable to more complex architectures like BERT while being lightweight and accessible.


# Requirements
python -v 3.7

pip install -r requirements.txt 


# Instructions
to get started with the chatbot, you will primarily work with two files:

1) chatbot_train.py

This script is used to train the model on the dataset.
It prepares the chatbot to understand and predict diseases based on user-provided symptom patterns.

2) chatbot_guifollowup.py

This script provides the graphical user interface (GUI) for the chatbot.
It interacts with the user, asks follow-up questions, and predicts diseases based on the responses.

# Steps to Use:
1) Training the Model:

Run chatbot_train.py to train the model.
Ensure you have the required dataset in place before starting.

2) Running the Chatbot:

Execute chatbot_guifollowup.py to launch the chatbot GUI.
Follow the on-screen instructions to interact with the chatbot and receive predictions.

# Details
chatbot_train.py is a python file in which we train the model with the help of available dataset.
Dataset is stored in the json file (intents2.json).
chatbot_guifollowup.py is a file which will open a GUI prompt where user can talk with chatbot and interact with it.

# About the Dataset
The dataset is specifically designed to describe symptoms in everyday language, avoiding complex medical terminology. This ensures that even non-expert users can interact with the chatbot effectively.
