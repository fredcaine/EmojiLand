# Fredrick Farouk. emoji_predictor.py
# This file runs the forward pass to predict emojis.
# Most logic is in the training files or in emotion_predictor.py, so I will not comment much.

import preprocessing
import numpy as np
from emotion_predictor import create_prediction_vector

emojis = ["😄","😢","😠","😨","😮","🤢","❤","🤩","🤔","😐"]

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# Yes, the item is called arr_0 as I never specified a name in preprocessing.
trained_data = np.load("emoji_model_weights_llm.npz")["arr_0"]

def predict(prompt, emoji, predictions, rounding=False):

    inputs = np.array(prompt)
    emoji_trained_data = trained_data[:,emojis.index(emoji)]
    weights = emoji_trained_data[:-1]
    bias = emoji_trained_data[-1]
    logit = inputs @ weights + bias
    prediction = float(sigmoid(logit))
    if rounding == True:
        predictions.append(round(prediction))
    elif rounding:
        predictions.append(round(prediction, rounding))
    else:
        predictions.append(prediction)
    
    return predictions

def create_emoji_prediction_vector(prompt):
    predictions = []
    for emoji in emojis:
        predictions = predict(prompt, emoji, predictions, rounding=15)
    return predictions

def emoji_dict(prompt):
    # The emoji weights take in a vector of predicted emotions, then return a respective emoji.
    emoji_preds = create_emoji_prediction_vector(create_prediction_vector(prompt))
    emoji_prediction_dict = {e:p for e, p in zip(emojis, emoji_preds)}
    return emoji_prediction_dict

# Here, I wrote a very brief script that iterates through every emoji and every line and prints the model's accuracy.
# This gave me a faster test than checking individual sentences myself.
# I primarily used this while adjusting the model's weights and biases manually.
emojis = ["😄","😢","😠","😨","😮","🤢","❤","🤩","🤔","😐"]
inputs = np.load("emoji_training_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
answer_sheet = np.load("emoji_training_ai_requirements/emoji_lookup.npz")["arr_0"]

for emoji in emojis:
    predictions = []
    for line in inputs:
        predictions = predict(line, emoji, predictions, True)
    y_pred = np.array(predictions)
    y_true = np.array(answer_sheet[:, emojis.index(emoji)])
    print(f"Accuracy for {emoji}: {round(100 * np.sum(y_pred == y_true) / len(predictions), 1)}%.\t\tPercentage frequency in dataset: {round(100 * np.sum(y_true) / len(predictions), 1)}%.\t\tPercentage positive predictions: {round(100 * np.sum(y_pred) / len(predictions), 1)}%.\t\tPercentage correct true predictions: {round(100 * np.sum(y_pred == y_true == 1) / len(predictions), 1)}%.")

print("\n\n")

# Code similar to emotion_predictor.py
# while True:
#     prompt = input("Enter your prompt:\n")
#     if prompt.lower() == "exit":
#         break
#     emoji_dict_solution = emoji_dict(prompt)
#     print(emoji_dict_solution)
#     print("Best emoji: ", max(emoji_dict_solution, key=emoji_dict_solution.get))