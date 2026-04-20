# Fredrick Farouk. emoji_predictor.py
# This file runs the forward pass to predict emojis.
# Most logic is in the training files or in emotion_predictor.py, so I will not comment much.
import preprocessing
import numpy as np
from emotion_predictor import create_prediction_vector

emojis = ["😄","😢","😠","😨","😮","🤢","❤","🤩","🤔","😐"]

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

trained_data = np.load("emoji_aigen_model_weights.npz")["arr_0"]

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

while True:
    prompt = input("Enter your prompt:\n")
    if prompt.lower() == "exit":
        break
    emoji_dict_solution = emoji_dict(prompt)
    print(create_prediction_dicts(prompt))
    print(emoji_dict_solution)
    print("best emoji: ", max(emoji_dict_solution, key=emoji_dict_solution.get))