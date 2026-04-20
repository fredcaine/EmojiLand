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
    prediction = sigmoid(logit)
    if rounding == True:
        predictions.append(round(prediction))
    elif rounding:
        predictions.append(float(round(prediction, rounding)))
    else:
        predictions.append(float(prediction))
    
    return predictions

def create_emoji_prediction_vector(prompt):
    predictions = []
    for emoji in emojis:
        predictions = predict(prompt, emoji, predictions, rounding=15)
    return predictions

def emoji_dict(prompt):
    emoji_preds = create_emoji_prediction_vector(create_prediction_vector(prompt)[1:])  # exclude clarity emotion with [1:]
    emoji_prediction_dict = {e:p for e, p in zip(emojis, emoji_preds)}
    return emoji_prediction_dict

while True:
    prompt = input("enter your prompt:\n")
    if prompt == "exit":
        break
    emoji_dict_solution = emoji_dict(prompt)
    print(emoji_dict_solution)
    print("best emoji: ", max(emoji_dict_solution, key=emoji_dict_solution.get))