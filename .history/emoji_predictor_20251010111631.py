import numpy as np
from emotion_predictor import create_prediction_vector

bce_tf = bool(input("Would you like a BCE or CCE model? Respond with anything for BCE and just press enter for CCE.\n").lower())
ai_tf = bool(input("""Would you like a model trained on another one of my models or a model trained on hand-labelled data?
Respond with anything for another model and just press enter for hand-labelled.\n""").lower())
emojis = ["😄","😢","😠","😨","😮","🤢","❤","🤩","🤔","😐"]

if bce_tf and ai_tf:
    weights_tbu = "ai_BCE_emoji_model_weights"
elif bce_tf:
    weights_tbu = "db_BCE_emoji_model_weights"
elif ai_tf:
    weights_tbu = "ai_CCE_emoji_model_weights"
else:
    weights_tbu = "db_CCE_emoji_model_weights"

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def predict(prompt, emoji, predictions, rounding=False):

    inputs = np.array(prompt)
    trained_data = np.load(f"{weights_tbu}/{emoji}.npz")
    weights = trained_data["weights"]
    bias = trained_data["bias"].item()
    print(bias)
    logit = inputs @ weights + bias
    prediction = sigmoid(logit)[0]
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
    emoji_preds = create_emoji_prediction_vector(create_prediction_vector(prompt))
    emoji_prediction_dict = {e:p for e, p in zip(emojis, emoji_preds)}
    return emoji_prediction_dict

while True:
    prompt = input("enter your prompt:\n")
    if prompt == "exit":
        break
    emoji_dict_solution = emoji_dict(prompt)
    print(emoji_dict_solution)
    print("best emoji: ", max(emoji_dict_solution, key=emoji_dict_solution.get))