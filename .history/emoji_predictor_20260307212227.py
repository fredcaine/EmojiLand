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



# Fredrick Farouk. emoji_ai_training.py
# All logic here is the same as emotion_ai_training.py, so commenting is minimal.

# Important NOTE: This gives not great predictions. There is no pre-existing database of GoEmotions text with emoji labelling.
# Thus, I had to label it myself, which I talk about more in project_notes.txt.
# Hence, the database is 300 lines long; pretty much nothing for learning emoji usage patterns.

# Unfortunately, it is pretty much impossible to train these models without overfitting.
# The models that are currently trained are incredibly overfit. Therefore, the model used in emoji_predictor.py is one where I manually wrote the weights.

import numpy as np
from pandas import read_csv

# Did not include a way to continue training, as the dataset is so small training is almost instant.
current_epochs = 0
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def bce(y_pred, y_true):
    total = 0
    for i in range(len(y_pred)):
        total += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
    mean = -total / len(y_pred)
    return mean

lr_w = 1e-5
lr_b = 1e-6

emojis = ["❤","😄","😐","😠","😢","😨","😮","🤔","🤢","🤩"]
# Hand-labelled csv of over 3000 labels.
db = read_csv("emoji_training_ai_requirements/emoji_db.csv")
# See preprocessing.py to see an explanation of how these files were created.
# This .npz encodes the best fitting emojis and the 28 emotions associated with several documents.
# There is only one array encoded in each of these, so I left their names as the default (arr_0)
inputs = np.load("emoji_training_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
answer_sheet = np.load("emoji_training_ai_requirements/emoji_lookup.npz")["arr_0"]
epochs = 1000001
for emoji in emojis:
    weights = np.zeros(shape=(28,1))
    bias = 0
    total_y_true = np.array(answer_sheet[:, emojis.index(emoji)]).reshape(-1, 1)

    train_y_true = total_y_true
    train_inputs = inputs

    valuation = []
    val_y_true = []
    # We find the last three instances of a '1' to create a valuation dataset (explained in emotion_ai_training.py and Project.txt).
    for i in range(len(total_y_true) - 1, -1, -1):
        if total_y_true[i] == 1:
            valuation.append(inputs[i])
            val_y_true.append(total_y_true[i])
            train_y_true = np.delete(train_y_true, i, axis=0)
            train_inputs = np.delete(train_inputs, i, axis=0)

            if len(valuation) >= 3:
                break

    valuation_inputs = np.array(valuation)
    val_y_true = np.array(val_y_true)

    for epoch in range(epochs):

        logits = train_inputs @ weights + bias 
        y_pred = sigmoid(logits)
        errors = (y_pred - train_y_true)/(y_pred * (1 - y_pred))
        grad_w = train_inputs.T.dot(errors) / len(y_pred)
        grad_b = np.mean(errors)
        weights -= grad_w * lr_w
        bias -= grad_b * lr_b

            # Again, writing this file was before I understood the importance of valuation loss to avoid overfitting.
            # Nevertheless, it's pretty much impossible to do anything with ~20-30 instances of each label, so there is nothing to be done about this.

        # Run a forward pass on valuation:
        val_logits = valuation @ weights + bias
        val_y_pred = sigmoid(val_logits)
        print(f"Valuation BCE: {bce(val_y_pred, val_y_true)}")



while True:
    prompt = input("Enter your prompt:\n")
    if prompt.lower() == "exit":
        break
    emoji_dict_solution = emoji_dict(prompt)
    print(emoji_dict_solution)
    print("Best emoji: ", max(emoji_dict_solution, key=emoji_dict_solution.get))