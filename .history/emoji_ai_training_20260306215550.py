# Fredrick Farouk. emoji_ai_training.py
# All logic here is the same as emotion_ai_training.py, so commenting is minimal.

# Important NOTE. This gives very bad predictions. There is no pre-existing database of GoEmotions text with emoji labelling.
# Thus, I had to label it myself, which I talk about in project_notes.txt.

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

lr_w = 0.01
lr_b = 1e-5

emojis = ["❤","😄","😐","😠","😢","😨","😮","🤔","🤢","🤩"]
# Hand-labelled csv of over 3000 labels.
db = read_csv("emoji_ai_requirements/emoji_db.csv")

# See preprocessing.py to see an explanation of how these files were created.
# This npz encodes the best fitting emojis and the 28 emotions associated with several documents.
# There is only one array encoded in each of these, so I left its name as the default (arr_0)
inputs = np.load("emoji_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
answer_sheet = np.load("emoji_ai_requirements/emoji_lookup.npz")["arr_0"]

epochs = 1000001

for emoji in emojis:
    weights = np.zeros(shape=(28,1))
    bias = 0
    # We find y_true by taking all entries on the column of the emoji index we are on.
    # Note that .index() is 0-indexed, as is slicing, so this does not illicit an off-by-one.
    # We also make the array's shape (doccount,1) via .reshape() to make matrix multiplication easier later on.
    y_true = np.array(answer_sheet[:, emojis.index(emoji)]).reshape(-1, 1)

    # This is the false-negative weighting discussed in emotion_ai_training.py, but applied this time.
    # In emotion_ai_training.py, filtering to half was used as a replacement. Here, that would've led to an incredibly small dataset.
    pos_weight = (len(y_true)/2) / sum(y_true)
    sample_weights = np.where(y_true == 1, pos_weight, 1.0)

    for epoch in range(epochs):

        logits = inputs @ weights + bias 
        y_pred = sigmoid(logits)
        errors = (y_pred - y_true) * sample_weights
        grad_w = inputs.T.dot(errors) / len(y_pred)
        grad_b = np.mean(errors)
        weights -= grad_w * lr_w
        bias -= grad_b * lr_b

        if epoch % 50000 == 0:
            print(f"training {emoji}. epoch {epoch}. bce loss: {bce(y_pred, y_true)}")
            print(f"""gradient of bias {grad_b * lr_b}. newest bias: {bias}
mean gradient of weights {np.mean(grad_w) * lr_w}. most recent mean of weights: {np.mean(weights)}""")
    np.savez(f"emoji_model_weights_posweighted/db_BCE_emoji_model_weights.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))