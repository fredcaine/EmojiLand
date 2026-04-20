import numpy as np
from pandas import read_csv
current_epochs = 0
def sigmoid(y):  # reduces inputs to a range 0-1
    return 1 / (1 + np.exp(-y))
def bce(y_pred, y_true):
    N = len(y_pred)
    total = 0
    for i in range(N):
        total += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
    mean = -total / len(y_pred)
    return mean  # takes in two lists and returns the mean BCE loss

lr_w = 0.01  # the speed the bot learns
lr_b = 1e-5

emojis = ["❤","😄","😐","😠","😢","😨","😮","🤔","🤢","🤩"]
db = read_csv("emoji_ai_requirements/emoji_db.csv")
inputs = np.load("emoji_ai_requirements/db_emoji_input_vector.npz")["arr_0"]  # gives 28-long emotion vectors

answer_sheet = np.load("emoji_ai_requirements/emoji_lookup.npz")["arr_0"]
epochs = 10000000

for emoji in emojis:
    weights = np.zeros(shape=(28,1))
    bias = 0
    y_true = np.array(answer_sheet[:, emojis.index(emoji)]).reshape(-1, 1)  # [:,x] slices to extract column x (0-indexed, as is .index)
    # made an array (doccount,1) via .reshape() to make multiplication easier later

    for epoch in range(epochs):

        logits = inputs @ weights + bias  # make a prediction and squash it down to {0,1}
        y_pred = sigmoid(logits)  # this actually runs because numpy arrays work element wise with arithmetic operators
        # y_true is made an array (doccount,1) such that it matches y_pred, which is an array and therefore (doccount,1)
        errors = y_pred - y_true  # rewards confident answers linearly. this is the partial derivative of the BCE loss function
        grad_w = inputs.T.dot(errors) / len(y_pred) # negative for a guess that's too low, positive for a guess that's too high
        grad_b = np.mean(errors)
        weights -= grad_w * lr_w  # negate it to adjust properly for the sign
        bias -= grad_b * lr_b

        if epoch % 50000 == 0:
            print(f"training {emoji}. epoch {epoch}. bce loss: {bce(y_pred, y_true)}")
            print(f"""gradient of bias {grad_b * lr_b}. newest bias: {bias}
mean gradient of weights {np.mean(grad_w) * lr_w}. most recent mean of weights: {np.mean(weights)}
first ten preds {y_pred[:10]}. first ten raw logits {logits[:10]}
first ten weights {weights.flatten()[:10]}""")
    np.savez(f"emoji_model_weights/db_BCE_emoji_model_weights.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))