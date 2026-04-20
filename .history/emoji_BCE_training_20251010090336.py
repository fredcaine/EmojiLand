import numpy as np
from pandas import read_csv

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
inputs = np.load("emoji_ai_requirements/ai_emoji_input_vector.npz").astype(float)  # gives 28-long emotion vectors

answer_sheet = np.load("emoji_ai_requirements/emoji_lookup.npz")["arr_0"]
epochs = 1000000

for emoji in emojis:
    weights = np.zeros(shape=(28,1))
    bias = 0
    y_true = answer_sheet[:,emojis.index(emoji)]  # [:,x] slices to extract column x (0-indexed, as is .index)

    for epoch in range(epochs):

        logits = inputs @ weights + bias  # make a prediction and squash it down to {0,1}
        y_pred = sigmoid(logits)  # this actually runs because numpy arrays work element wise with arithmetic operators
        errors = y_pred - y_true  # rewards confident answers linearly. this is the partial derivative of the BCE loss function
        grad_w = errors * lr_w * weights  # negative for a guess that's too low, positive for a guess that's too high
        grad_b = errors * lr_b
        weights -= grad_w  # negate it to adjust properly for the sign
        bias -= grad_b

        if epoch % 25000 == 0:
            print(f"training {emoji}. epoch {epoch}. bce loss: {bce(y_pred, y_true)}")