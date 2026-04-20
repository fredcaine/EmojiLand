import numpy as np
from pandas import read_csv
current_epochs = 0
def softmax(y):
    final_array = []
    for i in range(len(y)):
        final_array.append(np.exp(y[i]) / sum(np.exp(y)))
    return final_array
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
epochs = 1000000
inputs = np.load("emoji_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
current_epochs = 0
y_true = np.load("emoji_ai_requirements/emoji_lookup.npz")["arr_0"]
weights = np.zeros((28,1))
pos_frac = y_true.mean()    # e.g. 0.05 for 5% positives
bias = pos_frac

for epoch in range(epochs):

    logits = inputs @ weights  # (doccount, 1)
    preds = softmax(logits)
    errors = preds - y_true    # (doccount, 1)

    # (tokencount, doccount) @ (doccount, [1]) gives (tokencount, 1) gradient
    grad_w = inputs.T.dot(errors).reshape(-1, 1) / inputs.shape[0]
    grad_b = np.mean(errors)

    weights -= grad_w * lr_w  # slow it down
    bias -= grad_b * lr_b
if epoch % 25000 == 0:
    print()

np.savez(f"emoji_model_weights/db_CCE_emoji_model_weights.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))
