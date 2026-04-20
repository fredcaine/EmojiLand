import numpy as np
from scipy.sparse import load_npz, vstack
from pandas import read_csv
from ai_requirements.vectoriser import corpus

db = read_csv("emoji_ai_requirements/emoji_db.csv")
# completed: admiration, amusement, anger, annoyance, approval, v\caring, confusion, curiosity, desire, disappointment,
# disapproval, disgust, embarrassment, example_very_unclear, excitement, fear, gratitude, joy, grief, love, nervousness,
# optimism, pride, realization, relief, remorse, sadness, surprise, neutral
emojis = ["❤","😄","😐","😠","😢","😨","😮","🤔","🤢","🤩"]
def softmax(y):
    final_array = []
    for i in range(len(y)):
        final_array.append(np.exp(y[i]) / sum(np.exp(y)))
    return final_array

def bce(y_pred, y_true, pos_weight=20):  # technically doesn't need to be calculated
    # mostly to know how good the ai is performing.
    # we will do a mean loss, because there's 200k tokens to test over
    # pos_weight weights the loss so that it doesn't just guess zero bc it's the most common.
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)  # removes chance of a log(0)
    loss = -(pos_weight * y_true * np.log(y_pred) + ((1 - y_true) * np.log(1 - y_pred)))  # binary cross entropy formula
    return loss

emoji_checker = []
for emoji in emojis:
    emoji_checker.append(db[emoji])
# emoji_count = sum(emoji_checker)
epochs = 1000000
inputs = np.load("emoji_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
# trained_values = np.load(f"model_weights/{emoji}.npz")
# weights = trained_values["weights"]
# bias = trained_values["bias"].item()
# current_epochs = trained_values["epochs"].item()
lr = 0.01  # learning rate should be slow to avoid divergence bc machines are dumb
current_epochs = 0
y_true = np.load("emoji_ai_requirements/emoji_lookup.npz")["arr_0"]
weights = np.zeros((28,1))
pos_frac = y_true.mean()    # e.g. 0.05 for 5% positives
bias = 0

for epoch in range(epochs):

    logits = inputs @ weights  # (doccount, 1)
    preds = softmax(logits)
    errors = preds - y_true    # (doccount, 1)

    # (tokencount, doccount) @ (doccount, [1]) gives (tokencount, 1) gradient
    grad_w = inputs.T.dot(errors).reshape(-1, 1) / len(corpus)
    weights -= lr * grad_w  # slow it down

np.savez(f"emoji_model_weights/db_CCE_emoji_model_weights.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))
