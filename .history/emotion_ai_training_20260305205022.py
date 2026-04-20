import numpy as np
from scipy.sparse import load_npz, vstack
from pandas import read_csv
from ai_requirements.vectoriser import corpus

# Completed: admiration, amusement, anger, annoyance, approval, v\caring, confusion, curiosity, desire, disappointment,
# disapproval, disgust, embarrassment, example_very_unclear, excitement, fear, gratitude, joy, grief, love, nervousness,
# optimism, pride, realization, relief, remorse, sadness, surprise, neutral

db = read_csv("ai_requirements/go_emotions_dataset.csv")

# List of emotions left to train.
emotions = ["neutral"]

def sigmoid(y):
    # Note: I added this clipping here because there was an issue in my old training that led to strange infinite logits etc.
    # This is no longer an issue, but I thought I would leave it here anyways.
    y = np.clip(y, -709, 709)

    # The known sigmoid function. This reduces all logits to a range between 0 and 1.
    return 1 / (1 + np.exp(-y))

def bce(y_pred, y_true, pos_weight=20):

    # Technically doesn't need to be calculated.
    # Mostly to know how good the ai is performing, in order to allow early stopping to avoid overfitting, etc.
    # We will do a mean loss, because there's 200k tokens to test over.
    # pos_weight weights the loss: since our model is simple gradient descent, it is likely going to simply choose 0.
    # The database has 0 as the correct decision the majority of the time.
    # Thus, we weight getting false negatives as the worst thing possible to incentivize choosing positives when possible.
    eps = 1e-7
    # We use an epsilon of 1e-7 to avoid a math error with ln(0),
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(pos_weight * y_true * np.log(y_pred) + ((1 - y_true) * np.log(1 - y_pred)))  # binary cross entropy formula
    return loss

for emotion in emotions:
    print(f"now training: {emotion}")
    emotion_checker = db[emotion]
    emotion_count = sum(emotion_checker)
    epochs = 100000
    inputs = load_npz("ai_requirements/tfidf_vector/tfidf_vector.npz")
    trained_values = np.load(f"model_weights/{emotion}.npz")
    weights = trained_values["weights"]
    bias = trained_values["bias"].item()
    current_epochs = trained_values["epochs"].item()
    lr = 1 # learning rate should be slow to avoid divergence bc machines are dumb
    # current_epochs = 0
    # y_true = emotion_checker.to_numpy()   # shape: (doccount,)
    # weights = np.zeros((len(vocab), 1))
    # pos_frac = y_true.mean()    # e.g. 0.05 for 5% positives
    # bias = 0

    def filter_to_half(inputs):
        filtered_rows, filtered_labels = [], []
        i, loop = 0, 0
        while i <= emotion_count:

            if emotion_checker[loop] == 0:
                filtered_rows.append(inputs[loop])
                filtered_labels.append(0)
                i += 1
            else:
                filtered_rows.append(inputs[loop])
                filtered_labels.append(1)
            loop += 1
        return vstack(filtered_rows), np.array(filtered_labels)

    inputs, y_true = filter_to_half(inputs)

# weights = np.zeros(shape=(len(vocab), 1))
# original solution (WILDLY slow and probably doesn't work anymore)
# for epoch in range(epochs):
#     loss = []
#     print("inputs.shape:", inputs.shape)
#     print("len(emotion_checker):", len(emotion_checker))
#     logits = inputs @ weights + bias  # @ is a dot product. inputs.shape == doccount, tokencount. weights.shape == tokencount, 1. 
#     y_pred = sigmoid(logits)  # absolute magic that i can take the sigmoid of whatever monster of a data type this is
#     # logits.shape == y_pred.shape == 1, doccount
#     y_true = emotion_checker  # length = doccount
#     for doc in range(len(corpus)):
#         if doc % 5000 == 0:
#             print(f"training epoch {epoch} on doc {doc}")
#         doc_tfidf = inputs[doc]
#         doc_logit = weights @ doc_tfidf        
#         doc_y_pred = sigmoid(doc_logit)
#         doc_y_true = y_true[doc]
#         loss.append(bce(doc_y_pred, doc_y_true))
#         error = doc_y_pred - doc_y_true
#         for idx, val in zip(doc_tfidf.indices, doc_tfidf.data):
#             weights[idx, 0] -= lr * error * val  # gradient descent ITERATIVELY to avoid weird numpy errors to do
#             # with sparse + dense matrix subtraction
#         bias -= lr * error

#     logits = weights.T @ inputs + bias
#     y_pred  = sigmoid(logits)
#     print(f"epoch {epoch}. current bce loss: {np.mean(loss)}")

# new solution with everything done in batches
    for epoch in range(epochs):

        logits = inputs @ weights  # (doccount, 1)
        preds = sigmoid(logits).flatten()   # vectorized
        errors = preds - y_true    # (doccount,)
        pos_weight = 20  # weighting positive values to encourage not being negative
        weights_vector = np.where(y_true == 1, pos_weight, 1)  # shape: (doccount,)

        # apply it to the errors
        errors = errors * weights_vector  # shape: (doccount,)
        # (tokencount, doccount) @ (doccount, [1]) gives (tokencount, 1) gradient
        grad_w = inputs.T.dot(errors).reshape(-1, 1) / len(corpus)
        weights -= lr * grad_w  # slow it down
        if (current_epochs + epoch) % 10000 == 0:
            print(f"emotion {emotion}: epoch {current_epochs + epoch}, batch BCE: {np.mean(bce(preds, y_true)):.4f}")

    np.savez(f"model_weights/{emotion}.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))
