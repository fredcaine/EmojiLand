import numpy as np
from scipy.sparse import load_npz, vstack
from pandas import read_csv
from ai_requirements.vectoriser import corpus

db = read_csv("emoji_ai_requirements/emoji_db.csv")
# completed: admiration, amusement, anger, annoyance, approval, v\caring, confusion, curiosity, desire, disappointment,
# disapproval, disgust, embarrassment, example_very_unclear, excitement, fear, gratitude, joy, grief, love, nervousness,
# optimism, pride, realization, relief, remorse, sadness, surprise, neutral
emojis = ["❤","😄","😐","😠","😢","😨","😮","🤔","🤢","🤩"]
optimal_epochs = {"❤": 700000,"😄": 775000,"😐": 1275000,"😠": 950000, "😢": 825000,
                  "😨": 625000,"😮": 850000,"🤔": 900000,"🤢": 825000,"🤩": 475000}
def sigmoid(y):
    y = np.clip(y, -709, 709)  # making sure i don't get any overflow errors from ridiculously far off numbers.
    return 1 / (1 + np.exp(-y))

def bce(y_pred, y_true):  # technically doesn't need to be calculated
    # mostly to know how good the ai is performing.
    # pos_weight weights the loss so that it doesn't just guess zero bc it's the most common.
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)  # removes chance of a log(0)
    loss = -(y_true * np.log(y_pred) + ((1 - y_true) * np.log(1 - y_pred)))  # binary cross entropy formula
    return loss

def filter_to_half(inputs, emoji_checker, emoji_count):

    labels = np.asarray(emoji_checker).astype(int).flatten()
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    emoji_count = int(max(0, emoji_count))
    n_neg = min(emoji_count, len(neg_idx))
    chosen_neg = (np.random.choice(neg_idx, size=n_neg, replace=False)
                  if n_neg > 0 else np.array([], dtype=int))

    # keep positives first (preserves their original order), then random negatives
    chosen = np.concatenate([pos_idx, chosen_neg]) if len(pos_idx) + len(chosen_neg) > 0 else np.array([], dtype=int)
    if chosen.size == 0:
        try:
            n_features = inputs.shape[1]
        except Exception:
            n_features = 0
        return np.empty((0, n_features)), np.array([], dtype=int)

    X_sub = inputs[chosen]            # keeps dtype and shape
    y_sub = labels[chosen].astype(int)
    return X_sub, y_sub


for emoji in emojis:
    print(f"now training: {emoji}")
    emoji_checker = db[emoji]
    emoji_count = sum(emoji_checker)
    epochs = 1500000
    inputs = np.load("emoji_ai_requirements/db_emoji_input_vector.npz")["arr_0"]
    # trained_values = np.load(f"model_weights/{emoji}.npz")
    # weights = trained_values["weights"]
    # bias = trained_values["bias"].item()
    # current_epochs = trained_values["epochs"].item()
    lr = 1 # learning rate should be slow to avoid divergence bc machines are dumb
    current_epochs = 0
    weights = np.zeros((28,1))
    bias = 0

# weights = np.zeros(shape=(len(vocab), 1))
# original solution (WILDLY slow and probably doesn't work anymore)
# for epoch in range(epochs):
#     loss = []
#     print("inputs.shape:", inputs.shape)
#     print("len(emoji_checker):", len(emoji_checker))
#     logits = inputs @ weights + bias  # @ is a dot product. inputs.shape == doccount, tokencount. weights.shape == tokencount, 1. 
#     y_pred = sigmoid(logits)  # absolute magic that i can take the sigmoid of whatever monster of a data type this is
#     # logits.shape == y_pred.shape == 1, doccount
#     y_true = emoji_checker  # length = doccount
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
    inputs, y_true = filter_to_half(inputs, emoji_checker, emoji_count)
    for epoch in range(epochs):

        n_examples = inputs.shape[0]
        # logits: include bias
        logits = inputs.dot(weights).reshape(-1) + bias   # shape (n_examples,)
        preds  = sigmoid(logits)                          # (n_examples,)
        errors = preds - y_true                           # (n_examples,)

        # gradients normalized by current batch size
        grad_w = inputs.T.dot(errors).reshape(-1,1) / n_examples
        grad_b = np.mean(errors)                          # scalar

        weights -= lr * grad_w
        bias    -= lr * grad_b

        if (current_epochs + epoch) % 100000 == 0:
            X = inputs  # shape (150, 28) dense numpy
            y = y_true.astype(float)

            corrs = []
            for j in range(X.shape[1]):
                col = X[:, j].astype(float)
                if np.std(col) == 0:
                    corrs.append(0.0)
                else:
                    corrs.append(np.corrcoef(col, y)[0,1])
            print("per-feature Pearson correlations (first 28):")
            print(np.round(corrs, 4))
            print("max abs corr:", np.max(np.abs(corrs)))
            n_examples = inputs.shape[0]
            print("=== DIAGNOSTICS ===")
            print("inputs type:", type(inputs), "shape:", getattr(inputs, "shape", None))
            print("weights shape:", weights.shape, "dtype:", getattr(weights, "dtype", None))
            print("y_true shape/dtype:", y_true.shape, y_true.dtype)
            print("n_examples (batch):", n_examples)

            # quick stats
            label_mean = np.mean(y_true)
            pred_mean = float(np.mean(preds))
            print("label mean (p):", label_mean)
            print("pred mean:", pred_mean)
            print("logits min/max/mean/std:", float(np.min(logits)), float(np.max(logits)), float(np.mean(logits)), float(np.std(logits)))
            print("weights norm:", float(np.linalg.norm(weights)))
            print("bias:", float(bias))

            # gradient checks
            raw_grad = inputs.T.dot(errors).reshape(-1,1)
            print("raw grad norm:", float(np.linalg.norm(raw_grad)))
            print("grad norm normalized (divide by n_examples):", float(np.linalg.norm(raw_grad) / n_examples))

            # check for NaN / inf
            if np.isnan(preds).any() or np.isinf(preds).any():
                print("WARNING: preds contains NaN or Inf")
            if np.isnan(raw_grad).any() or np.isinf(raw_grad).any():
                print("WARNING: grad contains NaN or Inf")

            # sample pairs (first 10) to see label vs pred
            print("first 10 y_true:", y_true[:10])
            print("first 10 preds:", preds[:10])

            # one-step test: copy weights, apply update, compute loss before/after
            def bce_scalar(p, y, eps=1e-7):
                p = np.clip(p, eps, 1-eps)
                return -(y * np.log(p) + (1-y) * np.log(1-p))

            loss_before = np.mean(bce_scalar(preds, y_true))
            # compute update
            test_w = weights.copy()
            test_b = bias
            test_w -= lr * (raw_grad / n_examples)
            test_b -= lr * np.mean(errors)
            # new logits/preds (works for sparse and dense)
            new_logits = inputs.dot(test_w).reshape(-1) + test_b
            new_preds = 1.0 / (1.0 + np.exp(-np.clip(new_logits, -709, 709)))
            loss_after = np.mean(bce_scalar(new_preds, y_true))
            print(f"loss before: {loss_before:.6f}, loss after one lr-step: {loss_after:.6f}, delta: {loss_after - loss_before:.6e}")

            print("=== END DIAGNOSTICS ===")
            print(f"emoji {emoji}: epoch {current_epochs + epoch}, batch BCE: {np.mean(bce(preds, y_true)):.4f}")

    np.savez(f"db_BCE_emoji_model_weights/{emoji}.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))
