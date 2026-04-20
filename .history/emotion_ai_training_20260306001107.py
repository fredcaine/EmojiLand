import numpy as np
from scipy.sparse import load_npz, vstack
from pandas import read_csv
from sklearn.model_selection import train_test_split
from ai_requirements.vectoriser import corpus, vocab

# Completed: admiration, amusement, anger
# BCE Losses, respectively: 0.0749, 0.0177, 0.0193

# ------------------------- SETUP -------------------------

db = read_csv("ai_requirements/go_emotions_dataset.csv")

# List of emotions left to train.
emotions = ["annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment",
"disapproval", "disgust", "embarrassment", "example_very_unclear", "excitement", "fear", "gratitude", "joy", "grief", "love", "nervousness",
"optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

def sigmoid(y):
    # Note: I added this clipping here because there was an issue in my old training that led to strange infinite logits etc.
    # This is no longer an issue, but I thought I would leave it here anyways.
    y = np.clip(y, -709, 709)

    # The known sigmoid function. This reduces all logits to a range between 0 and 1.
    return 1 / (1 + np.exp(-y))

# A major issue arose in my first training: the model guessed the exact fraction of numbers with a '1' label, every single time.
# This is classic overfitting. I countered in two ways.
# 1) I added weighting to the loss function. This is technically useless.
# It is useless because I do not adjust the gradient descent to the loss function, instead defining them separately (gradient as the partial derivative of loss)
# However, I keep the weighting on the loss so that the printing logs give me a better understanding of the number of false-negatives.
# Since we sort into many different emotions, we would rather a false-positive than a false-negative, since we choose the highest prediction anyways.
# Nevertheless, it really is not useless, as I apply the exact same weighting to the gradient descent independently.

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
    # This is the known standard formula for Binary Cross-Entropy (BCE).
    loss = -(pos_weight * y_true * np.log(y_pred) + ((1 - y_true) * np.log(1 - y_pred)))
    return loss

# 2) I filtered the dataset provided to each training cycle into half '1' and half '0'.
# This makes it so the optimum bias is 0.5, *forcing* the model to train
# Though I did not know this was a real technique at the time, apparently this is called balanced batch sampling.
# Here is (likely the first) paper on it: https://doi.org/10.1214/14-AOS1220
def filter_to_half_with_val(inputs, val_frac=0.2, random_state=42):
    
    filtered_rows, filtered_labels = [], []
    i, loop = 0, 0

    # Here, i will represent the number of items labelled '0' are added.
    # For each '1' label for the emotion:
    while i <= emotion_count:
        # Go through the emotion_checker.
        # If an item is labelled '0', then add it, and *count*.
        if emotion_checker[loop] == 0:
            filtered_rows.append(inputs[loop])
            filtered_labels.append(0)
            i += 1
        # If it's labelled '1', then add it, but *do not count*.
        else:
            filtered_rows.append(inputs[loop])
            filtered_labels.append(1)
        loop += 1

    # Stack and convert
    X_balanced = vstack(filtered_rows)
    y_balanced = np.array(filtered_labels)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, y_balanced, test_size=val_frac, random_state=random_state, stratify=y_balanced
    )

    return X_train, X_val, y_train, y_val

print(len(vocab), len(corpus))

# ------------------------- TRAINING -------------------------

for emotion in emotions:
    print(f"Now training: {emotion}")

    # This allows easy checks for evaluating correctness.
    emotion_checker = db[emotion]
    # This gives us a good initial prediction for bias, which will be seen later. emotion_checker is binary.
    emotion_count = sum(emotion_checker)
    # For a simple gradient descent model, many, many epochs are used. This is because learning is very slow here.
    epochs = 100001

    # The TF-IDF vector is explained in vectoriser.py
    inputs = load_npz("ai_requirements/tfidf_vector/tfidf_vector.npz")

    # Yes, learning rate should be slow. That is true.
    # Logically, slow learning rate means slow training, which, on my CPU (i7-1355U), would not have been possible.
    # As it is, this took roughly 24 hours of training.
    # From my findings, my model actually managed to do fine at any learning rate I set it to.
    # The point is to avoid divergence, and I found it to work just fine at this learning rate.
    # This is likely because it is simple gradient descent, and the dataset is more than large enough to combat overfitting.
    lr = 1e-4

    # Apply balanced half + validation split
    X_train, X_val, y_train, y_val = filter_to_half_with_val(inputs)

    # This following segment is specifically for *beginning* training, not for continuing training.
    # Of course, if you're trying to test this, then uncomment this the first time.
    current_epochs = 0
    weights = np.zeros((len(vocab), 1))
    pos_frac = y_train.mean()    # e.g. 0.05 for 5% positives
    bias = 0

    print(X_train.shape)

    # ------------------------- BATCH TRAINING -------------------------

    for epoch in range(epochs):

        # First, the forward pass.
        # @ represents matrix multiplication since Python 3.5.
        logits = X_train @ weights
        y_pred = sigmoid(logits).flatten()
        # Now note: the current size of y_pred is (doccount,), as it has been flattened.
        # Again, y_train is flattened: np.array(list) --> a flattened array.
        # So errors = (doccount,) - (doccount,) = (doccount,)

        # Here is the aforementioned gradient descent function.
        # This is just the partial derivative of the BCE function. Pretty conveniently works out to (y_pred - y_true)/(y_pred(1 - y_pred)).
        # More precisely, the derivative is that for final *predictions*, not for logits.
        # Using the chain rule with sigmoid, you can find that for pure logits, it's just y_pred - y_true, which is even more beautiful.
        # However, since I have already converted to predictions, I must use the more complicated formula.
        # Again, I repeat the idea of adding epsilon to avoid an error (in this case, to avoid division by 0).
        eps = 1e-7
        errors = (y_pred - y_train)/(y_pred * (1 - y_pred) + eps)

        pos_weight = 20
        weights_vector = np.where(y_train == 1, pos_weight, 1)  # shape: (doccount,)

        # apply it to the errors
        errors = errors * weights_vector  # shape: (doccount,)
        # (tokencount, doccount) @ (doccount, [1]) gives (tokencount, 1) gradient
        grad_w = X_train.T.dot(errors).reshape(-1, 1) / len(corpus)
        weights -= lr * grad_w  # slow it down

        if (current_epochs + epoch) % 10000 == 0:
            # BCE on training
            train_bce = np.mean(bce(y_pred, y_train))
            
            # BCE on validation
            val_logits = X_val @ weights
            val_pred = sigmoid(val_logits).flatten()
            val_bce = np.mean(bce(val_pred, y_val))

            print(f"emotion {emotion}: epoch {current_epochs + epoch}, "
                  f"train BCE: {train_bce:.4f}, val BCE: {val_bce:.4f}")

    np.savez(f"{emotion}.npz", weights=weights, bias=bias, epochs=(current_epochs + epochs))