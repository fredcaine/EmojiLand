# Fredrick Farouk. emotion_ai_training.py
# This is the main file that trains the emotion models to interpret individual emotion.

import numpy as np
import os
from scipy.sparse import load_npz, vstack
from pandas import read_csv
from ai_requirements.vectoriser import corpus, vocab

# For clarity, this file refers to 'documents' or 'docs'. These are simply reddit messages from the GoEmotions dataset.
# Here is a link to the GoEmotions dataset. It is also found in ai_requirements/go_emotions_dataset, along with the vectoriser.
# https://doi.org/10.48550/arXiv.2005.00547

# Completed: admiration, amusement, anger
# BCE Losses, respectively: 0.0749, 0.0177, 0.0193

# ------------------------- SETUP -------------------------

db = read_csv("ai_requirements/go_emotions_dataset.csv")

# List of emotions left to train.
emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment",
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
def filter_to_half(inputs, val_frac=0.2):

    filtered_rows, filtered_labels = [], []
    i, loop = 0, 0

    # Here, idx will represent the number of items labelled '0' are added.
    # For each '1' label for the emotion:
    idx = 0
    while idx <= emotion_count:
        # Go through the emotion_checker.
        # If an item is labelled '0', then add it, and *count*.
        if emotion_checker[loop] == 0:
            filtered_rows.append(inputs[loop])
            filtered_labels.append(0)
            idx += 1
        # If it's labelled '1', then add it, but *do not count*.
        else:
            filtered_rows.append(inputs[loop])
            filtered_labels.append(1)
        loop += 1

        # This way, we have split filtered_labels into exactly one half 'Yes' and one half 'No'.

    # Now, we split the remaining information into 80% training and 20% valuation.
    # First, we find how many items go into the valuation set (20% of the total number of items).
    val_count = int(len(filtered_rows) * 0.2)

    # Now, we form a vstack and array out of the filtered rows and labels up until the number of items required.
    x_val = vstack(filtered_rows[:val_count])
    y_val = np.array(filtered_labels[:val_count])

    # And another vstack and array for the remainder of the items.
    x_train = vstack(filtered_rows[val_count:])
    y_train = np.array(filtered_labels[val_count:])

    # Note that, as per the standard literature, x represents the inputs (i.e. the TF-IDF vector), and y represents the labelled values.

    return x_train, x_val, y_train, y_val

# ------------------------- TRAINING -------------------------

for emotion in emotions:
    print(f"Now training: {emotion}")

    # This allows easy checks for evaluating correctness.
    emotion_checker = db[emotion]
    # This gives us a good initial prediction for bias, which will be seen later. emotion_checker is binary.
    emotion_count = sum(emotion_checker)
    # For a simple gradient descent model, many, many epochs are used. This is because learning is very slow here.
    epochs = 10000001

    # The TF-IDF vector is explained in vectoriser.py
    inputs = load_npz("ai_requirements/tfidf_vector/tfidf_vector.npz")

    # Learning rate *must* be slow.
    # This is, in fact, a hyperparameter.
    lr = 1e-2

    # ------------------- LOAD OR INITIALIZE -------------------


    # Apply balanced half + validation split
    x_train, x_val, y_train, y_val = filter_to_half(inputs)
    pos_frac = y_train.mean()

    npz_file = f"{emotion}.npz"
    if os.path.exists(npz_file):
        # load existing weights, bias, and epochs
        data = np.load(npz_file)
        weights = data["weights"]
        bias = data["bias"].item()
        current_epochs = data["epochs"].item()
        print(f"Resuming training for {emotion} from epoch {current_epochs}.")
    else:
        # This following segment is specifically for *beginning* training, not for continuing training.
        current_epochs = 0
        weights = np.zeros((len(vocab), 1))
        bias = 0
        # This gives information on how large the actual training dataset currently is (1.6 * number of documents with emotion)
        print(x_train.shape)

    # ------------------------- BATCH TRAINING -------------------------

    try:
        for epoch in range(current_epochs, epochs):
            # First, the forward pass.
            # @ represents matrix multiplication since Python 3.5.
            logits = x_train @ weights
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

            # Again, we weight positives higher than negatives to fully avoid false negatives.
            pos_weight = 20
            # This makes an array that multiplies all positives by 20, and leaves all negatives unnaffected.
            weights_vector = np.where(y_train == 1, pos_weight, 1)

            # Then, we multiply the errors by this to fully punish negatives.
            errors = errors * weights_vector

            # We multiply the inputs by the errors to
            grad_w = x_train.T.dot(errors).reshape(-1, 1) / len(corpus)
            weights -= lr * grad_w  # slow it down

            if epoch % 10000 == 0:
                # BCE on training
                train_bce = np.mean(bce(y_pred, y_train))

                # BCE on validation
                val_logits = x_val @ weights
                val_pred = sigmoid(val_logits).flatten()
                val_bce = np.mean(bce(val_pred, y_val))

                print(f"""Emotion: {emotion}. Epoch: {epoch}.
Train BCE: {train_bce:.4f}, Valuation BCE: {val_bce:.4f}""")

    # This exception is mostly KeyboardInterrupt so I can pause.
    except KeyboardInterrupt:
        print(f"\nTraining paused at epoch {epoch}. Progress saved.")
        np.savez(npz_file, weights=weights, bias=bias, epochs=epoch)
        continue

    # Save after finishing all epochs, just in case (though unlikely).
    np.savez(npz_file, weights=weights, bias=bias, epochs=epochs)