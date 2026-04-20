# Fredrick Farouk. emotion_predictor.py
# This file allows sending input through the forward pass to evaluate the model that predicts emotion.

import numpy as np
from re import findall
from ai_requirements.tfidf_vector.idfs_dict import global_idfs_dict
from ai_requirements.vocab_to_index import vocab_to_index
from scipy.sparse import csr_matrix

# This is the list of emotions that are evaluated.
emotions = ["example_very_unclear","admiration","amusement","anger",
            "annoyance","approval","caring","confusion","curiosity","desire",
            "disappointment","disapproval","disgust","embarrassment","excitement",
            "fear","gratitude","grief","joy","love","nervousness","optimism","pride",
            "realization","relief","remorse","sadness","surprise","neutral"]

# We first vectorize input by the same code as ai_requirements/vectorizer.py
def get_ngrams(tokenized_text, ngrams=2):
    new_tokens = []
    for token in range(1, len(tokenized_text)):
        new_tokens.append(" ".join(tokenized_text[token - ngrams : token]))
    return new_tokens

def tokenize(text):
    tokenized_text = findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return tokenized_text + get_ngrams(tokenized_text)

def compute_tf_dict(doc):
    tf = {}
    doc_len = len(doc)
    for token in doc:
        tf[token] = tf.get(token, 0) + 1
    for token in tf:
        tf[token] = tf[token] / doc_len
    return tf

def vectorize(prompt):
    
    rows, cols, values = [], [], []
    tfidf = {}
    prompt = tokenize(prompt)
    tf = compute_tf_dict(prompt)
    for token in prompt:
        if token in global_idfs_dict:
            tfidf[token] = global_idfs_dict[token] * tf[token]

    for token, value in tfidf.items():
        if token in vocab_to_index:
            cols.append(vocab_to_index[token])
            rows.append(0)
            values.append(value)
    x = csr_matrix((values, (rows, cols)), shape=(1, len(vocab_to_index)))
    return x

def sigmoid(y):
    y = np.clip(y, -709, 709)
    return 1 / (1 + np.exp(-y))

# This predictor has strange parameters to allow for clean I/O.
# It takes in the prompt and the emotion that must be predicted.
# It also takes in arrays of any predictions that have already been done to other emotions, then appends the new prediction to this array.
# This is useful for formatting the output, as will be seen later.
def predict(prompt, emotion, predictions, predictions2 = None, rounding=False):
    # Firstly, we vectorize the prompt so the model can understand it.
    inputs = vectorize(prompt)
    # Now we load the model.
    trained_data = np.load(f"emotion_model_weights/{emotion}.npz")
    weights = trained_data["weights"]
    bias = trained_data["bias"].item()
    # We run the forward pass as defined in emotion_model_training.py.
    logit = inputs @ weights + bias
    prediction = sigmoid(logit)[0, 0]
    # This gives us the model's prediction.

    # Sometimes, I used this function to evaluate two separate models and see which ones are more accurate or have more usable predictions.
    # For these, we run the exact same procedure but with the other model weights.
    if predictions2 is not None:
        trained_data2 = np.load(f"old_emotion_model_weights/{emotion}.npz")
        weights2 = trained_data2["weights"]
        bias2 = trained_data2["bias"].item()
        logit2 = inputs @ weights2 + bias2
        prediction2 = sigmoid(logit2)[0, 0]
    
    # Again, for the sake of I/O and usability, we sometimes need rounding.
    # In this case, if rounding is true then we give a '1' or a '0' directly.
    # If rounding is a number (i.e. not True but also not None/False), then we round to that many decimal places.
    # If rounding is False or None, then we don't round.
    if rounding == True:
        predictions.append(float(np.clip(round(prediction * 4)), 0, 1))
        if predictions2 is not None: predictions2.append(float(np.clip(round(prediction2)), 0, 1))
    elif rounding:
        predictions.append(float(np.clip(round(prediction * 4, rounding), 0, 1)))
        if predictions2 is not None: predictions2.append(float(np.clip(round(prediction2, rounding), 0, 1)))
    else:
        predictions.append(float(np.clip(prediction * 4), 0, 1))
        if predictions2 is not None: predictions2.append(float(np.clip(prediction2), 0, 1))
    
    # Return the lists with the new predictions appended.
    if predictions2 is not None:
        return predictions, predictions2
    else:
        return predictions

# This is a simple function that iterates through the emotions and appends their predictions.
# If I was not testing against another model, then I would use the same function with all instances of predictions2 removed.
# This would still pass correct parameters into other functions.
def create_prediction_vector(prompt):
    predictions, predictions2 = [], []
    for emotion in emotions:
        predictions, predictions2 = predict(prompt, emotion, predictions, predictions2, rounding=3)
    return predictions, predictions2

# Again, if I was not evaluating, I would just not return the second item.
def create_prediction_dicts(prompt):
    # Form a list of tuples then combine into a dictionary.
    # We also add "Values for newer/older model" in here so that it prints along with the other prediction dictionaries.
    # Yes, this is a bit awkward, but this I/O is never used in any other code, so it doesn't really matter.
    return "\n\nValues for newer model:\n\n", {e:p for e, p in zip(emotions, create_prediction_vector(prompt)[0])}, "\n\nVALUES FOR OLDER MODEL:\n\n", {e:p for e, p in zip(emotions, create_prediction_vector(prompt)[1])}

print("Type 'exit' at any point to exit the program.\n")
while True:
    prompt = input("Enter your prompt:\n")
    # Make the I/O nicer.
    if prompt.lower() == "exit":
        break
    # Call the function.
    emotion_dicts = create_prediction_dicts(prompt)


    formatted_emotion_dicts = []
    # We go through all the returned values to format them for pretty printing.
    for item in emotion_dicts:
        formatted_item = item
        # If they are dictionaries:
        if type(item) == dict:
            # Convert the formatted item to the string of the item.
            formatted_item = str(item)
            # We remove the braces.
            formatted_item = formatted_item[1:-1]
            # Split across items.
            dictionary_items = formatted_item.split(",")
            # Join together the split items with a newline and tab to make the I/O prettier.
            formatted_item = "\n\t".join(dictionary_items)
            # Finally, we add a newline and tab to the beginning so that all items are on the same tab (Fencepost problem)
            formatted_item = "\n\t" + formatted_item

        # Then we append the formatted item.
        formatted_emotion_dicts.append(formatted_item)

    # Finally, we print everything in the formatted list.
    for item in formatted_emotion_dicts:
        print(item)