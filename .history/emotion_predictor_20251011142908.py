import numpy as np
from re import findall
from ai_requirements.tfidf_vector.idfs_dict import global_idfs_dict
from ai_requirements.vocab_to_index import vocab_to_index
from scipy.sparse import csr_matrix
emotions = ["example_very_unclear","admiration","amusement","anger",
            "annoyance","approval","caring","confusion","curiosity","desire",
            "disappointment","disapproval","disgust","embarrassment","excitement",
            "fear","gratitude","grief","joy","love","nervousness","optimism","pride",
            "realization","relief","remorse","sadness","surprise"]
# neutral removed as it's very difficult to figure out if something is neutral from just individual (or couples of two) tokens

def get_ngrams(tokenized_text, ngrams=2):
    new_tokens = []
    for token in range(1, len(tokenized_text)):
        new_tokens.append(" ".join(tokenized_text[token - ngrams : token]))
    return new_tokens

def tokenize(text):
    tokenized_text = findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return tokenized_text + get_ngrams(tokenized_text)
    # \b: word boundary, \w+: word chars, (?:'\w+)?: optional apostrophe-contracted suffix

def sigmoid(y):
    y = np.clip(y, -709, 709)  # making sure i don't get any overflow errors from ridiculously far off numbers.
    # i don't really need to anymore because i fixed the thing that was causing it, but that's alright
    return 1 / (1 + np.exp(-y))

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
        if token in vocab_to_index:  # vocab_dict maps token -> index
            cols.append(vocab_to_index[token])
            rows.append(0)
            values.append(value)
    x = csr_matrix((values, (rows, cols)), shape=(1, len(vocab_to_index)))
    return x

def predict(prompt, emotion, predictions, rounding=False):
    inputs = vectorize(prompt)
    trained_data = np.load(f"model_weights/{emotion}.npz")
    weights = trained_data["weights"]
    bias = trained_data["bias"].item()

    logit = inputs @ weights + bias
    prediction = sigmoid(logit)[0, 0]
    if rounding == True:
        predictions.append(round(prediction))
    elif rounding:
        predictions.append(float(round(prediction, rounding)))
    else:
        predictions.append(float(prediction))
    
    # if round(prediction) == 1:
        # print(f"emotion {emotion} exists with certainty {prediction}")
    return predictions

def create_prediction_vector(prompt):
    predictions = []
    for emotion in emotions:
        predictions = predict(prompt, emotion, predictions, rounding=3)
    return predictions

def create_prediction_dict(prompt):
    return {e:p for e, p in zip(emotions, create_prediction_vector(prompt))}

while True:
    prompt = input("enter your prompt: ")
    if prompt == "exit":
        break
    emotion_dict = create_prediction_dict(prompt)
    print("\n\t",("\n\t".join(str(emotion_dict).split(",")))[1:-1])  # name a cleaner line of code