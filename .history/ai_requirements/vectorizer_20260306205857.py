# Fredrick Farouk. vectorizer.py
# This vectorizes the entire GoEmotions dataset for usage in training (as 'inputs').

from re import findall
from pandas import read_csv
from time import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import Counter

# Yes, the vectorizing took long enough that I kept timing logs.
code_beginning_time = time()

# We read in all the text.
db = read_csv("ai_requirements/go_emotions_dataset.csv")
corpus = db["text"]

# This is a helper function for the tokenizer.
def get_ngrams(tokenized_text, ngrams=2):
    new_tokens = []
    # First, we take every position in the tokenized list.
    for token in range(1, len(tokenized_text)):
        # Now, tokenized_text[token - ngrams : token] simply gives the ngram in tokenized_text.
        # " ".join() combines it into one string. We then append this to new tokens.
        new_tokens.append(" ".join(tokenized_text[token - ngrams : token]))
    return new_tokens

# I'll tokenize the text into sets of individual words, then add on the ngrams.
def tokenize(text):
    # This regular expression finds words by this logic:
    # \b is the boundary at the start of a word, then \w+ implies word characters beginning.
    # Optionally, (?:'\w+) gives the chance for apostrophes to be included, followed by more word characters.
    # Finally, we require \b at the end to give another word boundary to ensure we have reached the end of the word.
    tokenized_text = findall(r"\b\w+(?:'\w+)?\b", text.lower())
    # The tokens will include both the unigrams acquired above and ngrams from the get_ngrams function.
    return tokenized_text + get_ngrams(tokenized_text)

# We loop through the corpus to tokenize.
tokenized_corpus = []
for doc in corpus:
    tokenized_corpus.append(tokenize(doc))

# We create a vocabulary that is used several times, including in the training file.
def build_vocab(corpus):
    vocab = {}
    index = -1
    # For each token in each document, we add it to build our vocabulary.
    # We build a dictionary of vocab:index as it will be useful later.
    for doc in corpus:
        for token in doc:
            index += 1
            vocab[token] = index
    return vocab

vocab_to_index = build_vocab(tokenized_corpus)
vocab = sorted(vocab)

# The TF-IDF vectorizer is Term Frequency : Inverse Document Frequency
# We begin by calculating the term frequency for each comment in a document.
# def compute_tf_dict(doc):
#     tf = {}
#     # Total number of tokens in the document.
#     doc_len = len(doc)
#     # If the token hasn't been seen yet, it becomes 0+1=1.
#     # If it has been seen, we increment its value by 1.
#     for token in doc:
#         tf[token] = tf.get(token, 0) + 1
#     
#     # We then divide by the total number of tokens in a document to get the frequency.
#     for token in tf:
#         tf[token] = tf[token] / doc_len
#     return tf

# # # Not used here, but retained for reference.
# # # This worked out the IDF dictionary, but this only needs to be calculated once as it is global.
# # # That is why this is double-commented; uncommenting the vectorizer will keep this commented.
# # def compute_idf_dict(corpus):
# #     df_counts = Counter()
# #
# #     for i, doc in enumerate(corpus):
# #         # Yes, that is how long it took. The corpus has over 200000 documents.
# #         if i % 1000 == 0:
# #             print(f"Document number: {i}. Time taken so far in seconds: {round(time() - code_beginning_time)}")
# #         # We count a document exactly one if it contains the term.
# #         for token in set(doc):
# #             if token in vocab:
# #                 df_counts[token] += 1
# #
# #     # Now we compute the actual IDF by dividing corpus length by 1 + the number of documents containing the token (for each token).
# #     return {token: np.log(len(corpus) / (1 + df_counts.get(token, 0))) for token in vocab}

# # global_idfs_dict = compute_idf_dict(corpus)

# # # Dumping results into 
# # import json
# # with open("ai_requirements/tfidf_vector/idfs_dict.py", "w", encoding="utf-8") as f:
# #     f.write("global_idfs_dict = ")
# #     json.dump(global_idfs_dict, f, indent=4, ensure_ascii=False)

# def multiply_dict_values(dict1: dict, dict2: dict):
#     answer = dict1.copy()
#     for item in answer:
#         answer[item] *= dict2[item]
#     return answer

# data, rows, cols = [], [], []

# for doc_idx, doc in enumerate(tokenized_corpus):
#     if doc_idx % 1000 == 0 and doc_idx > 0:
#         print(f"flushing docs {doc_idx - 1000} to {doc_idx} to sparse_parts.csv")
#         buffer = pd.DataFrame({
#             "rows": rows,
#             "cols": cols,
#             "data": data,
#         })
#         buffer.to_csv("sparse_parts.csv", mode="a", header=(doc_idx == 1000), index=False)
#         rows.clear(); cols.clear(); data.clear()
#     current_tf_dict = compute_tf_dict(doc)
#     current_tfidf_dict = multiply_dict_values(current_tf_dict, global_idfs_dict)
#     for token in current_tfidf_dict:
#         rows.append(doc_idx)
#         cols.append(vocab_to_index[token])
#         data.append(current_tfidf_dict[token])

# df = pd.read_csv("ai_requirements/tfidf_vector/sparse_parts.csv")
# tfidf_matrix = csr_matrix((df["data"], (df["rows"], df["cols"])), shape=(len(corpus), len(vocab)))
# save_npz("tfidf_vector.npz", tfidf_matrix)
