# Fredrick Farouk. vectorizer.py
# This vectorizes the entire GoEmotions dataset for usage in training (as 'inputs').

from re import findall
from pandas import read_csv
from time import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import Counter

code_beginning_time = time()

db = read_csv("ai_requirements/go_emotions_dataset.csv")
corpus = db["text"]

def get_ngrams(tokenized_text, ngrams=2):
    new_tokens = []
    for token in range(1, len(tokenized_text)):
        new_tokens.append(" ".join(tokenized_text[token - ngrams : token]))
    return new_tokens

def tokenize(text):
    tokenized_text = findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return tokenized_text + get_ngrams(tokenized_text)
    # \b: word boundary, \w+: word chars, (?:'\w+)?: optional apostrophe-contracted suffix

tokenized_corpus = []
for doc in corpus:
    tokenized_corpus.append(tokenize(doc))

def build_vocab(corpus):
    vocab = {}
    index = -1
    for doc in corpus:
        for token in doc:
            index += 1
            vocab[token] = index
    return vocab

vocab = build_vocab(tokenized_corpus)
vocab = sorted(vocab)

vocab_to_index = {word: i for i, word in enumerate(vocab)}

# the following code was used for the vectorizer part. however, 
# def compute_tf_dict(doc):
#     tf = {}
#     doc_len = len(doc)
#     for token in doc:
#         tf[token] = tf.get(token, 0) + 1
#     for token in tf:
#         tf[token] = tf[token] / doc_len
#     return tf

# # Not used here, but retained for reference
# def compute_idf_dict(corpus):
#     N = len(corpus)
#     df_counts = Counter()

#     for i, doc in enumerate(corpus):
#         if i % 1000 == 0:
#             print(f"currently on doc {i}. time (s): {round(time() - code_beginning_time)}")
#         for token in set(doc):  # count token only once per doc
#             if token in vocab:
#                 df_counts[token] += 1

#     # Compute IDF
#     return {
#         token: np.log(N / (1 + df_counts.get(token, 0)))
#         for token in vocab
#     }


# print(len(vocab))
# global_idfs_dict = compute_idf_dict(corpus)
# import json

# with open("idfs_dict.py", "w", encoding="utf-8") as f:
#     f.write("global_idfs_dict = ")
#     json.dump(global_idfs_dict, f, indent=4, ensure_ascii=False)

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
