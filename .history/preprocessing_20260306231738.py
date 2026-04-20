# Fredrick Farouk. preprocessing.py
# This file is quite diffiult to use, but the majority is just different input generators that aren't of much use as the inputs are given in the folder.
# It is kept solely for reference.

import pandas as pd
import numpy as np

# UNCOMMENT THIS FOR A LOOKUP TABLE OF CORRECT EMOJI ANSWERS
# db = pd.read_csv("emoji_ai_requirements/emoji_db.csv")
# finish = db.drop(columns="text")
# np.savez("emoji_ai_requirements/emoji_lookup.npz", finish)


# UNCOMMENT THIS FOR AN AI INPUT FOR ALL 200K LINES
# from emotion_predictor import create_prediction_vector
# corpus = pd.read_csv("ai_requirements/go_emotions_dataset.csv")["text"]
# finish = []
# for i in range(len(corpus)):
#     finish.append(create_prediction_vector(corpus[i]))
#     if i % 1000 == 0:
#         print(i, "completed")
# np.savez("emoji_ai_requirements/ai_emoji_input_vector.npz", finish)


# UNCOMMENT THIS FOR AN AI INPUT FOR THE MAIN 300 LINES
# from emotion_predictor import create_prediction_vector
# corpus = pd.read_csv("ai_requirements/go_emotions_dataset.csv")["text"]
# finish = []
# for i in range(294):
#     finish.append(create_prediction_vector(corpus[i])[1:])
#     if i % 50 == 0:
#         print(i, "completed")
# np.savez("emoji_ai_requirements/ai_emoji_input_vector.npz", finish)

db = pd.read_csv("model_weights.csv")
finish = db.drop(columns="emotion")
np.savez("emoji_model_weights_llm.npz", finish)

# UNCOMMENT THIS FOR A DB INPUT
# finish = db.drop(columns="text").drop(columns="id").drop(columns="neutral").astype(int).iloc[:294].values  # beautiful line of code
# np.savez("emoji_ai_requirements/db_emoji_input_vector.npz", finish)
