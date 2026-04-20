import pandas as pd
import numpy as np

# UNCOMMENT THIS FOR A LOOKUP TABLE OF CORRECT EMOJI ANSWERS
# db = pd.read_csv("emoji_ai_requirements/emoji_db.csv")
# finish = db.drop(columns="text")
# np.savez("emoji_ai_requirements/emoji_lookup.npz", finish)


# # UNCOMMENT THIS FOR AN AI INPUT FOR ALL 200K LINES
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
finish = db.drop(columns="emotion").drop(columns="neutral")
np.savez("model_weights.npz", finish)

# UNCOMMENT THIS FOR A DB INPUT
# finish = db.drop(columns="text").drop(columns="id").drop(columns="neutral").astype(int).iloc[:294].values  # beautiful line of code
# np.savez("emoji_ai_requirements/db_emoji_input_vector.npz", finish)
