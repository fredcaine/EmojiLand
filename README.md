Fredrick Farouk. Project.txt
Below is a report on the journey of making this project.
I worked on this during the summer of 2025.

The goal of this project was to create an AI model that could take in any text and output a suitable emoji. That's why I called it EmojiLand.
I decided to take a somewhat novel approach: first, make a model that converts input to emotion, then, use a second model to convert emotion to emoji.

The primary goal, however, was to use zero machine learning (ML) libraries, so that I could learn exactly how logistic regression actually works.
I took a relatively simple model: logistic regression simple gradient descent.

The first step in this pipeline was to train a model to convert strings of text into emotion, and the first part of any language model is a vectorizer.
After a little bit of research, I found that my (likely) best choice was TF-IDF vectorizer (Term Frequency-Inverse Document Frequency).
The idea here is to give every token in every document a value based on its frequency in its individual document and its frequency in the corpus.
This then serves as a weight on each token that signifies its importance. At this point, we train a model that calculates model weights. These give a value to each token so that the weights multiply by the TF-IDF vector to give a single prediction for each document.
Though this model is somewhat explained here, more specificity can be found in my detailed comments throughout the files (in this case, emotion_ai_training.py).

In trying to implement a vectorizer myself, and making the tokenizer that pairs it, I learnt my first piece of experimental information: in order for a model to understand context, it needs to be able to see tokens that are more than just one word.
The other option would have been building in context through giant tensors and high-dimensional vectors, as is standard in Large Language Models (LLMs). However, this takes vast amounts of training (more than can be done with a dataset of only 211225 documents), and is a very complex task in general.
Due to my training constraints (a laptop CPU, i7-13900U), I had to keep it to bigrams (sets of two words at most), so context ends up being quite limited.
This is especially so because of the English language.
After all, important phrases are often three words long e.g.
"I love him."
"This disgusts me."
"You are amazing."
etc.
These phrases get pushed down by a number of other tokens within them, e.g. "You are", which is very neutral, or "I love", which could begin a sarcastic statement.
This hints towards the fact that perhaps other languages are easier to decipher with less training power!

The second thing I learnt was the importance of creating checkpoints. Models, especially when trained on CPUs, do not train quickly.
I had issues at the beginning with training models because I could not shut down my laptop... at all. That is when I thought to implement checkpoints to help.

Something I observed to some degree was the garbage-in garbage-out phenomenon. I stated earlier that "I love" could begin a sarcastic statement.
This is made clear by the sampling bias; the dataset consists of Reddit messages. As we know, Reddit is not the best representation of Standard English.
Often, messages include heavy sarcasm, the majority of labelled "aggressive" documents are only labelled that due to a few cuss words, and the dataset overall balances the model to learn the way Reddit users speak online, and not in real life.
The GoEmotions dataset was, nevertheless, the largest that served the purpose I intended. The Vent dataset (https://doi.org/10.5281/zenodo.2537838) gives exactly one label to each document. For my purposes, this would heavily restrict the proportion of documents with each emotion.

The fourth was overfitting. Overfitting was probably one of the more difficult issues to combat, as it exists for a lot of ML.
A technique I learnt after working on a different project, which I came back to apply to this one, is valuation. Overfitting comes when a model "memorizes" a dataset and is great at predicting results within the dataset, but terrible with anything outside of it.
The response is as follows: split the dataset into training and valuation. Every epoch, do standard gradient descent on the training dataset only.
Every thousand epochs, run the forward pass on the valuation dataset. Now, check the loss on the valuation. If 1000 epochs ago the loss was better, then we stop right there.

I also had an idea for avoiding sampling bias: I split up the dataset into half true and half false cases by filtering out an excess of false cases.
This let the model learn that it should try to find positive cases, instead of just predicting '0' as is typically accurate.
Later, I learnt that this is known as balanced batch sampling.


That is the first step of the pipeline: next was a model that could take the likelihoods for each emotion, then output an emoji.
There was a major issue. There is no labelled dataset anything like this. So, I labelled my own based on GoEmotions.

Of course, that is quite (very) boring, so I created a large text-based game, EmojiGame, which is also going onto my GitHub profile, and playing this helped me finish over 3000 labels by hand.

At this point, I had to decide something: should the input emotion values for each item be taken from the GoEmotions dataset, or should they be generated by the emotion model?
There were pros for both. For dataset inputs, the inputs were guaranteed to be accurate for each emotion, so the AI could really learn the patterns of converting emotion to emoji.
For AI-generated inputs, they were closer to the inputs the AI would practically see: non-binary, potentially incorrect. This might even train the model to counteract any subtle issues with the emotion model.

I made both (both weights are found in emoji_trained_model_weights/), but in the end, it didn't matter. Both models overfit instantly, because it didn't have a large enough dataset to work with.

I solved this by ditching machine learning altogether and spending a few hours writing my own weights; after all, there's only 28 * 10 = 280 weights + 10 biases = 290 parameters to find.

In the end, I got a model with ~65% accuracy on detecting emojis, leading to a generally good classifier when choosing the most likely.
Of course, it still suffers from the garbage-in garbage-out phenomenon: simple phrases like "I love you" are simply completely misunderstood and given the 😢 emoji (due to none of the emotions being triggered in the emotion model).
In general, there are more false-positives than false-negatives, which is something I'm quite happy with.

Finally, I put together the two parts in emoji_predictor.py, evaluating and running tests on the model.

Thank you for reading my synopsis. This was an amazing project to work on.
