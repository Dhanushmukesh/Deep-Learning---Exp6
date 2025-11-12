Deep-Learning-Exp6
Developing a Deep Learning Model for NER using LSTM
AIM
To develop an LSTM-based model for recognizing the named entities in the text.

THEORY
Neural Network Model
image
DESIGN STEPS
Step 1: Load the dataset (ner_dataset.csv) using pandas and fill missing values with .ffill().

Step 2: Extract all unique words and tags, then create mappings — word2idx and tag2idx.

Step 3: Group by "Sentence #" to form complete sentences as lists of (word, POS, tag) tuples.

Step 4: Convert each sentence’s words and tags into their corresponding integer indices.

Step 5: Apply padding to all sequences (e.g., max_len = 50) using keras.preprocessing.sequence.pad_sequences.

Step 6: Split the data into training and testing sets with train_test_split.

Step 7: Build a BiLSTM model using Embedding → SpatialDropout1D → Bidirectional(LSTM) → TimeDistributed(Dense(softmax)).

Step 8: Compile the model with Adam and sparse_categorical_crossentropy, train (~3 epochs), then predict and compare true vs predicted tags.

PROGRAM
Name: Rahul M R

Register Number: 2305002005

import matplotlib.pyplot as plt, pandas as pd, numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers, Model

# Load + preprocess
data = pd.read_csv("NER dataset.csv", encoding="latin1").ffill()  # replaces deprecated fillna(method='ffill')
print("Unique words:", data['Word'].nunique(), "| Unique tags:", data['Tag'].nunique())

words, tags = list(data['Word'].unique()) + ["ENDPAD"], list(data['Tag'].unique())
word2idx, tag2idx = {w:i+1 for i,w in enumerate(words)}, {t:i for i,t in enumerate(tags)}

# Group sentences safely
sents = data.groupby("Sentence #", group_keys=False).apply(
    lambda s:[(w,p,t) for w,p,t in zip(s.Word,s.POS,s.Tag)]
).tolist()

# Sequence preparation
max_len = 50
X = sequence.pad_sequences([[word2idx[w[0]] for w in s] for s in sents],
                           maxlen=max_len,padding="post",value=len(words)-1)
y = sequence.pad_sequences([[tag2idx[w[2]] for w in s] for s in sents],
                           maxlen=max_len,padding="post",value=tag2idx["O"])

# Convert labels to integer array
X, y = np.array(X, dtype="int32"), np.array(y, dtype="int32")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

# Model
inp = layers.Input(shape=(max_len,))
x = layers.Embedding(len(words), 50, input_length=max_len)(inp)
x = layers.SpatialDropout1D(0.13)(x)
x = layers.Bidirectional(layers.LSTM(250, return_sequences=True, recurrent_dropout=0.13))(x)
out = layers.TimeDistributed(layers.Dense(len(tags), activation="softmax"))(x)

model = Model(inp, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xte, yte), batch_size=45, epochs=3)

# Metrics plot
hist = pd.DataFrame(model.history.history)
hist[['accuracy','val_accuracy']].plot(); hist[['loss','val_loss']].plot()

# Sample prediction
i = 20
p = np.argmax(model.predict(np.array([Xte[i]])), axis=-1)[0]
print("{:15}{:5}\t{}".format("Word", "True", "Pred")); print("-"*30)
for w,t,pd_ in zip(Xte[i], yte[i], p):
    print("{:15}{}\t{}".format(words[w-1], tags[t], tags[pd_]))
OUTPUT
Epoch Training
image
Loss Vs Epoch Plot
image
image
Sample Text Prediction
image
RESULT
Thus, The program to develop an LSTM-based model for recognizing the named entities in the text has been successfully executed.
