from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

df = pd.read_csv("data/data.csv")
df["text"] = df["text"].apply(clean_text)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df["text"])
df["tokenized"] = tokenizer.texts_to_sequences(df["text"])
max_len = 500
padded_sequences = pad_sequences(df["tokenized"], maxlen = max_len, padding = "post")
df["padded"] = np.NAN
df["padded"] = df["padded"].astype("object")
df["padded"] = [list(seq) for seq in padded_sequences]

df = df[["padded","adj_rating"]]

print(df.head())

df.to_csv("data/token_data.csv")


