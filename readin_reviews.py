import pandas as pd
import os

path = r"C:\Users\brand\OneDrive\Documents\GitHub\TrueReview"
path_in = os.path.join(path, "data")
path_out = os.path.join(path, "data")

#reviews = pd.read_json(os.path.join(path_in, "TEST_Movies_and_TV.jsonl"), lines=True)
#reviews = reviews.filter(items = ["parent_asin", "rating", "text"]).dropna()
#reviews["text"] = reviews["text"].str.replace(r"<br />", " ", regex=True)

iter = pd.read_json(os.path.join(path_in, "Movies_and_TV.jsonl"), lines=True, chunksize=1000)
df = pd.concat(
    chunk.filter(items=['parent_asin', 'rating', 'text']).assign(
        text=lambda x: x['text'].astype(str).str.replace(r"<br />", " ", regex=True))
    for chunk in iter)

df.to_csv(os.path.join(path_out, r'reviews.csv'), index=False)