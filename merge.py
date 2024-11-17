import pandas as pd
import os

path = r"C:\Users\brand\OneDrive\Documents\GitHub\TrueReview"
path_in = os.path.join(path, "data")
path_out = os.path.join(path, "data")


iter = pd.read_csv(os.path.join(path_in, "reviews.csv"), iterator=True, chunksize=1000)
reviews = pd.concat(chunk for chunk in iter)
metadata = pd.read_csv(os.path.join(path_in, "metadata.csv"))

df = reviews.merge(metadata, on="parent_asin", how="left").dropna()
df.to_csv(os.path.join(path_out, "merge.csv"), index=False)