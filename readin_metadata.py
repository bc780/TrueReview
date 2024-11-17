import pandas as pd
import os

path = r"C:\Users\brand\OneDrive\Documents\GitHub\TrueReview"
path_in = os.path.join(path, "data")
path_out = os.path.join(path, "data")

df_md = pd.read_json(r"C:\Users\brand\OneDrive\Documents\GitHub\TrueReview\data\meta_Movies_and_TV.jsonl", lines=True)
df_md = df_md.filter(items = ['parent_asin', 'title', 'rating_number', 'categories']).dropna()
df_md.to_csv(os.path.join(path,r'metadata.csv'), index=False)

pd.set_option('display.max.columns', 4)
print(df_md.info)