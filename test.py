import pandas as pd
import ast

df = pd.read_csv("data/token_data.csv", dtype={"padded":"object","adj_rating":"float32"})
df["padded"] = df["padded"].apply(ast.literal_eval)

print(type(df.iloc[0,1]))