import pandas as pd

#data

prof_df = pd.read_csv("data/professional_reviews_cleaned.csv")
result = prof_df.groupby('title')['decimal_score'].mean().reset_index()
rev_df = pd.read_csv("data/merge.csv")
avg_rev_df = rev_df.groupby('title')['rating'].mean().reset_index()
avg_rev_df = avg_rev_df.rename(columns={'rating':'avg_rating'})
avg_rev_df['avg_rating'] = avg_rev_df['avg_rating']/5.0

data = pd.merge(rev_df,result,on = "title",how = "inner")
data = pd.merge(data,avg_rev_df,on = "title",how = "inner")
data["adjustment"] = data['decimal_score'] - data['avg_rating']
data["adj_rating"] = data['rating'] + data['adjustment']
data = data.dropna()

data.to_csv("data/data.csv",index = False)
print(data.head())
