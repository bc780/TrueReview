import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

bs = 16

df = pd.read_csv("data/data.csv")

class RevDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #TODO fix storage of strings
        features = torch.tensor(row["text"].values, dtype=torch.float32)
        label = torch.tensor(row["adj_rating"], dtype=torch.long)
        return features, label
    
revDataset = RevDataset(df)

torch.manual_seed(1)
train_dataset, val_dataset, test_dataset = random_split(revDataset, [0.8*len(revDataset),0.1*len(revDataset),0.1*len(revDataset)])

train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = True)


