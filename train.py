import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

bs = 16
epochs = 10

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

class revLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx = 0):
        super(revLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)

        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def attention(self, lstm_output):
        score = torch.matmul(lstm_output)
        weights = torch.softmax(score, dim=1)
        weighted_output = lstm_output*weights
        return weighted_output.sum(dim=1)
    
    def foward(self, x):
        lstm_output, _ = self.lstm(self.embedding(x))
        output = self.attention(lstm_output)
        return torch.sigmoid(self.fc(output))



