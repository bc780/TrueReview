import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

bs = 64
max_batches = 100
epochs = 100

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


class RevDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #TODO fix storage of strings
        features = torch.tensor(row["padded"], dtype=torch.long)
        label = torch.tensor(row["adj_rating"], dtype=torch.float32)
        return features, label
    
revDataset = RevDataset(df)

torch.manual_seed(1)
train_dataset, val_dataset, test_dataset = random_split(revDataset, [int(0.8*len(revDataset)),int(0.1*len(revDataset)),len(revDataset) - int(0.8*len(revDataset)) - int(0.1*len(revDataset))])

train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = True)

class revLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len, pad_idx = 0):
        super(revLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)

        self.attention_weights = nn.Parameter(torch.randn(seq_len))

    
    def forward(self, x):
        lstm_output, _ = self.lstm(self.embedding(x))
        weights = self.attention_weights.unsqueeze(0).unsqueeze(2)  # Shape: [1, seq_len, 1]
        weights = weights.expand(x.size(0), -1, -1)
        weighted_output = lstm_output * weights
        final_output = weighted_output.mean(dim=1)
        output = self.fc(final_output)
        return torch.sigmoid(output)

model = revLSTM(10000, 500, 16, 500)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.MSELoss(reduction="mean")

print("start train")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        if i > max_batches:
            break
        inputs, labels = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.2f}")
    
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i > max_batches:
                break
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch} Validation Loss: {val_loss:.2f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "checkpoint.pth")



