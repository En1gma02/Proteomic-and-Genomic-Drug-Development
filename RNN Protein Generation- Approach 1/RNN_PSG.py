import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, data, token_to_index):
        self.data = data
        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        sequence = [self.token_to_index[token] for token in sequence]  # Convert sequence to numerical indices
        return sequence

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    data = pd.read_csv('E:/Hackathons/IIT BHU/Protien Sequencing/preprocessed_encoded_vectorized.csv')

    all_amino_acids = set(''.join(data['sequence']))
    token_to_index = {token: idx for idx, token in enumerate(sorted(all_amino_acids))}
    input_size = len(token_to_index)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = ProteinDataset(train_data, token_to_index)
    test_dataset = ProteinDataset(test_data, token_to_index)

    def collate_fn(batch):
        sequences = [torch.tensor(item) for item in batch]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    hidden_size = 128
    output_size = len(token_to_index)

    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            target = batch[:, 1:].contiguous().view(-1)
            output = output[:, :-1, :].contiguous().view(-1, output.shape[-1])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    torch.save(model.state_dict(), 'trained_model.pth')


