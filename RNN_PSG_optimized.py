import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb  # For experiment tracking
from tqdm import tqdm
import math

class ProteinDataset(Dataset):
    def __init__(self, data, token_to_index):
        self.data = data
        self.token_to_index = token_to_index
        self.sequence_lengths = [len(seq) for seq in data['sequence']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        sequence = [self.token_to_index[token] for token in sequence]
        return torch.tensor(sequence), self.sequence_lengths[idx]

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # Apply attention weights
        attended = attention_weights * x
        return attended, attention_weights

class EnhancedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3, max_seq_length=5000):
        super(EnhancedRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer with learned positional encoding
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        
        # Positional encoding
        self.register_buffer(
            "pos_encoder",
            self._create_positional_encoding(max_seq_length, hidden_size)
        )
        
        # Bidirectional LSTM layers with residual connections
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
        
    def forward(self, x, lengths):
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x)
        
        # Add positional encoding
        positions = self.pos_encoder[:seq_len, :].unsqueeze(0)
        embedded = embedded + positions
        
        # Pack padded sequence for LSTM
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM layers
        lstm_out, _ = self.lstm(packed)
        
        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out)
        
        # Residual connection and layer normalization
        out = self.fc1(attended_out)
        out = self.dropout(out)
        out = self.layer_norm(out + embedded)  # Residual connection
        
        # Final output layer
        out = self.fc2(out)
        
        return out, attention_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch, lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output, _ = model(batch, lengths)
            target = batch[:, 1:].contiguous().view(-1)
            output = output[:, :-1, :].contiguous().view(-1, output.shape[-1])
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch, lengths in val_loader:
                batch = batch.to(device)
                output, _ = model(batch, lengths)
                target = batch[:, 1:].contiguous().view(-1)
                output = output[:, :-1, :].contiguous().view(-1, output.shape[-1])
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pth')
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

if __name__ == '__main__':
    # Initialize wandb
    wandb.init(project="protein-sequence-generation", name="enhanced-rnn")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
    data = pd.read_csv('preprocessed_encoded_vectorized.csv')
    
    # Calculate max sequence length
    max_seq_length = data['sequence'].str.len().max()
    print(f"Max sequence length in dataset: {max_seq_length}")
    
    all_amino_acids = set(''.join(data['sequence']))
    token_to_index = {token: idx + 1 for idx, token in enumerate(sorted(all_amino_acids))}
    token_to_index['<PAD>'] = 0  # Add padding token
    input_size = len(token_to_index)
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = ProteinDataset(train_data, token_to_index)
    val_dataset = ProteinDataset(val_data, token_to_index)
    
    def collate_fn(batch):
        sequences, lengths = zip(*batch)
        lengths = torch.tensor(lengths)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences, lengths
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Model parameters
    hidden_size = 256
    output_size = len(token_to_index)
    num_layers = 3
    dropout = 0.3
    
    # Initialize model with max_seq_length parameter
    model = EnhancedRNNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_length=max_seq_length
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model
    num_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'token_to_index': token_to_index,
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'max_seq_length': max_seq_length
        }
    }, 'final_model.pth')
    
    wandb.finish()


