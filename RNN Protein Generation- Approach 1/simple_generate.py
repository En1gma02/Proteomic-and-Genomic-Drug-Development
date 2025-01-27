import torch
import torch.nn as nn
import pandas as pd

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

def generate_sequence(model, index_to_token, max_length=100, start_token='M'):
    model.eval()
    with torch.no_grad():
        if start_token in index_to_token:
            start_idx = index_to_token[start_token]
        else:
            start_idx = 0  # Use the first token in the dictionary as the start token
        start_tensor = torch.tensor([start_idx]).unsqueeze(0)
        hidden = model.init_hidden(1)
        sequence = [index_to_token[start_idx]]
        input = start_tensor

        for _ in range(max_length - 1):
            output, hidden = model.forward(input, hidden)
            output_probs = output.squeeze().softmax(dim=0)
            token_idx = torch.multinomial(output_probs, num_samples=1).item()
            token = index_to_token[token_idx]
            sequence.append(token)
            input = torch.tensor([[token_idx]])

            if token == '*':  # Stop token
                break

    return ''.join(sequence)

if __name__ == '__main__':
    data = pd.read_csv('E:/Hackathons/IIT BHU/Protien Sequencing/preprocessed_encoded_vectorized.csv')
    all_amino_acids = set(''.join(data['sequence']))
    token_to_index = {token: idx for idx, token in enumerate(sorted(all_amino_acids))}
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    input_size = len(token_to_index)
    hidden_size = 128
    output_size = len(token_to_index)

    model = RNNModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('trained_model.pth'))

    num_sequences = 10
    for _ in range(num_sequences):
        generated_sequence = generate_sequence(model, index_to_token)
        print(generated_sequence)