import torch
import torch.nn as nn
import numpy as np
from RNN_PSG_optimized import EnhancedRNNModel
import argparse

class ProteinGenerator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model, self.token_to_index = self.load_model(model_path)
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        
    def load_model(self, model_path):
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        config = checkpoint['model_config']
        
        # Calculate max sequence length based on model's capabilities
        max_seq_length = 10287  # This should be at least as large as your training data's max length
        
        # Initialize model with saved configuration
        model = EnhancedRNNModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_length=max_seq_length  # Add this parameter
        ).to(self.device)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint['token_to_index']
    
    def generate_sequence(self, seed_sequence=None, max_length=1000, temperature=1.0):
        """
        Generate a protein sequence.
        Args:
            seed_sequence (str, optional): Starting sequence for generation
            max_length (int): Maximum length of the generated sequence
            temperature (float): Controls randomness (higher = more random)
        """
        with torch.no_grad():
            # Initialize sequence
            if seed_sequence is None:
                # Start with a random amino acid
                current_sequence = [np.random.choice(list(self.token_to_index.keys()))]
            else:
                current_sequence = list(seed_sequence)
            
            # Convert to indices
            current_indices = [self.token_to_index[aa] for aa in current_sequence]
            
            while len(current_sequence) < max_length:
                # Prepare input
                x = torch.tensor([current_indices]).to(self.device)
                lengths = torch.tensor([len(current_indices)])
                
                # Get model prediction
                output, attention_weights = self.model(x, lengths)
                
                # Get probabilities for next amino acid
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=0)
                
                # Sample from the distribution
                next_token_idx = torch.multinomial(next_token_probs, 1).item()
                
                # Add to sequence
                current_indices.append(next_token_idx)
                current_sequence.append(self.index_to_token[next_token_idx])
                
                # Optional: Stop if we generate a stop token (if implemented)
                # if next_token == stop_token:
                #     break
            
            return ''.join(current_sequence)
    
    def generate_multiple_sequences(self, n_sequences, min_length=50, max_length=1000, 
                                 temperature=1.0, seed_sequence=None):
        """
        Generate multiple protein sequences.
        """
        sequences = []
        for _ in range(n_sequences):
            length = np.random.randint(min_length, max_length + 1)
            sequence = self.generate_sequence(
                seed_sequence=seed_sequence,
                max_length=length,
                temperature=temperature
            )
            sequences.append(sequence)
        return sequences

def main():
    parser = argparse.ArgumentParser(description='Generate protein sequences using trained model')
    parser.add_argument('--model_path', type=str, default='RNN Protein Generation- Approach 1/Models/final_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--num_sequences', type=int, default=5,
                      help='Number of sequences to generate')
    parser.add_argument('--min_length', type=int, default=50,
                      help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=1000,
                      help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--seed_sequence', type=str, default=None,
                      help='Optional seed sequence to start generation')
    parser.add_argument('--output_file', type=str, default='generated_sequences.txt',
                      help='Output file to save generated sequences')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ProteinGenerator(args.model_path)
    
    # Generate sequences
    sequences = generator.generate_multiple_sequences(
        n_sequences=args.num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        temperature=args.temperature,
        seed_sequence=args.seed_sequence
    )
    
    # Save sequences to file
    with open(args.output_file, 'w') as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">Generated_Sequence_{i}\n{seq}\n")
    
    print(f"Generated {args.num_sequences} sequences and saved to {args.output_file}")
    
    # Print first sequence as example
    print("\nExample generated sequence:")
    print(sequences[0])

if __name__ == "__main__":
    main()
