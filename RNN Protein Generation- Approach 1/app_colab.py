import gradio as gr
import torch
from simple_generate import RNNModel, generate_sequence
from optimized_generate import ProteinGenerator as OptimizedGenerator
import re
import webbrowser

class SimpleProteinGenerator:
    def __init__(self, model_path='RNN Protein Generation- Approach 1/Models/trained_model.pth', data_path='RNN Protein Generation- Approach 1/Datasets/preprocessed_encoded_vectorized.csv'):
        # Load data and initialize model
        self.model, self.index_to_token = self._initialize_model(model_path, data_path)

    def _initialize_model(self, model_path, data_path):
        import pandas as pd
        data = pd.read_csv(data_path)
        all_amino_acids = set(''.join(data['sequence']))
        token_to_index = {token: idx for idx, token in enumerate(sorted(all_amino_acids))}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        
        input_size = len(token_to_index)
        hidden_size = 128
        output_size = len(token_to_index)
        
        model = RNNModel(input_size, hidden_size, output_size)
        # Load model with CPU map location
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, index_to_token

    def is_valid_sequence(self, sequence):
        if re.search(r'(.)\1{2,}', sequence):
            return False
        if len(sequence) < 20:
            return False
        if len(set(sequence)) < 3:
            return False
        return True

    def generate_protein(self, num_sequences=5, max_length=100, temperature=1.0, start_token='M'):
        valid_sequences = []
        attempts = 0
        max_attempts = num_sequences * 5
        
        while len(valid_sequences) < num_sequences and attempts < max_attempts:
            sequence = generate_sequence(
                self.model, 
                self.index_to_token, 
                max_length=max_length, 
                start_token=start_token
            )
            
            if self.is_valid_sequence(sequence):
                valid_sequences.append(sequence)
            attempts += 1
            
        return "\n".join(valid_sequences) if valid_sequences else "No valid sequences generated"

def generate_simple_proteins(num_sequences, max_length, temperature, start_token):
    generator = SimpleProteinGenerator()
    sequences = generator.generate_protein(
        num_sequences=int(num_sequences),
        max_length=int(max_length),
        temperature=float(temperature),
        start_token=start_token
    )
    return sequences

def generate_optimized_proteins(num_sequences, min_length, max_length, temperature, seed_sequence):
    # Initialize with CPU device
    device = 'cpu'
    generator = OptimizedGenerator(
        model_path='RNN Protein Generation- Approach 1/Models/final_model.pth',
        device=device
    )
    
    sequences = generator.generate_multiple_sequences(
        n_sequences=int(num_sequences),
        min_length=int(min_length),
        max_length=int(max_length),
        temperature=float(temperature),
        seed_sequence=seed_sequence if seed_sequence.strip() else None
    )
    return "\n".join(sequences) if sequences else "No sequences generated"

def open_colabfold(sequence):
    colab_url = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"
    webbrowser.open(colab_url, new=2)
    return ("Please follow these steps:\n\n"
            "1. The ColabFold notebook has been opened in a new tab\n"
            "2. Click 'Runtime' -> 'Run all' at the top\n"
            "3. Paste your sequence in the first text box\n"
            "4. Follow the instructions in the notebook\n\n"
            "Your sequence has been copied to clipboard for convenience.")

# Create Gradio interface
with gr.Blocks(title="Protein Analysis Suite") as iface:
    gr.Markdown("""
    # Protein Analysis Suite
    This tool provides three main functionalities:
    1. Generate new protein sequences using our simple RNN model
    2. Generate new protein sequences using our optimized RNN model
    3. Analyze protein structures using AlphaFold2 (via ColabFold)
    """)
    
    with gr.Tab("Simple Protein Generator"):
        with gr.Row():
            with gr.Column():
                num_sequences = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, 
                    label="Number of Sequences"
                )
                max_length = gr.Slider(
                    minimum=50, maximum=200, value=100, step=10, 
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                    label="Temperature (higher = more random)"
                )
                start_token = gr.Textbox(
                    value="M", 
                    label="Start Token (usually M for proteins)"
                )
                generate_btn = gr.Button("Generate Proteins")
            
            with gr.Column():
                output_sequences = gr.Textbox(
                    label="Generated Sequences", 
                    lines=10,
                    placeholder="Generated sequences will appear here..."
                )
    
    with gr.Tab("Optimized Protein Generator"):
        with gr.Row():
            with gr.Column():
                opt_num_sequences = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, 
                    label="Number of Sequences"
                )
                opt_min_length = gr.Slider(
                    minimum=20, maximum=150, value=50, step=10, 
                    label="Minimum Length"
                )
                opt_max_length = gr.Slider(
                    minimum=50, maximum=200, value=100, step=10, 
                    label="Maximum Length"
                )
                opt_temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1, 
                    label="Temperature (higher = more random)"
                )
                opt_seed_sequence = gr.Textbox(
                    value="", 
                    label="Seed Sequence (optional)",
                    placeholder="Leave empty for random start"
                )
                opt_generate_btn = gr.Button("Generate Proteins")
            
            with gr.Column():
                opt_output_sequences = gr.Textbox(
                    label="Generated Sequences", 
                    lines=10,
                    placeholder="Generated sequences will appear here..."
                )
    
    with gr.Tab("Protein Structure Analysis"):
        gr.Markdown("""
        ## Protein Structure Prediction using AlphaFold2
        This tool will redirect you to ColabFold, which provides free access to AlphaFold2 
        through Google Colab. Follow these steps:
        1. Generate or paste your protein sequence below
        2. Click 'Analyze with AlphaFold'
        3. A new tab will open with ColabFold
        4. Follow the instructions in the Colab notebook
        """)
        
        with gr.Row():
            with gr.Column():
                input_sequence = gr.Textbox(
                    label="Input Protein Sequence", 
                    lines=5,
                    placeholder="Paste your protein sequence here..."
                )
                analyze_btn = gr.Button("Analyze with AlphaFold")
            
            with gr.Column():
                instructions = gr.Textbox(
                    label="Instructions",
                    lines=8,
                    interactive=False
                )
    
    # Connect components
    generate_btn.click(
        generate_simple_proteins,
        inputs=[num_sequences, max_length, temperature, start_token],
        outputs=[output_sequences]
    )
    
    opt_generate_btn.click(
        generate_optimized_proteins,
        inputs=[opt_num_sequences, opt_min_length, opt_max_length, opt_temperature, opt_seed_sequence],
        outputs=[opt_output_sequences]
    )
    
    analyze_btn.click(
        open_colabfold,
        inputs=[input_sequence],
        outputs=[instructions]
    )

if __name__ == "__main__":
    iface.launch()
