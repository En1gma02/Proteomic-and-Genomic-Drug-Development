import gradio as gr
import torch
import pandas as pd
from simple_generate import RNNModel, generate_sequence
import re
import webbrowser

class ProteinGenerator:
    def __init__(self, model_path='trained_model.pth', data_path='preprocessed_encoded_vectorized.csv'):
        # Load data and initialize model
        data = pd.read_csv(data_path)
        all_amino_acids = set(''.join(data['sequence']))
        self.token_to_index = {token: idx for idx, token in enumerate(sorted(all_amino_acids))}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        
        input_size = len(self.token_to_index)
        hidden_size = 128
        output_size = len(self.token_to_index)
        
        self.model = RNNModel(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def is_valid_sequence(self, sequence):
        # Check if sequence has 3 or more continuous same amino acids
        if re.search(r'(.)\1{2,}', sequence):
            return False
        # Check if sequence is too short
        if len(sequence) < 20:
            return False
        # Check if sequence has enough variety
        if len(set(sequence)) < 3:
            return False
        return True

    def generate_protein(self, num_sequences=5, max_length=100, temperature=1.0, start_token='M'):
        valid_sequences = []
        attempts = 0
        max_attempts = num_sequences * 5  # Allow more attempts to find valid sequences
        
        while len(valid_sequences) < num_sequences and attempts < max_attempts:
            sequence = generate_sequence(
                self.model, 
                self.index_to_token, 
                max_length=max_length, 
                start_token=start_token,
                temperature=temperature
            )
            
            if self.is_valid_sequence(sequence):
                valid_sequences.append(sequence)
            attempts += 1
            
        return "\n".join(valid_sequences) if valid_sequences else "No valid sequences generated"

def generate_proteins(num_sequences, max_length, temperature, start_token):
    generator = ProteinGenerator()
    sequences = generator.generate_protein(
        num_sequences=int(num_sequences),
        max_length=int(max_length),
        temperature=float(temperature),
        start_token=start_token
    )
    return sequences

def open_colabfold(sequence):
    # Base URL for the ColabFold notebook
    colab_url = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"
    
    # Open the URL in a new browser tab
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
    This tool provides two main functionalities:
    1. Generate new protein sequences using our trained model
    2. Analyze protein structures using AlphaFold2 (via ColabFold)
    """)
    
    with gr.Tab("Protein Generator"):
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
        generate_proteins,
        inputs=[num_sequences, max_length, temperature, start_token],
        outputs=[output_sequences]
    )
    
    analyze_btn.click(
        open_colabfold,
        inputs=[input_sequence],
        outputs=[instructions]
    )

if __name__ == "__main__":
    iface.launch()
