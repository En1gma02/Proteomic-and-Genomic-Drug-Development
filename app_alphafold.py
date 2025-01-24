import gradio as gr
import torch
import pandas as pd
from simple_generate import RNNModel, generate_sequence
import re
import os
import py3Dmol
import time
import subprocess

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

def setup_alphafold():
    try:
        # Install required packages
        commands = [
            "pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'",
            "pip install -q --no-warn-conflicts -U dm-haiku==0.0.10 jax==0.3.25",
            "pip install -q py3Dmol",
            "pip install -q matplotlib",
            "pip install -q ipywidgets"
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error installing packages: {result.stderr}"
        
        return "Setup completed successfully"
    except Exception as e:
        return f"Setup failed: {str(e)}"

def predict_structure(sequence, jobname, num_recycle, template_mode, model_type, num_models):
    try:
        # Create directory for results
        os.makedirs(jobname, exist_ok=True)
        
        # Save sequence to FASTA file
        fasta_path = os.path.join(jobname, f"{jobname}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">query\n{sequence}")
        
        # Prepare command
        cmd = [
            "colabfold_batch",
            "--num-recycle", str(num_recycle),
            "--model-type", model_type,
            "--num-models", str(num_models)
        ]
        
        if template_mode != "none":
            cmd.extend(["--templates", template_mode])
        
        cmd.extend([fasta_path, jobname])
        
        # Run prediction
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                yield output.strip(), None
        
        # Check for results
        pdb_file = os.path.join(jobname, "ranked_0.pdb")
        if os.path.exists(pdb_file):
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            # Create 3D visualization
            viewer = py3Dmol.view(width=800, height=600)
            viewer.addModel(pdb_content, "pdb")
            viewer.setStyle({'cartoon': {'colorscheme': 'spectrum'}})
            viewer.zoomTo()
            
            yield "Prediction completed successfully", viewer.render()
        else:
            yield "Failed to generate structure", None
            
    except Exception as e:
        yield f"Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Protein Analysis Suite") as iface:
    gr.Markdown("# Protein Analysis Suite")
    
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
        First click 'Setup Environment' to install required packages, then enter your protein sequence and parameters.
        """)
        
        with gr.Row():
            setup_btn = gr.Button("Setup Environment")
            setup_status = gr.Textbox(label="Setup Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                input_sequence = gr.Textbox(
                    label="Input Protein Sequence",
                    lines=5,
                    placeholder="Enter protein sequence..."
                )
                jobname = gr.Textbox(
                    label="Job Name",
                    value="test"
                )
                model_type = gr.Dropdown(
                    choices=["auto", "alphafold2_ptm", "alphafold2_multimer_v1", 
                            "alphafold2_multimer_v2", "alphafold2_multimer_v3"],
                    value="auto",
                    label="Model Type"
                )
                num_recycle = gr.Dropdown(
                    choices=["auto", "0", "1", "3", "6", "12", "24", "48"],
                    value="3",
                    label="Number of Recycles"
                )
                template_mode = gr.Dropdown(
                    choices=["none", "pdb100", "custom"],
                    value="none",
                    label="Template Mode"
                )
                num_models = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Models"
                )
                predict_btn = gr.Button("Predict Structure")
            
            with gr.Column():
                progress_output = gr.Textbox(
                    label="Progress",
                    lines=10,
                    interactive=False
                )
                structure_viewer = gr.HTML(label="3D Structure Visualization")
    
    # Connect components
    setup_btn.click(setup_alphafold, outputs=[setup_status])
    
    predict_btn.click(
        predict_structure,
        inputs=[
            input_sequence, jobname, num_recycle,
            template_mode, model_type, num_models
        ],
        outputs=[progress_output, structure_viewer]
    )

if __name__ == "__main__":
    iface.launch()
