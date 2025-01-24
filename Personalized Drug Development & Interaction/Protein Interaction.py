import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F

import gradio as gr
from gradio.themes.utils import colors

class YellowGreenTheme(gr.themes.Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.lime,
            secondary_hue=colors.yellow,
            neutral_hue=colors.gray,
            spacing_size="md",
            radius_size="md",
            text_size="lg",
            font=["Arial", "sans-serif"],
        )
        self.set(
            body_background_fill="linear-gradient(135deg, #d4fc79, #96e6a1)",
            button_primary_background_fill="linear-gradient(90deg, #a8e063, #56ab2f)",
            button_primary_text_color="white",
            block_title_text_weight="600",
            block_border_width="2px",
            block_shadow="*shadow_drop_lg",
        )

class PLMinteract(nn.Module):
    def __init__(self, model_name, num_labels, embedding_size):
        super(PLMinteract, self).__init__()
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name)
        self.embedding_size = embedding_size
        self.classifier = nn.Linear(embedding_size, 1)
        self.num_labels = num_labels

    def forward_test(self, features):
        embedding_output = self.esm_mask.base_model(**features, return_dict=True)
        embedding = embedding_output.last_hidden_state[:, 0, :]
        embedding = F.relu(embedding)
        logits = self.classifier(embedding)
        logits = logits.view(-1)
        probability = torch.sigmoid(logits)
        return probability

def load_model():
    folder_huggingface_download = r'C:\Users\ppawa\Pranav\Study\Engineering\Hackathons\BioCode Breakers\Code Demos\Test/'
    model_name = 'facebook/esm2_t33_650M_UR50D'
    embedding_size = 1280
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    PLMinter = PLMinteract(model_name, 1, embedding_size)
    PLMinter.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name)
    load_model = torch.load(f"{folder_huggingface_download}pytorch_model.bin")
    PLMinter.load_state_dict(load_model)
    PLMinter.eval()
    PLMinter.to(DEVICE)
    
    return PLMinter, tokenizer, DEVICE

model, tokenizer, device = load_model()

def predict_ppi(protein1, protein2):
    texts = [protein1, protein2]
    tokenized = tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=1603)
    tokenized = tokenized.to(device)
    
    with torch.no_grad():
        probability = model.forward_test(tokenized)
    
    return probability.item()

def ppi_prediction(species, protein1, protein2):
    if protein1 and protein2:
        probability = predict_ppi(protein1, protein2)
        return f"Interaction Probability: {probability:.4f}"
    else:
        return "Please enter both protein sequences."

def virus_host_ppi(virus_protein, host_protein):
    if virus_protein and host_protein:
        probability = predict_ppi(virus_protein, host_protein)
        return f"Virus-Host Interaction Probability: {probability:.4f}"
    else:
        return "Please enter both virus and host protein sequences."

def interaction_disruption(original_sequence, modified_sequence, partner_sequence):
    if original_sequence and modified_sequence and partner_sequence:
        original_prob = predict_ppi(original_sequence, partner_sequence)
        modified_prob = predict_ppi(modified_sequence, partner_sequence)
        
        result = f"Original Interaction Probability: {original_prob:.4f}\n"
        result += f"Modified Interaction Probability: {modified_prob:.4f}\n"
        
        if abs(original_prob - modified_prob) > 0.1:
            result += "Significant change in interaction probability detected."
        else:
            result += "No significant change in interaction probability detected."
        
        return result
    else:
        return "Please enter all required sequences."

def sequence_modification(sequence_to_analyze, modification_position, new_amino_acid):
    if sequence_to_analyze and new_amino_acid:
        if 1 <= modification_position <= len(sequence_to_analyze):
            modified_sequence = sequence_to_analyze[:modification_position-1] + new_amino_acid + sequence_to_analyze[modification_position:]
            result = f"Modified Sequence:\n{modified_sequence}\n\n"
            result += "Potential impact on interactions:\n"
            
            for aa in "ACDEFGHIKLMNPQRSTVWY":
                if aa != new_amino_acid:
                    partner_seq = "A" * 50 + aa + "A" * 50
                    orig_prob = predict_ppi(sequence_to_analyze, partner_seq)
                    mod_prob = predict_ppi(modified_sequence, partner_seq)
                    result += f"With {aa}: Original: {orig_prob:.4f}, Modified: {mod_prob:.4f}\n"
            
            return result
        else:
            return "Invalid modification position."
    else:
        return "Please enter the sequence and new amino acid."

custom_css = """
.gradio-container {
    background-image: url('https://img.freepik.com/premium-photo/mesmerizing-macro-photograph-complex-biological-molecular-structure-vibrant-colors_1287633-9058.jpg');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
"""

with gr.Blocks(theme=YellowGreenTheme(), css=custom_css) as demo:
    gr.Markdown("# PLM-Interact: Protein-Protein Interaction Analysis")
    
    with gr.Tab("PPI Prediction"):
        gr.Markdown("## Protein-Protein Interaction (PPI) Prediction")
        species = gr.Dropdown(["Human", "Mouse", "Fly", "Worm", "E. coli", "Yeast"], label="Select Species")
        with gr.Row():
            protein1 = gr.Textbox(label="Protein 1 Sequence", lines=10)
            protein2 = gr.Textbox(label="Protein 2 Sequence", lines=10)
        ppi_button = gr.Button("Predict PPI")
        ppi_output = gr.Textbox(label="PPI Prediction Result")
        ppi_button.click(ppi_prediction, inputs=[species, protein1, protein2], outputs=ppi_output)
    
    with gr.Tab("Virus-Host PPI"):
        gr.Markdown("## Virus-Host PPI Prediction")
        virus_protein = gr.Textbox(label="Virus Protein Sequence", lines=8)
        host_protein = gr.Textbox(label="Host Protein Sequence", lines=8)
        virus_host_button = gr.Button("Predict Virus-Host PPI")
        virus_host_output = gr.Textbox(label="Virus-Host PPI Prediction Result")
        virus_host_button.click(virus_host_ppi, inputs=[virus_protein, host_protein], outputs=virus_host_output)
    
    with gr.Tab("Interaction Disruption"):
        gr.Markdown("## Detection of Interaction Disruptions")
        original_sequence = gr.Textbox(label="Original Protein Sequence", lines=8)
        modified_sequence = gr.Textbox(label="Modified Protein Sequence", lines=8)
        partner_sequence = gr.Textbox(label="Partner Protein Sequence", lines=8)
        disruption_button = gr.Button("Analyze Interaction Disruption")
        disruption_output = gr.Textbox(label="Interaction Disruption Analysis Result")
        disruption_button.click(interaction_disruption, inputs=[original_sequence, modified_sequence, partner_sequence], outputs=disruption_output)
    
    with gr.Tab("Sequence Analysis"):
        gr.Markdown("## Protein Sequence Analysis")
        sequence_to_analyze = gr.Textbox(label="Protein Sequence to Analyze", lines=8)
        modification_position = gr.Number(label="Position to modify (1-based indexing)", minimum=1, step=1)
        new_amino_acid = gr.Textbox(label="New amino acid (single letter code)", max_lines=1)
        modification_button = gr.Button("Analyze Sequence Modification")
        modification_output = gr.Textbox(label="Sequence Modification Analysis Result")
        modification_button.click(sequence_modification, inputs=[sequence_to_analyze, modification_position, new_amino_acid], outputs=modification_output)
    
    with gr.Tab("Integration"):
        gr.Markdown("## Integration with Other Tools")
        gr.Markdown("""
        This section would typically include integration with other bioinformatics tools. For demonstration purposes, we'll provide links to external tools.

        - [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi): Search for similar sequences
        - [PDB](https://www.rcsb.org/): Protein Data Bank for 3D structures
        - [UniProt](https://www.uniprot.org/): Comprehensive protein database
        """)

demo.launch()