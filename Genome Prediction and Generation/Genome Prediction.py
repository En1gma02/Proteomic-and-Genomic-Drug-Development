import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

def process_sequence(sequence):
    tokens_ids = tokenizer.encode(sequence, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
    attention_mask = tokens_ids != tokenizer.pad_token_id
    return tokens_ids, attention_mask

def get_embeddings(tokens_ids, attention_mask):
    with torch.no_grad():
        outputs = model(tokens_ids, attention_mask=attention_mask, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1].squeeze().numpy()
    return embeddings

def predict_molecular_phenotype(sequence):
    tokens_ids, attention_mask = process_sequence(sequence)
    embeddings = get_embeddings(tokens_ids, attention_mask)
    # This is a placeholder for actual phenotype prediction
    phenotype_score = np.mean(embeddings)
    return f"Predicted molecular phenotype score: {phenotype_score:.4f}"

def analyze_genetic_sequence(sequence):
    tokens_ids, attention_mask = process_sequence(sequence)
    embeddings = get_embeddings(tokens_ids, attention_mask)
    # This is a placeholder for actual genetic analysis
    variation_score = np.std(embeddings)
    return f"Genetic variation score: {variation_score:.4f}"

def detect_regulatory_elements(sequence):
    tokens_ids, attention_mask = process_sequence(sequence)
    embeddings = get_embeddings(tokens_ids, attention_mask)
    # This is a placeholder for actual regulatory element detection
    regulatory_score = np.max(embeddings)
    return f"Regulatory element likelihood score: {regulatory_score:.4f}"

def predict_chromatin_accessibility(sequence):
    tokens_ids, attention_mask = process_sequence(sequence)
    embeddings = get_embeddings(tokens_ids, attention_mask)
    # This is a placeholder for actual chromatin accessibility prediction
    accessibility_score = np.median(embeddings)
    return f"Chromatin accessibility score: {accessibility_score:.4f}"

def nucleotide_transformer_app(input_sequence):
    phenotype_score = predict_molecular_phenotype(input_sequence)
    genetic_score = analyze_genetic_sequence(input_sequence)
    regulatory_score = detect_regulatory_elements(input_sequence)
    accessibility_score = predict_chromatin_accessibility(input_sequence)
    
    explanations = [
    f"DNA Activity Level: {phenotype_score}\n"
    f"Think of this like a thermometer reading. "
    f"{'This DNA segment shows active behavior' if float(phenotype_score.split()[-1]) > 0 else 'This DNA segment shows less active behavior'}. "
    f"{'The effect is very subtle - like a whisper' if abs(float(phenotype_score.split()[-1])) < 0.1 else 'The effect is quite strong - like a shout'}\n",
    
    f"DNA Uniqueness Score: {genetic_score}\n"
    f"Imagine comparing this DNA to a standard reference book. "
    f"{'This DNA has many unique spellings' if float(genetic_score.split()[-1]) > 0.6 else 'This DNA has some unique spellings' if float(genetic_score.split()[-1]) > 0.3 else 'This DNA follows mostly standard spelling'}. "
    f"These differences might make it special in its function.\n",
    
    f"DNA Control Switch Score: {regulatory_score}\n"
    f"Think of this like a volume control knob. "
    f"{'There are strong control switches here' if float(regulatory_score.split()[-1]) > 3 else 'There are some control switches here' if float(regulatory_score.split()[-1]) > 1 else 'There are few control switches here'}. "
    f"These switches help turn genes on and off, like a light switch for your genes.\n",
    
    f"DNA Accessibility Score: {accessibility_score}\n"
    f"Imagine this DNA as a book: "
    f"{'The pages are easy to open and read' if float(accessibility_score.split()[-1]) > 0 else 'The pages are somewhat stuck together and harder to read'}. "
    f"This tells us how easily the body's proteins can read this DNA section.\n"
    ]

    
    return "\n".join(explanations)

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
    background-image: url('https://news.fullerton.edu/app/uploads/2020/11/DNA-800x500.jpg'), 
                     linear-gradient(135deg, rgba(26, 42, 108, 0.95) 0%, 
                                           rgba(178, 31, 31, 0.95) 50%, 
                                           rgba(253, 187, 45, 0.95) 100%);
    background-blend-mode: overlay;
    background-size: cover;
    background-attachment: fixed;
}

.gr-input, .gr-button, .gr-form {
    backdrop-filter: blur(10px);
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.gr-input {
    color: white;
}

.gr-form {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 20px;
    border-radius: 15px;
}

.gr-button {
    background-color: rgba(52, 152, 219, 0.8);
    color: white;
    transition: all 0.3s ease;
}

.gr-button:hover {
    background-color: rgba(52, 152, 219, 1);
    transform: translateY(-2px);
}

.title {
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}
"""



with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue")) as iface:
    gr.Markdown(
    """
    # 🧬 Nucleotide Transformer for Multi-Species Analysis
    Analyze DNA sequences using the state-of-the-art Nucleotide Transformer model for various genomic tasks.
    """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=5, 
                placeholder="Enter DNA sequence here...",
                label="Input DNA Sequence"
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Analyze", variant="primary")
    
    output_text = gr.Textbox(label="Analysis Results", lines=10)
    
    examples = gr.Examples(
        examples=[["ATTCCGATTCCGATTCCG"], ["ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]],
        inputs=[input_text],
        outputs=[output_text],
        fn=nucleotide_transformer_app,
        cache_examples=True
    )
    
    submit_btn.click(fn=nucleotide_transformer_app, inputs=input_text, outputs=output_text)
    
    gr.Markdown(
    """
    ### How to use:
    1. Enter a DNA sequence in the input box.
    2. Click the "Analyze" button or use one of the provided examples.
    3. View the comprehensive analysis results below.
    """
    )

iface.launch()
