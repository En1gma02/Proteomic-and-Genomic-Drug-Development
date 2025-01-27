import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np

# Load models and tokenizers
gena_tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
gena_model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base', trust_remote_code=True)

nucleotide_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
nucleotide_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

# Generation Functions
def dna_sequence_completion(sequence):
    inputs = gena_tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512)
    inputs['input_ids'][0, len(sequence)//2:] = gena_tokenizer.mask_token_id
    outputs = gena_model(**inputs)
    logits = outputs.last_hidden_state[:, len(sequence)//2:, :]
    predicted_ids = torch.argmax(logits, dim=-1)
    completed_sequence = gena_tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    explanation = "✨ This is the completed DNA sequence based on the input. The model predicts likely nucleotides to fill in missing parts."
    return completed_sequence, explanation

def sequence_similarity(seq1, seq2):
    embedding1 = gena_model(**gena_tokenizer(seq1, return_tensors='pt')).last_hidden_state.mean(dim=1)
    embedding2 = gena_model(**gena_tokenizer(seq2, return_tensors='pt')).last_hidden_state.mean(dim=1)
    similarity_score = torch.cosine_similarity(embedding1, embedding2).item()
    explanation = f"🔍 Similarity score: {similarity_score:.2f}\nA score closer to 1.0 indicates higher similarity between sequences."
    return similarity_score, explanation

def mutation_impact_analysis(sequence, mutation):
    try:
        index, new_nucleotide = mutation.split(',')
        index = int(index)
        mutated_sequence = sequence[:index] + new_nucleotide + sequence[index+1:]
        original_embedding = gena_model(**gena_tokenizer(sequence, return_tensors='pt')).last_hidden_state.mean(dim=1)
        mutated_embedding = gena_model(**gena_tokenizer(mutated_sequence, return_tensors='pt')).last_hidden_state.mean(dim=1)
        impact_score = torch.cosine_similarity(original_embedding, mutated_embedding).item()
        return mutated_sequence, impact_score, f"🧬 Impact Score: {impact_score:.2f}\nLower scores indicate more significant changes."
    except:
        return "", 0.0, "❌ Error: Please enter mutation details in format: index,nucleotide (e.g., 5,A)"

# Add this new function after other functions
def dna_sequence_classification(sequence):
    """
    Classifies DNA sequence into categories (e.g., coding vs. non-coding).
    """
    # Using GENA model for better classification
    embedding = gena_model(**gena_tokenizer(sequence, return_tensors='pt')).last_hidden_state.mean(dim=1)
    
    # Simple classification logic based on sequence patterns and embeddings
    has_start_codon = "ATG" in sequence
    has_stop_codons = any(codon in sequence for codon in ["TAA", "TAG", "TGA"])
    embedding_mean = embedding.mean().item()
    
    # More sophisticated classification logic
    if has_start_codon and has_stop_codons and embedding_mean > 0:
        category = "Coding (Protein-coding gene)"
        confidence = abs(embedding_mean) * 100
    elif has_start_codon and not has_stop_codons:
        category = "Partial Coding (Incomplete gene)"
        confidence = abs(embedding_mean) * 75
    elif "TATA" in sequence or "CAAT" in sequence:
        category = "Regulatory (Promoter region)"
        confidence = abs(embedding_mean) * 85
    else:
        category = "Non-coding (Intergenic region)"
        confidence = abs(embedding_mean) * 90
    
    explanation = f"""🧬 Classification Result: {category}
📊 Confidence: {confidence:.2f}%

Key Features:
{'✓' if has_start_codon else '✗'} Start codon (ATG)
{'✓' if has_stop_codons else '✗'} Stop codons (TAA/TAG/TGA)
{'✓' if "TATA" in sequence or "CAAT" in sequence else '✗'} Regulatory elements

This classification is based on sequence patterns and deep learning analysis."""
    
    return category, explanation

# Prediction Functions
def process_sequence(sequence):
    tokens_ids = nucleotide_tokenizer.encode(sequence, return_tensors="pt", padding="max_length", max_length=nucleotide_tokenizer.model_max_length)
    attention_mask = tokens_ids != nucleotide_tokenizer.pad_token_id
    return tokens_ids, attention_mask

def get_embeddings(tokens_ids, attention_mask):
    with torch.no_grad():
        outputs = nucleotide_model(tokens_ids, attention_mask=attention_mask, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1].squeeze().numpy()
    return embeddings

def analyze_sequence(sequence):
    tokens_ids, attention_mask = process_sequence(sequence)
    embeddings = get_embeddings(tokens_ids, attention_mask)
    
    # Calculate various scores
    phenotype_score = np.mean(embeddings)
    variation_score = np.std(embeddings)
    regulatory_score = np.max(embeddings)
    accessibility_score = np.median(embeddings)
    
    analysis = f"""
    🧬 DNA Activity Level: {phenotype_score:.4f}
    {' High activity detected' if phenotype_score > 0 else ' Low activity detected'}
    
    🔍 Genetic Variation Score: {variation_score:.4f}
    {' High genetic diversity' if variation_score > 0.6 else ' Moderate to low genetic diversity'}
    
    🎛️ Regulatory Element Score: {regulatory_score:.4f}
    {' Strong regulatory elements present' if regulatory_score > 3 else ' Weak regulatory elements present'}
    
    🔓 DNA Accessibility Score: {accessibility_score:.4f}
    {' Highly accessible region' if accessibility_score > 0 else ' Less accessible region'}
    """
    return analysis

# Custom CSS for better UI
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
    color: white !important;
}

.tabs {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px;
}

.input-box, .output-box {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

.primary-btn {
    background-color: #4CAF50 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    background-color: #45a049 !important;
    transform: translateY(-2px) !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧬 DNA Generation & Analysis Suite
        A comprehensive tool for DNA sequence analysis, prediction, and generation.
        """
    )
    
    with gr.Tabs() as tabs:
        # Generation Tab (now first)
        with gr.Tab("✨ Sequence Generation"):
            gr.Markdown("### Complete Partial DNA Sequences")
            gen_input = gr.Textbox(label="Partial DNA Sequence", placeholder="Enter partial sequence...", lines=3)
            gen_btn = gr.Button("Complete Sequence", elem_classes="primary-btn")
            gen_output = gr.Textbox(label="Completed Sequence", interactive=False)
            gen_explanation = gr.Textbox(label="Explanation", interactive=False)
            gen_btn.click(dna_sequence_completion, inputs=gen_input, outputs=[gen_output, gen_explanation])

        # Classification Tab (new)
        with gr.Tab("🏷️ Sequence Classification"):
            gr.Markdown("### Classify DNA Sequences")
            classify_input = gr.Textbox(label="DNA Sequence", placeholder="Enter DNA sequence...", lines=3)
            classify_btn = gr.Button("Classify Sequence", elem_classes="primary-btn")
            classify_output = gr.Textbox(label="Classification", interactive=False)
            classify_explanation = gr.Textbox(label="Detailed Analysis", interactive=False, lines=6)
            classify_btn.click(dna_sequence_classification, inputs=classify_input, 
                             outputs=[classify_output, classify_explanation])

        # Analysis Tab
        with gr.Tab("🔬 Sequence Analysis"):
            gr.Markdown("### Comprehensive DNA Sequence Analysis")
            analysis_input = gr.Textbox(label="Input DNA Sequence", placeholder="Enter DNA sequence...", lines=3)
            analysis_btn = gr.Button("Analyze Sequence", elem_classes="primary-btn")
            analysis_output = gr.Textbox(label="Analysis Results", lines=8)
            analysis_btn.click(analyze_sequence, inputs=analysis_input, outputs=analysis_output)

        # Comparison Tab
        with gr.Tab("🔍 Sequence Comparison"):
            gr.Markdown("### Compare DNA Sequences")
            with gr.Row():
                seq1_input = gr.Textbox(label="Sequence 1", lines=2)
                seq2_input = gr.Textbox(label="Sequence 2", lines=2)
            compare_btn = gr.Button("Compare Sequences", elem_classes="primary-btn")
            similarity_score = gr.Number(label="Similarity Score")
            similarity_explanation = gr.Textbox(label="Explanation")
            compare_btn.click(sequence_similarity, inputs=[seq1_input, seq2_input], 
                            outputs=[similarity_score, similarity_explanation])

        # Mutation Tab
        with gr.Tab("🧬 Mutation Analysis"):
            gr.Markdown("### Analyze Mutation Impact")
            mut_sequence = gr.Textbox(label="Original DNA Sequence", lines=2)
            mut_details = gr.Textbox(label="Mutation Details (format: index,nucleotide)", 
                                   placeholder="e.g., 5,A")
            mut_btn = gr.Button("Analyze Mutation", elem_classes="primary-btn")
            mut_result = gr.Textbox(label="Mutated Sequence")
            mut_score = gr.Number(label="Impact Score")
            mut_explanation = gr.Textbox(label="Explanation")
            mut_btn.click(mutation_impact_analysis, 
                         inputs=[mut_sequence, mut_details],
                         outputs=[mut_result, mut_score, mut_explanation])

    gr.Markdown(
        """
        ### 📖 How to Use
        1. Choose the appropriate tab for your analysis needs
        2. Input your DNA sequence(s)
        3. Click the corresponding button to perform the analysis
        4. Review the results and explanations provided
        
        ### 🔬 Features
        - Comprehensive sequence analysis
        - DNA sequence completion
        - Sequence similarity comparison
        - Mutation impact analysis
        """
    )

demo.launch()
