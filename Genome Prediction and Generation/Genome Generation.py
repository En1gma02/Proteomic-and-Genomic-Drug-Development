import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained GENA-LM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base', trust_remote_code=True)

# Functions for each feature
def dna_sequence_completion(sequence):
    """
    Completes a partial DNA sequence by predicting masked tokens.
    """
    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512)
    inputs['input_ids'][0, len(sequence)//2:] = tokenizer.mask_token_id  # Mask part of the sequence
    outputs = model(**inputs)
    logits = outputs.last_hidden_state[:, len(sequence)//2:, :]
    predicted_ids = torch.argmax(logits, dim=-1)
    completed_sequence = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    explanation = "This is the completed DNA sequence based on the input. The model predicts likely nucleotides to fill in missing parts."
    return completed_sequence, explanation

def dna_sequence_classification(sequence):
    """
    Classifies DNA sequence into categories (e.g., coding vs. non-coding).
    """
    # Placeholder for actual classification logic
    category = "Coding" if "ATG" in sequence else "Non-Coding"
    explanation = f"The sequence was classified as '{category}'. Coding sequences typically contain start codons like 'ATG'."
    return category, explanation

def sequence_similarity(seq1, seq2):
    """
    Computes similarity between two DNA sequences.
    """
    embedding1 = model(**tokenizer(seq1, return_tensors='pt')).last_hidden_state.mean(dim=1)
    embedding2 = model(**tokenizer(seq2, return_tensors='pt')).last_hidden_state.mean(dim=1)
    similarity_score = torch.cosine_similarity(embedding1, embedding2).item()
    explanation = f"The similarity score between the two sequences is {similarity_score:.2f}. A higher score indicates more similarity."
    return similarity_score, explanation

def mutation_impact_analysis(sequence, mutation):
    """
    Analyzes the impact of a mutation on a DNA sequence.
    """
    mutated_sequence = sequence[:mutation[0]] + mutation[1] + sequence[mutation[0]+1:]
    original_embedding = model(**tokenizer(sequence, return_tensors='pt')).last_hidden_state.mean(dim=1)
    mutated_embedding = model(**tokenizer(mutated_sequence, return_tensors='pt')).last_hidden_state.mean(dim=1)
    impact_score = torch.cosine_similarity(original_embedding, mutated_embedding).item()
    explanation = f"The mutation changes the sequence's structure or function with an impact score of {impact_score:.2f}. Lower scores indicate more significant changes."
    return mutated_sequence, impact_score, explanation

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🌟 GENA-LM DNA Analysis App")
    
    with gr.Tabs():
        # Tab 1: DNA Sequence Completion
        with gr.Tab("DNA Sequence Completion"):
            gr.Markdown("### Predict missing parts of a DNA sequence.")
            input_seq_comp = gr.Textbox(label="Input Partial DNA Sequence")
            output_seq_comp = gr.Textbox(label="Completed DNA Sequence", interactive=False)
            explanation_comp = gr.Textbox(label="Explanation", interactive=False)
            complete_btn = gr.Button("Complete Sequence")
            complete_btn.click(dna_sequence_completion, inputs=input_seq_comp, outputs=[output_seq_comp, explanation_comp])
        
        # Tab 2: DNA Sequence Classification
        with gr.Tab("DNA Sequence Classification"):
            gr.Markdown("### Classify DNA sequences into categories.")
            input_seq_classify = gr.Textbox(label="Input DNA Sequence")
            output_classify = gr.Textbox(label="Category", interactive=False)
            explanation_classify = gr.Textbox(label="Explanation", interactive=False)
            classify_btn = gr.Button("Classify Sequence")
            classify_btn.click(dna_sequence_classification, inputs=input_seq_classify, outputs=[output_classify, explanation_classify])
        
        # Tab 3: Sequence Similarity Analysis
        with gr.Tab("Sequence Similarity Analysis"):
            gr.Markdown("### Compare similarity between two DNA sequences.")
            input_seq1_sim = gr.Textbox(label="Sequence 1")
            input_seq2_sim = gr.Textbox(label="Sequence 2")
            output_sim_score = gr.Number(label="Similarity Score", interactive=False)
            explanation_sim = gr.Textbox(label="Explanation", interactive=False)
            similarity_btn = gr.Button("Compute Similarity")
            similarity_btn.click(sequence_similarity, inputs=[input_seq1_sim, input_seq2_sim], outputs=[output_sim_score, explanation_sim])
        
        # Tab 4: Mutation Impact Analysis
        with gr.Tab("Mutation Impact Analysis"):
            gr.Markdown("### Analyze the impact of mutations on a DNA sequence.")
            input_seq_mutate = gr.Textbox(label="Original DNA Sequence")
            mutation_details = gr.Textbox(label="Mutation Details (e.g., index,new_nucleotide)")
            output_mutated_seq = gr.Textbox(label="Mutated Sequence", interactive=False)
            output_impact_score = gr.Number(label="Impact Score", interactive=False)
            explanation_mutate = gr.Textbox(label="Explanation", interactive=False)
            mutate_btn = gr.Button("Analyze Mutation Impact")
            
            def parse_mutation_details(details):
                index, new_nucleotide = details.split(',')
                return int(index), new_nucleotide
            
            mutate_btn.click(lambda seq, mut: mutation_impact_analysis(seq, parse_mutation_details(mut)),
                             inputs=[input_seq_mutate, mutation_details],
                             outputs=[output_mutated_seq, output_impact_score, explanation_mutate])

# Launch the app
demo.launch()