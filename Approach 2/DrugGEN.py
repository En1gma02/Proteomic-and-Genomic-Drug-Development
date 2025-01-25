import os
import json
import torch
import logging
import tempfile
import gradio as gr
from gradio.themes import Soft
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen
import py3Dmol

# Global logging setup
def setup_logging(output_file="app.log"):
    log_filename = os.path.splitext(output_file)[0] + ".log"
    logging.getLogger().handlers.clear()
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            logging.info("Moving model to CUDA device.")
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

# Load the dataset
def load_uniprot_dataset(dataset_name, dataset_key):
    try:
        dataset = load_dataset(dataset_name, dataset_key)
        uniprot_to_sequence = {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}
        logging.info("Dataset loaded and processed successfully.")
        return uniprot_to_sequence
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise RuntimeError(f"Failed to load dataset: {e}")

def save_smiles_to_file(results):
    file_path = os.path.join(tempfile.gettempdir(), "generated_smiles.json")
    
    # Create a new dictionary to store the serializable data
    serializable_results = {}
    
    for key, value in results.items():
        serializable_results[key] = {
            "sequence": value["sequence"],
            "smiles_results": []
        }
        
        if "smiles_results" in value:
            for smiles_result in value["smiles_results"]:
                serializable_smiles = {
                    "SMILES": smiles_result["SMILES"],
                    "Drug_Likeness_Score": smiles_result["Drug_Likeness_Score"]
                }
                serializable_results[key]["smiles_results"].append(serializable_smiles)
        
        if "error" in value:
            serializable_results[key]["error"] = value["error"]
    
    with open(file_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    
    return file_path

def visualize_molecule(mol):
    # 2D visualization
    img_2d = Draw.MolToImage(mol)
    
    # 3D visualization with escaped HTML
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(Chem.MolToMolBlock(mol_3d), "mol")
    viewer.setStyle({'stick':{}})
    viewer.zoomTo()
    
    html_content = f'''
    <div style="height: 400px;">
        {viewer.render()}
    </div>
    '''
    return img_2d, html_content

def calculate_drug_likeness(mol):
    mw = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    psa = Descriptors.TPSA(mol)
    
    # Lipinski's Rule of Five
    lipinski_score = 0
    if mw <= 500: lipinski_score += 1
    if logp <= 5: lipinski_score += 1
    if hbd <= 5: lipinski_score += 1
    if hba <= 10: lipinski_score += 1
    
    # Additional criteria
    if psa <= 140: lipinski_score += 1
    
    return lipinski_score / 5  # Normalized score

class SMILESGenerator:
    def __init__(self, model, tokenizer, uniprot_to_sequence):
        self.model = model
        self.tokenizer = tokenizer
        self.uniprot_to_sequence = uniprot_to_sequence
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.generation_kwargs = {
            "do_sample": True,
            "top_k": 9,
            "max_length": 1024,
            "top_p": 0.9,
            "num_return_sequences": 10,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

    def generate_smiles(self, sequence, num_generated, progress_callback=None):
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        logging.info(f"Generating SMILES for sequence: {sequence[:10]}...")
        retries = 0
        results = []
        
        while len(generated_smiles_set) < num_generated:
            if retries >= 30:
                logging.warning("Max retries reached. Returning what has been generated so far.")
                break

            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for i, sample_output in enumerate(sample_outputs):
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        generated_smiles_set.add(generated_smiles)
                        
                        # Visualize the molecule
                        mol = Chem.MolFromSmiles(generated_smiles)
                        if mol is not None:
                            img_2d, viewer_3d = visualize_molecule(mol)
                            
                            # Calculate drug-likeness score
                            drug_likeness_score = calculate_drug_likeness(mol)
                            
                            results.append({
                                "SMILES": generated_smiles,
                                "2D_Structure": img_2d,
                                "3D_Structure": viewer_3d,
                                "Drug_Likeness_Score": f"{drug_likeness_score:.2f}"
                            })
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Failed to parse small molecule due to error: {str(e)}. Skipping.")
            
            if progress_callback:
                progress_callback((retries + 1) / 30)

            retries += 1

        logging.info(f"Small molecules generation completed. Generated {len(generated_smiles_set)} Small molecules.")
        return results

def generate_smiles_gradio(sequence_input=None, uniprot_id=None, num_generated=10):
    results = {}
    uniprot_counter = 0
    first_result = None

    def process_sequence(seq, identifier):
        nonlocal first_result
        try:
            smiles_results = generator.generate_smiles(seq, num_generated)
            results[identifier] = {
                "sequence": seq,
                "smiles_results": smiles_results
            }
            if first_result is None and smiles_results:
                first_result = smiles_results[0]
        except Exception as e:
            results[identifier] = {
                "sequence": seq,
                "error": f"Error generating small molecules: {str(e)}"
            }

    if sequence_input:
        sequences = [seq.strip() for seq in sequence_input.split(",") if seq.strip()]
        for seq in sequences:
            uniprot_id_for_seq = [uid for uid, s in uniprot_to_sequence.items() if s == seq]
            identifier = uniprot_id_for_seq[0] if uniprot_id_for_seq else str(uniprot_counter)
            process_sequence(seq, identifier)
            if not uniprot_id_for_seq:
                uniprot_counter += 1

    if uniprot_id:
        uniprot_ids = [uid.strip() for uid in uniprot_id.split(",") if uid.strip()]
        for uid in uniprot_ids:
            sequence = uniprot_to_sequence.get(uid, "N/A")
            if sequence != "N/A":
                process_sequence(sequence, uid)
            else:
                results[uid] = {
                    "sequence": "N/A",
                    "error": f"UniProt ID {uid} not found in the dataset."
                }

    if not results:
        return {"error": "No small molecules generated. Please try again with different inputs."}, None, None, None, None

    file_path = save_smiles_to_file(results)

    if first_result:
        return (
            results,
            file_path,
            first_result["2D_Structure"],
            first_result["3D_Structure"],
            float(first_result["Drug_Likeness_Score"])
        )
    else:
        return results, file_path, None, None, None

if __name__ == "__main__":
    setup_logging()
    model_name = "alimotahharynia/DrugGen"
    dataset_name = "alimotahharynia/approved_drug_target"
    dataset_key = "uniprot_sequence"

    model, tokenizer = load_model_and_tokenizer(model_name)
    uniprot_to_sequence = load_uniprot_dataset(dataset_name, dataset_key)

    generator = SMILESGenerator(model, tokenizer, uniprot_to_sequence)

    theme = Soft(primary_hue="indigo", secondary_hue="teal")

    css = """
        #app-title {
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        #description {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        #file-output {
            height: 90px;
        }
        #generate-button {
            height: 40px;
            color: #333333 !important;
        }      
    """

    with gr.Blocks(theme=theme, css=css) as iface:
        gr.Markdown("## DrugGen", elem_id="app-title")
        gr.Markdown(
            "Generate **drug-like small molecules structures** from protein sequences or UniProt IDs. "
            "Input data, specify parameters, and download the results.",
            elem_id="description"
        )

        with gr.Row():
            sequence_input = gr.Textbox(
                label="Protein Sequences",
                placeholder="Enter sequences separated by commas (e.g., MGAASGRRGP..., MGETLGDSPI..., ...)",
                lines=3,
            )
            uniprot_id_input = gr.Textbox(
                label="UniProt IDs",
                placeholder="Enter UniProt IDs separated by commas (e.g., P12821, P37231, ...)",
                lines=3,
            )

        num_generated_slider = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=10,
            label="Number of Unique Small Molecules to Generate",
        )

        output = gr.JSON(label="Generated Small Molecules")
        file_output = gr.File(
            label="Download Results as JSON",
            elem_id=["file-output"]
        )

        with gr.Row():
            image_2d = gr.Image(label="2D Structure")
            # Remove the sanitize parameter
            image_3d = gr.HTML(label="3D Structure")
            
        drug_likeness_score = gr.Number(label="Drug-Likeness Score")
        
        generate_button = gr.Button("Generate Small Molecule", elem_id="generate-button")
        generate_button.click(
            generate_smiles_gradio,
            inputs=[sequence_input, uniprot_id_input, num_generated_slider],
            outputs=[output, file_output, image_2d, image_3d, drug_likeness_score]
        )
        
        iface.launch(allowed_paths=["/tmp"])