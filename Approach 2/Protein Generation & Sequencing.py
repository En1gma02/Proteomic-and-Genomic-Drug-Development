import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import py3Dmol
from stmol import showmol

# Load ProteinForceGPT model
ForceGPT_model_name = 'lamm-mit/ProteinForceGPT'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(ForceGPT_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    ForceGPT_model_name,
    trust_remote_code=True
).to(device)
model.config.use_cache = False

# Function to generate protein sequences
def generate_protein(sequence):
    prompt = f"Sequence<{sequence}"
    generated = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0).to(device)
    sample_outputs = model.generate(
        inputs=generated,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=500,
        max_length=300,
        top_p=0.9,
        num_return_sequences=1,
        temperature=1,
    )
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

# Function to calculate force for a sequence
def calculate_force(sequence):
    prompt = f"CalculateForce<{sequence}>"
    generated = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0).to(device)
    sample_outputs = model.generate(
        inputs=generated,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=500,
        max_length=300,
        top_p=0.9,
        num_return_sequences=1,
        temperature=1,
    )
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)


# Function to analyze protein stability
def analyze_stability(sequence):
    length = len(sequence)
    if length > 200:
        return "Rich and Stable Protein"
    elif length > 100:
        return "Moderately Stable Protein"
    else:
        return "Unstable Protein"

import streamlit as st

# Optional page configuration to set the page title, icon, layout, etc.
st.set_page_config(
    page_title="Protein Engineering Suite",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Add a custom background image ---
def set_custom_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://s7d1.scene7.com/is/image/CENODS/10225-feature1-protein?$responsive$&wid=700&qlt=90,0&resMode=sharp2");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_bg()

# --- Add a title with emojis ---
st.title("Protein Generation Interface 🧬⚗️")

# --- Create a brief description or instructions in an expander ---
with st.expander("Instructions ℹ️", expanded=True):
    st.write(
        "Welcome to the **Protein Engineering Suite**! "
        "Use the tabs below to generate protein sequences, calculate force, "
        "or analyze stability. Provide an input sequence and click the corresponding button."
    )

# --- Extra elements: sidebar with a simple display and an emoji ---
st.sidebar.header("Navigation 🗺️")
st.sidebar.write("Select a tab above to explore the app features.")

# --- Create tabs for different functionalities ---
tabs = st.tabs(["Generate Protein 🧪", "Calculate Force 🔬", "Analyze Stability 🏗️"])

# --- Tab 1: Generate Protein ---
with tabs[0]:
    st.subheader("Generate a Protein Sequence")
    seq_input_gen = st.text_input("Input Sequence", placeholder="e.g. AAKGKR...")
    if st.button("Generate 🧬"):
        if seq_input_gen.strip():
            output_seq = generate_protein(seq_input_gen)
            st.text_area("Generated Protein Sequence", output_seq, height=150)
        else:
            st.warning("Please enter a valid sequence.")

# --- Tab 2: Calculate Force ---
with tabs[1]:
    st.subheader("Calculate Force")
    seq_input_force = st.text_input("Input Sequence", placeholder="e.g. MATKPG...")
    if st.button("Calculate Force ⚙️"):
        if seq_input_force.strip():
            force_result = calculate_force(seq_input_force)
            st.text_area("Calculated Force", force_result, height=150)
        else:
            st.warning("Please enter a valid sequence.")

# --- Tab 3: Analyze Stability ---
with tabs[2]:
    st.subheader("Analyze Stability")
    seq_input_stability = st.text_input("Input Sequence", placeholder="e.g. PQIYRP...")
    if st.button("Analyze Stability 🏗️"):
        if seq_input_stability.strip():
            stability_result = analyze_stability(seq_input_stability)
            st.text_area("Stability Analysis", stability_result, height=100)
        else:
            st.warning("Please enter a valid sequence.")