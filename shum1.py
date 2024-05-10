import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from Pfeature.pfeature import  atc_wp, btc_wp
import os
import numpy as np
import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
from Bio import SeqIO
from io import StringIO


def main():
    # Set the color scheme
    primary_color = '#A7C957'
    secondary_color = '#3C3C3C'
    tertiary_color = '#FFFFFF'
    background_color = '#F4F4F4'
    text_color = '#333333'
    font = 'sans serif'

    # Set the page config
    st.set_page_config(
        page_title='ACPF',
        layout= 'wide',
        initial_sidebar_state='expanded',
        page_icon='🧬',
    )

    # Set the theme
    st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .sidebar .sidebar-content {{
            background-color: {secondary_color};
            color: {tertiary_color};
        }}
        .streamlit-button {{
            background-color: {primary_color};
            color: {tertiary_color};
        }}
        footer {{
            font-family: {font};
        }}
    </style>
    """, unsafe_allow_html=True)

    # Add university logos to the page
    left_logo, center, right_logo = st.columns([1, 2, 1])
    left_logo.image("PU.png", width=280)
    right_logo.image("LOGO_u.png", width=280)

    # Add header with application title and description
    with center:
      st.markdown("<h1 style='font-family:Bodoni MT Black;font-size:40px;'>ACP-Finder(ACPF)</h1>", unsafe_allow_html=True)
      st.write("")
      st.markdown("<p style='font-family:Bodoni MT;font-size:20px;font-style: italic;'>ACP-Finder (ACPF) is a powerful web application for predicting the likelihood of a peptide being anticancer or non-anticancer. Our machine learning-based model, built on a large balanced dataset and using a random forest classifier, achieves an accuracy of 89%. ACPF also includes structure visualization of anticancer peptides and highlights the best features in it. Try ACPF today and discover the potential of your peptide sequences!</p>", unsafe_allow_html=True)  

if __name__ == "__main__":
    main()

# Load the trained model
model_file = "model.pkl"  # Ensure this path is correct
model = joblib.load(model_file)

if 'current_seq_idx' not in st.session_state:
    st.session_state.current_seq_idx = 0

#         # Define the feature extraction section

            
def atc(input):
            a = input.rstrip("txt")
            output = a + 'atc.csv'
            input_file = os.path.join(os.getcwd(), 'Pfeature', 'input_sam.csv')
            with open(input_file, 'w') as f:
                f.write(input)
            df_out = atc_wp(input_file, output)
            df_in = pd.read_csv(output)
            os.remove(input_file)
            os.remove(output)
            return df_in
def btc(input):
            a = input.rstrip("txt")
            output = a + 'btc.csv'
            input_file = os.path.join(os.getcwd(), 'Pfeature', 'input_sam.csv')
            with open(input_file, 'w') as f:
                f.write(input)
            df_out = btc_wp(input_file, output)
            df_in = pd.read_csv(output)
            os.remove(input_file)
            os.remove(output)
            return df_in
            
def is_valid_sequence(sequence):
    valid_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    if not sequence or not all(char.upper() in valid_amino_acids for char in sequence):
        raise ValueError("You have entered an invalid sequence. Please check your input.")
    return True

def update(sequence_list):
    pdb_strings = []
    for sequence in sequence_list:
        # Convert the sequence to uppercase for API compatibility
        uppercase_sequence = sequence.upper()

        if not is_valid_sequence(uppercase_sequence):
            st.error(f"Invalid sequence: {sequence}")
            continue

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=uppercase_sequence, verify=False)
        if response.status_code == 200:
            pdb_string = response.content.decode('utf-8')
            pdb_strings.append(pdb_string)
        else:
            st.error(f"Error with sequence {sequence}: Status code {response.status_code}")
    return pdb_strings

  
# 3D Structure Prediction Functions
def render_mol(pdb):
    if not pdb.strip():
        st.error("Empty PDB data, cannot render.")
        return

    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, 'pdb')
    pdbview.setStyle({'cartoon': {'color': 'spectrum'}})
    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)
    
def show_next():
    if 'pdb_strings' in st.session_state:
        st.session_state.current_seq_idx = (st.session_state.current_seq_idx + 1) % len(st.session_state.pdb_strings)
        render_current_structure()

def show_previous():
    if 'pdb_strings' in st.session_state:
        st.session_state.current_seq_idx = (st.session_state.current_seq_idx - 1) % len(st.session_state.pdb_strings)
        render_current_structure()
        

def render_current_structure():
    if 'pdb_strings' in st.session_state and st.session_state.pdb_strings:
        current_pdb = st.session_state.pdb_strings[st.session_state.current_seq_idx]
        with structure_container:
            # Displaying the index of the current structure
            st.markdown(f"**Displaying Structure {st.session_state.current_seq_idx + 1} of {len(st.session_state.pdb_strings)}**")

            render_mol(current_pdb)

            # Adding a download button for the current structure
            st.download_button(
                label="Download this Structure",
                data=current_pdb,
                file_name=f"structure_{st.session_state.current_seq_idx + 1}.pdb",
                mime='chemical/x-pdb'
            )

# Function to parse FASTA format
def parse_fasta(file_content):
    sequences = []
    current_sequence = ""
    for line in file_content:
        if line.startswith('>'):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = ""
        else:
            current_sequence += line.strip()
    if current_sequence:
        sequences.append(current_sequence)
    return sequences

# Predict function using the model
def predict_peptide_structure(sequences):
    atc_df_list = [atc(seq) for seq in sequences if seq]
    btc_df_list = [btc(seq) for seq in sequences if seq]
    

    df_features = pd.concat([pd.concat(atc_df_list, axis=0), 
                             pd.concat(btc_df_list, axis=0)], axis=1)

    feature_cols =  ['ATC_H', 'ATC_C', 'ATC_O', 'ATC_N', 'ATC_S', 'BTC_T', 'BTC_H', 'BTC_S', 'BTC_D']
    df_features = df_features.reindex(columns=feature_cols)
    y_pred = model.predict(df_features)
    prediction_probability = model.predict_proba(df_features)[:,1]
    

    return y_pred, prediction_probability
  
# Streamlit app setup
#st.title("Protein Sequence Submission")

if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'submit_count' not in st.session_state:
        st.session_state.submit_count = 0
# Page 1: Input
st.title("Protein Sequence Submission")

if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'submit_count' not in st.session_state:
        st.session_state.submit_count = 0
# Page 1: Input
if st.session_state.page == 'input':
    st.subheader("Enter Protein Sequences")
    protein_sequences = st.text_area("Protein Sequences (Enter multiple sequences separated by new lines)", height=150)
    fasta_file = st.file_uploader("Or upload FASTA file", type=["fasta", "txt"])

    submit_button = st.button("Submit")

    if submit_button:
        st.session_state.submit_count += 1

    if fasta_file:
        fasta_content = fasta_file.getvalue().decode("utf-8").splitlines()
        protein_sequences = parse_fasta(fasta_content)
        st.info("File uploaded. Please click on 'Submit' to process.")

    if submit_button:
        if protein_sequences:
            sequences_list = protein_sequences.split('\n') if isinstance(protein_sequences, str) else protein_sequences
            valid_sequences = []
            for seq in sequences_list:
                try:
                    if is_valid_sequence(seq):
                        valid_sequences.append(seq)
                except ValueError as e:
                    st.error(str(e))
                    break

            if valid_sequences:
                st.session_state.protein_sequences = valid_sequences
                y_pred, prediction_probability = predict_peptide_structure(st.session_state.protein_sequences)
                st.session_state.prediction = y_pred
                st.session_state.prediction_probability = prediction_probability
                st.session_state.page = 'output'
        else:
            st.warning("Please enter protein sequences or upload a file.")

# Page 2: Output (including prediction results)
elif st.session_state.page == 'output':
    st.subheader("Prediction Results")

    # Creating the DataFrame
    results_df = pd.DataFrame({
        'Index': range(1, len(st.session_state.protein_sequences) + 1),
        'Peptide Sequence': st.session_state.protein_sequences,
        'Predicted Probability': st.session_state.prediction_probability,
        'Class Label': st.session_state.prediction
    })

    # Display the DataFrame as a table
    st.table(results_df)

    # Convert DataFrame to CSV string for download
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv',
    )

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'input'))
    structure_container = st.container()

    # Check if any ACPs are identified and trigger 3D structure prediction
    predict_3d_button = st.button("Predict 3D Structure")
    acp_sequences = []
    if predict_3d_button:
        predictions_list = st.session_state.prediction
        acp_sequences = [seq for seq, pred in zip(st.session_state.protein_sequences, predictions_list) if pred == 'ACPs']
    
    if acp_sequences:
        st.session_state.pdb_strings = update(acp_sequences)
        st.session_state.current_seq_idx = 0  # Initialize the sequence index
        render_current_structure()

    # Display navigation buttons regardless of the condition
    if 'pdb_strings' in st.session_state and len(st.session_state.pdb_strings) > 1:
        col1, col2 = st.columns([1, 1])
        if st.session_state.current_seq_idx > 0:
            col1.button("Previous", on_click=show_previous)
        if st.session_state.current_seq_idx < len(st.session_state.pdb_strings) - 1:
            col2.button("Next", on_click=show_next)
            

# Add a section with the developers' information at the bottom of the page
st.markdown("---")
st.header("Developers:")     

# Add the profiles as individual cards
row1, row2 = st.columns([1, 1])
row3 = st.columns(1)


with row1:
 #st.image("my-photo.jpg", width=100)
 st.write("")
 st.write("### Dr. Naeem Mahmood Ashraf")
 st.write("Assistant Professor")
 st.write("School of Biochemistry and Biotechnology")
 st.write("University of the Punjab")
 st.write("Email: Naeem.sbb@pu.edu.pk")

with row2:
 #st.image("colleague-photo.jpg", width=100)
 st.write("")
 st.write("### Arslan Hamid")
 st.write("Email: ahamid@uni-bonn.de")

#row3[0].image("teacher-photo.jpg", width=100)
row3[0].write("")
row3[0].write("### Kashaf Azam")
row3[0].write("Email: kashafazam09@gmail.com")