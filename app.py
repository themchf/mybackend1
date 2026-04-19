import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.ensemble import RandomForestRegressor

# --- UI and Page Setup ---
st.set_page_config(page_title="Drug Discovery ML", page_icon="🧬", layout="wide")

st.title("🧬 ML - Drug Discovery by MKF")
st.markdown("""
Welcome to the Drug Discovery prediction tool. 
This application processes chemical compounds using **SMILES** notation, calculates their molecular descriptors based on Lipinski's Rule of Five, and uses a Machine Learning model to predict potential bioactivity.
""")

# --- Chemistry Feature Extraction ---
def calculate_descriptors(smiles):
    """Calculates molecular properties from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate fundamental drug-likeness descriptors
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    
    return [mol_wt, logp, h_donors, h_acceptors]

# --- Machine Learning Model ---
@st.cache_resource
def load_ml_model():
    """
    In a world-class production app, you would load a pre-trained .pkl file here 
    that was trained on a massive database like ChEMBL. 
    For this deployment, we generate a template model trained on simulated data 
    so your app runs perfectly out-of-the-box on Streamlit.
    """
    # Simulated training data based on molecular weights, LogP, etc.
    X_train = np.random.rand(500, 4) * [500, 5, 10, 10] 
    y_train = np.random.rand(500) * 10 # Simulated pIC50 bioactivity values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_ml_model()

# --- User Interface: Sidebar ---
st.sidebar.header("🧪 Input Parameters")
st.sidebar.write("Enter a chemical compound to evaluate:")

# Default is Aspirin
smiles_input = st.sidebar.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O") 

# --- Processing & Output ---
if st.sidebar.button("Run Prediction"):
    with st.spinner("Calculating molecular descriptors and running ML model..."):
        features = calculate_descriptors(smiles_input)
        
        if features is None:
            st.error("Error: Invalid SMILES string. Please check your chemical notation.")
        else:
            col1, col2 = st.columns(2)
            
            # Display Extracted Features
            with col1:
                st.subheader("📊 Molecular Descriptors")
                st.write("These features define the drug-likeness of the molecule:")
                feature_df = pd.DataFrame([features], columns=["Molecular Weight", "LogP", "H-Bond Donors", "H-Bond Acceptors"])
                st.dataframe(feature_df, hide_index=True)
            
            # Display ML Prediction
            with col2:
                st.subheader("🎯 ML Bioactivity Prediction")
                prediction = model.predict([features])
                
                # Create a visual metric
                st.metric(label="Predicted pIC50 (Simulated)", value=f"{prediction[0]:.2f}")
                st.write("*Note: Higher values typically indicate higher drug efficacy against a target.*")
                
            st.success("Analysis Complete!")
            st.info("Architecture Note: To upgrade this to a production-grade tool, train a model locally on a dataset like ChEMBL, save it as a `model.pkl` file, upload it to your GitHub, and replace the `load_ml_model` function to load that file via the `joblib` library.")
