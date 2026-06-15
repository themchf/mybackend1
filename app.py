import os
import sys

# Maintain robust path targeting across container deployments
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

import streamlit as st
import pandas as pd

try:
    from src.orchestrator import ProcessOrchestrator
    from src.visualizer import create_volcano_plot
except ModuleNotFoundError:
    from orchestrator import ProcessOrchestrator
    from visualizer import create_volcano_plot

# Application initialization matching the MKF Informatics brand identity
st.set_page_config(page_title="MKF Informatics", page_icon="🧬", layout="wide")

openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

# Initialize data processing variables
if "computed_state" not in st.session_state:
    st.session_state.computed_state = None

st.title("🧬 MKF Informatics")
st.subheader("Enterprise-Grade Bioinformatics Intelligence Platform")
st.markdown("---")

col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("Pipeline Control")
    uploaded_files = st.file_uploader(
        "Upload raw computational assets:", 
        accept_multiple_files=True,
        type=["fasta", "fa", "fna", "csv", "tsv", "txt"]
    )
    
    if not openai_key:
        openai_key = st.text_input("Enter OpenAI API Key manually:", type="password")

    if uploaded_files:
        if st.button("Run Analytics Engine", type="primary"):
            with st.spinner("Processing biological data matrices..."):
                # Call the orchestrator which executes our cached subroutines
                engine = ProcessOrchestrator()
                st.session_state.computed_state = engine.execution_flow(uploaded_files, openai_key)
                st.rerun()

with col_main:
    if st.session_state.computed_state:
        res = st.session_state.computed_state
        
        st.success(res["summary"])
        
        if res.get("anomalies"):
            for anomaly in res["anomalies"]:
                st.warning(f"⚠️ {anomaly}")
                
        tab_viz, tab_report, tab_lit = st.tabs(["📊 Interactive Visualization", "🔬 Automated Core Insights", "📖 Live PubMed References"])
        
        with tab_viz:
            st.subheader("Interactive Graphic Profiling Workspace")
            if res["data_type"] == "expression":
                # Data payload references a structurally isolated cached execution footprint
                fig = create_volcano_plot(res["raw_meta"])
                st.plotly_chart(fig, use_container_width=True)
            elif res["data_type"] == "fasta":
                df_metrics = res["raw_meta"]["metrics"]
                st.dataframe(df_metrics, use_container_width=True)
                
                import plotly.express as px
                fig = px.histogram(df_metrics, x="gc_content", title="GC Percentage Frequency Mapping Profile", labels={"gc_content": "GC (%)"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active visualization template associated with uploaded file signature parameters.")
                
        with tab_report:
            st.header("Explain My Results")
            st.markdown(res["detailed_explanation"])
            
            st.subheader("Targeted Experimental Iterations")
            for step in res["next_steps"]:
                st.markdown(f"- {step}")
                
        with tab_lit:
            st.subheader("Dynamic Verified Literature Cross-Matching")
            for paper in res["citations"]:
                st.markdown(paper)
    else:
        st.info("Drop analytical metrics or FASTA files inside the workspace drop zone to initialize real-time calculation arrays.")
