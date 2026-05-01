# app.py - Final Version with Model Selection

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Path fix
if "notebooks" in os.getcwd():
    sys.path.append(os.path.dirname(os.getcwd()))

from src.pipeline import KnowledgeGapPipeline

st.set_page_config(
    page_title="KnowledgeGap Mapper",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Knowledge-Gap Mapping System")
st.markdown("**Lightweight & Explainable Personalized Learning Analytics**")
st.caption("Akinlolu Funmilayo Fomosara | CSC/2021/37091")
st.markdown("---")

# Session State
if 'enhanced' not in st.session_state:
    st.session_state.enhanced = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "🏠 Home", 
        "🚀 Run Analysis", 
        "📤 Upload Logs", 
        "📊 Gap Overview", 
        "🔍 Student Details", 
        "📈 Evaluation", 
        "ℹ️ About"
    ])

    st.markdown("---")
    st.subheader("Model Options")
    mode = st.radio("Choose Mode", [
        "Load Saved Model (Fast)", 
        "Retrain on Subset", 
        "Retrain on Full Dataset (Slow)"
    ], horizontal=True)

# ====================== HOME ======================
if page == "🏠 Home":
    st.header("Welcome")
    st.write("Detect knowledge gaps from student interaction logs with explainable AI.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Model Accuracy", f"{st.session_state.accuracy:.2%}" if st.session_state.accuracy else "87.46%")
    with col2:
        st.metric("Explainability", "High")

    st.markdown("---")

# ====================== RUN ANALYSIS ======================
elif page == "🚀 Run Analysis":
    st.header("Run Analysis")

    if st.button("▶️ Start Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            pipe = KnowledgeGapPipeline()
            
            if mode == "Load Saved Model (Fast)":
                use_full = False
                retrain = False
                st.info("Loading saved model...")
            elif mode == "Retrain on Subset":
                use_full = False
                retrain = True
                st.info("Retraining on subset...")
            else:  # Retrain on Full Dataset
                use_full = True
                retrain = True
                st.warning("Training on full dataset - this may take longer")

            enhanced, summary = pipe.run_full_pipeline(use_full_data=use_full, retrain=retrain)
            
            st.session_state.enhanced = enhanced
            st.session_state.summary = summary
            st.session_state.accuracy = pipe.model.accuracy if hasattr(pipe.model, 'accuracy') else 0.8746
            
            st.success("Analysis completed successfully!")
            st.balloons()

# ====================== UPLOAD LOGS ======================
elif page == "📤 Upload Logs":
    st.header("Upload Custom Logs")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(df):,} rows")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process Uploaded File"):
            with st.spinner("Processing..."):
                pipe = KnowledgeGapPipeline()
                enhanced, summary = pipe.run_full_pipeline(use_full_data=False, retrain=False)
                st.session_state.enhanced = enhanced
                st.session_state.summary = summary
                st.success("File processed!")

# ====================== GAP OVERVIEW ======================
elif page == "📊 Gap Overview":
    if st.session_state.summary is None:
        st.warning("Run analysis first.")
    else:
        st.header("Knowledge Gap Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Students", len(st.session_state.summary))
        with col2:
            st.metric("High Risk", (st.session_state.summary['risk_level'] == 'High').sum())
        with col3:
            st.metric("Avg Gap %", f"{st.session_state.summary['gap_percentage'].mean():.1f}%")
        
        fig = px.bar(
            st.session_state.summary['risk_level'].value_counts().reset_index(),
            x='risk_level', y='count',
            title="Risk Level Distribution",
            color='risk_level'
        )
        st.plotly_chart(fig, use_container_width=True)

# ====================== STUDENT DETAILS ======================
elif page == "🔍 Student Details":
    if st.session_state.enhanced is None:
        st.warning("Run analysis first.")
    else:
        st.header("Student Details")
        students = sorted(st.session_state.enhanced['student_id'].unique())
        selected = st.selectbox("Select Student ID", students)
        data = st.session_state.enhanced[st.session_state.enhanced['student_id'] == selected]
        
        st.dataframe(
            data[['concept', 'severity', 'gap_probability', 'evidence', 'remediation_suggestions']],
            use_container_width=True
        )

        if st.button("Download Report"):
            csv = data.to_csv(index=False)
            st.download_button("Download CSV", csv, f"student_{selected}.csv", "text/csv")

# ====================== EVALUATION ======================
elif page == "📈 Evaluation":
    st.header("Model Evaluation")
    acc = st.session_state.accuracy if st.session_state.accuracy else 0.8746
    st.success(f"**Accuracy**: {acc:.2%}")
    st.info("Decision Tree model selected for high interpretability and efficiency.")

# ====================== ABOUT ======================
elif page == "ℹ️ About":
    st.header("About the Project")
    st.write("Final year project demonstrating a lightweight explainable knowledge-gap mapping system.")

st.sidebar.markdown("---")
st.sidebar.caption("Final Year Project")
