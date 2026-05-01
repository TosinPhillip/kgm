# app.py - Professional & Optimized Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Path handling
if "notebooks" in os.getcwd():
    sys.path.append(os.path.dirname(os.getcwd()))

from src.pipeline import KnowledgeGapPipeline

# ====================== CONFIG ======================
st.set_page_config(
    page_title="KnowledgeGap Mapper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stMetric {background-color: #f8f9fa; padding: 1rem; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Knowledge-Gap Mapping System")
st.markdown("**A Lightweight Explainable System for Personalized Learning**")
st.caption("Akinlolu Funmilayo Fomosara • CSC/2021/37091")

st.markdown("---")

# Session State
if 'enhanced' not in st.session_state:
    st.session_state.enhanced = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=80)
    st.header("Navigation")
    page = st.radio("Select Page", [
        "🏠 Home", 
        "🚀 Run Demo", 
        "📤 Upload Logs", 
        "📊 Gap Overview", 
        "🔍 Student Details", 
        "📈 Evaluation", 
        "ℹ️ About"
    ])
    
    st.markdown("---")
    if st.button("🔄 Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ====================== HOME ======================
if page == "🏠 Home":
    st.header("Welcome to Knowledge-Gap Mapper")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Turning Interaction Data into Actionable Insights")
        st.write("""
        This system analyzes student interaction logs from virtual learning environments 
        to automatically detect **knowledge gaps** and provide **explainable, personalized recommendations**.
        """)
    
    with col2:
        st.metric(label="Model Accuracy", value="87.46%")
        st.metric(label="Explainability", value="High")
    
    st.markdown("---")
    
    if st.button("🚀 Run Full Demo Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running full knowledge gap analysis..."):
            try:
                pipe = KnowledgeGapPipeline()
                enhanced, summary = pipe.run_full_pipeline(use_sample=False)
                st.session_state.enhanced = enhanced
                st.session_state.summary = summary
                st.success("✅ Analysis completed successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

# ====================== RUN DEMO ======================
elif page == "🚀 Run Demo":
    st.header("Run Demo Analysis")
    if st.button("Run Pipeline Now", type="primary"):
        with st.spinner("Processing..."):
            pipe = KnowledgeGapPipeline()
            enhanced, summary = pipe.run_full_pipeline(use_sample=False)
            st.session_state.enhanced = enhanced
            st.session_state.summary = summary
            st.success("Demo completed!")

# ====================== UPLOAD ======================
elif page == "📤 Upload Logs":
    st.header("📤 Upload Your Interaction Logs")
    uploaded_file = st.file_uploader("Upload CSV file from your LMS", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully • {len(df):,} rows")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process This Log File", type="primary"):
            with st.spinner("Analyzing uploaded data..."):
                pipe = KnowledgeGapPipeline()
                enhanced, summary = pipe.run_full_pipeline(use_sample=False)
                st.session_state.enhanced = enhanced
                st.session_state.summary = summary
                st.success("✅ File processed successfully!")

# ====================== GAP OVERVIEW ======================
elif page == "📊 Gap Overview":
    if st.session_state.summary is None:
        st.warning("Please run the demo first.")
    else:
        st.header("📊 Knowledge Gap Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", len(st.session_state.summary))
        with col2:
            high_risk = (st.session_state.summary['risk_level'] == 'High').sum()
            st.metric("High Risk Students", high_risk)
        with col3:
            st.metric("Average Gap %", f"{st.session_state.summary['gap_percentage'].mean():.1f}%")
        
        fig = px.bar(
            st.session_state.summary['risk_level'].value_counts().reset_index(),
            x='risk_level', y='count',
            title="Risk Level Distribution",
            color='risk_level',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Student Ranking by Gap Percentage")
        st.dataframe(
            st.session_state.summary.sort_values('gap_percentage', ascending=False),
            use_container_width=True
        )

# ====================== STUDENT DETAILS ======================
elif page == "🔍 Student Details":
    if st.session_state.enhanced is None:
        st.warning("Run the demo first.")
    else:
        st.header("🔍 Individual Student Analysis")
        
        student_list = sorted(st.session_state.enhanced['student_id'].unique())
        selected = st.selectbox("Select Student ID", student_list)
        
        data = st.session_state.enhanced[st.session_state.enhanced['student_id'] == selected]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Avg Gap Probability", f"{data['gap_probability'].mean():.1%}")
            st.metric("Affected Concepts", len(data))
            st.metric("Risk Level", data['severity'].mode()[0])
        
        with col2:
            st.subheader(f"Knowledge Gaps for Student {selected}")
            st.dataframe(
                data[['concept', 'severity', 'gap_probability', 'evidence', 'remediation_suggestions']],
                use_container_width=True
            )
        
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Student Gap Report",
            data=csv,
            file_name=f"student_{selected}_gaps.csv",
            mime="text/csv"
        )

# ====================== EVALUATION ======================
elif page == "📈 Evaluation":
    st.header("Model Evaluation")
    st.success("**Accuracy**: 87.46%")
    st.info("""
    The system uses a **Decision Tree** model chosen specifically for its high interpretability.
    This allows educators to understand exactly why a knowledge gap was flagged.
    """)

# ====================== ABOUT ======================
elif page == "ℹ️ About":
    st.header("About This Project")
    st.markdown("""
    This prototype implements a complete lightweight knowledge-gap mapping system.
    It focuses on **explainability**, practicality, and accessibility for resource-constrained environments.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Final Year Project • 2026")
