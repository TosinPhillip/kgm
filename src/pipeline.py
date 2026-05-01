# src/pipeline.py - Multi-module Training Version

import pandas as pd
import streamlit as st
from src.ingestion import LogIngestion
from src.features import FeatureEngineer
from src.model import KnowledgeGapModel
from src.evaluation import GapEvaluator

class KnowledgeGapPipeline:
    def __init__(self):
        self.ingestor = LogIngestion()
        self.engineer = FeatureEngineer()
        self.model = KnowledgeGapModel(max_depth=6)   # Slightly deeper tree
        self.evaluator = GapEvaluator()

    @st.cache_data(show_spinner=False)
    def run_full_pipeline(_self):
        """Train on multiple modules for better generalization"""
        self = _self
        
        modules_to_use = ['AAA', 'BBB', 'CCC']   # You can add more later
        
        try:
            # Load data from multiple modules
            raw_df = self.ingestor.load_oulad_vle()
            # Filter to selected modules
            raw_df = raw_df[raw_df['code_module'].isin(modules_to_use)]
            st.success(f"✅ Using data from modules: {modules_to_use}")
        except Exception:
            st.warning("Using sample data for demonstration.")
            raw_df = self.ingestor.get_sample_data()

        enriched = self.ingestor.enrich_with_activity_type(raw_df)
        std_df = self.ingestor.standardize_logs(enriched)

        # Load student info for selected modules
        student_info = None
        try:
            student_info = pd.read_csv("data/studentInfo.csv")
            student_info = student_info[student_info['code_module'].isin(modules_to_use)]
        except:
            pass

        feature_df = self.engineer.build_full_feature_set(std_df, student_info)

        # Train model
        self.model.train(feature_df)

        # Generate gaps
        enhanced = self.model.predict_gaps(feature_df)
        enhanced, summary = self.evaluator.create_knowledge_gap_map(enhanced)

        return enhanced, summary