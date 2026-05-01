# src/pipeline.py - Cloud-Friendly Version with Fallback

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
        self.model = KnowledgeGapModel(max_depth=5)
        self.evaluator = GapEvaluator()

    @st.cache_data(show_spinner=False)
    def run_full_pipeline(self):
        """Cloud-safe pipeline with fallback to sample data"""
        
        try:
            # Try to load full data first
            raw_df = self.ingestor.load_oulad_vle()
            st.success("✅ Using full AAA-2013J dataset")
        except Exception:
            st.warning("Full dataset not found. Using sample data for demonstration.")
            raw_df = self.ingestor.get_sample_data()

        # Continue with the rest of the pipeline
        enriched = self.ingestor.enrich_with_activity_type(raw_df)
        std_df = self.ingestor.standardize_logs(enriched)

        # Load student info if available
        student_info = None
        try:
            student_info = pd.read_csv("data/studentInfo.csv")
            student_info = student_info[
                (student_info['code_module'] == 'AAA') & 
                (student_info['code_presentation'] == '2013J')
            ]
        except:
            pass

        # Feature Engineering
        feature_df = self.engineer.build_full_feature_set(std_df, student_info)

        # Train Model
        self.model.train(feature_df)

        # Predict Gaps
        enhanced = self.model.predict_gaps(feature_df)
        enhanced, summary = self.evaluator.create_knowledge_gap_map(enhanced)

        return enhanced, summary