# src/pipeline.py - Adjusted for Model Selection

import pandas as pd
import streamlit as st
from src.ingestion import LogIngestion
from src.features import FeatureEngineer
from src.model import KnowledgeGapModel
from src.evaluation import GapEvaluator
import os

class KnowledgeGapPipeline:
    def __init__(self):
        self.ingestor = LogIngestion()
        self.engineer = FeatureEngineer()
        self.model = KnowledgeGapModel(max_depth=5)
        self.evaluator = GapEvaluator()
        self.model_path = "models/knowledge_gap_model.joblib"

    def run_full_pipeline(self, use_full_data=False, retrain=False):
        """Main entry point with model selection logic"""
        
        if retrain or not os.path.exists(self.model_path):
            st.warning("Training new model...")
            self._train_new_model(use_full_data)
        else:
            if self.model.load_model(self.model_path):
                st.info("✅ Loaded saved model")
            else:
                st.warning("Failed to load model. Training new one...")
                self._train_new_model(use_full_data)

        # Generate predictions
        try:
            raw_df = self.ingestor.load_oulad_vle() if use_full_data else self.ingestor.get_sample_data()
        except:
            raw_df = self.ingestor.get_sample_data()

        enriched = self.ingestor.enrich_with_activity_type(raw_df)
        std_df = self.ingestor.standardize_logs(enriched)

        feature_df = self.engineer.build_full_feature_set(std_df, None)

        enhanced = self.model.predict_gaps(feature_df)
        enhanced, summary = self.evaluator.create_knowledge_gap_map(enhanced)

        return enhanced, summary

    def _train_new_model(self, use_full_data):
        """Train and save new model"""
        raw_df = self.ingestor.load_oulad_vle() if use_full_data else self.ingestor.get_sample_data()
        enriched = self.ingestor.enrich_with_activity_type(raw_df)
        std_df = self.ingestor.standardize_logs(enriched)

        student_info = None
        try:
            student_info = pd.read_csv("data/studentInfo.csv")
            if not use_full_data:
                student_info = student_info[(student_info['code_module'] == 'AAA') & 
                                          (student_info['code_presentation'] == '2013J')]
        except:
            pass

        feature_df = self.engineer.build_full_feature_set(std_df, student_info)

        self.model.train(feature_df)
        self.model.save_model(self.model_path)