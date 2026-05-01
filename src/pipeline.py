# src/pipeline.py - Optimized for deployment

import pandas as pd
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
        self.enhanced = None
        self.summary = None

    def run_full_pipeline(self, use_sample=True):
        """Optimized pipeline"""
        print("Starting optimized pipeline...")

        # Load data
        raw_df = self.ingestor.get_sample_data() if use_sample else self.ingestor.load_oulad_vle()
        enriched = self.ingestor.enrich_with_activity_type(raw_df)
        std_df = self.ingestor.standardize_logs(enriched)

        # Feature Engineering
        student_info = None
        try:
            student_info = pd.read_csv("data/studentInfo.csv")
            student_info = student_info[
                (student_info['code_module'] == 'AAA') & 
                (student_info['code_presentation'] == '2013J')
            ]
        except:
            pass

        feature_df = self.engineer.build_full_feature_set(std_df, student_info)

        # Train model
        self.model.train(feature_df)

        # Predict and evaluate
        self.enhanced = self.model.predict_gaps(feature_df)
        self.enhanced, self.summary = self.evaluator.create_knowledge_gap_map(self.enhanced)

        print("Pipeline completed successfully.")
        return self.enhanced, self.summary