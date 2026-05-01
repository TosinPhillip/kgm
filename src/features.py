import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    """Feature Engineering for Knowledge-Gap Mapping System"""
    
    def __init__(self):
        pass
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing and cleaning"""
        print("Starting preprocessing...")
        
        df = df.copy()
        
        # Convert timestamp (days since start) to proper format if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=['student_id'])
        df = df[df['num_interactions'] >= 0]
        
        print(f"Preprocessed data: {len(df):,} rows")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral, engagement, and performance-related features"""
        print("Extracting features per student and per concept...")
        
        # Group by student and concept (activity_type)
        grouped = df.groupby(['student_id', 'concept'])
        
        features = pd.DataFrame()
        
        # 1. Behavioral Features
        features['total_interactions'] = grouped['num_interactions'].sum()
        features['unique_days_active'] = grouped['timestamp'].nunique()
        features['avg_interactions_per_day'] = features['total_interactions'] / features['unique_days_active']
        
        # 2. Temporal / Persistence Features
        features['first_interaction_day'] = grouped['timestamp'].min()
        features['last_interaction_day'] = grouped['timestamp'].max()
        features['span_days'] = features['last_interaction_day'] - features['first_interaction_day']
        
        # 3. Engagement Intensity
        features['max_daily_clicks'] = grouped['num_interactions'].max()
        features['interaction_consistency'] = features['unique_days_active'] / (features['span_days'] + 1)  # avoid division by zero
        
        # Reset index to make student_id and concept columns
        features = features.reset_index()
        
        # 4. Global Student-level features (merge back)
        student_total = df.groupby('student_id').agg(
            total_student_interactions=('num_interactions', 'sum'),
            total_concepts_accessed=('concept', 'nunique')
        ).reset_index()
        
        features = features.merge(student_total, on='student_id', how='left')
        
        # 5. Normalized features (relative to student)
        features['interaction_ratio'] = features['total_interactions'] / features['total_student_interactions']
        
        print(f"✅ Extracted {features.shape[1]} features for {features['student_id'].nunique():,} students")
        return features
    
    def create_target(self, student_info: pd.DataFrame) -> pd.DataFrame:
        """Create target variable (proxy for knowledge gap) using final_result"""
        target = student_info[['id_student', 'final_result']].copy()
        target = target.rename(columns={'id_student': 'student_id'})
        
        # Binary target: 1 = knowledge gap / at-risk (Fail or Withdrawn), 0 = Pass/Distinction
        target['knowledge_gap'] = target['final_result'].apply(
            lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0
        )
        
        print("Target distribution:")
        print(target['knowledge_gap'].value_counts())
        return target
    
    def build_full_feature_set(self, interaction_df: pd.DataFrame, student_info: pd.DataFrame = None) -> pd.DataFrame:
        """End-to-end feature engineering"""
        df_clean = self.preprocess(interaction_df)
        features = self.extract_features(df_clean)
        
        if student_info is not None:
            target = self.create_target(student_info)
            # Merge target to features (at student level)
            features = features.merge(target, on='student_id', how='left')
        
        print("\n🎉 Feature Engineering Complete!")
        print(f"Final feature shape: {features.shape}")
        print("Sample columns:", features.columns.tolist()[:10])
        
        return features


# Quick test
if __name__ == "__main__":
    from src.ingestion import LogIngestion
    
    ingestor = LogIngestion()
    raw_df = ingestor.load_oulad_vle()
    enriched = ingestor.enrich_with_activity_type(raw_df)
    std_df = ingestor.standardize_logs(enriched)
    
    engineer = FeatureEngineer()
    feature_df = engineer.build_full_feature_set(std_df)
    print(feature_df.head())