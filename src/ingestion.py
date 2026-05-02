import pandas as pd
import os
from typing import Dict, Optional

class LogIngestion:
    """Handles student interaction logs from OULAD and other platforms"""
    
    def __init__(self):
        # Expected standard schema for our knowledge-gap system
        self.standard_schema = ['student_id', 'timestamp', 'activity_type', 
                               'concept', 'num_interactions', 'duration_seconds', 
                               'performance_score']
        
        # Mapping from OULAD columns to our standard names
        self.oulad_mapping = {
            'id_student': 'student_id',
            'date': 'timestamp',
            'activity_type': 'activity_type',   # from vle.csv via id_site
            'id_site': 'concept',               # we'll enrich with activity_type
            'sum_click': 'num_interactions'
        }
    
    def load_oulad_vle(self) -> pd.DataFrame:
        """Load the full OULAD studentVle.csv without module filtering"""
        path = "data/studentVle.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        print("Loading full OULAD studentVle.csv (this may take a while)...")
        
        df = pd.read_csv(
            path,
            usecols=['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'],
            dtype={'id_student': 'int32', 'id_site': 'int32', 'sum_click': 'int16'},
            low_memory=False
        )
        
        print(f"✅ Loaded full dataset: {len(df):,} interaction records across all modules")
        return df
    
    def enrich_with_activity_type(self, df: pd.DataFrame, vle_df: pd.DataFrame = None) -> pd.DataFrame:
        """Join with vle.csv to get meaningful activity_type and concept"""
        if vle_df is None:
            vle_path = "data/vle.csv"
            if os.path.exists(vle_path):
                vle_df = pd.read_csv(vle_path, usecols=['id_site', 'activity_type', 'code_module', 'code_presentation'])
        
        if vle_df is not None:
            # Merge to bring activity_type
            df = df.merge(
                vle_df[['id_site', 'activity_type']], 
                on='id_site', 
                how='left'
            )
            print("✅ Enriched with activity_type from vle.csv")
        
        return df
    
    def standardize_logs(self, df: pd.DataFrame, manual_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """Convert raw logs to standard format"""
        std_df = pd.DataFrame()
        
        # Apply mapping for OULAD
        std_df['student_id'] = df['id_student']
        std_df['timestamp'] = df['date']           # days since start
        std_df['num_interactions'] = df['sum_click']
        
        if 'activity_type' in df.columns:
            std_df['activity_type'] = df['activity_type']
        else:
            std_df['activity_type'] = 'unknown'
            
        # Use activity_type as concept proxy for now (good enough for prototype)
        std_df['concept'] = std_df['activity_type']
        
        # Basic cleaning
        std_df = std_df.dropna(subset=['student_id'])
        std_df['timestamp'] = pd.to_numeric(std_df['timestamp'], errors='coerce')
        
        print(f"✅ Standardized DataFrame: {len(std_df):,} rows")
        print(f"Columns: {list(std_df.columns)}")
        return std_df
    
    def get_sample_data(self) -> pd.DataFrame:
        """Return sample for quick testing"""
        sample_path = "data/sample_studentVle.csv"
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        return self.load_oulad_vle().sample(5000, random_state=42)


# Quick test
if __name__ == "__main__":
    ingestor = LogIngestion()
    raw_df = ingestor.load_oulad_vle()
    enriched = ingestor.enrich_with_activity_type(raw_df)
    std_df = ingestor.standardize_logs(enriched)
    print("\nSample of standardized data:")
    print(std_df.head())
    print("\n🎉 Ingestion module ready!")