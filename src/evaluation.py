import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class GapEvaluator:
    """Evaluation, Severity Assessment and Remediation for Knowledge Gaps"""
    
    def __init__(self):
        self.remediation_rules = {
            'Low': [
                "Review key concepts briefly",
                "Practice with similar questions",
                "Watch short explanatory videos"
            ],
            'Medium': [
                "Dedicated revision session required",
                "Solve additional practice problems on this concept",
                "Join discussion forum or ask instructor for clarification",
                "Re-watch relevant lecture/material"
            ],
            'High': [
                "Urgent intervention needed",
                "Schedule one-on-one session with tutor",
                "Complete targeted remedial exercises",
                "Break down concept into smaller sub-topics",
                "Consider prerequisite review"
            ]
        }
    
    def evaluate_model(self, y_true, y_pred, y_prob=None):
        """Comprehensive model evaluation"""
        print("=== MODEL EVALUATION ===")
        print(classification_report(y_true, y_pred, target_names=['No Gap', 'Knowledge Gap']))
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
                print(f"\nROC-AUC Score: {auc:.4f}")
            except:
                pass
        
        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Gap', 'Gap'], 
                    yticklabels=['No Gap', 'Gap'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def assess_severity(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add severity level and evidence for each gap"""
        df = predictions_df.copy()
        
        # Severity based on probability and interaction patterns
        conditions = [
            (df['gap_probability'] >= 0.75) | 
            ((df['interaction_ratio'] < 0.1) & (df['unique_days_active'] < 3)),
            (df['gap_probability'] >= 0.45),
            (df['gap_probability'] < 0.45)
        ]
        
        choices = ['High', 'Medium', 'Low']
        df['severity'] = np.select(conditions, choices, default='Low')
        
        # Add supporting evidence
        df['evidence'] = df.apply(self._generate_evidence, axis=1)
        
        return df
    
    def _generate_evidence(self, row):
        """Generate human-readable explanation for the gap"""
        evidence = []
        if row['total_interactions'] < 5:
            evidence.append(f"Very low interactions ({row['total_interactions']})")
        if row['interaction_consistency'] < 0.3:
            evidence.append("Low consistency across days")
        if row['avg_interactions_per_day'] < 2:
            evidence.append("Minimal daily engagement")
        if row['gap_probability'] > 0.7:
            evidence.append(f"High model confidence ({row['gap_probability']:.2f})")
        
        return " + ".join(evidence) if evidence else "Based on behavioral patterns"
    
    def generate_remediation(self, row) -> list:
        """Return personalized remediation suggestions"""
        severity = row.get('severity', 'Medium')
        return self.remediation_rules.get(severity, self.remediation_rules['Medium'])
    
    def create_knowledge_gap_map(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Final Knowledge Gap Map per student"""
        enhanced = self.assess_severity(predictions_df)
        
        # Add remediation suggestions as string
        enhanced['remediation_suggestions'] = enhanced.apply(
            lambda row: " | ".join(self.generate_remediation(row)), axis=1
        )
        
        # Student-level summary
        student_map = enhanced.groupby('student_id').agg(
            total_concepts=('concept', 'nunique'),
            gaps_detected=('predicted_gap', 'sum'),
            high_severity_gaps=('severity', lambda x: (x == 'High').sum()),
            avg_gap_probability=('gap_probability', 'mean'),
            risk_level=('severity', lambda x: x.mode()[0] if not x.empty else 'Low')
        ).reset_index()
        
        student_map['gap_percentage'] = (student_map['gaps_detected'] / student_map['total_concepts'] * 100).round(1)
        
        print(f"✅ Generated knowledge gap map for {len(student_map)} students")
        return enhanced, student_map


# Quick test
if __name__ == "__main__":
    from src.ingestion import LogIngestion
    from src.features import FeatureEngineer
    from src.model import KnowledgeGapModel
    
    print("=== DAY 5: Evaluation & Remediation ===\n")
    
    # Load pipeline
    ingestor = LogIngestion()
    raw = ingestor.load_oulad_vle()
    enriched = ingestor.enrich_with_activity_type(raw)
    std_df = ingestor.standardize_logs(enriched)
    
    engineer = FeatureEngineer()
    student_info = pd.read_csv("data/studentInfo.csv")
    student_info = student_info[(student_info['code_module'] == 'AAA') & 
                               (student_info['code_presentation'] == '2013J')]
    
    feature_df = engineer.build_full_feature_set(std_df, student_info)
    
    # Train model
    model = KnowledgeGapModel(max_depth=5)
    model.train(feature_df)
    predictions = model.predict_gaps(feature_df)
    
    # Evaluation + Gap Mapping
    evaluator = GapEvaluator()
    enhanced_predictions, student_map = evaluator.create_knowledge_gap_map(predictions)
    
    print("\nSample Knowledge Gap Map:")
    sample_student = enhanced_predictions['student_id'].iloc[0]
    print(enhanced_predictions[enhanced_predictions['student_id'] == sample_student][['concept', 'severity', 'evidence', 'remediation_suggestions']].head())
    
    print("\n Evaluation and remediation system is complete.")