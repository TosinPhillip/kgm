# src/model.py - Fixed Load Model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class KnowledgeGapModel:
    def __init__(self, max_depth=5, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, 
            random_state=random_state,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.feature_names = None
        self.is_trained = False
        self.accuracy = None

    def prepare_data(self, feature_df: pd.DataFrame):
        drop_cols = ['student_id', 'concept', 'knowledge_gap', 'final_result']
        feature_cols = [col for col in feature_df.columns if col not in drop_cols]
        self.feature_names = feature_cols
        X = feature_df[feature_cols]
        y = feature_df['knowledge_gap'] if 'knowledge_gap' in feature_df.columns else None
        return X, y, feature_cols

    def train(self, feature_df: pd.DataFrame, test_size=0.25):
        X, y, _ = self.prepare_data(feature_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Accuracy: {self.accuracy:.4f}")
        return self.accuracy

    def predict_gaps(self, feature_df: pd.DataFrame):
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded first!")
        
        X, _, _ = self.prepare_data(feature_df)
        feature_df = feature_df.copy()
        feature_df['gap_probability'] = self.model.predict_proba(X)[:, 1]
        feature_df['predicted_gap'] = self.model.predict(X)
        return feature_df

    def save_model(self, path="models/knowledge_gap_model.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="models/knowledge_gap_model.joblib"):
        if os.path.exists(path):
            loaded = joblib.load(path)
            self.model = loaded.model
            self.feature_names = loaded.feature_names
            self.is_trained = True
            self.accuracy = getattr(loaded, 'accuracy', None)
            
            acc_str = f"{self.accuracy:.4f}" if self.accuracy is not None else "N/A"
            print(f"✅ Model loaded from {path} (Accuracy: {acc_str})")
            return True
        print(f"Model file not found at {path}")
        return False