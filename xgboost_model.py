"""
XGBoost Prediction Model
Gradient boosting classifier for next-day direction prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import pickle


class XGBoostPredictor:
    """XGBoost model for direction prediction"""
    
    def __init__(self, random_state: int = 42):
        """Initialize XGBoost predictor"""
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}
        
    def train(self,
             X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model hyperparameters (optional)
            
        Returns:
            Dictionary with training metrics
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Default parameters optimized for trading
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'n_estimators': 200,
                'early_stopping_rounds': 20
            }
        
        print(f"\n📊 Training data: {len(X_train)} samples, {len(self.feature_names)} features")
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with optional validation
        if X_val is not None and y_val is not None:
            print(f"📊 Validation data: {len(X_val)} samples")
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        print("\n✓ Model training complete")
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred),
            'train_recall': recall_score(y_train, train_pred),
            'train_f1': f1_score(y_train, train_pred),
            'train_auc': roc_auc_score(y_train, train_proba)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred),
                'val_recall': recall_score(y_val, val_pred),
                'val_f1': f1_score(y_val, val_pred),
                'val_auc': roc_auc_score(y_val, val_proba)
            })
        
        # Extract feature importance
        self._extract_feature_importance()
        
        # Print metrics
        self._print_metrics()
        
        return self.training_metrics
    
    def calibrate_probabilities(self,
                               X_cal: pd.DataFrame,
                               y_cal: pd.Series,
                               method: str = 'isotonic') -> None:
        """
        Calibrate probability predictions
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            method: Calibration method ('sigmoid' or 'isotonic')
        """
        print(f"\n🎯 Calibrating probabilities using {method} method...")
        
        # Create calibrated model
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=method,
            cv='prefit'
        )
        
        # Fit on calibration data
        self.calibrated_model.fit(X_cal, y_cal)
        
        print("✓ Probability calibration complete")
    
    def predict(self,
               X: pd.DataFrame,
               calibrated: bool = True) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            calibrated: Use calibrated probabilities
            
        Returns:
            Predicted labels
        """
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self,
                     X: pd.DataFrame,
                     calibrated: bool = True) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            calibrated: Use calibrated probabilities
            
        Returns:
            Predicted probabilities [P(down), P(up)]
        """
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def predict_with_confidence(self,
                               X: pd.DataFrame,
                               confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Predict with confidence filtering
        
        Args:
            X: Features
            confidence_threshold: Minimum probability for high-confidence predictions
            
        Returns:
            DataFrame with predictions, probabilities, and confidence levels
        """
        # Get predictions and probabilities
        pred_proba = self.predict_proba(X, calibrated=True)
        pred_labels = self.predict(X, calibrated=True)
        
        # Calculate confidence (max probability)
        confidence = np.max(pred_proba, axis=1)
        
        # Create results dataframe
        results = pd.DataFrame({
            'prediction': pred_labels,
            'prob_down': pred_proba[:, 0],
            'prob_up': pred_proba[:, 1],
            'confidence': confidence,
            'confidence_level': ['high' if c >= confidence_threshold else 
                               'medium' if c >= 0.55 else 
                               'low' for c in confidence]
        }, index=X.index)
        
        return results
    
    def evaluate(self,
                X_test: pd.DataFrame,
                y_test: pd.Series) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.predict(X_test, calibrated=True)
        y_proba = self.predict_proba(X_test, calibrated=True)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n📊 Test Set Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   F1 Score:  {metrics['f1']:.1%}")
        print(f"   ROC AUC:   {metrics['auc']:.3f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Down  Up")
        print(f"   Actual Down {cm[0,0]:>4} {cm[0,1]:>4}")
        print(f"   Actual Up   {cm[1,0]:>4} {cm[1,1]:>4}")
        
        # Classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Down', 'Up'],
                                   digits=3))
        
        return metrics
    
    def _extract_feature_importance(self) -> None:
        """Extract and store feature importance"""
        importance = self.model.feature_importances_
        
        self.feature_importance = {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                  key=lambda x: x[1], 
                  reverse=True)
        )
    
    def get_top_features(self, n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            n: Number of top features
            
        Returns:
            Dictionary of top features and their importance
        """
        return dict(list(self.feature_importance.items())[:n])
    
    def explain_prediction(self,
                          X: pd.DataFrame,
                          idx: int = 0,
                          top_n: int = 10) -> Dict:
        """
        Explain a single prediction
        
        Args:
            X: Features
            idx: Index of prediction to explain
            top_n: Number of top features to show
            
        Returns:
            Dictionary with prediction explanation
        """
        # Get prediction
        pred_proba = self.predict_proba(X.iloc[[idx]], calibrated=True)[0]
        pred_label = self.predict(X.iloc[[idx]], calibrated=True)[0]
        
        # Get feature values
        feature_values = X.iloc[idx]
        
        # Get top features by importance
        top_features = self.get_top_features(top_n)
        
        # Build explanation
        explanation = {
            'prediction': 'UP' if pred_label == 1 else 'DOWN',
            'prob_up': pred_proba[1],
            'prob_down': pred_proba[0],
            'confidence': max(pred_proba),
            'top_features': {}
        }
        
        for feature, importance in top_features.items():
            if feature in feature_values.index:
                explanation['top_features'][feature] = {
                    'value': float(feature_values[feature]),
                    'importance': float(importance)
                }
        
        return explanation
    
    def _print_metrics(self) -> None:
        """Print training metrics"""
        print("\n📊 Training Metrics:")
        print(f"   Accuracy:  {self.training_metrics['train_accuracy']:.1%}")
        print(f"   Precision: {self.training_metrics['train_precision']:.1%}")
        print(f"   Recall:    {self.training_metrics['train_recall']:.1%}")
        print(f"   F1 Score:  {self.training_metrics['train_f1']:.1%}")
        print(f"   ROC AUC:   {self.training_metrics['train_auc']:.3f}")
        
        if 'val_accuracy' in self.training_metrics:
            print("\n📊 Validation Metrics:")
            print(f"   Accuracy:  {self.training_metrics['val_accuracy']:.1%}")
            print(f"   Precision: {self.training_metrics['val_precision']:.1%}")
            print(f"   Recall:    {self.training_metrics['val_recall']:.1%}")
            print(f"   F1 Score:  {self.training_metrics['val_f1']:.1%}")
            print(f"   ROC AUC:   {self.training_metrics['val_auc']:.3f}")
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        
        print(f"\n✓ Model loaded from {filepath}")


def main():
    """Test XGBoost predictor"""
    from demo_synthetic import generate_synthetic_price_data
    from technical_indicators import TechnicalIndicators
    from feature_engineering import FeatureEngineering
    
    print("Generating data...")
    spy_data = generate_synthetic_price_data('SPY', days=1260)
    
    print("\nCalculating indicators...")
    spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_data)
    
    print("\nCreating ML dataset...")
    fe = FeatureEngineering()
    ml_dataset = fe.create_ml_dataset(spy_with_indicators, target_horizon=1)
    
    print("\nSplitting data...")
    train_df, val_df, test_df = fe.train_test_split_time_series(ml_dataset)
    
    # Prepare features and targets
    X_train = train_df[fe.feature_columns]
    y_train = train_df['target_direction_1d']
    
    X_val = val_df[fe.feature_columns]
    y_val = val_df['target_direction_1d']
    
    X_test = test_df[fe.feature_columns]
    y_test = test_df['target_direction_1d']
    
    # Train model
    predictor = XGBoostPredictor()
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Calibrate probabilities
    predictor.calibrate_probabilities(X_val, y_val)
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    # Show top features
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    
    top_features = predictor.get_top_features(15)
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"{i:2d}. {feature[:45]:<45} {importance:.4f}")
    
    # Example prediction with explanation
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION WITH EXPLANATION")
    print("="*60)
    
    explanation = predictor.explain_prediction(X_test, idx=0, top_n=5)
    
    print(f"\nPrediction: {explanation['prediction']}")
    print(f"Confidence: {explanation['confidence']:.1%}")
    print(f"Prob Up:    {explanation['prob_up']:.1%}")
    print(f"Prob Down:  {explanation['prob_down']:.1%}")
    
    print(f"\nTop Contributing Features:")
    for feature, data in explanation['top_features'].items():
        print(f"  {feature[:40]:<40} Value: {data['value']:>8.2f}  Importance: {data['importance']:.4f}")
    
    return predictor, metrics


if __name__ == "__main__":
    predictor, metrics = main()
