"""
Model evaluation and comparison module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import os


class ModelEvaluator:
    """Handles model evaluation and comparison"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name=None, label_encoder=None):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Optional model name for tracking
            label_encoder: Optional label encoder for XGBoost models
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        
        # Handle XGBoost predictions (convert back to string labels)
        if hasattr(model, 'classes_') and label_encoder is not None:
            # XGBoost produces numeric predictions, convert back to strings
            y_pred = label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'predictions': y_pred
        }
        
        # Store results if model name provided
        if model_name:
            self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def compare_models(self, results_dict):
        """
        Compare multiple model results
        
        Args:
            results_dict: Dictionary of {model_name: results}
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1-Macro': results['f1_macro'],
                'F1-Weighted': results.get('f1_weighted', 0),
                'Precision': results.get('precision_macro', 0),
                'Recall': results.get('recall_macro', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = sorted(set(y_true))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_model_comparison(self, comparison_df, save_path=None):
        """
        Plot model comparison metrics
        
        Args:
            comparison_df: DataFrame from compare_models
            save_path: Optional path to save plot
            
        Returns:
            matplotlib figure
        """
        metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                axes[i].bar(comparison_df['Model'], comparison_df[metric])
                axes[i].set_title(metric)
                axes[i].set_ylim(0, 1)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_classification_report(self, y_true, y_pred, class_names=None):
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            
        Returns:
            String with classification report
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        
        report = classification_report(y_true, y_pred, target_names=class_names)
        return report
    
    def analyze_prediction_errors(self, y_true, y_pred, X_test_text=None):
        """
        Analyze prediction errors to understand model weaknesses
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X_test_text: Optional test text for error analysis
            
        Returns:
            Dictionary with error analysis
        """
        # Create error mask
        error_mask = y_true != y_pred
        errors = pd.DataFrame({
            'true_label': y_true[error_mask],
            'predicted_label': y_pred[error_mask]
        })
        
        if X_test_text is not None:
            errors['text'] = X_test_text[error_mask]
        
        # Error statistics
        error_stats = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'errors_by_true_class': errors['true_label'].value_counts().to_dict(),
            'errors_by_predicted_class': errors['predicted_label'].value_counts().to_dict()
        }
        
        return error_stats, errors
    
    def calculate_class_performance(self, y_true, y_pred):
        """
        Calculate per-class performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        classes = sorted(set(y_true))
        class_metrics = []
        
        for class_name in classes:
            # Create binary masks for this class
            true_binary = (y_true == class_name)
            pred_binary = (y_pred == class_name)
            
            # Calculate metrics
            tp = np.sum(true_binary & pred_binary)
            fp = np.sum(~true_binary & pred_binary)
            fn = np.sum(true_binary & ~pred_binary)
            tn = np.sum(~true_binary & ~pred_binary)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': np.sum(true_binary)
            })
        
        return pd.DataFrame(class_metrics)
    
    def save_evaluation_report(self, results, filepath='reports/evaluation/evaluation_report.txt'):
        """
        Save comprehensive evaluation report to file
        
        Args:
            results: Dictionary with evaluation results
            filepath: Path to save the report
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("TWITTER SENTIMENT ANALYSIS - EVALUATION REPORT\\n")
            f.write("=" * 60 + "\\n\\n")
            
            for model_name, metrics in results.items():
                f.write(f"Model: {model_name}\\n")
                f.write("-" * 30 + "\\n")
                for metric_name, value in metrics.items():
                    if metric_name != 'predictions':
                        f.write(f"{metric_name}: {value:.4f}\\n")
                f.write("\\n")
        
        print(f"Evaluation report saved to {filepath}")
    
    def get_best_model_summary(self, results_dict):
        """
        Get summary of the best performing model
        
        Args:
            results_dict: Dictionary of model results
            
        Returns:
            Dictionary with best model info
        """
        if not results_dict:
            return None
        
        # Find best model by F1-macro score
        best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['f1_macro'])
        best_metrics = results_dict[best_model]
        
        summary = {
            'best_model': best_model,
            'metrics': best_metrics,
            'rank': 1,
            'total_models': len(results_dict)
        }
        
        return summary
