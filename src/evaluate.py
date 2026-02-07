"""
Model Evaluation Module
Evaluates trained model and logs metrics with MLflow
"""

import yaml
import logging
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_test_data(params):
    """
    Load test data and labels
    
    Args:
        params: Configuration parameters
        
    Returns:
        X_test, y_test
    """
    try:
        output_config = params['output']
        
        logger.info("Loading test data...")
        X_test = joblib.load(output_config['test_features_path'])
        y_test = joblib.load(output_config['test_labels_path'])
        
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test labels shape: {y_test.shape}")
        
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def load_model(params):
    """
    Load trained model
    
    Args:
        params: Configuration parameters
        
    Returns:
        Trained model
    """
    try:
        model_path = params['output']['model_path']
        
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_test, y_test, params):
    """
    Evaluate model on test data and log with MLflow
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        params: Configuration parameters
        
    Returns:
        Dictionary of metrics
    """
    try:
        logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
        
        logger.info("\n" + "="*50)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")
        logger.info("="*50)
        
        # Prepare metrics dictionary
        metrics = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'confusion_matrix': {
                'true_negatives': int(cm[0][0]),
                'false_positives': int(cm[0][1]),
                'false_negatives': int(cm[1][0]),
                'true_positives': int(cm[1][1])
            }
        }
        
        # Log to MLflow
        mlflow_config = params['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        with mlflow.start_run():
            # Log test metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            
            # Log confusion matrix values
            mlflow.log_metric("true_negatives", cm[0][0])
            mlflow.log_metric("false_positives", cm[0][1])
            mlflow.log_metric("false_negatives", cm[1][0])
            mlflow.log_metric("true_positives", cm[1][1])
            
            logger.info("Evaluation metrics logged to MLflow")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_metrics(metrics, output_path='metrics.json'):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Load parameters
        params = load_params()
        
        # Load test data
        X_test, y_test = load_test_data(params)
        
        # Load model
        model = load_model(params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, params)
        
        # Save metrics
        save_metrics(metrics)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
