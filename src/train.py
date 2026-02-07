"""
Model Training Module
Trains ML model and logs experiments with MLflow
"""

import os
import yaml
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def load_data(params):
    """
    Load training data and labels
    
    Args:
        params: Configuration parameters
        
    Returns:
        X_train, y_train
    """
    try:
        output_config = params['output']
        
        logger.info("Loading training data...")
        X_train = joblib.load(output_config['train_features_path'])
        y_train = joblib.load(output_config['train_labels_path'])
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Training labels shape: {y_train.shape}")
        
        return X_train, y_train
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def create_model(params):
    """
    Create ML model based on configuration
    
    Args:
        params: Configuration parameters
        
    Returns:
        Initialized model
    """
    model_config = params['model']
    algorithm = model_config['algorithm']
    
    logger.info(f"Creating {algorithm} model...")
    
    if algorithm == 'logistic_regression':
        lr_config = model_config['logistic_regression']
        model = LogisticRegression(
            C=lr_config['C'],
            max_iter=lr_config['max_iter'],
            solver=lr_config['solver'],
            random_state=lr_config['random_state']
        )
    elif algorithm == 'naive_bayes':
        nb_config = model_config['naive_bayes']
        model = MultinomialNB(
            alpha=nb_config['alpha']
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def train_model(model, X_train, y_train, params):
    """
    Train the ML model and log with MLflow
    
    Args:
        model: ML model
        X_train, y_train: Training data
        params: Configuration parameters
        
    Returns:
        Trained model
    """
    try:
        # Setup MLflow
        mlflow_config = params['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        with mlflow.start_run():
            logger.info("Training model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions on training data
            y_train_pred = model.predict(X_train)
            
            # Calculate training metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='binary')
            train_recall = recall_score(y_train, y_train_pred, average='binary')
            train_f1 = f1_score(y_train, y_train_pred, average='binary')
            
            logger.info(f"Training Accuracy: {train_accuracy:.4f}")
            logger.info(f"Training Precision: {train_precision:.4f}")
            logger.info(f"Training Recall: {train_recall:.4f}")
            logger.info(f"Training F1-Score: {train_f1:.4f}")
            
            # Log parameters
            model_config = params['model']
            algorithm = model_config['algorithm']
            mlflow.log_param("algorithm", algorithm)
            
            if algorithm == 'logistic_regression':
                for key, value in model_config['logistic_regression'].items():
                    mlflow.log_param(f"lr_{key}", value)
            elif algorithm == 'naive_bayes':
                for key, value in model_config['naive_bayes'].items():
                    mlflow.log_param(f"nb_{key}", value)
            
            # Log feature parameters
            feature_config = params['features']
            mlflow.log_param("max_features", feature_config['max_features'])
            mlflow.log_param("ngram_range", str(feature_config['ngram_range']))
            mlflow.log_param("test_size", feature_config['train_test_split']['test_size'])
            
            # Log training metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1_score", train_f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("Model and metrics logged to MLflow")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def save_model(model, params):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        params: Configuration parameters
    """
    try:
        model_path = params['output']['model_path']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Load parameters
        params = load_params()
        
        # Load training data
        X_train, y_train = load_data(params)
        
        # Create model
        model = create_model(params)
        
        # Train model
        model = train_model(model, X_train, y_train, params)
        
        # Save model
        save_model(model, params)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()