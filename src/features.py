"""
Feature Engineering Module
Converts text to numerical features using TF-IDF and splits data
"""

import os
import pandas as pd
import yaml
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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


def create_features(input_path, params):
    """
    Create TF-IDF features from processed text
    
    Args:
        input_path: Path to processed data
        params: Configuration parameters
        
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    try:
        logger.info(f"Loading processed data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Extract text and labels
        texts = df['cleaned_text'].values
        labels = df['label_encoded'].values
        
        logger.info(f"Dataset size: {len(texts)}")
        logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        # Get feature configuration
        feature_config = params['features']
        split_config = feature_config['train_test_split']
        
        # Create TF-IDF vectorizer
        logger.info("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=feature_config['max_features'],
            ngram_range=tuple(feature_config['ngram_range']),
            min_df=feature_config['min_df'],
            max_df=feature_config['max_df']
        )
        
        # Transform text to features
        X = vectorizer.fit_transform(texts)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        stratify = labels if split_config['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=split_config['test_size'],
            random_state=split_config['random_state'],
            stratify=stratify
        )
        
        logger.info(f"Train set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        logger.info(f"Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")
        logger.info(f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, vectorizer
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


def save_features(X_train, X_test, y_train, y_test, vectorizer, params):
    """
    Save feature matrices, labels, and vectorizer
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        vectorizer: Fitted TF-IDF vectorizer
        params: Configuration parameters
    """
    try:
        # Get output paths
        output_config = params['output']
        
        # Create directories
        os.makedirs(os.path.dirname(output_config['train_features_path']), exist_ok=True)
        os.makedirs(os.path.dirname(output_config['vectorizer_path']), exist_ok=True)
        
        # Save feature matrices
        logger.info("Saving feature matrices...")
        joblib.dump(X_train, output_config['train_features_path'])
        joblib.dump(X_test, output_config['test_features_path'])
        
        # Save labels
        logger.info("Saving labels...")
        joblib.dump(y_train, output_config['train_labels_path'])
        joblib.dump(y_test, output_config['test_labels_path'])
        
        # Save vectorizer
        logger.info("Saving vectorizer...")
        joblib.dump(vectorizer, output_config['vectorizer_path'])
        
        logger.info("All features and artifacts saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving features: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Load parameters
        params = load_params()
        
        # Extract paths
        processed_path = params['data']['processed_path']
        
        # Create features
        X_train, X_test, y_train, y_test, vectorizer = create_features(
            processed_path, params
        )
        
        # Save features
        save_features(X_train, X_test, y_train, y_test, vectorizer, params)
        
        logger.info("Feature engineering completed successfully")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()