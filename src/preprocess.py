"""
Preprocessing Module
Cleans and preprocesses text data for SMS Spam Classification
"""

import os
import pandas as pd
import yaml
import logging
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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


def clean_text(text, config):
    """
    Clean and preprocess text
    
    Args:
        text: Input text string
        config: Preprocessing configuration from params.yaml
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Lowercase
    if config['lowercase']:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    if config['remove_punctuation']:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    if config['remove_stopwords']:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        text = ' '.join([word for word in tokens if word not in stop_words])
    
    return text


def preprocess_data(input_path, output_path, params):
    """
    Preprocess the SMS dataset
    
    Args:
        input_path: Path to raw data
        output_path: Path to save processed data
        params: Configuration parameters
    """
    try:
        logger.info(f"Loading raw data from {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info(f"Initial dataset shape: {df.shape}")
        
        # Get configuration
        preprocessing_config = params['preprocessing']
        text_column = preprocessing_config['text_column']
        label_column = preprocessing_config['label_column']
        min_length = preprocessing_config['min_text_length']
        
        # Handle missing values
        logger.info("Handling missing values...")
        initial_count = len(df)
        df = df.dropna(subset=[text_column, label_column])
        logger.info(f"Removed {initial_count - len(df)} rows with missing values")
        
        # Clean text
        logger.info("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(
            lambda x: clean_text(x, preprocessing_config)
        )
        
        # Filter out very short texts
        df = df[df['cleaned_text'].str.len() >= min_length]
        logger.info(f"Removed texts shorter than {min_length} characters")
        
        # Encode labels (spam=1, ham=0)
        logger.info("Encoding labels...")
        df['label_encoded'] = df[label_column].map({'spam': 1, 'ham': 0})
        
        # Select final columns
        df_processed = df[[label_column, 'label_encoded', text_column, 'cleaned_text']]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data
        df_processed.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Final dataset shape: {df_processed.shape}")
        logger.info(f"\nLabel distribution:\n{df_processed['label'].value_counts()}")
        
        # Display sample
        logger.info("\nSample processed data:")
        logger.info(f"\n{df_processed.head(3)}")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Load parameters
        params = load_params()
        
        # Extract paths
        raw_path = params['data']['raw_path']
        processed_path = params['data']['processed_path']
        
        # Preprocess data
        df = preprocess_data(raw_path, processed_path, params)
        
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()