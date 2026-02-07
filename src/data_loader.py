"""
Data Loader Module
Downloads and saves raw dataset for SMS Spam Classification
"""

import os
import pandas as pd
import yaml
import logging

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


def download_dataset(url, output_path):
    """
    Download SMS Spam dataset from URL
    
    Args:
        url: Dataset URL
        output_path: Path to save raw data
    """
    try:
        logger.info(f"Downloading dataset from {url}")
        
        # Read the TSV file
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nLabel distribution:\n{df['label'].value_counts()}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Load parameters
        params = load_params()
        
        # Extract configuration
        dataset_url = params['data']['dataset_url']
        raw_path = params['data']['raw_path']
        
        # Download and save dataset
        df = download_dataset(dataset_url, raw_path)
        
        logger.info("Data loading completed successfully")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise


if __name__ == "__main__":
    main()