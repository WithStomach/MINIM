import pandas as pd
import os
from typing import Dict, List, Tuple
import logging
from ..utils.medical_processor import MedicalImageProcessor

class RatingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.ratings_cache = {}
        self.medical_processor = MedicalImageProcessor(config)
        
    def load_ratings(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Ratings file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        required_columns = ['image_path', 'rating', 'modality', 'expert_level', 'prompt']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
            
        if self.config.get("raters", {}).get("expert_weights", False):
            df['rating'] = df['rating'] * df['expert_level'] / 3.0
            
        return df
    
    def process_ratings(self, ratings_df: pd.DataFrame) -> Tuple[List[str], List[float], List[str]]:
        rating_scale = self.config["data"]["rating_range"]
        min_rating, max_rating = min(rating_scale), max(rating_scale)
        
        valid_ratings = ratings_df[
            (ratings_df['rating'] >= min_rating) & 
            (ratings_df['rating'] <= max_rating)
        ]
        
        if len(valid_ratings) < len(ratings_df):
            logging.warning(
                f"Filtered out {len(ratings_df) - len(valid_ratings)} ratings "
                f"outside range [{min_rating}, {max_rating}]"
            )
        
        return valid_ratings['image_path'].tolist(), valid_ratings['rating'].tolist(), valid_ratings['prompt'].tolist()
    
    def get_batch_ratings(self, csv_path: str, batch_size: int = None) -> Tuple[List[str], List[float], List[str]]:
        if csv_path not in self.ratings_cache:
            logging.info(f"Loading ratings from {csv_path}")
            df = self.load_ratings(csv_path)
            self.ratings_cache[csv_path] = df
            logging.info(f"Loaded {len(df)} ratings")
        
        df = self.ratings_cache[csv_path]
        
        if batch_size is None:
            logging.info("Using full dataset")
            return self.process_ratings(df)
        
        batch_df = df.sample(n=min(batch_size, len(df)))
        logging.info(f"Sampled batch of size {len(batch_df)}")
        return self.process_ratings(batch_df)