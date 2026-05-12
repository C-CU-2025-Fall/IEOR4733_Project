"""
Data Module - Data loading and preprocessing

Contents:
    - loader.py: Data loader
    - preprocessor.py: Data preprocessing
    - contract_manager.py: Futures contract management

Main classes:
    - DataLoader: General-purpose data loader
    - DataPreprocessor: Data preprocessing
    - ContractManager: Contract management

Usage example:
    from src.core.data import DataLoader, DataPreprocessor
    
    loader = DataLoader(data_dir="data/")
    df = loader.load_price_data("Commodity", "2011-01-01", "2019-12-31")
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process(df)
"""

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "ContractManager",
]
