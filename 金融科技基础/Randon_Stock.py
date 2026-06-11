import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import random
import tushare as ts  # Correct import for Tushare API

# Initialize API once
pro = ts.pro_api('a394376ad4ede9c1214c06479e7d1bee32919045f909fcbfc39cabde')

def random_stock(stocknumber):
    """
    Select random unique stocks from HS300 index
    
    Args:
        stocknumber (int): Number of unique stocks to select
    
    Returns:
        list: List of unique stock codes
    """
    try:
        # Get HS300 constituent stocks
        hs300 = pro.index_weight(index_code='000300.SH')
        hs300_codes = hs300['con_code'].tolist()
        
        # Validate input
        if stocknumber <= 0:
            raise ValueError("Stock number must be positive")
        if stocknumber > len(hs300_codes):
            raise ValueError(f"Cannot select {stocknumber} stocks. Only {len(hs300_codes)} available in HS300")
        
        # Use random.sample to ensure unique selection
        selected_tickers = random.sample(hs300_codes, stocknumber)
        
        return selected_tickers
        
    except Exception as e:
        print(f"Error selecting random stocks: {e}")
        return []

# Call the function and print the result
selected_stocks = random_stock(10)
print(f"Selected {len(selected_stocks)} unique stocks:")
print(selected_stocks)