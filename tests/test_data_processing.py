import pandas as pd
import numpy as np
import sys
import os

# Ensure the test file can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import optimize_memory

def test_optimize_memory():
    """
    Tests if the optimize_memory function successfully reduces memory footprint
    by downcasting a float64 column to float32.
    """
    # 1. Create fake data using large 64-bit numbers
    df = pd.DataFrame({'test_col': np.array([1.5, 2.5, 3.5], dtype=np.float64)})
    start_mem = df.memory_usage().sum()
    
    # 2. Run your optimization function
    df_optimized = optimize_memory(df)
    end_mem = df_optimized.memory_usage().sum()
    
    # 3. Assert (Verify) that it worked
    assert end_mem < start_mem, "Memory was not reduced!"
    assert df_optimized['test_col'].dtype == np.float32, "Column was not downcasted to float32!"
