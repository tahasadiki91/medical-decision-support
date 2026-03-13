import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def optimize_memory(df):
    """
    Optimizes memory usage of a pandas DataFrame by downcasting numeric types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    # Using .4f because this medical dataset is small, so we want to see the exact decimals
    print(f'Memory usage before optimization: {start_mem:.4f} MB')

    for col in df.columns:
        # STRICT CHECK: Only process this column if it is actually a number
        if is_numeric_dtype(df[col]):
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization: {end_mem:.4f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df
