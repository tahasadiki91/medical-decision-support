import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit l'utilisation de la mémoire d'un DataFrame en convertissant 
    les types de données vers les formats les plus compacts possibles. """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Utilisation mémoire initiale : {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Ignorer l'optimisation des dates si elles existent
        if 'datetime' in str(col_type):
            continue
            
        if col_type != object and not pd.api.types.is_categorical_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            # Optimisation des flottants
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Conversion des objets (strings) en catégories si pertinent
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
                
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Utilisation mémoire après optimisation : {end_mem:.2f} MB")
    print(f"Réduction de mémoire de : {100 * (start_mem - end_mem) / start_mem:.1f}%\n")
    
    return df
