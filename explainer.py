from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

def calculate_do_probability(model, X_train, x_instance, explanations, n_samples=10000, random_seed=42, ohe_groups=[]):
    is_numpy = isinstance(X_train, np.ndarray)
    
    if is_numpy:
        df_train = pd.DataFrame(X_train)
        if isinstance(x_instance, np.ndarray):
            x_arr = x_instance if x_instance.ndim > 1 else np.array([x_instance])
        else:
            x_arr = np.array([x_instance])
        df_instance = pd.DataFrame(x_arr)
    else:
        df_train = X_train.copy()
        if isinstance(x_instance, pd.Series):
            df_instance = x_instance.to_frame().T.reset_index(drop=True)
        else:
            df_instance = pd.DataFrame(x_instance).reset_index(drop=True)

    generated_data = {}
    rng = np.random.default_rng(seed=random_seed)

    processed_cols = set()

    for group in ohe_groups:
        is_explained = all(col in explanations for col in group)
        
        if is_explained:
            for col in group:
                generated_data[col] = np.full(n_samples, df_instance.at[0, col])
                processed_cols.add(col)
        else:
            n_categories = len(group)
            chosen_indices = rng.integers(0, n_categories, size=n_samples)
            
            for i, col in enumerate(group):
                generated_data[col] = (chosen_indices == i).astype(int)
                processed_cols.add(col)
    
    for col in df_train.columns:
        if col in processed_cols:
            continue
            
        if col in explanations:
            fixed_val = df_instance.at[0, col]
            generated_data[col] = np.full(n_samples, fixed_val)
        else:
            col_data = df_train[col].dropna()
            
            is_binary = set(col_data.unique()).issubset({0, 0.0, 1, 1.0})
            is_continuous = pd.api.types.is_float_dtype(col_data) and not is_binary
            
            if is_continuous:
                mean_val = col_data.mean()
                std_val = col_data.std()
                generated_data[col] = rng.normal(loc=mean_val, scale=std_val, size=n_samples)
            else:
                unique_vals = col_data.unique()
                generated_data[col] = rng.choice(unique_vals, size=n_samples)

    df_generated = pd.DataFrame(generated_data, columns=df_train.columns)
    
    all_probs = model.predict_proba(df_generated)
    
    return np.mean(all_probs, axis=0)
    

def get_bounded_lime(model, X_train, x_instance, random_seed=42):
    pass

def get_bounded_shap(model, X_train, x_instance):
    pass

def get_bounded_pdp(model, X_train, random_seed=42):
    pass

def get_bounded_saliency():
    pass

def get_bounded_prototype():
    pass

class PrototypeExplainer(ABC):

    def __init__(self, X_train, cognitive_method):
        self.X_train = X_train
        self.cognitive = cognitive_method

    @classmethod
    @abstractmethod
    def explain_instance(self, x_instance):
        pass

    @classmethod
    @abstractmethod
    def extract_explanation(self):
        pass


class PrototypeTabularExplainer:
    pass

class PrototypeImageExplainer:
    pass