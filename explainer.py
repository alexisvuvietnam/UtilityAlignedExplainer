from abc import ABC, abstractmethod
import lime
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from utils import *

def calculate_do_probability_tabular(model, X_train, x_instance, explanations, n_samples=10000, random_seed=42, ohe_groups=[]):    
    df_train = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    x_inst_vals = np.array(x_instance).flatten() if isinstance(x_instance, (np.ndarray, list)) else x_instance.iloc[0].values
    
    rng = np.random.default_rng(seed=random_seed)
    cols = df_train.columns
    n_features = len(cols)
    
    X_gen = np.empty((n_samples, n_features), dtype=np.float64)
    
    exp_mask = cols.isin(explanations)
    X_gen[:, exp_mask] = x_inst_vals[exp_mask]
    
    processed_idxs = set(np.where(exp_mask)[0])
    
    for group in ohe_groups:
        group_idxs = [cols.get_loc(c) for c in group]
        if not all(cols[i] in explanations for i in group_idxs):
            choices = rng.integers(0, len(group), size=n_samples)
            X_gen[:, group_idxs] = np.eye(len(group))[choices]
            processed_idxs.update(group_idxs)
            
    rem_idxs = [i for i in range(n_features) if i not in processed_idxs]
    
    if rem_idxs:
        rem_train = df_train.iloc[:, rem_idxs]
        is_cont = rem_train.apply(lambda x: pd.api.types.is_float_dtype(x) and not set(x.dropna().unique()).issubset({0, 1}))
        
        cont_idxs = [rem_idxs[i] for i, b in enumerate(is_cont) if b]
        cat_idxs = [rem_idxs[i] for i, b in enumerate(is_cont) if not b]
        
        if cont_idxs:
            means = rem_train.iloc[:, is_cont.values].mean().values
            stds = rem_train.iloc[:, is_cont.values].std().values
            X_gen[:, cont_idxs] = rng.normal(loc=means, scale=stds, size=(n_samples, len(cont_idxs)))
            
        for idx in cat_idxs:
            X_gen[:, idx] = rng.choice(df_train.iloc[:, idx].dropna().unique(), size=n_samples)
            
    df_gen = pd.DataFrame(X_gen, columns=cols)
    return np.mean(model.predict_proba(df_gen), axis=0)
    
def explanation_form(features, values, exp_type, gain, proba_matrix):
    return {"features": np.array(features), "values": np.array(values), "type": exp_type, "information": gain, "probability": proba_matrix}

def get_lime_result(model, X_train, x_instance, random_seed=42):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values if hasattr(X_train, 'values') else X_train,
        mode='classification',
        random_state=random_seed
    )
    instance_1d = x_instance.iloc[0] if hasattr(x_instance, 'iloc') else x_instance[0]
    exp = explainer.explain_instance(
        data_row=instance_1d,
        predict_fn=model.predict_proba 
    )
    lime_results = exp.as_list()
    features, feature_values = zip(*lime_results) # Pythonic way
    return list(features), list(feature_values)

def get_shap_result(model, X_train, x_instance):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(x_instance)
    
    single_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    if single_shap.feature_names is not None:
        features = list(single_shap.feature_names)
    elif hasattr(x_instance, 'columns'):
        features = list(x_instance.columns)
    else:
        features = [f"Feature {i}" for i in range(len(single_shap.values))]
        
    shap_vals = np.array(single_shap.values)
    if shap_vals.ndim > 1:
        shap_vals = np.mean(np.abs(shap_vals), axis=1)
        
    feature_values = shap_vals.tolist()
    return (features, feature_values)

def get_saliency():
    pass

def get_image_prototype():
    pass

class PrototypeExplainer(ABC):
    def __init__(self, model, utility_matrix, X_train, cognitive_method, base, random_seed=42, **kwargs):
        self.model = model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.cognitive = cognitive_method
        self.seed = random_seed
        self.base = base
        self.explanations = []

    @abstractmethod
    def explain_instance(self, x_instance):
        pass

    @abstractmethod
    def extract_explanation(self):
        pass

class PrototypeTabularExplainer(PrototypeExplainer):
    def __init__(self, model, utility_matrix, X_train, cognitive_method="hick", base=np.e, random_seed=42, **kwargs):
        super().__init__(model, utility_matrix, X_train, cognitive_method, base, random_seed=random_seed)
        
        if cognitive_method == "hick":
            assert "allow_rational_time" in kwargs and "observation_time" in kwargs, "Lack of needed parameters"
            if "reflexion_time" in kwargs:
                self.bounded_rational = entropy_by_hick(kwargs["allow_rational_time"], kwargs["observation_time"], reflexion_time = kwargs["reflexion_time"])
            else:
                self.bounded_rational = entropy_by_hick(kwargs["allow_rational_time"], kwargs["observation_time"])
        elif cognitive_method == "miller":
            assert "chunk_size" in kwargs, "Lack of needed parameters"
            self.bounded_rational = entropy_by_miller(kwargs["chunk_size"])
        else:
            self.bounded_rational = np.inf

    def explain_instance(self, x_instance):
        self.explanations = []

        features, feature_values = get_lime_result(self.model, self.X_train, x_instance, random_seed=self.seed)
        lime_result = {"features": np.array(features), "values": np.array(feature_values), "absolute": np.abs(np.array(feature_values))}

        features, feature_values = get_shap_result(self.model, self.X_train, x_instance)
        shap_result = {"features": np.array(features), "values": np.array(feature_values), "absolute": np.abs(np.array(feature_values))}

        based_proba = self.model.predict_proba(x_instance)[0]
        based_entropy = shannon_entropy(based_proba)
        
        ### Phase 1: Over LIME, true value
        sort_idx_lime_true = np.argsort(lime_result["values"])[::-1]
        feature_rank = lime_result["features"][sort_idx_lime_true]
        value_rank = lime_result["values"][sort_idx_lime_true]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "true_lime", based_entropy - causal_entropy, causal_proba))

        ### Phase 2: Over LIME, absolute value
        sort_idx_lime_abs = np.argsort(lime_result["absolute"])[::-1]
        feature_rank = lime_result["features"][sort_idx_lime_abs]
        value_rank = lime_result["absolute"][sort_idx_lime_abs]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "absolute_lime", based_entropy - causal_entropy, causal_proba))

        ### Phase 3: Over SHAP, true value
        sort_idx_shap_true = np.argsort(shap_result["values"])[::-1]
        feature_rank = shap_result["features"][sort_idx_shap_true]
        value_rank = shap_result["values"][sort_idx_shap_true]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "true_shap", based_entropy - causal_entropy, causal_proba))

        ### Phase 4: Over SHAP, absolute value
        sort_idx_shap_abs = np.argsort(shap_result["absolute"])[::-1]
        feature_rank = shap_result["features"][sort_idx_shap_abs]
        value_rank = shap_result["absolute"][sort_idx_shap_abs]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "absolute_shap", based_entropy - causal_entropy, causal_proba))

        self.explanations.sort(key=lambda e: self.utility_matrix @ e["probability"].T, reverse = True)

    def extract_explanation(self):
        explanation = self.explanations[0]
        explanation["expected utility"] = self.utility_matrix @ explanation["probability"].T
        return explanation

class PrototypeImageExplainer(PrototypeExplainer):
    pass