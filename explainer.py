from abc import ABC, abstractmethod
import lime
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from utils import *

def calculate_do_probability_tabular(
    model,
    X_train,
    x_instance,
    explanations,
    ohe_groups=[],
    n_samples=10000,
    random_seed=42
):
    """
    Monte Carlo estimation of

        p(y|-x) = ∫ p(y|x,z)p(z)dz

    Parameters
    ----------
    model : sklearn classifier
        Must implement predict_proba().

    X_train : ndarray (n_samples, n_features)

    x_instance : ndarray (n_features,)

    explanations : list[int]
        Features to KEEP fixed.
        Remaining features are marginalized.

    ohe_groups : list[list[int]]
        One-hot encoded groups.

    n_samples : int

    Returns
    -------
    ndarray (n_classes,)
        Marginalized class probabilities.
    """

    rng = np.random.default_rng(random_seed)

    X_train = np.asarray(X_train)
    x_instance = np.asarray(x_instance)

    n_train = len(X_train)

    # sample rows from empirical distribution p(z)
    sampled_idx = rng.choice(
        n_train,
        size=n_samples,
        replace=True
    )

    samples = X_train[sampled_idx].copy()

    # features to keep fixed
    keep_features = set(explanations)

    # handle one-hot groups
    for group in ohe_groups:
        if any(f in keep_features for f in group):
            keep_features.update(group)

    # overwrite kept features by x_instance
    samples[:, list(keep_features)] = x_instance[list(keep_features)]

    # Monte Carlo expectation
    probs = model.predict_proba(samples)

    return probs.mean(axis=0)
    
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
    def __init__(self, model, utility_matrix, X_train, cognitive_method, base, random_seed=42, ohe_group = [], **kwargs):
        self.model = model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.cognitive = cognitive_method
        self.seed = random_seed
        self.base = base
        self.explanations = []
        self.ohe_group = ohe_group

    @abstractmethod
    def explain_instance(self, x_instance):
        pass

    @abstractmethod
    def extract_explanation(self):
        pass

class PrototypeTabularExplainer(PrototypeExplainer):
    def __init__(self, model, utility_matrix, X_train, cognitive_method="hick", base=np.e, random_seed=42, ohe_group = [], **kwargs):
        super().__init__(model, utility_matrix, X_train, cognitive_method, base, random_seed=random_seed, ohe_group=ohe_group)
        
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
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed, ohe_group=self.ohe_group)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "true_lime", based_entropy - causal_entropy, causal_proba))

        ### Phase 2: Over LIME, absolute value
        sort_idx_lime_abs = np.argsort(lime_result["absolute"])[::-1]
        feature_rank = lime_result["features"][sort_idx_lime_abs]
        value_rank = lime_result["absolute"][sort_idx_lime_abs]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed, ohe_group=self.ohe_group)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "absolute_lime", based_entropy - causal_entropy, causal_proba))

        ### Phase 3: Over SHAP, true value
        sort_idx_shap_true = np.argsort(shap_result["values"])[::-1]
        feature_rank = shap_result["features"][sort_idx_shap_true]
        value_rank = shap_result["values"][sort_idx_shap_true]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed, ohe_group=self.ohe_group)
            causal_entropy = shannon_entropy(causal_proba)
            if causal_entropy <= self.bounded_rational:
                self.explanations.append(explanation_form(feature_rank[:i], value_rank[:i], "true_shap", based_entropy - causal_entropy, causal_proba))

        ### Phase 4: Over SHAP, absolute value
        sort_idx_shap_abs = np.argsort(shap_result["absolute"])[::-1]
        feature_rank = shap_result["features"][sort_idx_shap_abs]
        value_rank = shap_result["absolute"][sort_idx_shap_abs]
        for i in range(1, len(feature_rank) + 1):
            causal_proba = calculate_do_probability_tabular(self.model, self.X_train, x_instance, feature_rank[:i], random_seed=self.seed, ohe_group=self.ohe_group)
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