from abc import ABC, abstractmethod
import lime
import numpy as np
import pandas as pd
import pyagrum as gum
import shap
from sklearn.inspection import PartialDependenceDisplay
from typing import Literal, Sequence
from utils import *

def estimate_interventional_probability_tabular(model, X_train, x_instance, intervention_features):
    tmp = X_train.copy()
    
    intervention_values = x_instance[intervention_features].values
    
    if isinstance(tmp, pd.DataFrame):
        tmp.loc[:, intervention_features] = intervention_values
    else:
        tmp[:, intervention_features] = intervention_values
        
    probs = model.predict_proba(tmp)
    return probs.mean(axis=0)

def decision_making_explanation_form(features, expected_utility, information_score):
    return {"features": features, "utility score": expected_utility, "information score": information_score}

class PrototypeExplainer(ABC):
    def __init__(self, *, model, actions, causal_model:CausalModel, utility_matrix, X_train, cognitive_method, information_method, base=np.e, random_seed=42, **kwargs):
        assert set(actions) == causal_model.outcomes, "Invalid causal model"
        self.model = model
        self.actions = actions
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.cognitive = cognitive_method
        self.information = information_method
        self.seed = random_seed
        self.base = base
        self.explanations = []

    @abstractmethod
    def explain_instance(self, x_instance, **kwargs):
        pass

    @abstractmethod
    def extract_explanation(self):
        pass

class PrototypeTabularExplainer(PrototypeExplainer):
    def __init__(self, features, model, actions, causal_model:CausalModel, utility_matrix, X_train, minimalism=False, cognitive_method:Sequence[Literal["quantity", "time"]]=("quantity",), information_method:Literal["shannon", "gini"]="shannon", base=np.e, random_seed=42, ohe_group=None, **kwargs):
        
        super().__init__(
            model=model, 
            actions=actions, 
            causal_model=causal_model, 
            utility_matrix=utility_matrix, 
            X_train=X_train, 
            cognitive_method=cognitive_method, 
            information_method=information_method,
            base=base, 
            random_seed=random_seed
        )
        
        self.features = features
        self.ohe_group = ohe_group if ohe_group is not None else {}

        # Information and cognitive constraint
        self.minimalism = minimalism
        self.max_features = len(features)
        if "quantity" in self.cognitive and "max_features" in kwargs:
            self.max_features = kwargs["max_features"]
            
        self.max_entropy = np.inf
        if "time" in self.cognitive and "max_rational_time" in kwargs and "observation_time" in kwargs:
            self.max_entropy = kwargs["max_rational_time"]/kwargs["observation_time"]
            
        self.max_impurity = np.inf

        # Global extraction
        self.critical_features = set()
        for f in self.features:
            if self.causal_model.backdoor_satisfaction(f):
                self.critical_features.add(f)

    def explain_instance(self, x_instance, max_k_features=5):
        max_size = min(self.max_features, max_k_features)
        explanation_combination = get_combinations_up_to_k(self.critical_features, max_size)
        self.explanations = []
        for e in explanation_combination:
            probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, list(e))
            
            information_score = 0
            if self.information == "shannon":
                information_score = shannon_entropy(probs, base=self.base)
                if information_score > self.max_entropy: continue
            elif self.information == "gini":
                information_score = gini_impurity(probs)
                if information_score > self.max_impurity: continue
                
            expected_utility = np.sum(self.utility_matrix @ probs.T)
            self.explanations.append(decision_making_explanation_form(set(e), expected_utility, information_score))
            
        self.explanations.sort(
            reverse=True, 
            key=lambda e: (
                e["utility score"], 
                -e["information score"], 
                len(e["features"]) if self.minimalism else -len(e["features"])
            )
        )

    def extract_explanation(self):
        return self.explanations

    def extract_critical_features(self):
        return self.critical_features

class PrototypeImageExplainer(PrototypeExplainer):
    pass