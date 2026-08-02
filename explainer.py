from abc import ABC, abstractmethod
import itertools
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

def decision_making_explanation_form(features, probability, expected_utility, information_score, action):
    return {"features": str(features), "prediction probability": probability, "utility score": expected_utility, "preferrable action": action, "information score": information_score}

def explanation_form(feature, weight):
    return {"feature": feature, "signed score": weight, "absolute score": abs(weight)}

def lime_explanation_form(model, lime_explainer, x_instance_1d, importance=True):
    raw_pred = model.predict(x_instance_1d.to_frame().T)[0]
    class_idx = np.where(model.classes_ == raw_pred)[0][0]
    exp = lime_explainer.explain_instance(data_row=x_instance_1d.to_numpy(), predict_fn=model.predict_proba,labels=(class_idx,))
    res_map = exp.as_map()[class_idx]
    feature_names = x_instance_1d.index.tolist()
    attribution = []
    for feat_idx, weight in res_map:
        feature = feature_names[feat_idx]
        attribution.append(explanation_form(feature, weight))
    attribution.sort(reverse=True, key=lambda x: (x["absolute score"] if importance else x["signed score"]))
    return attribution

def shap_explanation_form(model, shap_explainer, x_instance_2d, importance=True):
    raw_pred = model.predict(x_instance_2d)[0]
    class_idx = np.where(model.classes_ == raw_pred)[0][0]

    shap_vals = shap_explainer.shap_values(x_instance_2d)
    if isinstance(shap_vals, list):
        instance_shap = shap_vals[class_idx][0] 
    else:
        if len(shap_vals.shape) == 3: 
            instance_shap = shap_vals[0, :, class_idx]
        else:
            instance_shap = shap_vals[0]
            
    if isinstance(x_instance_2d, pd.DataFrame):
        feature_names = x_instance_2d.columns.tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(len(instance_shap))]
        
    attribution = []
    for feature, weight in zip(feature_names, instance_shap):
        attribution.append(explanation_form(feature, weight))
        
    attribution.sort(reverse=True, key=lambda x: (x["absolute score"] if importance else x["signed score"]))
    
    return attribution

class UtilityAlignedExplainer(ABC):
    def __init__(self, model, X_train, *, actions, causal_model:CausalModel, utility_matrix, information_method, base=np.e, **kwargs):
        assert set(actions) == causal_model.outcomes, "Invalid causal model"
        self.model = model
        self.actions = actions
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.information = information_method
        self.base = base
        self.explanations = []
        self.attributions = []

    @abstractmethod
    def explain_instance(self, x_instance, **kwargs):
        pass

    @abstractmethod
    def extract_explanation(self):
        pass

class UtilityAlignedTabularExplainer(UtilityAlignedExplainer):
    def __init__(self, model, X_train, features, actions, causal_model:CausalModel, utility_matrix, information_method:Literal["shannon", "gini"]="shannon", base=np.e, ohe_group=None, **kwargs):
        
        super().__init__(
            model,
            X_train, 
            actions=actions, 
            causal_model=causal_model, 
            utility_matrix=utility_matrix,
            information_method=information_method,
            base=base
        )
        
        self.features = features
        self.ohe_group = ohe_group if ohe_group is not None else {}

        # Information and cognitive constraints
        self.max_features = len(features)
            
        self.information_bound = np.inf

        # Global extraction
        self.critical_features = set()
        for f in self.features:
            if self.causal_model.backdoor_satisfaction(f):
                self.critical_features.add(f)

    def explain_instance(self, x_instance):
        explanation_combination = get_combinations_up_to_k(self.critical_features, self.max_features)
        self.explanations = []
        for e in explanation_combination:
            standard_e = list(e)[:]
            for f in e:
                if f in self.ohe_group:
                    standard_e.remove(f)
                    standard_e += self.ohe_group[f]
            probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, list(standard_e))
            
            information_score = 0
            if self.information == "shannon":
                information_score = shannon_entropy(probs, base=self.base)
            elif self.information == "gini":
                information_score = gini_impurity(probs)
                
            expected_utility = np.max(self.utility_matrix @ probs.T)
            self.explanations.append(decision_making_explanation_form(set(e), probs, expected_utility, information_score, self.actions[np.argmax(self.utility_matrix @ probs.T)]))
            
        self.explanations.sort(
            reverse=True, 
            key=lambda e: (
                e["utility score"], 
                -e["information score"], 
                len(e["features"])
            )
        )

        return self.extract_explanation()

    def explain_instance_k_features(self, x_instance, num_features):
        if num_features > len(self.critical_features):
            return
        explanation_combination = itertools.combinations(self.critical_features, num_features)
        self.explanations = []
        for e in explanation_combination:
            standard_e = list(e)[:]
            for f in e:
                if f in self.ohe_group:
                    standard_e.remove(f)
                    standard_e += self.ohe_group[f]
            probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, list(standard_e))
                    
            information_score = 0
            if self.information == "shannon":
                information_score = shannon_entropy(probs, base=self.base)
            elif self.information == "gini":
                information_score = gini_impurity(probs)
                    
            expected_utility = np.max(self.utility_matrix @ probs.T)
            self.explanations.append(decision_making_explanation_form(set(e), probs, expected_utility, information_score, self.actions[np.argmax(self.utility_matrix @ probs.T)]))
                    
        self.explanations.sort(
            reverse=True, 
            key=lambda e: (
                e["utility score"], 
                -e["information score"], 
                len(e["features"])
            )
        )

        return self.extract_explanation()

    def explain_instance_with_rationality(self, x_instance, minimalism=False, cognitive_method:Sequence[Literal["quantity", "time"]]=("quantity",), **kwargs):
        max_k_features = self.max_features
        information_bound = self.information_bound
        if "quantity" in cognitive_method:
            assert "max_k_features" in kwargs, "Invalid call"
            max_k_features = kwargs["max_k_features"]
        if "time" in cognitive_method:
            assert "allow_rational_time" in kwargs and "observation_time" in kwargs, "Invalid call"
            if "reflexion_time" in kwargs:
                information_bound = entropy_by_hick(kwargs["allow_rational_time"], kwargs["observation_time"], reflexion_time="reflexion_time")
            else:
                information_bound = entropy_by_hick(kwargs["allow_rational_time"], kwargs["observation_time"])
        explanation_combination = get_combinations_up_to_k(self.critical_features, max_k_features)
        self.explanations = []
        for e in explanation_combination:
            standard_e = list(e)[:]
            for f in e:
                if f in self.ohe_group:
                    standard_e.remove(f)
                    standard_e += self.ohe_group[f]
            probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, list(standard_e))
            
            information_score = 0
            if self.information == "shannon":
                information_score = shannon_entropy(probs, base=self.base)
                if information_score > information_bound: continue
            elif self.information == "gini":
                information_score = gini_impurity(probs)
                if information_score > information_bound: continue
                
            expected_utility = np.max(self.utility_matrix @ probs.T)
            self.explanations.append(decision_making_explanation_form(set(e), probs, expected_utility, information_score, self.actions[np.argmax(self.utility_matrix @ probs.T)]))
            
        self.explanations.sort(
            reverse=True, 
            key=lambda e: (
                e["utility score"], 
                -e["information score"], 
                -len(e["features"]) if minimalism else len(e["features"])
            )
        )

    def extract_attribution(self, x_instance):
        self.attributions = []
        for f in self.features:
            probs = None
            if f in self.critical_features:
                if f in self.ohe_group:
                    probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, self.ohe_group[f])
                else:
                    probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, [f])
            else:
                probs = self.model.predict_proba(x_instance)[0]
            information_score = 0
            if self.information == "shannon":
                information_score = shannon_entropy(probs, base=self.base)
            elif self.information == "gini":
                information_score = gini_impurity(probs)
            self.attributions.append(decision_making_explanation_form(f, probs, np.max(self.utility_matrix @ probs.T), information_score, self.actions[np.argmax(self.utility_matrix @ probs.T)]))
        self.attributions.sort(
            reverse=True, 
            key=lambda e: (
                e["utility score"], 
                -e["information score"]
            )
        )
        return self.attributions

    def extract_explanation(self):
        return self.explanations

class UtilityAlignedImageExplainer(UtilityAlignedExplainer):
    pass