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

def decision_making_explanation_form(features, parents, probability, expected_utility, action):
    return {"features": features, "parents": parents, "prediction probability": probability, "utility score": expected_utility, "preferrable action": action}

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

def top_k_lime_explanation_form(model, lime_explainer, x_instance_1d, importance=True, top_k=5):
    assert top_k <= len(x_instance_1d.index.tolist()), "The number of features extracted is invalid"
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
    return attribution[:top_k]

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

def top_k_shap_explanation_form(model, shap_explainer, x_instance_2d, importance=True, top_k=5):
    if isinstance(x_instance_2d, pd.DataFrame):
        num_features = len(x_instance_2d.columns)
    else:
        num_features = x_instance_2d.shape[1]
    assert top_k <= num_features, "The number of features extracted is invalid"
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
    return attribution[:top_k]

class UtilityAlignedExplainer(ABC):
    def __init__(self, model, X_train, *, actions, causal_model:CausalModel, utility_matrix, **kwargs):
        assert set(actions) == causal_model.outcomes, "Invalid causal model"
        self.model = model
        self.actions = np.array(actions)
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.X_train = X_train

    @abstractmethod
    def explain_instance(self, x_instance, **kwargs):
        pass

class UtilityAlignedTabularExplainer(UtilityAlignedExplainer):
    def __init__(self, model, X_train, features, actions, causal_model:CausalModel, utility_matrix, base=np.e, ohe_group=None, **kwargs):
        
        super().__init__(
            model,
            X_train, 
            actions=actions, 
            causal_model=causal_model, 
            utility_matrix=utility_matrix,
            base=base
        )
        
        self.features = features
        self.ohe_group = ohe_group if ohe_group is not None else {}

        # Information and cognitive constraints
        self.max_features = len(features)

        # Global extraction
        self.critical_features = set()
        for f in self.features:
            if self.causal_model.causal_validity(f):
                self.critical_features.add(f)

    def explain_instance(self, x_instance):
        attributions = []
        for f in self.features:
            probs = None
            if f in self.critical_features:
                if f in self.ohe_group:
                    probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, self.ohe_group[f])
                else:
                    probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, [f])
                possible_actions = self.causal_model.possible_actions({f})
                element_mask = np.isin(self.actions, list(possible_actions))
                action_indices = np.where(element_mask)[0]
                expected_utility = np.max(self.utility_matrix[action_indices] @ probs.T)
                attributions.append(decision_making_explanation_form(f, self.causal_model.extract_parents(f), probs, expected_utility, self.actions[np.argmax(self.utility_matrix[action_indices] @ probs.T)]))    
        attributions.sort(
            reverse=True,
            key=lambda e: e["utility score"]
        )
        return attributions

    def explain_instance_optimal_combination(self, x_instance):
        explanations = []
        if len(self.critical_features) == 0:
            probs = self.model.predict_proba(x_instance.values.reshape(1, -1))[0]
            expected_utility = np.max(self.utility_matrix @ probs.T)
            preferrable_action = self.actions[np.argmax(self.utility_matrix @ probs.T)]
            explanations = [decision_making_explanation_form(set(), set(), probs, expected_utility, preferrable_action)]
            return explanations
        explanation_combination = get_combinations_up_to_k(self.critical_features, self.max_features)
        for e in explanation_combination:
            parents = set()
            standard_e = list(e)[:]
            for f in e:
                if f in self.ohe_group:
                    standard_e.remove(f)
                    standard_e += self.ohe_group[f]
                parents = parents | self.causal_model.extract_parents(f)
            probs = estimate_interventional_probability_tabular(self.model, self.X_train, x_instance, list(standard_e))
            possible_actions = self.causal_model.possible_actions(e)
            element_mask = np.isin(self.actions, list(possible_actions))
            action_indices = np.where(element_mask)[0]
            expected_utility = np.max(self.utility_matrix[action_indices] @ probs.T)
            explanations.append(decision_making_explanation_form(set(e), parents, probs, expected_utility, self.actions[np.argmax(self.utility_matrix[action_indices] @ probs.T)]))
        explanations.sort(
            reverse=True,
            key=lambda e: e["utility score"]
        )
        return explanations

class UtilityAlignedTabularExplainerWithInformation(UtilityAlignedTabularExplainer):
    pass

class UtilityAlignedImageExplainer(UtilityAlignedExplainer):
    pass