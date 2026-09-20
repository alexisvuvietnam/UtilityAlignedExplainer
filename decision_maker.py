import copy
from explainer import *
import matplotlib.pyplot as plt
import lime
import numpy as np
import pandas as pd
import py_ciu_master.ciu as ciu
import random
from sklearn.base import BaseEstimator, clone
from scipy.stats import spearmanr
import seaborn as sns
import shap
from utils import *
import xgboost as xgb

def generate_utility_matrix(category='crossing', seed=None, decided_label=None, epsilon=1e-4):
    if seed is not None:
        np.random.seed(seed)
        
    U21, U22 = np.random.uniform(-10, 10, size=2)
    
    if category == 'action_strictly_dominates':
        A = np.random.uniform(0.1, 5.0)
        B = A + np.random.uniform(0.0, 5.0)
    elif category == 'action_weakly_dominates':
        if decided_label == 0:
            B = np.random.uniform(0.0, 5.0)
            A = B + np.random.uniform(0.1, 5.0)
        elif decided_label == 1:
            A = np.random.uniform(0.0, 5.0)
            B = A + np.random.uniform(0.1, 5.0)
        else:
            raise ValueError("Invalid parameters: decided_label must be 0 or 1 for 'action_weakly_dominates' category")
    elif category == 'crossing':
        A = np.random.uniform(-5.0, -0.1)
        B = np.random.uniform(0.1, 5.0)
    elif category == 'indifference':
        A = 0.0
        B = 0.0
    else:
        raise ValueError("Invalid category")
        
    U11 = U21 + A
    U12 = U22 + B
    
    return np.array([[U11, U12], [U21, U22]])

class ExplainerTester:

    def __init__(self, name, model, features, actions, causal_graph, X_train):
        self.name = name
        self.model = model
        self.features = features
        self.actions = actions
        self.causal_graph = causal_graph
        self.X_train = X_train

    def test_explain_instance(self, x_instance, num_variants=100, category='crossing', display=False):
        feature_ranks = dict()
        for i in range(num_variants):
            utility_matrix = generate_utility_matrix(category=category, seed=i, decided_label=self.model.predict(x_instance)[0])
            print(utility_matrix)
            explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, utility_matrix)
            explanation = explainer.explain_instance(x_instance)
            i = 0
            for feature_info in explanation:
                i += 1
                feature_name = feature_info['features']
                if feature_name not in feature_ranks:
                    feature_ranks[feature_name] = []
                feature_ranks[feature_name].append(i)
        if display:
            plt.figure(figsize=(12, 6))
            runs = range(1, num_variants + 1)
            
            for feature, ranks in feature_ranks.items():
                plt.scatter(runs, ranks, label=feature, alpha=0.8, linewidth=2)
            
            plt.title(f"Feature Rank Variation across {num_variants} Variants ({category})")
            plt.xlabel("Variant (runs)")
            plt.ylabel("Rank")
            
            plt.gca().invert_yaxis()
            
            max_rank = max([max(r) for r in feature_ranks.values()]) if feature_ranks else 10
            plt.yticks(range(1, max_rank + 1))
            
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        return feature_ranks

class BaselineTester:

    def __init__(self, name, model, features, actions, causal_graph, utility_matrix, X_train, y_train, X_val, y_val, X_test, lime_explainers, shap_explainer, random_seed=42, n_samples=100):
        self.name = name
        self.model = model
        self.features = features
        self.actions = actions
        self.causal_graph = causal_graph
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.lime_explainers = lime_explainers
        self.shap_explainer = shap_explainer
        self.ciu_explainer = ciu.CIU(self.model.predict_proba, self.model.classes_, data=self.X_train, input_names=self.features, output_inds=self.model.classes_.tolist())
        self.causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, self.utility_matrix)
        self.random_seed = random_seed
        self.num_instance = min(len(self.X_test), n_samples)