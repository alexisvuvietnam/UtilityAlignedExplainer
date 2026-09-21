import copy
from explainer import *
import matplotlib.cm as cm
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

def generate_initial_utility_matrix(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    U11, U12, U21, U22 = np.random.uniform(-10, 10, size=4)
    return np.array([[U11, U12], [U21, U22]])

def generate_utility_matrix(category='crossing', seed=None, decided_label=None):
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

def generate_utility_matrix_priotizing_true(category='action_dominates', seed=None, true_label=0):
    if seed is not None:
        np.random.seed(seed)
        
    U21, U22 = np.random.uniform(-10, 10, size=2)
    if true_label == 0:
        while U21 <= U22:
            U21, U22 = np.random.uniform(-10, 10, size=2)
    else:
        while U21 >= U22:
            U21, U22 = np.random.uniform(-10, 10, size=2)
    
    if category == 'action_dominates':
        if true_label == 0:
            A = np.random.uniform(0.1, 5.0)
            B = np.random.uniform(0.0, A)
        else:
            B = np.random.uniform(0.1, 5.0)
            A = np.random.uniform(0.0, B)
    elif category == 'indifference':
        A = 0.0
        B = 0.0
    else:
        raise ValueError("Invalid category")
        
    U11 = U21 + A
    U12 = U22 + B
    
    return np.array([[U11, U12], [U21, U22]])

def generate_utility_matrix_priotizing_false(category='crossing', seed=None, true_label=0):
    if seed is not None:
        np.random.seed(seed)
        
    U21, U22 = np.random.uniform(-10, 10, size=2)
    if true_label == 0:
        while U21 >= U22:
            U21, U22 = np.random.uniform(-10, 10, size=2)
    else:
        while U21 <= U22:
            U21, U22 = np.random.uniform(-10, 10, size=2)
    
    if category == 'action_dominates':
        if true_label == 1:
            A = np.random.uniform(0.1, 5.0)
            B = np.random.uniform(0.0, A)
        else:
            B = np.random.uniform(0.1, 5.0)
            A = np.random.uniform(0.0, B)
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

    def _format_inputs(self, x_instances, categories=None):
        if isinstance(x_instances, pd.DataFrame):
            x_list = [row for _, row in x_instances.iterrows()]
        elif isinstance(x_instances, np.ndarray):
            x_list = list(x_instances) if x_instances.ndim > 1 else [x_instances]
        elif isinstance(x_instances, list):
            x_list = x_instances if (len(x_instances) > 0 and isinstance(x_instances[0], (list, np.ndarray, pd.Series))) else [x_instances]
        else:
            x_list = [x_instances]
            
        if categories is not None:
            cat_list = categories if isinstance(categories, list) else [categories]
            return x_list, cat_list
        return x_list

    def test_explain_instance_phase_1(self, x_instances, num_variants=100, display=False):
        x_list = self._format_inputs(x_instances)
        feature_ranks = dict()
        
        for c_idx, x_inst in enumerate(x_list):
            feature_ranks[c_idx] = dict()
            for i in range(num_variants):
                utility_matrix = generate_initial_utility_matrix(seed=i)
                explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, utility_matrix)
                explanation = explainer.explain_instance(x_inst)
                
                rank_idx = 0
                for feature_info in explanation:
                    rank_idx += 1
                    feature_name = feature_info['features']
                    if feature_name not in feature_ranks[c_idx]:
                        feature_ranks[c_idx][feature_name] = []
                    feature_ranks[c_idx][feature_name].append(rank_idx)
                    
        if display:
            fig, axs = plt.subplots(nrows=1, ncols=len(x_list), figsize=(8 * len(x_list), 6), squeeze=False)
            runs = range(1, num_variants + 1)
            cmap = cm.get_cmap('tab20')
            color_map = {feat: cmap(idx % 20) for idx, feat in enumerate(self.features)}
            
            for c_idx in range(len(x_list)):
                ax = axs[0, c_idx]
                for feature, ranks in feature_ranks[c_idx].items():
                    ax.scatter(runs, ranks, label=feature, color=color_map.get(feature, 'black'), alpha=0.8, linewidth=2)
                ax.set_title(f"Instance {c_idx}")
                ax.set_xlabel("Variant (runs)")
                ax.set_ylabel("Rank")
                ax.invert_yaxis()
                ax.grid(True, linestyle='--', alpha=0.6)
                max_rank = max([max(r) for r in feature_ranks[c_idx].values()]) if feature_ranks[c_idx] else 10
                ax.set_yticks(range(1, max_rank + 1))
                
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.95), loc='upper left')
            plt.tight_layout()
            plt.show()
            
        return feature_ranks

    def test_explain_instance_phase_2(self, x_instances, num_variants=100, category='crossing', display=False):
        x_list, cat_list = self._format_inputs(x_instances, category)
        feature_ranks = dict()
        
        for cat in cat_list:
            feature_ranks[cat] = dict()
            for c_idx, x_inst in enumerate(x_list):
                feature_ranks[cat][c_idx] = dict()
                try:
                    true_label = self.model.predict(x_inst)[0]
                except:
                    true_label = self.model.predict(np.array([x_inst]))[0]
                    
                for i in range(num_variants):
                    utility_matrix = generate_utility_matrix(category=cat, seed=i, decided_label=true_label)
                    explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, utility_matrix)
                    explanation = explainer.explain_instance(x_inst)
                    
                    rank_idx = 0
                    for feature_info in explanation:
                        rank_idx += 1
                        feature_name = feature_info['features']
                        if feature_name not in feature_ranks[cat][c_idx]:
                            feature_ranks[cat][c_idx][feature_name] = []
                        feature_ranks[cat][c_idx][feature_name].append(rank_idx)
                        
        if display:
            fig, axs = plt.subplots(nrows=len(cat_list), ncols=len(x_list), figsize=(8 * len(x_list), 6 * len(cat_list)), squeeze=False)
            runs = range(1, num_variants + 1)
            cmap = cm.get_cmap('tab20')
            color_map = {feat: cmap(idx % 20) for idx, feat in enumerate(self.features)}
            
            for r_idx, cat in enumerate(cat_list):
                for c_idx in range(len(x_list)):
                    ax = axs[r_idx, c_idx]
                    for feature, ranks in feature_ranks[cat][c_idx].items():
                        ax.scatter(runs, ranks, label=feature, color=color_map.get(feature, 'black'), alpha=0.8, linewidth=2)
                    ax.set_title(f"Instance {c_idx} | {cat}")
                    ax.set_xlabel("Variant (runs)")
                    ax.set_ylabel("Rank")
                    ax.invert_yaxis()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    max_rank = max([max(r) for r in feature_ranks[cat][c_idx].values()]) if feature_ranks[cat][c_idx] else 10
                    ax.set_yticks(range(1, max_rank + 1))
                    
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.95), loc='upper left')
            plt.tight_layout()
            plt.show()
            
        return feature_ranks

    def test_explain_instance_phase_3(self, x_instances, num_variants=100, category='action_dominates', display=False):
        x_list, cat_list = self._format_inputs(x_instances, category)
        feature_ranks = dict()
        
        for cat in cat_list:
            feature_ranks[cat] = dict()
            for c_idx, x_inst in enumerate(x_list):
                feature_ranks[cat][c_idx] = {True: dict(), False: dict()}
                try:
                    true_label = self.model.predict(x_inst)[0]
                except:
                    true_label = self.model.predict(np.array([x_inst]))[0]
                    
                for i in range(num_variants):
                    utility_matrix_for_true = generate_utility_matrix_priotizing_true(category=cat, seed=i, true_label=true_label)
                    utility_matrix_for_false = generate_utility_matrix_priotizing_false(category=cat, seed=i, true_label=true_label)
                    
                    explainer_1 = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, utility_matrix_for_true)
                    explainer_2 = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, utility_matrix_for_false)
                    
                    explanation_1 = explainer_1.explain_instance(x_inst)
                    explanation_2 = explainer_2.explain_instance(x_inst)
                    
                    rank_idx = 0
                    for feature_info in explanation_1:
                        rank_idx += 1
                        feature_name = feature_info['features']
                        if feature_name not in feature_ranks[cat][c_idx][True]:
                            feature_ranks[cat][c_idx][True][feature_name] = []
                        feature_ranks[cat][c_idx][True][feature_name].append(rank_idx)
                        
                    rank_idx = 0
                    for feature_info in explanation_2:
                        rank_idx += 1
                        feature_name = feature_info['features']
                        if feature_name not in feature_ranks[cat][c_idx][False]:
                            feature_ranks[cat][c_idx][False][feature_name] = []
                        feature_ranks[cat][c_idx][False][feature_name].append(rank_idx)
                        
        if display:
            fig, axs = plt.subplots(nrows=len(cat_list) * len(x_list), ncols=2, figsize=(16, 6 * len(cat_list) * len(x_list)), squeeze=False)
            runs = range(1, num_variants + 1)
            cmap = plt.get_cmap('tab20')
            color_map = {feat: cmap(idx % 20) for idx, feat in enumerate(self.features)}
            
            for cat_idx, cat in enumerate(cat_list):
                for c_idx in range(len(x_list)):
                    r_idx = cat_idx * len(x_list) + c_idx
                    ax_true = axs[r_idx, 0]
                    ax_false = axs[r_idx, 1]
                    
                    for feature, ranks in feature_ranks[cat][c_idx][True].items():
                        ax_true.scatter(runs, ranks, label=feature, color=color_map.get(feature, 'black'), alpha=0.8, linewidth=2)
                    ax_true.set_title(f"Inst {c_idx} | {cat} | Prioritizing Predicted output")
                    ax_true.set_xlabel("Variant (runs)")
                    ax_true.set_ylabel("Rank")
                    ax_true.invert_yaxis()
                    ax_true.grid(True, linestyle='--', alpha=0.6)
                    
                    for feature, ranks in feature_ranks[cat][c_idx][False].items():
                        ax_false.scatter(runs, ranks, label=feature, color=color_map.get(feature, 'black'), alpha=0.8, linewidth=2)
                    ax_false.set_title(f"Inst {c_idx} | {cat} | Prioritizing Non-predicted output")
                    ax_false.set_xlabel("Variant (runs)")
                    ax_false.set_ylabel("Rank")
                    ax_false.invert_yaxis()
                    ax_false.grid(True, linestyle='--', alpha=0.6)
                    
                    max_t = max([max(r) for r in feature_ranks[cat][c_idx][True].values()]) if feature_ranks[cat][c_idx][True] else 10
                    max_f = max([max(r) for r in feature_ranks[cat][c_idx][False].values()]) if feature_ranks[cat][c_idx][False] else 10
                    max_rank = max(max_t, max_f)
                    
                    ax_true.set_yticks(range(1, max_rank + 1))
                    ax_false.set_yticks(range(1, max_rank + 1))
                    
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.95), loc='upper left')
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