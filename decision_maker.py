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
            plt.savefig("rq1_1.svg", format="svg", bbox_inches="tight")
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
            plt.savefig("rq1_2.svg", format="svg", bbox_inches="tight")
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
            plt.savefig("rq1_3.svg", format="svg", bbox_inches="tight")
            plt.show()
            
        return feature_ranks

class BaselineTester:

    def __init__(self, name, model, features, classes, actions, causal_graph, X_train, y_train, X_val, y_val, X_test, lime_seeds=[0, 1, 2], random_seed=42, n_samples=100):
        self.name = name
        self.model = model
        self.features = features
        self.classes = classes
        self.actions = actions
        self.causal_graph = causal_graph
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.lime_explainers = dict()
        for seed in lime_seeds:
            self.lime_explainers[seed] = lime.lime_tabular.LimeTabularExplainer(training_data=self.X_train.values, feature_names=self.features, class_names=self.classes, random_state=seed)
        self.shap_explainer = shap.Explainer(self.model.predict, X_train)
        self.ciu_explainer = ciu.CIU(self.model.predict_proba, self.classes, data=self.X_train, input_names=self.features, output_inds=self.classes)
        self.biased_utility_matrix = np.array([[100, 50]])
        self.neutral_utility_matrix = np.array([[100, 100]])
        self.biased_causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, self.biased_utility_matrix)
        self.neutral_causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, self.neutral_utility_matrix)
        self.random_seed = random_seed
        self.num_instances = min(len(self.X_test), n_samples)
        self.explanation_types  = ["Biased-utility Utility-aligned", "Neutral-utility Utility-aligned", "SHAP", "CI-based CIU", "CU-based CIU", "Influence-based CIU"] + [f"Seed-{s} LIME" for s in lime_seeds]

    def _format_number_list(self, number):
        if isinstance(number, int) or isinstance(number, float):
            return [number]
        
        return number

    def _explain_instances_feature(self, instance_1d, instance_2d):
        explanations = {etype: [] for etype in self.explanation_types}

        biased_explanation = self.biased_causal_explainer.explain_instance(instance_2d)
        for feature_info in biased_explanation:
            feature_name = feature_info['features']
            explanations["Biased-utility Utility-aligned"].append(feature_name)

        neutral_explanation = self.neutral_causal_explainer.explain_instance(instance_2d)
        for feature_info in neutral_explanation:
            feature_name = feature_info['features']
            explanations["Neutral-utility Utility-aligned"].append(feature_name)

        shap_exp = shap_explanation_form(self.model, self.shap_explainer, instance_2d)
        for feature_info in shap_exp:
            feature_name = feature_info['feature']
            explanations["SHAP"].append(feature_name)

        ci_exp = ci_based_CIU_explanation_form(self.ciu_explainer, instance_2d, [np.argmax(self.model.predict_proba(instance_2d)[0])])
        for feature_info in ci_exp:
            feature_name = feature_info['feature']
            explanations["CI-based CIU"].append(feature_name)

        cu_exp = cu_based_CIU_explanation_form(self.ciu_explainer, instance_2d, [np.argmax(self.model.predict_proba(instance_2d)[0])])
        for feature_info in cu_exp:
            feature_name = feature_info['feature']
            explanations["CU-based CIU"].append(feature_name)

        influence_exp = influence_based_CIU_explanation_form(self.ciu_explainer, instance_2d, [np.argmax(self.model.predict_proba(instance_2d)[0])])
        for feature_info in influence_exp:
            feature_name = feature_info['feature']
            explanations["Influence-based CIU"].append(feature_name)

        for seed, lime_explainer in self.lime_explainers.items():
            lime_exp = lime_explanation_form(self.model, lime_explainer, instance_1d)
            for feature_info in lime_exp:
                feature_name = feature_info['feature']
                explanations[f"Seed-{seed} LIME"].append(feature_name)

        return explanations

    def model_sensitivity_test(self, display=False):
        if isinstance(self.model, xgb.XGBClassifier):
            corrupted_model = xgb.XGBClassifier()
            corrupted_model.set_params(**self.model.get_params())
        else:
            corrupted_model = clone(self.model)
        
        np.random.seed(self.random_seed)
        shuffled_y_train = np.random.permutation(self.y_train)
                
        if isinstance(corrupted_model, xgb.XGBClassifier):
            corrupted_model.fit(self.X_train, shuffled_y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        else:
            corrupted_model.fit(self.X_train, shuffled_y_train)

        corrupted_lime_explainers = dict()
        for seed in self.lime_explainers.keys():
            corrupted_lime_explainers[seed] = lime.lime_tabular.LimeTabularExplainer(training_data=self.X_train.values, feature_names=self.features, class_names=self.classes, random_state=seed)

        corrupted_shap_explainer = shap.Explainer(corrupted_model.predict, self.X_train)
        corrupted_ciu_explainer = ciu.CIU(corrupted_model.predict_proba, self.classes, data=self.X_train, input_names=self.features, output_inds=self.classes)
        corrupted_biased_causal_explainer = UtilityAlignedTabularExplainer(corrupted_model, self.X_train, self.features, self.actions, self.causal_graph, self.biased_utility_matrix)
        corrupted_neutral_causal_explainer = UtilityAlignedTabularExplainer(corrupted_model, self.X_train, self.features, self.actions, self.causal_graph, self.neutral_utility_matrix)

        all_datas = dict()
        for explainer in self.explanation_types:
            all_datas[explainer] = []
        
        for i in range(self.num_instances):
            corrupted_explanations = {etype: [] for etype in self.explanation_types}
            instance_1d = self.X_test.iloc[i]
            instance_2d = self.X_test.iloc[[i]]
            
            original = self._explain_instances_feature(instance_1d, instance_2d)
            
            biased_explanation = corrupted_biased_causal_explainer.explain_instance(instance_1d)
            for feature_info in biased_explanation:
                feature_name = feature_info['features']
                corrupted_explanations["Biased-utility Utility-aligned"].append(feature_name)

            all_datas["Biased-utility Utility-aligned"].append(spearman_similarity(original["Biased-utility Utility-aligned"], corrupted_explanations["Biased-utility Utility-aligned"]))
            
            neutral_explanation = corrupted_neutral_causal_explainer.explain_instance(instance_1d)
            for feature_info in neutral_explanation:
                feature_name = feature_info['features']
                corrupted_explanations["Neutral-utility Utility-aligned"].append(feature_name)

            all_datas["Neutral-utility Utility-aligned"].append(spearman_similarity(original["Neutral-utility Utility-aligned"], corrupted_explanations["Neutral-utility Utility-aligned"]))
            
            shap_exp = shap_explanation_form(corrupted_model, corrupted_shap_explainer, instance_2d)
            for feature_info in shap_exp:
                feature_name = feature_info['feature']
                corrupted_explanations["SHAP"].append(feature_name)

            all_datas["SHAP"].append(spearman_similarity(original["SHAP"], corrupted_explanations["SHAP"]))
            
            ci_exp = ci_based_CIU_explanation_form(corrupted_ciu_explainer, instance_2d, [np.argmax(corrupted_model.predict_proba(instance_2d)[0])])
            for feature_info in ci_exp:
                feature_name = feature_info['feature']
                corrupted_explanations["CI-based CIU"].append(feature_name)

            all_datas["CI-based CIU"].append(spearman_similarity(original["CI-based CIU"], corrupted_explanations["CI-based CIU"]))
            
            cu_exp = cu_based_CIU_explanation_form(corrupted_ciu_explainer, instance_2d, [np.argmax(corrupted_model.predict_proba(instance_2d)[0])])
            for feature_info in cu_exp:
                feature_name = feature_info['feature']
                corrupted_explanations["CU-based CIU"].append(feature_name)

            all_datas["CU-based CIU"].append(spearman_similarity(original["CU-based CIU"], corrupted_explanations["CU-based CIU"]))
        
            influence_exp = influence_based_CIU_explanation_form(corrupted_ciu_explainer, instance_2d, [np.argmax(corrupted_model.predict_proba(instance_2d)[0])])
            for feature_info in influence_exp:
                feature_name = feature_info['feature']
                corrupted_explanations["Influence-based CIU"].append(feature_name)

            all_datas["Influence-based CIU"].append(spearman_similarity(original["Influence-based CIU"], corrupted_explanations["Influence-based CIU"]))

            for seed, lime_explainer in self.lime_explainers.items():
                lime_exp = lime_explanation_form(corrupted_model, lime_explainer, instance_1d)
                for feature_info in lime_exp:
                    feature_name = feature_info['feature']
                    corrupted_explanations[f"Seed-{seed} LIME"].append(feature_name)

                all_datas[f"Seed-{seed} LIME"].append(spearman_similarity(original[f"Seed-{seed} LIME"], corrupted_explanations[f"Seed-{seed} LIME"]))

        if display:
            df_plot = pd.DataFrame(all_datas)
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_plot, palette="Set3", showfliers=False)
            sns.stripplot(data=df_plot, color="black", alpha=0.4, jitter=True, size=3)
            plt.title("Model Sensitivity Test: Sanity Checks over Feature Rankings")
            plt.ylabel("Spearman Similarity")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("rq2_1.svg", format="svg", bbox_inches="tight")
            plt.show()

        return all_datas

    def descriptive_faithfulness_test(self, top_k=5, display=False):
        assert top_k < len(self.biased_causal_explainer.critical_features), "top_k should be less than the number of critical features in the causal graph."
        all_datas = dict()
        for explainer in self.explanation_types:
            all_datas[explainer] = []

        for i in range(self.num_instances):
            instance_1d = self.X_test.iloc[i]
            instance_2d = self.X_test.iloc[[i]]
            original = self._explain_instances_feature(instance_1d, instance_2d)

            for explainer in self.explanation_types:
                explanation = original[explainer][:top_k]
                all_datas[explainer].append(top_k_RoAR(self.model, instance_2d, explanation, self.X_train, self.y_train, self.X_val, self.y_val).tolist())

        if display:
            plot_data = []
            for explainer, drop_curves in all_datas.items():
                for i, drop_curve in enumerate(drop_curves):
                    for k_idx, drop_val in enumerate(drop_curve):
                        plot_data.append({
                            "Explainer": explainer,
                            "Instance": i,
                            "Number of Features Removed (k)": k_idx + 1,
                            "RoAR Metric (Performance Degradation)": drop_val
                        })
            
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(14, 7))
            
            sns.lineplot(
                data=df_plot, 
                x="Number of Features Removed (k)", 
                y="RoAR Metric (Performance Degradation)", 
                hue="Explainer", 
                marker="o", 
                linewidth=2,
                errorbar='sd',
                err_kws={'alpha': 0.1}
            )
            
            plt.title(f"Descriptive Faithfulness Test: RoAR Performance Drop\n(Top-{top_k} Features Removed)", fontsize=14)
            plt.xlabel("Number of Features Removed (k)", fontsize=12) 
            plt.ylabel("Average Performance Drop", fontsize=12)
            plt.xticks(range(1, top_k + 1))
            plt.legend(title="Explainers", bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.grid(True, axis='x', linestyle=':', alpha=0.4)
            plt.tight_layout()
            plt.savefig("rq2_2.svg", format="svg", bbox_inches="tight")
            plt.show()

        return all_datas
        
    def do_all_test(self, top_k=5, display=False):
        data1 = self.model_sensitivity_test(display=display)
        data2 = self.descriptive_faithfulness_test(top_k=top_k, display=display)
        return {
            "model sensitivity": data1,
            "descriptive faithfulness": data2,
        }

class DecisionMaker:
    def __init__(self, name, model, features, actions, causal_graph, utility_matrix, X_train, y_train, X_test, random_seed=42):
        self.name = name
        self.model = model
        self.features = features
        self.actions = actions
        self.causal_graph = causal_graph
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.utility_matrix = utility_matrix
        self.causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, self.utility_matrix)
        self.random_seed = random_seed


    def decision_utility(self, top_k=5, n_samples=100, display=False):
        assert top_k < len(self.causal_explainer.critical_features), "top_k should be less than the number of critical features in the causal graph."
        assert n_samples <= len(self.X_test), "number of samples should fit the size of test dataset"
        filted_instances = self.X_test.sample(n=n_samples, random_state=self.random_seed)
        
        absolute_results = []
        delta_results = []
        delta_results_2 = []
        
        for i in range(n_samples):
            instance_2d = filted_instances.iloc[[i]]

            explanation = self.causal_explainer.explain_instance(instance_2d)
            features_list = [e["features"] for e in explanation]

            probs_0 = estimate_interventional_probability_tabular(self.model, self.X_train, instance_2d, [])
            u_0 = np.max(self.utility_matrix @ probs_0.T)

            tmp_abs = []
            tmp_delta = []
            tmp_delta_2 = []
            
            for k in range(1, top_k + 1):
                top_k_explanation = features_list[:k]
                subsets = get_combinations_up_to_k(top_k_explanation, k)
                utility_list = []
                for subset in subsets:
                    probs = estimate_interventional_probability_tabular(self.model, self.X_train, instance_2d, list(subset))
                    utility_list.append(np.max(self.utility_matrix @ probs.T))
                
                max_u_k = max(utility_list)
                if len(tmp_abs) > 0:
                    tmp_delta_2.append(max_u_k - tmp_abs[-1])
                #else:
                    #tmp_delta_2.append(max_u_k)
                tmp_abs.append(max_u_k)
                tmp_delta.append(max_u_k - u_0) 
                
            absolute_results.append(tmp_abs)
            delta_results.append(tmp_delta)
            delta_results_2.append(tmp_delta_2)

        if display:
            abs_arr = np.array(absolute_results)
            delta_arr = np.array(delta_results)
            delta_arr_2 = np.array(delta_results_2)
            
            mean_abs = np.mean(abs_arr, axis=0)
            std_abs = np.std(abs_arr, axis=0)
            
            mean_delta = np.mean(delta_arr, axis=0)
            #std_delta = np.std(delta_arr, axis=0)

            mean_delta_2 = np.mean(delta_arr_2, axis=0)
            
            k_values = range(1, top_k + 1)
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            
            axes[0].plot(k_values, mean_abs, marker='o', linewidth=2, color='#1f77b4')
            axes[0].fill_between(k_values, mean_abs - std_abs, mean_abs + std_abs, color='#1f77b4', alpha=0.15)
            axes[0].set_title('Average Absolute Decision Utility', fontsize=14)
            axes[0].set_xlabel('Number of Features (k)', fontsize=12)
            axes[0].set_ylabel('Expected Utility', fontsize=12)
            axes[0].set_xticks(k_values)
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            axes[1].plot(k_values, mean_delta, marker='s', linewidth=2, color='#ff7f0e')
            #axes[1].fill_between(k_values, mean_delta - std_delta, mean_delta + std_delta, color='#ff7f0e', alpha=0.15)
            axes[1].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6) 
            axes[1].set_title(r'Average $\Delta$ Decision Utility ($U_k - U_0$)', fontsize=14)
            axes[1].set_xlabel('Number of Features (k)', fontsize=12)
            axes[1].set_ylabel(r'$\Delta$ Expected Utility', fontsize=12)
            axes[1].set_xticks(k_values)
            axes[1].grid(True, linestyle='--', alpha=0.6)

            axes[2].plot(range(2, top_k + 1), mean_delta_2, marker='x', linewidth=2, color="#ff0ee7")
            #axes[1].fill_between(k_values, mean_delta - std_delta, mean_delta + std_delta, color='#ff7f0e', alpha=0.15)
            axes[2].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6) 
            axes[2].set_title(r'Average $\Delta$ Decision Utility ($U_k - U_{k-1}$)', fontsize=14)
            axes[2].set_xlabel('Number of Features (k)', fontsize=12)
            axes[2].set_ylabel(r'$\Delta$ Expected Utility', fontsize=12)
            axes[2].set_xticks(range(2, top_k + 1))
            axes[2].grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig("rq3.svg", format="svg", bbox_inches="tight")
            plt.show()

        return absolute_results, delta_results


