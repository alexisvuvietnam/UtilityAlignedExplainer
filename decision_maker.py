import copy
from explainer import *
import matplotlib.pyplot as plt
import lime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from scipy.stats import spearmanr
import seaborn as sns
import shap
from utils import *
import xgboost as xgb

class ExplainerTester:

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
        self.causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_graph, self.utility_matrix)
        self.random_seed = random_seed
        self.num_instance = min(len(self.X_test), n_samples)

    def fidelity_test(self, display=False):
        exp_size = len(self.causal_explainer.critical_features)
        if exp_size == 0: exp_size = 1
        #baseline_vector = self.X_train.median()
        
        causal_fid = {"RoAR Score": [], "RoAR Trending": []}
        lime_fids = {f"LIME (Seed: {seed})": {"RoAR Score": [], "RoAR Trending": []} for seed in self.lime_explainers}
        shap_fid = {"RoAR Score": [], "RoAR Trending": []}

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
        
        if isinstance(corrupted_model, xgb.XGBClassifier):
            corrupted_shap_explainer = shap.Explainer(corrupted_model.predict, self.X_train)
        else:
            corrupted_shap_explainer = shap.Explainer(corrupted_model)
        
        corrupted_causal_explainer = UtilityAlignedTabularExplainer(
            corrupted_model, 
            self.X_train, 
            self.features, 
            self.actions, 
            self.causal_graph, 
            self.utility_matrix
        )
        
        causal_sanity = {"Top-k Jaccard": [], "Spearman": []}
        lime_sanity = {f"LIME (Seed: {seed})": {"Top-k Jaccard": [], "Spearman": []} for seed in self.lime_explainers}
        shap_sanity = {"Top-k Jaccard": [], "Spearman": []}

        #causal_sanity = []
        #lime_sanity = {f"LIME (Seed: {seed})": [] for seed in self.lime_explainers}
        #shap_sanity = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]
                
            c_exp = self.causal_explainer.explain_instance(inst_1d)
            c_f = [v["features"] for v in c_exp]
            c_s = [v["utility score"] for v in c_exp]

            roar_score, roar_cumul = RoAR(self.model, inst_2d, c_f[:exp_size], c_s[:exp_size], self.X_train, self.y_train, self.X_val, self.y_val)
            
            causal_fid["RoAR Score"].append(roar_score)
            causal_fid["RoAR Trending"].append(roar_cumul)

            s_exp = shap_explanation_form(self.model, self.shap_explainer, inst_2d) 
            s_f = np.array([v["feature"] for v in s_exp])
            s_s = np.array([v["absolute score"] for v in s_exp])

            roar_score, roar_cumul = RoAR(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], self.X_train, self.y_train, self.X_val, self.y_val)
            
            shap_fid["RoAR Score"].append(roar_score)
            shap_fid["RoAR Trending"].append(roar_cumul)

            l_f = dict()
            l_s = dict()

            for seed, lime_explainer in self.lime_explainers.items():
                l_exp = lime_explanation_form(self.model, lime_explainer, inst_1d)
                l_f[seed] = np.array([v["feature"] for v in l_exp])
                l_s[seed] = np.array([v["absolute score"] for v in l_exp])

                roar_score, roar_cumul = RoAR(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], self.X_train, self.y_train, self.X_val, self.y_val)
                
                lime_fids[f"LIME (Seed: {seed})"]["RoAR Score"].append(roar_score)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR Trending"].append(roar_cumul)

            corr_c_exp = corrupted_causal_explainer.explain_instance(inst_1d)
            corr_c_f = [v["features"] for v in corr_c_exp]

            corr_s_exp = shap_explanation_form(corrupted_model, corrupted_shap_explainer, inst_2d)
            corr_s_f = [v["feature"] for v in corr_s_exp]

            corr_l_f = dict()
            for seed, explainer in self.lime_explainers.items():
                corr_l_exp = lime_explanation_form(corrupted_model, explainer, inst_1d)
                corr_l_f[seed] = [v["feature"] for v in corr_l_exp]

            causal_sanity["Top-k Jaccard"].append(jaccard_similarity(list(c_f)[:exp_size - 1], corr_c_f[:exp_size - 1]))
            shap_sanity["Top-k Jaccard"].append(jaccard_similarity(list(s_f)[:exp_size - 1], corr_s_f[:exp_size - 1]))
            
            for seed in self.lime_explainers:
                lime_sanity[f"LIME (Seed: {seed})"]["Top-k Jaccard"].append(jaccard_similarity(list(l_f[seed])[:exp_size - 1], corr_l_f[seed][:exp_size - 1]))

            causal_sanity["Spearman"].append(spearman_similarity(list(c_f), corr_c_f))
            shap_sanity["Spearman"].append(spearman_similarity(list(s_f), corr_s_f))

            for seed in self.lime_explainers:
                lime_sanity[f"LIME (Seed: {seed})"]["Spearman"].append(spearman_similarity(list(l_f[seed]), corr_l_f[seed]))
                
        if display:

            #data_list = []
            #for i in range(self.num_instance):
            #    for label, lime_fid in lime_fids.items():
            #        data_list.append({"Explainer": label, "Score": lime_fid["RoAR Score"][i]})
            #    data_list.append({"Explainer": "SHAP", "Score": shap_fid["RoAR Score"][i]})
            #    data_list.append({"Explainer": "Utility-aligned", "Score": causal_fid["RoAR Score"][i]})
            #df_roar_box = pd.DataFrame(data_list)
            
            trend_data = []
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                for label, lime_fid in lime_fids.items():
                    mean_val = np.mean(lime_fid["RoAR Trending"], axis=0)[k_idx]
                    trend_data.append({"Explainer": label, "k": k_val, "Mean Score": mean_val})
                
                mean_shap = np.mean(shap_fid["RoAR Trending"], axis=0)[k_idx]
                trend_data.append({"Explainer": "SHAP", "k": k_val, "Mean Score": mean_shap})
                
                mean_causal = np.mean(causal_fid["RoAR Trending"], axis=0)[k_idx]
                trend_data.append({"Explainer": "Utility-aligned", "k": k_val, "Mean Score": mean_causal})
            df_roar_trend = pd.DataFrame(trend_data)

            sanity_list = []
            for i in range(self.num_instance):
                # LIME
                for seed, lime_san in lime_sanity.items():
                    sanity_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Top-k Jaccard", "Score": lime_san["Top-k Jaccard"][i]})
                    sanity_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": lime_san["Spearman"][i]})
                # SHAP
                sanity_list.append({"Explainer": "SHAP", "Metric": "Top-k Jaccard", "Score": shap_sanity["Top-k Jaccard"][i]})
                sanity_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_sanity["Spearman"][i]})
                # Causal
                sanity_list.append({"Explainer": "Utility-aligned", "Metric": "Top-k Jaccard", "Score": causal_sanity["Top-k Jaccard"][i]})
                sanity_list.append({"Explainer": "Utility-aligned", "Metric": "Spearman", "Score": causal_sanity["Spearman"][i]})
            
            df_sanity = pd.DataFrame(sanity_list)
            #df_jaccard = df_sanity[df_sanity["Metric"] == "Top-k Jaccard"]
            df_spearman = df_sanity[df_sanity["Metric"] == "Spearman"]

            #sns.set_theme(style="whitegrid")
            #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
            #fig.suptitle("Explainer Evaluation: Fidelity (RoAR) & Sanity Checks", fontsize=18, fontweight="bold")
            #
            #sns.boxplot(data=df_roar_box, x="Explainer", y="Score", hue="Explainer", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5, legend=False)
            #sns.stripplot(data=df_roar_box, x="Explainer", y="Score", hue="Explainer", ax=axes[0, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, legend=False)
            #axes[0, 0].set_title("RoAR Distribution (Higher is better / drop is sharper)", fontsize=13)
            #axes[0, 0].set_ylabel("Overall RoAR Score", fontsize=12)
            #axes[0, 0].set_xlabel("")
            #
            #sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[0, 1], palette="Set2")
            #axes[0, 1].set_title("Cumulative RoAR Trend", fontsize=13)
            #axes[0, 1].set_ylabel("Mean Cumulative RoAR", fontsize=12)
            #axes[0, 1].set_xlabel("Number of Masked Features (k)", fontsize=12)
            #axes[0, 1].set_xticks(range(1, exp_size + 1))
            #
            #sns.boxplot(data=df_jaccard, x="Explainer", y="Score", hue="Explainer", ax=axes[1, 0], palette="Pastel1", showfliers=False, width=0.5, legend=False)
            #sns.stripplot(data=df_jaccard, x="Explainer", y="Score", hue="Explainer", ax=axes[1, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, legend=False)
            #axes[1, 0].set_title("Sanity: Top-k Jaccard Similarity (Lower is better)", fontsize=13)
            #axes[1, 0].set_ylabel("Jaccard Index", fontsize=12)
            #axes[1, 0].set_xlabel("Explainer Method", fontsize=12)
            #
            #sns.boxplot(data=df_spearman, x="Explainer", y="Score", hue="Explainer", ax=axes[1, 1], palette="Pastel1", showfliers=False, width=0.5, legend=False)
            #sns.stripplot(data=df_spearman, x="Explainer", y="Score", hue="Explainer", ax=axes[1, 1], palette="dark:black", size=4, alpha=0.6, jitter=True, legend=False)
            #axes[1, 1].set_title("Sanity: Spearman Rank Correlation (Lower is better)", fontsize=13)
            #axes[1, 1].set_ylabel("Spearman Correlation", fontsize=12)
            #axes[1, 1].set_xlabel("Explainer Method", fontsize=12)

            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
            fig, axes = plt.subplots(2, figsize=(10, 12))
            fig.suptitle("Fidelity Evaluation", fontsize=18, fontweight="bold")

            sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[0], palette="Set2")
            axes[0].set_title("Cumulative RoAR Trend", fontsize=13)
            axes[0].set_ylabel("Mean Cumulative RoAR", fontsize=12)
            axes[0].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[0].set_xticks(range(1, exp_size + 1))

            sns.boxplot(data=df_spearman, x="Explainer", y="Score", hue="Explainer", ax=axes[1], palette="Pastel1", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_spearman, x="Explainer", y="Score", hue="Explainer", ax=axes[1], palette="dark:black", size=4, alpha=0.6, jitter=True, legend=False)
            axes[1].set_title("Sanity: Spearman Rank Correlation (Lower is better)", fontsize=13)
            axes[1].set_ylabel("Spearman Correlation", fontsize=12)
            axes[1].set_xlabel("Explainer Method", fontsize=12)

            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=15)
                
            plt.tight_layout()
            plt.savefig(f"{self.name}_fidelity_sanity_test.png", dpi=300, bbox_inches='tight')
            plt.show()

        return (causal_fid, causal_sanity), (lime_fids, lime_sanity), (shap_fid, shap_sanity)
    
    def stability_test(self, noise_stds=[0.01, 0.05, 0.1], display=False):
        all_causal_results = {std: [] for std in noise_stds}
        all_shap_results = {std: [] for std in noise_stds}
        all_lime_results = {std: {f"LIME (Seed: {seed})": [] for seed in self.lime_explainers} for std in noise_stds}
        
        numeric_cols = self.X_test.select_dtypes(include=["float64", "int64"]).columns
        
        def calc_spearman(base_f, compare_f, all_features):
            if len(all_features) < 2: return 1.0
            ranks_base = [base_f.index(f) if f in base_f else len(all_features) for f in all_features]
            ranks_compare = [compare_f.index(f) if f in compare_f else len(all_features) for f in all_features]
            corr, _ = spearmanr(ranks_base, ranks_compare)
            return corr if not np.isnan(corr) else 0.0

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()
            
            causal_explanation = self.causal_explainer.explain_instance(inst_1d)
            causal_features = [v["features"] for v in causal_explanation]
            
            shap_explanation = shap_explanation_form(self.model, self.shap_explainer, inst_2d)
            shap_features = [v["feature"] for v in shap_explanation]
            #shap_scores = {v["feature"]: v.get("score", v.get("absolute score", 0.0)) for v in shap_explanation}
            
            lime_features_dict = {}
            lime_scores_dict = {}
            for seed, explainer in self.lime_explainers.items():
                lime_explanation = lime_explanation_form(self.model, explainer, inst_1d)
                lime_features_dict[seed] = [v["feature"] for v in lime_explanation]
                lime_scores_dict[seed] = {v["feature"]: v.get("score", v.get("absolute score", 0.0)) for v in lime_explanation}

            for std in noise_stds:
                inst_2d_noisy = inst_2d.copy()
                inst_2d_noisy[numeric_cols] += np.random.normal(0, std, len(numeric_cols))
                inst_1d_noisy = inst_2d_noisy.iloc[0].copy()
                    
                causal_explanation_noisy = self.causal_explainer.explain_instance(inst_1d_noisy)
                causal_features_noisy = [v["features"] for v in causal_explanation_noisy]
                
                shap_explanation_noisy = shap_explanation_form(self.model, self.shap_explainer, inst_2d_noisy)
                shap_features_noisy = [v["feature"] for v in shap_explanation_noisy]
                
                lime_features_noisy_dict = {}
                for seed, explainer in self.lime_explainers.items():
                    lime_explanation_noisy = lime_explanation_form(self.model, explainer, inst_1d_noisy)
                    lime_features_noisy_dict[seed] = [v["feature"] for v in lime_explanation_noisy]

                all_causal_results[std].append(calc_spearman(causal_features, causal_features_noisy, self.features))

                all_shap_results[std].append(calc_spearman(shap_features, shap_features_noisy, self.features))
                
                for seed in self.lime_explainers:
                    l_lbl = f"LIME (Seed: {seed})"
                    all_lime_results[std][l_lbl].append(calc_spearman(lime_features_dict[seed], lime_features_noisy_dict[seed], self.features))

        if display:
            rows = (len(noise_stds) // 2) + (len(noise_stds) % 2)
            cols = 2
            
            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
            fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows), squeeze=False)
            fig.suptitle("Stability Evaluation Across Different Noise Levels", fontsize=16, fontweight="bold")
            
            for idx, std in enumerate(noise_stds):
                data_list = []
                for i in range(self.num_instance):
                    data_list.append({"Explainer": "Utility-aligned", "Metric": "Robustness", "Score": all_causal_results[std][i]})
                    data_list.append({"Explainer": "SHAP", "Metric": "Robustness", "Score": all_shap_results[std][i]})
                    for label, results in all_lime_results[std].items():
                        data_list.append({"Explainer": label, "Metric": "Robustness", "Score": results[i]})
                        
                df_sta = pd.DataFrame(data_list)
                df_rob = df_sta[df_sta["Metric"] == "Robustness"]
                
                ax = axes[idx // 2, idx % 2]
                sns.boxplot(data=df_rob, x="Explainer", y="Score", ax=ax, palette="Set2", showfliers=False, width=0.5)
                sns.stripplot(data=df_rob, x="Explainer", y="Score", ax=ax, color=".2", size=4, alpha=0.6, jitter=True)
                
                ax.set_title(f"Robustness (Rank Correlation) [std = {std}]", fontsize=13)
                ax.set_ylabel("Score (Higher is better)", fontsize=11)
                ax.set_xlabel("")
                ax.set_ylim(-1.05, 1.05)
                ax.tick_params(axis='x', rotation=15)
                #ax.set_xticklabels([])
                #if idx < rows - 1:
                #    ax.set_xticklabels([])

            plt.tight_layout()
            plt.savefig(f"{self.name}_stability_test.png")
            plt.show()

        return all_causal_results, all_lime_results, all_shap_results

    def decision_utility_test(self, display=False, top_k=5):
        exp_size = min(top_k, len(self.causal_explainer.critical_features))
        if exp_size == 0:
            exp_size = 1

        shap_results = {"SHAP": []}
        lime_results = {f"LIME (Seed: {seed})": [] for seed in self.lime_explainers}
        causal_results = {"Utility-aligned": []}
        
        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]
            
            c_exp = self.causal_explainer.explain_instance(inst_1d)
            c_f = [v["features"] for v in c_exp if v["features"] in self.causal_explainer.critical_features]

            shap_explanation = shap_explanation_form(self.model, self.shap_explainer, inst_2d)
            s_f = [v["feature"] for v in shap_explanation if v["feature"] in self.causal_explainer.critical_features]
            
            l_f_dict = {}
            for seed, explainer in self.lime_explainers.items():
                lime_explanation = lime_explanation_form(self.model, explainer, inst_1d)
                l_f_dict[seed] = [v["feature"] for v in lime_explanation if v["feature"] in self.causal_explainer.critical_features]
            
            c_utilities = []
            s_utilities = []
            l_utilities = {seed: [] for seed in self.lime_explainers}

            for k in range(1, exp_size + 1):
                # --- Causal (Utility-aligned) ---
                c_f_k = c_f[:k]
                if len(c_f_k) > 0:
                    c_subsets = get_combinations_up_to_k(c_f_k, len(c_f_k))
                    c_utilities_list = []
                    for c_subset in c_subsets:
                        c_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_1d, list(c_subset))
                        c_utilities_list.append(np.max(self.utility_matrix @ c_probs.T))
                    c_utilities.append(max(c_utilities_list))
                else:
                    c_probs = self.model.predict_proba(inst_2d.values.reshape(1, -1))[0]
                    c_utilities.append(np.max(self.utility_matrix @ c_probs.T))

                # --- SHAP ---
                s_f_k = s_f[:k]
                if len(s_f_k) > 0:
                    s_subsets = get_combinations_up_to_k(s_f_k, len(s_f_k))
                    s_utilities_list = []
                    for s_subset in s_subsets:
                        s_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_1d, list(s_subset))
                        s_utilities_list.append(np.max(self.utility_matrix @ s_probs.T))
                    s_utilities.append(max(s_utilities_list))
                else:
                    s_probs = self.model.predict_proba(inst_2d.values.reshape(1, -1))[0]
                    s_utilities.append(np.max(self.utility_matrix @ s_probs.T))

                # --- LIME ---
                for seed in self.lime_explainers:
                    l_f_k = l_f_dict[seed][:k]
                    if len(l_f_k) > 0:
                        l_subsets = get_combinations_up_to_k(l_f_k, len(l_f_k))
                        l_utilities_list = []
                        for l_subset in l_subsets:
                            l_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_1d, list(l_subset))
                            l_utilities_list.append(np.max(self.utility_matrix @ l_probs.T))
                        l_utilities[seed].append(max(l_utilities_list)) 
                    else:
                        l_probs = self.model.predict_proba(inst_2d.values.reshape(1, -1))[0]
                        l_utilities[seed].append(np.max(self.utility_matrix @ l_probs.T))

            causal_results["Utility-aligned"].append(c_utilities)
            shap_results["SHAP"].append(s_utilities)
            for seed in self.lime_explainers:
                lime_results[f"LIME (Seed: {seed})"].append(l_utilities[seed])
                
        if display:
            trend_data = []
            
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                
                c_mean = np.mean([causal_results["Utility-aligned"][i][k_idx] for i in range(self.num_instance)])
                s_mean = np.mean([shap_results["SHAP"][i][k_idx] for i in range(self.num_instance)])
                
                trend_data.append({"Explainer": "Utility-aligned", "k": k_val, "Mean Expected Utility": c_mean})
                trend_data.append({"Explainer": "SHAP", "k": k_val, "Mean Expected Utility": s_mean})
                
                for seed in self.lime_explainers:
                    label = f"LIME (Seed: {seed})"
                    l_mean = np.mean([lime_results[label][i][k_idx] for i in range(self.num_instance)])
                    trend_data.append({"Explainer": label, "k": k_val, "Mean Expected Utility": l_mean})
                    
            df_trend = pd.DataFrame(trend_data)
            
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 6))
            
            sns.lineplot(data=df_trend, x="k", y="Mean Expected Utility", hue="Explainer", marker="o", palette="Set2", linewidth=2, markersize=8)
            
            plt.title("Mean Expected Utility by Top-k Features", fontsize=16, fontweight="bold")
            plt.ylabel("Mean Expected Utility", fontsize=12)
            plt.xlabel("Number of selected features (k)", fontsize=12)
            plt.xticks(range(1, exp_size + 1))
            
            plt.tight_layout()
            plt.savefig(f"{self.name}_decision_utility_test.png")
            plt.show()

        return causal_results, lime_results, shap_results
    
    def do_all_tests(self, noise_stds=[0.05, 0.1, 0.5, 1.0], display_test=False):
        fidelity_results = self.fidelity_test(display=display_test)
        
        stability_results = self.stability_test(noise_stds=noise_stds, display=display_test)

        utility_results = self.decision_utility_test(display=display_test, top_k=len(self.causal_explainer.critical_features))
        
        return {
            "fidelity": fidelity_results,
            "stability": stability_results,
            "utility": utility_results
        }

class ReducedExplainerTester:

    def __init__(self, name, model, features, X_train, y_train, X_val, y_val, X_test, lime_explainers, shap_explainer, n_samples=100):
        self.name = name
        self.model = model
        self.features = features
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.lime_explainers = lime_explainers
        self.shap_explainer = shap_explainer
        self.num_instance = min(len(self.X_test), n_samples)

    def _get_lime_shap_attributions(self, instance_1d, instance_2d):
        lime_features_dict = dict()
        lime_scores_dict = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            lime_attr_raw = lime_explanation_form(self.model, lime_explainer, instance_1d)
            lime_features = np.array([v["feature"] for v in lime_attr_raw])
            lime_scores = np.array([v["absolute score"] for v in lime_attr_raw])
            lime_features_dict[seed] = lime_features
            lime_scores_dict[seed] = lime_scores

        shap_attr_raw = shap_explanation_form(self.model, self.shap_explainer, instance_2d)
        shap_features = np.array([v["feature"] for v in shap_attr_raw])
        shap_scores = np.array([v["absolute score"] for v in shap_attr_raw])

        return (lime_features_dict, lime_scores_dict), (shap_features, shap_scores)

    def fidelity_test(self, display=False, top_k=5):
        lime_fids = dict()
        for seed in self.lime_explainers:
            lime_fids[f"LIME (Seed: {seed})"] = {"ABPC score": [], "RoAR score": [], "ABPC trending": [], "RoAR trending": []}
        shap_fid = {"ABPC score": [], "RoAR score": [], "ABPC trending": [], "RoAR trending": []}
        baseline_vector = self.X_train.median()

        exp_size = min(len(self.features) - 1, top_k)
        
        lime_sanity = {f"LIME (Seed: {seed})": {"Top-k Jaccard": [], "Spearman": []} for seed in self.lime_explainers}
        shap_sanity = {"Top-k Jaccard": [], "Spearman": []}
        
        if isinstance(self.model, xgb.XGBClassifier):
            corrupted_model = xgb.XGBClassifier()
            corrupted_model.set_params(**self.model.get_params())
        else:
            corrupted_model = clone(self.model)

        np.random.seed(42)
        shuffled_y_train = np.random.permutation(self.y_train)
        
        if isinstance(corrupted_model, xgb.XGBClassifier):
            corrupted_model.fit(self.X_train, shuffled_y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
            corrupted_shap_explainer = shap.Explainer(corrupted_model.predict, self.X_train)
        else:
            corrupted_model.fit(self.X_train, shuffled_y_train)
            corrupted_shap_explainer = shap.Explainer(corrupted_model)

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (l_f, l_s), (s_f, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)
            exp_size = min(top_k, len(self.features) - 1)

            for seed in self.lime_explainers:
                scores_abpc, trending_abpc = ABPC(self.model, inst_2d, l_f[seed], l_s[seed], baseline_vector)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC score"].append(scores_abpc)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC trending"].append(trending_abpc[:exp_size])

                scores_roar, trending_roar = RoAR(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], self.X_train, self.y_train, self.X_val, self.y_val)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR score"].append(scores_roar)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR trending"].append(trending_roar)

            scores_shap_abpc, trending_shap_abpc = ABPC(self.model, inst_2d, s_f, s_s, baseline_vector)
            shap_fid["ABPC score"].append(scores_shap_abpc)
            shap_fid["ABPC trending"].append(trending_shap_abpc[:exp_size])

            scores_shap_roar, trending_shap_roar = RoAR(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], self.X_train, self.y_train, self.X_val, self.y_val)
            shap_fid["RoAR score"].append(scores_shap_roar)
            shap_fid["RoAR trending"].append(trending_shap_roar)

            corr_s_exp = shap_explanation_form(corrupted_model, corrupted_shap_explainer, inst_2d)
            corr_s_f = [v["feature"] for v in corr_s_exp]
            
            shap_sanity["Spearman"].append(spearman_similarity(list(s_f), corr_s_f))
            shap_sanity["Top-k Jaccard"].append(jaccard_similarity(list(s_f)[:exp_size], corr_s_f[:exp_size]))

            for seed, explainer in self.lime_explainers.items():
                corr_l_exp = lime_explanation_form(corrupted_model, explainer, inst_1d)
                corr_l_f = [v["feature"] for v in corr_l_exp]
                
                label = f"LIME (Seed: {seed})"
                lime_sanity[label]["Spearman"].append(spearman_similarity(list(l_f[seed]), corr_l_f))
                lime_sanity[label]["Top-k Jaccard"].append(jaccard_similarity(list(l_f[seed])[:exp_size], corr_l_f[:exp_size]))

        if display:
            data_list = []
            for i in range(self.num_instance):
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "ABPC", "Score": lime_fid["ABPC score"][i]})
                    data_list.append({"Explainer": label, "Metric": "RoAR", "Score": lime_fid["RoAR score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "ABPC", "Score": shap_fid["ABPC score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "RoAR", "Score": shap_fid["RoAR score"][i]})
            df_box = pd.DataFrame(data_list)
            
            mean_lime_abpc = {label: np.mean(lime_fid["ABPC trending"], axis=0) for label, lime_fid in lime_fids.items()}
            mean_shap_abpc = np.mean(shap_fid["ABPC trending"], axis=0)
            mean_lime_roar = {label: np.mean(lime_fid["RoAR trending"], axis=0) for label, lime_fid in lime_fids.items()}
            mean_shap_roar = np.mean(shap_fid["RoAR trending"], axis=0)

            trend_data = []
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                for label in lime_fids.keys():
                    trend_data.append({"Explainer": label, "Metric": "ABPC", "k": k_val, "Mean Score": mean_lime_abpc[label][k_idx]})
                    trend_data.append({"Explainer": label, "Metric": "RoAR", "k": k_val, "Mean Score": mean_lime_roar[label][k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "ABPC", "k": k_val, "Mean Score": mean_shap_abpc[k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "RoAR", "k": k_val, "Mean Score": mean_shap_roar[k_idx]})
            df_trend = pd.DataFrame(trend_data)
            
            sanity_list = []
            for i in range(self.num_instance):
                sanity_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_sanity["Spearman"][i]})
                sanity_list.append({"Explainer": "SHAP", "Metric": "Top-k Jaccard", "Score": shap_sanity["Top-k Jaccard"][i]})
                
                for label, l_data in lime_sanity.items():
                    sanity_list.append({"Explainer": label, "Metric": "Spearman", "Score": l_data["Spearman"][i]})
                    sanity_list.append({"Explainer": label, "Metric": "Top-k Jaccard", "Score": l_data["Top-k Jaccard"][i]})
                    
            df_sanity = pd.DataFrame(sanity_list)
            df_spearman = df_sanity[df_sanity["Metric"] == "Spearman"]
            df_jaccard = df_sanity[df_sanity["Metric"] == "Top-k Jaccard"]
            
            #sns.set_theme(style="whitegrid")
            #fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 18))
            #fig.suptitle("Fidelity (ABPC, RoAR) & Sanity Check Evaluation", fontsize=18, fontweight="bold")
            #
            #df_abpc_box = df_box[df_box["Metric"] == "ABPC"]
            #sns.boxplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5, hue="Explainer", legend=False)
            #sns.stripplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            #axes[0, 0].set_title("ABPC Distribution (Overall)", fontsize=13)
            #
            #df_abpc_trend = df_trend[df_trend["Metric"] == "ABPC"]
            #sns.lineplot(data=df_abpc_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[0, 1], palette="Set2")
            #axes[0, 1].set_title("ABPC Trend", fontsize=13)
            #axes[0, 1].set_xticks(range(1, exp_size + 1))
            #
            #df_roar_box = df_box[df_box["Metric"] == "RoAR"]
            #sns.boxplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[1, 0], palette="Set2", showfliers=False, width=0.5, hue="Explainer", legend=False)
            #sns.stripplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[1, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            #axes[1, 0].set_title("RoAR Distribution (Overall)", fontsize=13)
            #
            #df_roar_trend = df_trend[df_trend["Metric"] == "RoAR"]
            #sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 1], palette="Set2")
            #axes[1, 1].set_title("RoAR Trend", fontsize=13)
            #axes[1, 1].set_xticks(range(1, exp_size + 1))
#
            #sns.boxplot(data=df_spearman, x="Explainer", y="Score", ax=axes[2, 0], palette="Pastel1", showfliers=False, width=0.5, hue="Explainer", legend=False)
            #sns.stripplot(data=df_spearman, x="Explainer", y="Score", ax=axes[2, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            #axes[2, 0].set_title("Sanity: Spearman Rank (Original vs Corrupted)", fontsize=13)
            #axes[2, 0].set_ylabel("Spearman Score (Lower is better)", fontsize=12)
            #axes[2, 0].set_ylim(-1.05, 1.05)
#
            #sns.boxplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[2, 1], palette="Pastel1", showfliers=False, width=0.5, hue="Explainer", legend=False)
            #sns.stripplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[2, 1], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            #axes[2, 1].set_title(f"Sanity: Top-{exp_size} Jaccard (Original vs Corrupted)", fontsize=13)
            #axes[2, 1].set_ylabel("Jaccard Index (Lower is better)", fontsize=12)
            #axes[2, 1].set_ylim(-0.05, 1.05)

            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
            fig.suptitle("Fidelity Evaluation", fontsize=18, fontweight="bold")


            df_abpc_trend = df_trend[df_trend["Metric"] == "ABPC"]
            sns.lineplot(data=df_abpc_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[0, 0], palette="Set2")
            axes[0, 0].set_title("ABPC Trend", fontsize=13)
            axes[0, 0].set_xticks(range(1, exp_size + 1))


            df_roar_trend = df_trend[df_trend["Metric"] == "RoAR"]
            sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[0, 1], palette="Set2")
            axes[0, 1].set_title("RoAR Trend", fontsize=13)
            axes[0, 1].set_xticks(range(1, exp_size + 1))


            sns.boxplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1, 0], palette="Pastel1", showfliers=False, width=0.5, hue="Explainer", legend=False)
            sns.stripplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1, 0], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            axes[1, 0].set_title("Sanity: Spearman Rank (Original vs Corrupted)", fontsize=13)
            axes[1, 0].set_ylabel("Spearman Score (Lower is better)", fontsize=12)
            axes[1, 0].set_ylim(-1.05, 1.05)


            sns.boxplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[1, 1], palette="Pastel1", showfliers=False, width=0.5, hue="Explainer", legend=False)
            sns.stripplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[1, 1], palette="dark:black", size=4, alpha=0.6, jitter=True, hue="Explainer", legend=False)
            axes[1, 1].set_title(f"Sanity: Top-{exp_size} Jaccard (Original vs Corrupted)", fontsize=13)
            axes[1, 1].set_ylabel("Jaccard Index (Lower is better)", fontsize=12)
            axes[1, 1].set_ylim(-0.05, 1.05)


            for ax in axes.flat:
                if ax.has_data():
                    ax.tick_params(axis='x', rotation=15)
            
            plt.tight_layout()
            plt.savefig(f"{self.name}_reduced_fidelity_sanity_test.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return (lime_fids, lime_sanity), (shap_fid, shap_sanity)

    def robustness_test(self, noise_stds=[0.01, 0.05, 0.1], display=False, top_k=5):
        all_lime_robs_spearman = {std: {seed: [] for seed in self.lime_explainers} for std in noise_stds}
        all_shap_rob_spearman = {std: [] for std in noise_stds}

        all_lime_robs_jaccard = {std: {seed: {k: [] for k in range(1, top_k + 1)} for seed in self.lime_explainers} for std in noise_stds}
        all_shap_rob_jaccard = {std: {k: [] for k in range(1, top_k + 1)} for std in noise_stds}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)
            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns

            for std in noise_stds:
                inst_2d_noisy = inst_2d.copy()
                inst_2d_noisy[numeric_cols] += np.random.normal(0, std, len(numeric_cols))
                inst_1d_noisy = inst_2d_noisy.iloc[0].copy()

                (l_f_n, _), (s_f_n, _) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

                for seed in self.lime_explainers:
                    all_lime_robs_spearman[std][seed].append(spearman_similarity(list(l_f[seed]), list(l_f_n[seed])))
                all_shap_rob_spearman[std].append(spearman_similarity(list(s_f), list(s_f_n)))

                for k in range(1, top_k + 1):
                    exp_size = min(k, len(self.features) - 1)
                    for seed in self.lime_explainers:
                        all_lime_robs_jaccard[std][seed][k].append(jaccard_similarity(list(l_f[seed])[:exp_size], list(l_f_n[seed])[:exp_size]))
                    all_shap_rob_jaccard[std][k].append(jaccard_similarity(list(s_f)[:exp_size], list(s_f_n)[:exp_size]))

        if display:
            rows = len(noise_stds)
            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
            fig, axes = plt.subplots(rows, 2, figsize=(20, 6 * rows), squeeze=False)
            fig.suptitle("Robustness Test Between Explainers Across Different Noise Levels", fontsize=16, fontweight="bold")

            for idx, std in enumerate(noise_stds):
                spearman_data = []
                for i in range(self.num_instance):
                    for seed in self.lime_explainers:
                        spearman_data.append({"Explainer": f"LIME (Seed: {seed})", "Spearman Score": all_lime_robs_spearman[std][seed][i]})
                    spearman_data.append({"Explainer": "SHAP", "Spearman Score": all_shap_rob_spearman[std][i]})
                df_spearman = pd.DataFrame(spearman_data)

                jaccard_trend = []
                for k in range(1, top_k + 1):
                    for seed in self.lime_explainers:
                        mean_jac = np.mean(all_lime_robs_jaccard[std][seed][k])
                        jaccard_trend.append({"Explainer": f"LIME (Seed: {seed})", "k": k, "Mean Jaccard": mean_jac})
                    mean_jac_shap = np.mean(all_shap_rob_jaccard[std][k])
                    jaccard_trend.append({"Explainer": "SHAP", "k": k, "Mean Jaccard": mean_jac_shap})
                df_jaccard = pd.DataFrame(jaccard_trend)
                
                sns.boxplot(data=df_spearman, x="Explainer", y="Spearman Score", ax=axes[idx, 0], palette="Set2", showfliers=False, width=0.5)
                sns.stripplot(data=df_spearman, x="Explainer", y="Spearman Score", ax=axes[idx, 0], color=".2", size=4, alpha=0.6, jitter=True)
                axes[idx, 0].set_title(f"Spearman Rank Correlation [std = {std}]", fontsize=13)
                axes[idx, 0].set_ylim(-1.05, 1.05)
                
                sns.lineplot(data=df_jaccard, x="k", y="Mean Jaccard", hue="Explainer", marker="o", ax=axes[idx, 1], palette="Set2")
                axes[idx, 1].set_title(f"Top-k Jaccard Similarity Trend [std = {std}]", fontsize=13)
                axes[idx, 1].set_xticks(range(1, top_k + 1))
                axes[idx, 1].set_ylim(-0.05, 1.05)
            
            plt.tight_layout(pad = 2.0)
            plt.savefig(f"{self.name}_reduced_robustness_test.png")
            plt.show()
        
        return all_lime_robs_spearman, all_shap_rob_spearman, all_lime_robs_jaccard, all_shap_rob_jaccard

    def sensitivity_test(self, noise_stds=[0.01, 0.05, 0.1], display=False):
        all_sensitivity_data = {std: {"SHAP": []} for std in noise_stds}
        for std in noise_stds:
            for seed in self.lime_explainers:
                all_sensitivity_data[std][f"LIME (Seed: {seed})"] = []

        def _normalize_attr(attr_vector):
            norm = np.linalg.norm(attr_vector)
            return attr_vector / norm if norm > 0 else attr_vector

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (_, l_s), (_, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)
            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns

            for std in noise_stds:
                inst_2d_noisy = inst_2d.copy()
                noise = np.random.normal(0, std, len(numeric_cols))
                inst_2d_noisy[numeric_cols] += noise
                inst_1d_noisy = inst_2d_noisy.iloc[0].copy()

                (_, l_s_n), (_, s_s_n) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

                input_distance = np.linalg.norm(noise)
                if input_distance == 0: 
                    input_distance = 1e-9

                for seed in self.lime_explainers:
                    norm_l_s = _normalize_attr(l_s[seed])
                    norm_l_s_n = _normalize_attr(l_s_n[seed])
                    
                    output_distance = np.linalg.norm(norm_l_s - norm_l_s_n)
                    all_sensitivity_data[std][f"LIME (Seed: {seed})"].append(output_distance / input_distance)
                
                norm_s_s = _normalize_attr(s_s)
                norm_s_s_n = _normalize_attr(s_s_n)
                
                output_distance_shap = np.linalg.norm(norm_s_s - norm_s_s_n)
                all_sensitivity_data[std]["SHAP"].append(output_distance_shap / input_distance)

        if display:
            rows = (len(noise_stds) // 2) + (len(noise_stds) % 2)
            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1.5})
            fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), squeeze=False)
            fig.suptitle("Sensitivity (Normalized Lipschitz) Evaluation", fontsize=16, fontweight="bold")

            for idx, std in enumerate(noise_stds):
                df = pd.DataFrame(all_sensitivity_data[std]).melt(var_name="Explainer", value_name="Lipschitz Estimate")
                sns.boxplot(data=df, x="Explainer", y="Lipschitz Estimate", ax=axes[idx // 2, idx % 2], palette="Set2", showfliers=False, width=0.5)
                sns.stripplot(data=df, x="Explainer", y="Lipschitz Estimate", ax=axes[idx // 2, idx % 2], color=".2", size=5, alpha=0.6, jitter=True)
                axes[idx // 2, idx % 2].set_title(f"Sensitivity Test [std = {std}]", fontsize=13)
                axes[idx // 2, idx % 2].set_ylabel("Lipschitz Constant (Lower is better)")

            plt.tight_layout()
            plt.savefig(f"{self.name}_reduced_sensitivity_test.png")
            plt.show()

        return all_sensitivity_data

    def do_all_tests(self, noise_stds=[0.05, 0.1, 0.5, 1.0], display=False, top_k=5):
        fidelity_results = self.fidelity_test(display=display, top_k=top_k)
        robustness_results = self.robustness_test(noise_stds=noise_stds, display=display, top_k=top_k)
        sensitivity_results = self.sensitivity_test(noise_stds=noise_stds, display=display)
    
        return {
            "fidelity": fidelity_results,
            "robustness": robustness_results,
            "sensitivity": sensitivity_results,
        }