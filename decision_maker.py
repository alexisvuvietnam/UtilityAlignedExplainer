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

    def __init__(self, model, features, actions, causal_model, utility_matrix, X_train, y_train, X_test, lime_explainers, shap_explainer, n_samples=100):
        self.model = model
        self.features = features
        self.actions = actions
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.lime_explainers = lime_explainers
        self.shap_explainer = shap_explainer
        self.causal_explainer = UtilityAlignedTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_model, self.utility_matrix)
        self.num_instance = min(len(self.X_test), n_samples)

    def fidelity_test(self, display=False):
        exp_size = len(self.causal_explainer.critical_features)
        baseline_vector = self.X_train.median()
        
        causal_fid = {"MoRF Score": [], "RoAR Score": [], "MoRF Trending": [], "RoAR Trending": []}
        lime_fids = dict()
        for seed in self.lime_explainers:
            lime_fids[f"LIME (Seed: {seed})"] = {"MoRF Score": [], "RoAR Score": [], "MoRF Trending": [], "RoAR Trending": []}
        shap_fid = {"MoRF Score": [], "RoAR Score": [], "MoRF Trending": [], "RoAR Trending": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]
                
            c_exp = self.causal_explainer.explain_instance(inst_1d)
            c_f = np.array([v["features"] for v in c_exp])
            c_s = np.array([v["utility score"] for v in c_exp])
            #c_s = np.array([v["utility score"] for v in c_exp])
            #c_s_i = np.array([-v["information score"] for v in c_exp])

            morf_score, morf_cumul = MoRF(self.model, inst_2d, c_f, c_s, baseline_vector)
            roar_score, roar_cumul = RoAR(self.model, inst_2d, c_f, c_s, self.X_train, self.y_train)
            causal_fid["MoRF Score"].append(morf_score)
            causal_fid["RoAR Score"].append(roar_score)
            causal_fid["MoRF Trending"].append(morf_cumul)
            causal_fid["RoAR Trending"].append(roar_cumul)

            s_exp = top_k_shap_explanation_form(self.model, self.shap_explainer, inst_2d, top_k=exp_size)
            s_f = np.array([v["feature"] for v in s_exp])
            s_s = np.array([v["absolute score"] for v in s_exp])

            morf_score, morf_cumul = MoRF(self.model, inst_2d, s_f, s_s, baseline_vector)
            roar_score, roar_cumul = RoAR(self.model, inst_2d, s_f, s_s, self.X_train, self.y_train)
            shap_fid["MoRF Score"].append(morf_score)
            shap_fid["RoAR Score"].append(roar_score)
            shap_fid["MoRF Trending"].append(morf_cumul)
            shap_fid["RoAR Trending"].append(roar_cumul)

            for seed, lime_explainer in self.lime_explainers.items():
                l_exp = top_k_lime_explanation_form(self.model, lime_explainer, inst_1d, top_k=exp_size)
                l_f = np.array([v["feature"] for v in l_exp])
                l_s = np.array([v["absolute score"] for v in l_exp])

                morf_score, morf_cumul = MoRF(self.model, inst_2d, l_f, l_s, baseline_vector)
                roar_score, roar_cumul = RoAR(self.model, inst_2d, l_f, l_s, self.X_train, self.y_train)
                lime_fids[f"LIME (Seed: {seed})"]["MoRF Score"].append(morf_score)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR Score"].append(roar_score)
                lime_fids[f"LIME (Seed: {seed})"]["MoRF Trending"].append(morf_cumul)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR Trending"].append(roar_cumul)

        if display:
            data_list = []
            for i in range(self.num_instance):
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "MoRF", "Score": lime_fid["MoRF Score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "MoRF", "Score": shap_fid["MoRF Score"][i]})
                data_list.append({"Explainer": "Utility-aligned", "Metric": "MoRF", "Score": causal_fid["MoRF Score"][i]})
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "RoAR", "Score": lime_fid["RoAR Score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "RoAR", "Score": shap_fid["RoAR Score"][i]})
                data_list.append({"Explainer": "Utility-aligned", "Metric": "RoAR", "Score": causal_fid["RoAR Score"][i]})
                
            df_box = pd.DataFrame(data_list)
            
            mean_lime_morf = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_morf[label] = np.mean(lime_fid["MoRF Trending"], axis=0)
            mean_shap_morf = np.mean(shap_fid["MoRF Trending"], axis=0)
            mean_causal_morf = np.mean(causal_fid["MoRF Trending"], axis=0)
            mean_lime_roar = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_roar[label] = np.mean(lime_fid["RoAR Trending"], axis=0)
            mean_shap_roar = np.mean(shap_fid["RoAR Trending"], axis=0)
            mean_causal_roar = np.mean(causal_fid["RoAR Trending"], axis=0)

            trend_data = []
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                for label, lime_fid in lime_fids.items():
                    trend_data.append({"Explainer": label, "Metric": "MoRF", "k": k_val, "Mean Score": mean_lime_morf[label][k_idx]})
                    trend_data.append({"Explainer": label, "Metric": "RoAR", "k": k_val, "Mean Score": mean_lime_roar[label][k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "MoRF", "k": k_val, "Mean Score": mean_shap_morf[k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "RoAR", "k": k_val, "Mean Score": mean_shap_roar[k_idx]})
                trend_data.append({"Explainer": "Utility-aligned", "Metric": "MoRF", "k": k_val, "Mean Score": mean_causal_morf[k_idx]})
                trend_data.append({"Explainer": "Utility-aligned", "Metric": "RoAR", "k": k_val, "Mean Score": mean_causal_roar[k_idx]})
                
            df_trend = pd.DataFrame(trend_data)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle("Fidelity/Faithfulness Evaluation", fontsize=16, fontweight="bold")
            
            df_abpc_box = df_box[df_box["Metric"] == "MoRF"]
            df_roar_box = df_box[df_box["Metric"] == "RoAR"]
            
            sns.boxplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 0].set_title("MoRF Distribution (Overall)", fontsize=13)
            axes[0, 0].set_ylabel("Overall MoRF Score", fontsize=12)
            axes[0, 0].set_xlabel("Explainer Method", fontsize=12)
            
            sns.boxplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[0, 1], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[0, 1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 1].set_title("RoAR Distribution (Overall)", fontsize=13)
            axes[0, 1].set_ylabel("Overall RoAR Score", fontsize=12)
            axes[0, 1].set_xlabel("Explainer Method", fontsize=12)
        
            df_abpc_trend = df_trend[df_trend["Metric"] == "MoRF"]
            df_roar_trend = df_trend[df_trend["Metric"] == "RoAR"]
            
            sns.lineplot(data=df_abpc_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 0], palette="Set2")
            axes[1, 0].set_title("MoRF Trend (Mean over Instances)", fontsize=13)
            axes[1, 0].set_ylabel("Mean Cumulative MoRF", fontsize=12)
            axes[1, 0].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 0].set_xticks(range(1, exp_size + 1))
            
            sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 1], palette="Set2")
            axes[1, 1].set_title("RoAR Trend (Mean over Instances)", fontsize=13)
            axes[1, 1].set_ylabel("Mean Cumulative RoAR", fontsize=12)
            axes[1, 1].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 1].set_xticks(range(1, exp_size + 1))
            
            plt.tight_layout()
            plt.show()

        return causal_fid, lime_fids, shap_fid
    
    def stability_test(self, noise_stds=[0.01, 0.05, 0.1], display=False):
        all_causal_results = {std: {"Robustness": [], "Sensitivity": []} for std in noise_stds}
        all_shap_results = {std: {"Robustness": [], "Sensitivity": []} for std in noise_stds}
        all_lime_results = {std: {f"LIME (Seed: {seed})": {"Robustness": [], "Sensitivity": []} for seed in self.lime_explainers} for std in noise_stds}
        
        numeric_cols = self.X_test.select_dtypes(include=["float64", "int64"]).columns
        epsilon_smooth = 1e-5
        
        def calc_spearman(base_f, compare_f, all_features):
            if len(all_features) < 2: return 1.0
            ranks_base = [base_f.index(f) if f in base_f else len(all_features) for f in all_features]
            ranks_compare = [compare_f.index(f) if f in compare_f else len(all_features) for f in all_features]
            corr, _ = spearmanr(ranks_base, ranks_compare)
            return corr if not np.isnan(corr) else 0.0

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()
            x_i = inst_2d[numeric_cols].values[0]
            
            causal_explanation = self.causal_explainer.explain_instance(inst_1d)
            causal_features = [v["features"] for v in causal_explanation]
            causal_scores = {v["features"]: v["utility score"] for v in causal_explanation}
            
            shap_explanation = shap_explanation_form(self.model, self.shap_explainer, inst_2d)
            shap_features = [v["feature"] for v in shap_explanation]
            shap_scores = {v["feature"]: v.get("score", v.get("absolute score", 0.0)) for v in shap_explanation}
            
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
                x_i_noisy = inst_2d_noisy[numeric_cols].values[0]
                
                d_x_noise = max(np.linalg.norm(x_i - x_i_noisy), epsilon_smooth)
                    
                causal_explanation_noisy = self.causal_explainer.explain_instance(inst_1d_noisy)
                causal_features_noisy = [v["features"] for v in causal_explanation_noisy]
                causal_scores_noisy = {v["features"]: v["utility score"] for v in causal_explanation_noisy}
                
                shap_explanation_noisy = shap_explanation_form(self.model, self.shap_explainer, inst_2d_noisy)
                shap_features_noisy = [v["feature"] for v in shap_explanation_noisy]
                shap_scores_noisy = {v["feature"]: v.get("score", v.get("absolute score", 0.0)) for v in shap_explanation_noisy}
                
                lime_features_noisy_dict = {}
                lime_scores_noisy_dict = {}
                for seed, explainer in self.lime_explainers.items():
                    lime_explanation_noisy = lime_explanation_form(self.model, explainer, inst_1d_noisy)
                    lime_features_noisy_dict[seed] = [v["feature"] for v in lime_explanation_noisy]
                    lime_scores_noisy_dict[seed] = {v["feature"]: v.get("score", v.get("absolute score", 0.0)) for v in lime_explanation_noisy}
                    
                all_causal_results[std]["Robustness"].append(calc_spearman(causal_features, causal_features_noisy, self.features))
                c_vec = np.array([causal_scores.get(f, 0.0) for f in self.features])
                c_vec_noisy = np.array([causal_scores_noisy.get(f, 0.0) for f in self.features])
                all_causal_results[std]["Sensitivity"].append(np.linalg.norm(c_vec - c_vec_noisy) / d_x_noise)

                all_shap_results[std]["Robustness"].append(calc_spearman(shap_features, shap_features_noisy, self.features))
                s_vec = np.array([shap_scores.get(f, 0.0) for f in self.features])
                s_vec_noisy = np.array([shap_scores_noisy.get(f, 0.0) for f in self.features])
                all_shap_results[std]["Sensitivity"].append(np.linalg.norm(s_vec - s_vec_noisy) / d_x_noise)
                
                for seed in self.lime_explainers:
                    l_lbl = f"LIME (Seed: {seed})"
                    all_lime_results[std][l_lbl]["Robustness"].append(calc_spearman(lime_features_dict[seed], lime_features_noisy_dict[seed], self.features))
                    l_vec = np.array([lime_scores_dict[seed].get(f, 0.0) for f in self.features])
                    l_vec_noisy = np.array([lime_scores_noisy_dict[seed].get(f, 0.0) for f in self.features])
                    all_lime_results[std][l_lbl]["Sensitivity"].append(np.linalg.norm(l_vec - l_vec_noisy) / d_x_noise)

        if display:
            rows = len(noise_stds)
            cols = 2
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
            fig.suptitle("Stability Evaluation Across Different Noise Levels", fontsize=16, fontweight="bold")
            
            if rows == 1: axes = np.expand_dims(axes, axis=0)

            for idx, std in enumerate(noise_stds):
                data_list = []
                for i in range(self.num_instance):
                    data_list.append({"Explainer": "Utility-aligned", "Metric": "Robustness", "Score": all_causal_results[std]["Robustness"][i]})
                    data_list.append({"Explainer": "SHAP", "Metric": "Robustness", "Score": all_shap_results[std]["Robustness"][i]})
                    for label, results in all_lime_results[std].items():
                        data_list.append({"Explainer": label, "Metric": "Robustness", "Score": results["Robustness"][i]})
                        
                    data_list.append({"Explainer": "Utility-aligned", "Metric": "Sensitivity", "Score": all_causal_results[std]["Sensitivity"][i]})
                    data_list.append({"Explainer": "SHAP", "Metric": "Sensitivity", "Score": all_shap_results[std]["Sensitivity"][i]})
                    for label, results in all_lime_results[std].items():
                        data_list.append({"Explainer": label, "Metric": "Sensitivity", "Score": results["Sensitivity"][i]})
                
                df_sta = pd.DataFrame(data_list)
                df_rob = df_sta[df_sta["Metric"] == "Robustness"]
                df_sens = df_sta[df_sta["Metric"] == "Sensitivity"]
                
                sns.boxplot(data=df_rob, x="Explainer", y="Score", ax=axes[idx, 0], palette="Set2", showfliers=False, width=0.5)
                sns.stripplot(data=df_rob, x="Explainer", y="Score", ax=axes[idx, 0], color=".2", size=4, alpha=0.6, jitter=True)
                axes[idx, 0].set_title(f"Robustness (Rank Correlation) [std = {std}]", fontsize=13)
                axes[idx, 0].set_ylabel("Score (Higher is better)", fontsize=11)
                axes[idx, 0].set_xlabel("")
                axes[idx, 0].set_ylim(-1.05, 1.05)
                axes[idx, 0].tick_params(axis='x', rotation=15)
                
                sns.boxplot(data=df_sens, x="Explainer", y="Score", ax=axes[idx, 1], palette="Set2", showfliers=False, width=0.5)
                sns.stripplot(data=df_sens, x="Explainer", y="Score", ax=axes[idx, 1], color=".2", size=4, alpha=0.6, jitter=True)
                axes[idx, 1].set_title(f"Sensitivity (Local Lipschitz) [std = {std}]", fontsize=13)
                axes[idx, 1].set_ylabel("Lipschitz Ratio (Lower is better)", fontsize=11)
                axes[idx, 1].set_xlabel("")
                axes[idx, 1].tick_params(axis='x', rotation=15)

                if idx < rows - 1:
                    axes[idx, 0].set_xticklabels([])
                    axes[idx, 1].set_xticklabels([])

            plt.tight_layout()
            plt.show()

        return all_causal_results, all_lime_results, all_shap_results

    def sanity_check_test(self, display=False):

        if isinstance(self.model, xgb.XGBClassifier):
            corrupted_model = xgb.XGBClassifier()
            corrupted_model.set_params(**self.model.get_params())
        else:
            corrupted_model = clone(self.model)

        np.random.seed(42)
        shuffled_y_train = np.random.permutation(self.y_train)
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
            self.causal_model, 
            self.utility_matrix
        )

        #lime_sanity = {seed: {"jaccard": [], "spearman": []} for seed in self.lime_explainers}
        #shap_sanity = {"jaccard": [], "spearman": []}
        #causal_sanity = {"jaccard": [], "spearman": []}

        lime_sanity = {seed: [] for seed in self.lime_explainers}
        shap_sanity = []
        causal_sanity = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            orig_c_exp = self.causal_explainer.explain_instance(inst_1d)
            orig_c_f = [v["features"] for v in orig_c_exp]

            orig_s_exp = shap_explanation_form(self.model, self.shap_explainer, inst_2d)
            orig_s_f = [v["feature"] for v in orig_s_exp]

            orig_l_f = dict()
            for seed, explainer in self.lime_explainers.items():
                orig_l_exp = lime_explanation_form(self.model, explainer, inst_1d)
                orig_l_f[seed] = [v["feature"] for v in orig_l_exp]

            corr_c_exp = corrupted_causal_explainer.explain_instance(inst_1d)
            corr_c_f = [v["features"] for v in corr_c_exp]

            corr_s_exp = shap_explanation_form(corrupted_model, corrupted_shap_explainer, inst_2d)
            corr_s_f = [v["feature"] for v in corr_s_exp]

            corr_l_f = dict()
            for seed, explainer in self.lime_explainers.items():
                corr_l_exp = top_k_lime_explanation_form(corrupted_model, explainer, inst_1d, top_k=len(self.features))
                corr_l_f[seed] = [v["feature"] for v in corr_l_exp]

            #causal_sanity["jaccard"].append(jaccard_similarity(orig_c_f, corr_c_f))
            #causal_sanity["spearman"].append(spearman_similarity(orig_c_f, corr_c_f))
            causal_sanity.append(spearman_similarity(orig_c_f, corr_c_f))

            #shap_sanity["jaccard"].append(jaccard_similarity(orig_s_f, corr_s_f))
            #shap_sanity["spearman"].append(spearman_similarity(orig_s_f, corr_s_f))
            shap_sanity.append(spearman_similarity(orig_s_f, corr_s_f))

            for seed in self.lime_explainers:
                #lime_sanity[seed]["jaccard"].append(jaccard_similarity(orig_l_f[seed], corr_l_f[seed]))
                #lime_sanity[seed]["spearman"].append(spearman_similarity(orig_l_f[seed], corr_l_f[seed]))
                lime_sanity[seed].append(spearman_similarity(orig_l_f[seed], corr_l_f[seed]))

        if display:
            data_list = []
            for i in range(self.num_instance):
                #data_list.append({"Explainer": "Utility-aligned", "Metric": "Jaccard", "Score": causal_sanity["jaccard"][i]})
                #data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_sanity["jaccard"][i]})
                #for seed, l_data in lime_sanity.items():
                #    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Jaccard", "Score": l_data["jaccard"][i]})
                #
                #data_list.append({"Explainer": "Utility-aligned", "Metric": "Spearman", "Score": causal_sanity["spearman"][i]})
                #data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_sanity["spearman"][i]})
                #for seed, l_data in lime_sanity.items():
                #    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": l_data["spearman"][i]})

                data_list.append({"Explainer": "Utility-aligned", "Metric": "Spearman", "Score": causal_sanity[i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_sanity[i]})
                for seed, l_data in lime_sanity.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": l_data[i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle("Sanity Check: Label Randomization", fontsize=16, fontweight="bold")
            
            #df_jaccard = df[df["Metric"] == "Jaccard"]
            df_spearman = df[df["Metric"] == "Spearman"]
            
            #sns.boxplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], palette="Set2", showfliers=False, width=0.5)
            #sns.stripplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], color=".2", size=4, alpha=0.6, jitter=True)
            #axes[0].set_title("Jaccard Similarity (Original vs Corrupted)", fontsize=13)
            #axes[0].set_ylabel("Jaccard Score", fontsize=12)
            #axes[0].set_ylim(-0.05, 1.05)
            
            sns.boxplot(data=df_spearman, x="Explainer", y="Score", ax=ax, palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_spearman, x="Explainer", y="Score", ax=ax, color=".2", size=4, alpha=0.6, jitter=True)
            ax.set_title("Spearman Rank Correlation (Original vs Corrupted)", fontsize=13)
            ax.set_ylabel("Spearman Score", fontsize=12)
            ax.set_ylim(-1.05, 1.05)
            
            #plt.tight_layout()
            plt.show()

        return causal_sanity, lime_sanity, shap_sanity

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
            
            # Tiền xử lý dữ liệu: Tính trung bình cộng của Utility tại mỗi giá trị k
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
            
            # Vẽ biểu đồ Lineplot
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 6))
            
            sns.lineplot(data=df_trend, x="k", y="Mean Expected Utility", hue="Explainer", marker="o", palette="Set2", linewidth=2, markersize=8)
            
            plt.title("Mean Expected Utility by Top-k Features", fontsize=16, fontweight="bold")
            plt.ylabel("Mean Expected Utility", fontsize=12)
            plt.xlabel("Number of selected features (k)", fontsize=12)
            plt.xticks(range(1, exp_size + 1))
            
            plt.tight_layout()
            plt.show()

        return causal_results, lime_results, shap_results
    
    def do_all_tests(self, noise_stds=[0.05, 0.1, 0.5, 1.0], display_test=False):
        fidelity_results = self.fidelity_test(display=display_test)
        
        stability_results = self.stability_test(noise_stds=noise_stds, display=display_test)

        sanity_results = self.sanity_check_test(display=display_test)

        utility_results = self.decision_utility_test(display=display_test, top_k=3)
        
        return {
            "fidelity": fidelity_results,
            "stability": stability_results,
            "sanity": sanity_results,
            "utility": utility_results
        }



class ReducedExplainerTester:

    def __init__(self, model, features, X_train, y_train, X_test, lime_explainers, shap_explainer, n_samples=100):
        self.model = model
        self.features = features
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.lime_explainers = lime_explainers
        self.shap_explainer = shap_explainer
        self.num_instance = min(len(self.X_test), n_samples)

    ### XAI EMPIRICAL TESTS FROM HERE
    def consistency_test(self, display=False, top_k=5):
        base_instance_1d = self.X_test.iloc[0]
        base_instance_2d = self.X_test.iloc[[0]]
        base_lime_explanations = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            base_lime_explanation = lime_explanation_form(self.model, lime_explainer, base_instance_1d)
            base_lime_explanations[seed] = base_lime_explanation
        base_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, base_instance_2d)
        
        base_lime_samples = dict()
        for seed, base_lime_explanation in base_lime_explanations.items():
            base_lime_sample = [v["feature"] for v in base_lime_explanation]
            base_lime_samples[seed] = base_lime_sample
        base_shap_sample = [v["feature"] for v in base_shap_explanation]
        exp_size = min(top_k, len(self.features) - 1)
        
        lime_datas = dict()
        for seed in self.lime_explainers:
            lime_datas[seed] = {"jaccard": [], "spearman": []}
        shap_data = {"jaccard": [], "spearman": []}
        
        for i in range(1, self.num_instance + 1):
            example_instance_1d = self.X_test.iloc[i]
            example_instance_2d = self.X_test.iloc[[i]]
            new_lime_explanations = dict()
            for seed, lime_explainer in self.lime_explainers.items():
                new_lime_explanation = lime_explanation_form(self.model, lime_explainer, example_instance_1d)
                new_lime_explanations[seed] = new_lime_explanation
            new_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, example_instance_2d)
            
            new_lime_samples = dict()
            for seed, new_lime_explanation in new_lime_explanations.items():
                new_lime_sample = [v["feature"] for v in new_lime_explanation]
                new_lime_samples[seed] = new_lime_sample
            new_shap_sample = [v["feature"] for v in new_shap_explanation]
            
            for seed, lime_data in lime_datas.items():
                lime_data["jaccard"].append(jaccard_similarity(base_lime_samples[seed][:exp_size], new_lime_samples[seed][:exp_size]))
            shap_data["jaccard"].append(jaccard_similarity(base_shap_sample[:exp_size], new_shap_sample[:exp_size]))
            
            for seed, lime_data in lime_datas.items():
                lime_data["spearman"].append(spearman_similarity(base_lime_samples[seed], new_lime_samples[seed]))
            shap_data["spearman"].append(spearman_similarity(base_shap_sample, new_shap_sample))

        if display:
            data_list = []
            for i in range(self.num_instance):
                for seed, lime_data in lime_datas.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Jaccard", "Score": lime_data["jaccard"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_data["jaccard"][i]})
                
                for seed, lime_data in lime_datas.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": lime_data["spearman"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_data["spearman"][i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            fig.suptitle("Consistency Test Between Explainers", fontsize=16, fontweight="bold")

            df_jaccard = df[df["Metric"] == "Jaccard"]
            df_spearman = df[df["Metric"] == "Spearman"]
                        
            sns.boxplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0].set_title("Jaccard Similarity Distribution", fontsize=13)
            axes[0].set_ylabel("Jaccard Score", fontsize=12)
            axes[0].set_xlabel("Explainer Method", fontsize=12)
            axes[0].set_ylim(-0.05, 1.05)
            
            sns.boxplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[1].set_title("Spearman Rank Correlation Distribution", fontsize=13)
            axes[1].set_ylabel("Spearman Score", fontsize=12)
            axes[1].set_xlabel("Explainer Method", fontsize=12)
            axes[1].set_ylim(-1.05, 1.05)
            
            plt.tight_layout()
            plt.show()

        return (lime_data, shap_data)

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

    def robustness_test(self, noise_std=0.01, display=False, top_k=5):
        lime_robs = dict()
        for seed in self.lime_explainers:
            lime_robs[seed] = {"jaccard": [], "spearman": []}
        shap_rob = {"jaccard": [], "spearman": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_2d_noisy = inst_2d.copy()
            inst_2d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_1d_noisy = inst_2d_noisy.iloc[0].copy()

            (l_f_n, _), (s_f_n, _) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

            exp_size = min(top_k, len(self.features) - 1)
            for seed, lime_rob in lime_robs.items():
                lime_robs[seed]["jaccard"].append(jaccard_similarity(list(l_f[seed])[:exp_size], list(l_f_n[seed])[:exp_size]))
            shap_rob["jaccard"].append(jaccard_similarity(list(s_f)[:exp_size], list(s_f_n)[:exp_size]))

            for seed, lime_rob in lime_robs.items():
                lime_robs[seed]["spearman"].append(spearman_similarity(list(l_f[seed]), list(l_f_n[seed])))
            shap_rob["spearman"].append(spearman_similarity(list(s_f), list(s_f_n)))

        if display:
            data_list = []
            for i in range(self.num_instance):
                for seed, lime_rob in lime_robs.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Jaccard", "Score": lime_rob["jaccard"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_rob["jaccard"][i]})
                
                for seed, lime_rob in lime_robs.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": lime_rob["spearman"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_rob["spearman"][i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            fig.suptitle(f"Robustness Test Between Explainers: Noise std = {noise_std}", fontsize=16, fontweight="bold")
            
            df_jaccard = df[df["Metric"] == "Jaccard"]
            df_spearman = df[df["Metric"] == "Spearman"]
            
            sns.boxplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_jaccard, x="Explainer", y="Score", ax=axes[0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0].set_title("Jaccard Similarity Distribution", fontsize=13)
            axes[0].set_ylabel("Jaccard Score", fontsize=12)
            axes[0].set_xlabel("Explainer Method", fontsize=12)
            axes[0].set_ylim(-0.05, 1.05)
            
            sns.boxplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_spearman, x="Explainer", y="Score", ax=axes[1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[1].set_title("Spearman Rank Correlation Distribution", fontsize=13)
            axes[1].set_ylabel("Spearman Score", fontsize=12)
            axes[1].set_xlabel("Explainer Method", fontsize=12)
            axes[1].set_ylim(-1.05, 1.05)
            
            plt.tight_layout()
            plt.show()
        
        return lime_robs, shap_rob

    def sensitivity_test(self, noise_std=0.01, display=False):
        sensitivity_data = {"SHAP": []}
        for seed in self.lime_explainers:
            sensitivity_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (_, l_s), (_, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_2d_noisy = inst_2d.copy()
            inst_2d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_1d_noisy = inst_2d_noisy.iloc[0].copy()

            (_, l_s_n), (_, s_s_n) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

            for seed in self.lime_explainers:
                sensitivity_data[f"LIME (Seed: {seed})"].append(np.linalg.norm(l_s[seed] - l_s_n[seed]))
            sensitivity_data["SHAP"].append(np.linalg.norm(s_s - s_s_n))

        if display:
            df = pd.DataFrame(sensitivity_data).melt(var_name="Explainer", value_name="Sensitivity")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Sensitivity", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Sensitivity", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title(f"Sensitivity Test: Noise std = {noise_std}")
            plt.ylabel("L2 Norm of Score Differences")
            plt.tight_layout()
            plt.show()

        return sensitivity_data

    def fidelity_test(self, display=False, top_k=5):
        lime_fids = dict()
        for seed in self.lime_explainers:
            lime_fids[f"LIME (Seed: {seed})"] = {"ABPC score": [], "RoAR score": [], "ABPC trending": [], "RoAR trending": []}
        shap_fid = {"ABPC score": [], "RoAR score": [], "ABPC trending": [], "RoAR trending": []}
        baseline_vector = self.X_train.median()

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (l_f, l_s), (s_f, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            exp_size = min(top_k, len(self.features) - 1)

            for seed in self.lime_explainers:
                scores, trending = ABPC(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], baseline_vector)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC score"].append(scores)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC trending"].append(trending)

            scores, trending = ABPC(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], baseline_vector)
            shap_fid["ABPC score"].append(scores)
            shap_fid["ABPC trending"].append(trending)

            for seed in self.lime_explainers:
                scores, trending = RoAR(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], self.X_train, self.y_train)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR score"].append(scores)
                lime_fids[f"LIME (Seed: {seed})"]["RoAR trending"].append(trending)

            scores, trending = RoAR(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], self.X_train, self.y_train)
            shap_fid["RoAR score"].append(scores)
            shap_fid["RoAR trending"].append(trending)

        if display:
            data_list = []
            for i in range(self.num_instance):
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "ABPC", "Score": lime_fid["ABPC score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "ABPC", "Score": shap_fid["ABPC score"][i]})
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "RoAR", "Score": lime_fid["RoAR score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "RoAR", "Score": shap_fid["RoAR score"][i]})
                
            df_box = pd.DataFrame(data_list)
            
            mean_lime_abpc = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_abpc[label] = np.mean(lime_fid["ABPC trending"], axis=0)
            mean_shap_abpc = np.mean(shap_fid["ABPC trending"], axis=0)
            mean_lime_roar = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_roar[label] = np.mean(lime_fid["RoAR trending"], axis=0)
            mean_shap_roar = np.mean(shap_fid["RoAR trending"], axis=0)

            trend_data = []
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                for label, lime_fid in lime_fids.items():
                    trend_data.append({"Explainer": label, "Metric": "ABPC", "k": k_val, "Mean Score": mean_lime_abpc[label][k_idx]})
                    trend_data.append({"Explainer": label, "Metric": "RoAR", "k": k_val, "Mean Score": mean_lime_roar[label][k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "ABPC", "k": k_val, "Mean Score": mean_shap_abpc[k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "RoAR", "k": k_val, "Mean Score": mean_shap_roar[k_idx]})
                
            df_trend = pd.DataFrame(trend_data)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle("Fidelity/Faithfulness Evaluation", fontsize=16, fontweight="bold")
            
            df_abpc_box = df_box[df_box["Metric"] == "ABPC"]
            df_roar_box = df_box[df_box["Metric"] == "RoAR"]
            
            sns.boxplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 0].set_title("ABPC Distribution (Overall)", fontsize=13)
            axes[0, 0].set_ylabel("Overall ABPC Score", fontsize=12)
            axes[0, 0].set_xlabel("Explainer Method", fontsize=12)
            
            sns.boxplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[0, 1], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_roar_box, x="Explainer", y="Score", ax=axes[0, 1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 1].set_title("RoAR Distribution (Overall)", fontsize=13)
            axes[0, 1].set_ylabel("Overall RoAR Score", fontsize=12)
            axes[0, 1].set_xlabel("Explainer Method", fontsize=12)
        
            df_abpc_trend = df_trend[df_trend["Metric"] == "ABPC"]
            df_roar_trend = df_trend[df_trend["Metric"] == "RoAR"]
            
            sns.lineplot(data=df_abpc_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 0], palette="Set2")
            axes[1, 0].set_title("ABPC Trend (Mean over Instances)", fontsize=13)
            axes[1, 0].set_ylabel("Mean Cumulative ABPC", fontsize=12)
            axes[1, 0].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 0].set_xticks(range(1, exp_size + 1))
            
            sns.lineplot(data=df_roar_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 1], palette="Set2")
            axes[1, 1].set_title("RoAR Trend (Mean over Instances)", fontsize=13)
            axes[1, 1].set_ylabel("Mean Cumulative RoAR", fontsize=12)
            axes[1, 1].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 1].set_xticks(range(1, exp_size + 1))
            
            plt.tight_layout()
            plt.show()
        
        return lime_fids, shap_fid

    def fairness_test(self, sensitive_features, display=False):
        fairness_data = {"SHAP": []}
        for seed in self.lime_explainers:
            fairness_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            for seed in self.lime_explainers:
                fairness_data[f"LIME (Seed: {seed})"].append(fairness_metric(l_f[seed], sensitive_features))
            fairness_data["SHAP"].append(fairness_metric(s_f, sensitive_features))

        if display:
            df = pd.DataFrame(fairness_data).melt(var_name="Explainer", value_name="Fairness Score")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Fairness Score", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Fairness Score", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Fairness Test")
            plt.ylabel("Fairness Score (Lower is better)")
            plt.tight_layout()
            plt.show()

        return fairness_data

    def do_all_tests(self, sensitive_features, noise_stds=[0.1, 0.5, 1.0], display=False, top_k=5):
        consistency_results = self.consistency_test(display=display, top_k=top_k)
        robustness_results = []
        sensitivity_results = []
        for noise_std in noise_stds:
            robustness_results.append(self.robustness_test(noise_std=noise_std, display=display, top_k=top_k))
            sensitivity_results.append(self.sensitivity_test(noise_std=noise_std, display=display))
        fidelity_results = self.fidelity_test(display=display, top_k=top_k)
        fairness_results = self.fairness_test(sensitive_features=sensitive_features, display=display)
    
        return {
            "consistency": consistency_results,
            "robustness": robustness_results,
            "sensitivity": sensitivity_results,
            "fidelity": fidelity_results,
            "fairness": fairness_results
        }

class UtilityAlignedExplainerUser:
    def __init__(self, model, features, actions, X_train, X_test, causal_model, utility_matrix, n_samples=100, information_method="shannon", information_bound=np.inf):
        self.model = model
        self.features = features
        self.actions = actions
        self.X_train = X_train
        self.X_test = X_test
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.n_samples = n_samples
        self.information_method = information_method
        self.information_bound = information_bound
        self.explainer = UtilityAlignedTabularExplainer(
            model = self.model,
            X_train = self.X_train,
            features = self.features,
            actions = self.actions,
            causal_model = self.causal_model,
            utility_matrix = self.utility_matrix,
            information_method = information_method,
            information_bound = self.information_bound
        )

#    def similarity_measurement(self, display=False):
#        bu_jaccard_scores = []
#        bu_utility_diffs = []
#        bu_info_diffs = []
#        bu_action_matches = []
#
#        td_jaccard_scores = []
#        td_utility_diffs = []
#        td_info_diffs = []
#        td_action_matches = []
#        
#        num_instances = min(len(self.X_test), self.n_samples)
#        instance_indices = list(range(1, num_instances + 1))
#        
#        for i in range(num_instances):
#            inst_1d = self.X_test.iloc[i]
#            inst_2d = self.X_test.iloc[[i]]
#            
#            # --- Optimal Search ---
#            optimal_exps = self.explainer.explain_instance(inst_1d)
#            best_optimal = optimal_exps[0]
#            opt_f = list(best_optimal["features"])
#            opt_u = best_optimal["utility score"]
#            opt_i = best_optimal["information score"]
#            
#            opt_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, opt_f)
#            opt_action = np.argmax(self.utility_matrix @ opt_probs.T)
#            
#            # --- Bottom-Up Heuristic ---
#            best_bottom_up_heuristic = self.explainer.explain_instance_by_bottom_up_heuristic(inst_1d)
#            bu_heur_f_raw = best_bottom_up_heuristic["features"]
#            bu_heur_f = list(bu_heur_f_raw)
#                
#            bu_heur_u = best_bottom_up_heuristic["utility score"]
#            bu_heur_i = best_bottom_up_heuristic["information score"]
#            
#            bu_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, bu_heur_f)
#            bu_action = np.argmax(self.utility_matrix @ bu_probs.T)
#            
#            bu_jaccard_scores.append(jaccard_similarity(opt_f, bu_heur_f))
#            bu_utility_diffs.append(abs(opt_u - bu_heur_u))
#            bu_info_diffs.append(abs(opt_i - bu_heur_i))
#            bu_action_matches.append(1 if opt_action == bu_action else 0)
#
#            # --- Top-Down Heuristic ---
#            best_top_down_heuristic = self.explainer.explain_instance_by_top_down_heuristic(inst_1d)
#            td_heur_f_raw = best_top_down_heuristic["features"]
#            td_heur_f = list(td_heur_f_raw)
#                
#            td_heur_u = best_top_down_heuristic["utility score"]
#            td_heur_i = best_top_down_heuristic["information score"]
#            
#            td_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, td_heur_f)
#            td_action = np.argmax(self.utility_matrix @ td_probs.T)
#            
#            td_jaccard_scores.append(jaccard_similarity(opt_f, td_heur_f))
#            td_utility_diffs.append(abs(opt_u - td_heur_u))
#            td_info_diffs.append(abs(opt_i - td_heur_i))
#            td_action_matches.append(1 if opt_action == td_action else 0)
#            
#        bottom_up_results_data = {
#            "jaccard_similarity": bu_jaccard_scores,
#            "utility_difference": bu_utility_diffs,
#            "information_difference": bu_info_diffs,
#            "action_match": bu_action_matches
#        }
#
#        top_down_results_data = {
#            "jaccard_similarity": td_jaccard_scores,
#            "utility_difference": td_utility_diffs,
#            "information_difference": td_info_diffs,
#            "action_match": td_action_matches
#        }
#        
#        if display:
#            
#            # Consolidate data for seaborn plotting
#            data_list = []
#            for i, idx in enumerate(instance_indices):
#                # Bottom-up data
#                data_list.append({"Strategy": "Bottom-up", "Metric": "Jaccard Similarity", "Score": bu_jaccard_scores[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Bottom-up", "Metric": "Utility Abs Diff", "Score": bu_utility_diffs[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Bottom-up", "Metric": "Info Abs Diff", "Score": bu_info_diffs[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Bottom-up", "Metric": "Action Match", "Score": bu_action_matches[i], "Instance Index": idx})
#                
#                # Top-down data
#                data_list.append({"Strategy": "Top-down", "Metric": "Jaccard Similarity", "Score": td_jaccard_scores[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Top-down", "Metric": "Utility Abs Diff", "Score": td_utility_diffs[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Top-down", "Metric": "Info Abs Diff", "Score": td_info_diffs[i], "Instance Index": idx})
#                data_list.append({"Strategy": "Top-down", "Metric": "Action Match", "Score": td_action_matches[i], "Instance Index": idx})
#                
#            df = pd.DataFrame(data_list)
#            
#            df_jaccard = df[df["Metric"] == "Jaccard Similarity"]
#            df_utility = df[df["Metric"] == "Utility Abs Diff"]
#            df_info = df[df["Metric"] == "Info Abs Diff"]
#            df_action = df[df["Metric"] == "Action Match"]
#            
#            sns.set_theme(style="whitegrid")
#            
#            fig, axes = plt.subplots(2, 4, figsize=(26, 12))
#            fig.suptitle("Similarity Measurement: Optimal vs Heuristic Search", fontsize=16, fontweight="bold")
#            
#            # --- Boxplots (Top Row) ---
#            sns.boxplot(data=df_jaccard, x="Strategy", y="Score", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5)
#            sns.stripplot(data=df_jaccard, x="Strategy", y="Score", ax=axes[0, 0], color=".2", size=4, alpha=0.6, jitter=True, dodge=True)
#            axes[0, 0].set_title("Feature Sets Jaccard Similarity", fontsize=13)
#            axes[0, 0].set_ylim(-0.05, 1.05)
#            axes[0, 0].set_ylabel("Jaccard Score", fontsize=12)
#            axes[0, 0].set_xlabel("")
#            
#            sns.boxplot(data=df_utility, x="Strategy", y="Score", ax=axes[0, 1], palette="Set2", showfliers=False, width=0.5)
#            sns.stripplot(data=df_utility, x="Strategy", y="Score", ax=axes[0, 1], color=".2", size=4, alpha=0.6, jitter=True, dodge=True)
#            axes[0, 1].set_title("Utility Score Absolute Difference", fontsize=13)
#            axes[0, 1].set_ylabel("Absolute Difference", fontsize=12)
#            axes[0, 1].set_xlabel("")
#            
#            sns.boxplot(data=df_info, x="Strategy", y="Score", ax=axes[0, 2], palette="Set2", showfliers=False, width=0.5)
#            sns.stripplot(data=df_info, x="Strategy", y="Score", ax=axes[0, 2], color=".2", size=4, alpha=0.6, jitter=True, dodge=True)
#            axes[0, 2].set_title("Information Score Absolute Difference", fontsize=13)
#            axes[0, 2].set_ylabel("Absolute Difference", fontsize=12)
#            axes[0, 2].set_xlabel("")
#
#            sns.barplot(data=df_action, x="Strategy", y="Score", ax=axes[0, 3], palette="Set2", errorbar=None)
#            axes[0, 3].set_title("Action Match Rate", fontsize=13)
#            axes[0, 3].set_ylim(0, 1.05)
#            axes[0, 3].set_ylabel("Match Rate", fontsize=12)
#            axes[0, 3].set_xlabel("")
#
#            # --- Lineplots (Bottom Row) ---
#            sns.lineplot(data=df_jaccard, x="Instance Index", y="Score", hue="Strategy", ax=axes[1, 0], marker="o", alpha=0.8, palette="Set2")
#            axes[1, 0].set_title("Jaccard Trend across Instances", fontsize=13)
#            axes[1, 0].set_ylim(-0.05, 1.05)
#            axes[1, 0].set_ylabel("Jaccard Score", fontsize=12)
#            axes[1, 0].set_xlabel("Instance Index", fontsize=12)
#            
#            sns.lineplot(data=df_utility, x="Instance Index", y="Score", hue="Strategy", ax=axes[1, 1], marker="o", alpha=0.8, palette="Set2")
#            axes[1, 1].set_title("Utility Difference Trend across Instances", fontsize=13)
#            axes[1, 1].set_ylabel("Absolute Difference", fontsize=12)
#            axes[1, 1].set_xlabel("Instance Index", fontsize=12)
#            
#            sns.lineplot(data=df_info, x="Instance Index", y="Score", hue="Strategy", ax=axes[1, 2], marker="o", alpha=0.8, palette="Set2")
#            axes[1, 2].set_title("Information Difference Trend across Instances", fontsize=13)
#            axes[1, 2].set_ylabel("Absolute Difference", fontsize=12)
#            axes[1, 2].set_xlabel("Instance Index", fontsize=12)
#            
#            sns.lineplot(data=df_action, x="Instance Index", y="Score", hue="Strategy", ax=axes[1, 3], marker="o", alpha=0.8, palette="Set2")
#            axes[1, 3].set_title("Action Match Trend across Instances", fontsize=13)
#            axes[1, 3].set_ylim(-0.05, 1.05)
#            axes[1, 3].set_ylabel("Match (1=Yes, 0=No)", fontsize=12)
#            axes[1, 3].set_xlabel("Instance Index", fontsize=12)
#            
#            plt.tight_layout()
#            plt.show()
#            
#        return top_down_results_data