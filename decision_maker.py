import ast
from explainer import *
import matplotlib.pyplot as plt
import lime
import numpy as np
import seaborn as sns
import shap
from utils import *

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

    ### XAI EMPIRICAL TESTS FROM HERE
    def consistency_test(self, display=False, top_k=5):
        base_instance_1d = self.X_test.iloc[0]
        base_instance_2d = self.X_test.iloc[[0]]
        base_causal_explanation = self.causal_explainer.explain_instance(base_instance_2d)
        base_lime_explanations = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            base_lime_explanation = lime_explanation_form(self.model, lime_explainer, base_instance_1d)
            base_lime_explanations[seed] = base_lime_explanation
        base_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, base_instance_2d)
        
        base_causal_sample = [v["features"] for v in base_causal_explanation]
        base_lime_samples = dict()
        for seed, base_lime_explanation in base_lime_explanations.items():
            base_lime_sample = [v["feature"] for v in base_lime_explanation]
            base_lime_samples[seed] = base_lime_sample
        base_shap_sample = [v["feature"] for v in base_shap_explanation]
        exp_size = min(top_k, len(self.features) - 1, len(base_causal_explanation) - 1)
        
        causal_data = {"jaccard": [], "spearman": []}
        lime_datas = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            lime_datas[seed] = {"jaccard": [], "spearman": []}
        shap_data = {"jaccard": [], "spearman": []}
        
        for i in range(1, self.num_instance + 1):
            example_instance_1d = self.X_test.iloc[i]
            example_instance_2d = self.X_test.iloc[[i]]
            new_causal_explanation = self.causal_explainer.explain_instance(example_instance_2d)
            new_lime_explanations = dict()
            for seed, lime_explainer in self.lime_explainers.items():
                new_lime_explanation = lime_explanation_form(self.model, lime_explainer, example_instance_1d)
                new_lime_explanations[seed] = new_lime_explanation
            new_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, example_instance_2d)
            
            new_causal_sample = [v["features"] for v in new_causal_explanation]
            new_lime_samples = dict()
            for seed, new_lime_explanation in new_lime_explanations.items():
                new_lime_sample = [v["feature"] for v in new_lime_explanation]
                new_lime_samples[seed] = new_lime_sample
            new_shap_sample = [v["feature"] for v in new_shap_explanation]
            
            causal_data["jaccard"].append(jaccard_similarity(base_causal_sample[:exp_size], new_causal_sample[:exp_size]))
            for seed, lime_data in lime_datas.items():
                lime_data["jaccard"].append(jaccard_similarity(base_lime_samples[seed][:exp_size], new_lime_samples[seed][:exp_size]))
            shap_data["jaccard"].append(jaccard_similarity(base_shap_sample[:exp_size], new_shap_sample[:exp_size]))
            
            causal_data["spearman"].append(spearman_similarity(base_causal_sample, new_causal_sample))
            for seed, lime_data in lime_datas.items():
                lime_data["spearman"].append(spearman_similarity(base_lime_samples[seed], new_lime_samples[seed]))
            shap_data["spearman"].append(spearman_similarity(base_shap_sample, new_shap_sample))

        if display:
            data_list = []
            for i in range(self.num_instance):
                data_list.append({"Explainer": "Utility-aligned", "Metric": "Jaccard", "Score": causal_data["jaccard"][i]})
                for seed, lime_data in lime_datas.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Jaccard", "Score": lime_data["jaccard"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_data["jaccard"][i]})
                
                data_list.append({"Explainer": "Utility-aligned", "Metric": "Spearman", "Score": causal_data["spearman"][i]})
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

        return (causal_data, lime_data, shap_data)

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
        causal_rob = {"jaccard": [], "spearman": []}
        lime_robs = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            lime_robs[seed] = {"jaccard": [], "spearman": []}
        shap_rob = {"jaccard": [], "spearman": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            proto_exp = self.causal_explainer.explain_instance(inst_2d)
            p_f = np.array([v["features"] for v in proto_exp])
            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

            proto_exp_noisy = self.causal_explainer.explain_instance(inst_2d_noisy)
            p_f_n = np.array([v["features"] for v in proto_exp_noisy])
            (l_f_n, _), (s_f_n, _) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

            exp_size = min(top_k, len(self.features) - 1, len(p_f_n) - 1)
            causal_rob["jaccard"].append(jaccard_similarity(list(p_f)[:exp_size], list(p_f_n)[:exp_size]))
            for seed, lime_rob in lime_robs.items():
                lime_robs[seed]["jaccard"].append(jaccard_similarity(list(l_f[seed])[:exp_size], list(l_f_n[seed])[:exp_size]))
            shap_rob["jaccard"].append(jaccard_similarity(list(s_f)[:exp_size], list(s_f_n)[:exp_size]))

            causal_rob["spearman"].append(spearman_similarity(list(p_f), list(p_f_n)))
            for seed, lime_rob in lime_robs.items():
                lime_robs[seed]["spearman"].append(spearman_similarity(list(l_f[seed]), list(l_f_n[seed])))
            shap_rob["spearman"].append(spearman_similarity(list(s_f), list(s_f_n)))

        if display:
            data_list = []
            for i in range(self.num_instance):
                data_list.append({"Explainer": "Utility-aligned", "Metric": "Jaccard", "Score": causal_rob["jaccard"][i]})
                for seed, lime_rob in lime_robs.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Jaccard", "Score": lime_rob["jaccard"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_rob["jaccard"][i]})
                
                data_list.append({"Explainer": "Utility-aligned", "Metric": "Spearman", "Score": causal_rob["spearman"][i]})
                for seed, lime_rob in lime_robs.items():
                    data_list.append({"Explainer": f"LIME (Seed: {seed})", "Metric": "Spearman", "Score": lime_rob["spearman"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_rob["spearman"][i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            fig.suptitle("Robustness Test Between Explainers", fontsize=16, fontweight="bold")
            
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
        
        return causal_rob, lime_robs, shap_rob

    def sensitivity_test(self, noise_std=0.01, display=False):
        sensitivity_data = {"Utility-aligned": [], "SHAP": []}
        for seed, lime_explainer in self.lime_explainers.items():
            sensitivity_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            proto_exp = self.causal_explainer.explain_instance(inst_2d)
            p_s = np.array([v["utility score"] for v in proto_exp])
            (_, l_s), (_, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

            proto_exp_noisy = self.causal_explainer.explain_instance(inst_2d_noisy)
            p_s_n = np.array([v["utility score"] for v in proto_exp_noisy])
            (_, l_s_n), (_, s_s_n) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

            sensitivity_data["Utility-aligned"].append(np.linalg.norm(p_s - p_s_n))
            for seed, lime_explainer in self.lime_explainers.items():
                sensitivity_data[f"LIME (Seed: {seed})"].append(np.linalg.norm(l_s[seed] - l_s_n[seed]))
            sensitivity_data["SHAP"].append(np.linalg.norm(s_s - s_s_n))

        if display:
            df = pd.DataFrame(sensitivity_data).melt(var_name="Explainer", value_name="Sensitivity")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Sensitivity", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Sensitivity", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Sensitivity Test")
            plt.ylabel("L2 Norm of Score Differences")
            plt.tight_layout()
            plt.show()

        return sensitivity_data

    def fidelity_test(self, display=False, top_k=5):
        causal_fid = {"Local MORF": [], "Local LOCO": []}
        lime_fids = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            lime_fids[f"LIME (Seed: {seed})"] = {"Local MORF": [], "Local LOCO": []}
        shap_fid = {"Local MORF": [], "Local LOCO": []}
        baseline_vector = self.X_train.median()

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            proto_exp = self.causal_explainer.explain_instance(inst_2d)
            p_f = np.array([v["features"] for v in proto_exp])
            p_s = np.array([v["utility score"] for v in proto_exp])
            (l_f, l_s), (s_f, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            exp_size = min(top_k, len(self.features) - 1, len(p_f) - 1)

            causal_fid["Local MORF"].append(local_MoRF(self.model, inst_2d, np.array(list(ast.literal_eval(p_f[0]))), p_s[0], baseline_vector, exp_size))
            for seed, lime_explainer in self.lime_explainers.items():
                lime_fids[f"LIME (Seed: {seed})"]["Local MORF"].append(local_MoRF(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], baseline_vector, exp_size))
            shap_fid["Local MORF"].append(local_MoRF(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], baseline_vector, exp_size))

            causal_fid["Local LOCO"].append(local_LOCO(self.model, inst_2d, np.array(list(ast.literal_eval(p_f[0]))), p_s[0], baseline_vector, exp_size, self.X_train, self.y_train))
            for seed, lime_explainer in self.lime_explainers.items():
                lime_fids[f"LIME (Seed: {seed})"]["Local LOCO"].append(local_LOCO(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], baseline_vector, exp_size, self.X_train, self.y_train))
            shap_fid["Local LOCO"].append(local_LOCO(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], baseline_vector, exp_size, self.X_train, self.y_train))

        if display:
            data_list = []
            for i in range(self.num_instance):
                data_list.append({"Explainer": "Utility-aligned", "Metric": "Local MORF", "Score": causal_fid["Local MORF"][i]})
                for seed, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": seed, "Metric": "Local MORF", "Score": lime_fid["Local MORF"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Local MORF", "Score": shap_fid["Local MORF"][i]})

                data_list.append({"Explainer": "Utility-aligned", "Metric": "Local LOCO", "Score": causal_fid["Local LOCO"][i]})
                for seed, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": seed, "Metric": "Local LOCO", "Score": lime_fid["Local LOCO"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Local LOCO", "Score": shap_fid["Local LOCO"][i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            fig.suptitle("Fidelity Test Between Explainers", fontsize=16, fontweight="bold")
            
            df_morf = df[df["Metric"] == "Local MORF"]
            df_loco = df[df["Metric"] == "Local LOCO"]
            
            sns.boxplot(data=df_morf, x="Explainer", y="Score", ax=axes[0], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_morf, x="Explainer", y="Score", ax=axes[0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0].set_title("Local MORF Distribution", fontsize=13)
            axes[0].set_ylabel("Local MORF Score", fontsize=12)
            axes[0].set_xlabel("Explainer Method", fontsize=12)
            axes[0].set_ylim(-0.05, 1.05)
            
            sns.boxplot(data=df_loco, x="Explainer", y="Score", ax=axes[1], palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df_loco, x="Explainer", y="Score", ax=axes[1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[1].set_title("Local LOCO Distribution", fontsize=13)
            axes[1].set_ylabel("Local LOCO Score", fontsize=12)
            axes[1].set_xlabel("Explainer Method", fontsize=12)
            axes[1].set_ylim(-1.05, 1.05)
            
            plt.tight_layout()
            plt.show()
        
        return causal_fid, lime_fids, shap_fid

    def causality_test(self, display=False):
        causality_data = {"Utility-aligned": [], "SHAP": []}
        for seed, lime_explainer in self.lime_explainers.items():
            causality_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            num_critical = len(self.causal_explainer.critical_features)

            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            causality_data["Utility-aligned"].append(1.0)

            lime_cvs = dict()
            for seed, lime_explainer in self.lime_explainers.items():
                lime_cv = self.causal_model.causal_consistency(l_f[seed][:num_critical])
                lime_cvs[seed] = lime_cv
                causality_data[f"LIME (Seed: {seed})"].append(lime_cv)

            shap_cv = self.causal_model.causal_consistency(s_f[:num_critical])
            causality_data["SHAP"].append(shap_cv)

        if display:
            df = pd.DataFrame(causality_data).melt(var_name="Explainer", value_name="Causal Consistency Score")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Causal Consistency Score", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Causal Consistency Score", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Causality Test (Causal Consistency)")
            plt.ylabel("Causal Consistency Score (Higher is better)")
            plt.tight_layout()
            plt.show()

        return causality_data

    def utility_test(self, display=False):
        utility_data = {"Utility-aligned": [], "SHAP": []}
        for seed, lime_explainer in self.lime_explainers.items():
            utility_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]
        
            p_f = self.causal_explainer.explain_instance(inst_2d)
            best_proto_features = list(ast.literal_eval(p_f[0]["features"]))
            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)
    
            prototype_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, best_proto_features)
            utility_data["Utility-aligned"].append(np.max(self.utility_matrix @ prototype_probs.T))

            cv_s_f = []
            for i in s_f:
                if self.causal_model.backdoor_satisfaction(i):
                    cv_s_f.append(i)
            shap_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, cv_s_f)
            utility_data["SHAP"].append(np.max(self.utility_matrix @ shap_probs.T))

            cv_l_fs = dict()
            for seed, lime_explainer in self.lime_explainers.items():
                cv_l_fs[seed] = []
                for i in l_f[seed]:
                    if self.causal_model.backdoor_satisfaction(i):
                        cv_l_fs[seed].append(i)
            for seed, cv_l_f in cv_l_fs.items():
                lime_probs = estimate_interventional_probability_tabular(self.model, self.X_train, inst_2d, cv_l_f)
                utility_data[f"LIME (Seed: {seed})"].append(np.max(self.utility_matrix @ lime_probs.T))

        if display:
            df = pd.DataFrame(utility_data).melt(var_name="Explainer", value_name="Utility Score")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Utility Score", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Utility Score", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Utility Test (Interventional Probability)")
            plt.ylabel("Utility Score (Higher is better)")
            plt.tight_layout()
            plt.show()

        return utility_data

    def fairness_test(self, sensitive_features, display=False):
        fairness_data = {"Utility-aligned": [], "SHAP": []}
        for seed, lime_explainer in self.lime_explainers.items():
            fairness_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            p_f = self.causal_explainer.explain_instance(inst_2d)
            best_proto_features = list(ast.literal_eval(p_f[0]["features"]))
            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            fairness_data["Utility-aligned"].append(fairness_metric(best_proto_features, sensitive_features))
            for seed, lime_explainer in self.lime_explainers.items():
                fairness_data[f"LIME (Seed: {seed})"].append(fairness_metric(l_f[seed], sensitive_features))
            fairness_data["SHAP"].append(fairness_metric(s_f, sensitive_features))

        if display:
            df = pd.DataFrame(fairness_data).melt(var_name="Explainer", value_name="Fairness Score")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Fairness Score", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Fairness Score", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Fairness Test (Interventional Probability)")
            plt.ylabel("Fairness Score (Lower is better)")
            plt.tight_layout()
            plt.show()
        return fairness_data

    def do_all_tests(self, sensitive_features, noise_std=0.01, display=False, top_k=5):
        consistency_results = self.consistency_test(display=display, top_k=top_k)
        robustness_results = self.robustness_test(noise_std=noise_std, display=display, top_k=top_k)
        sensitivity_results = self.sensitivity_test(noise_std=noise_std, display=display)
        fidelity_results = self.fidelity_test(display=display, top_k=top_k)
        causality_results = self.causality_test(display=display)
        utility_results = self.utility_test(display=display)
        fairness_results = self.fairness_test(sensitive_features=sensitive_features, display=display)

        return {
            "consistency": consistency_results,
            "robustness": robustness_results,
            "sensitivity": sensitivity_results,
            "fidelity": fidelity_results,
            "causality": causality_results,
            "utility": utility_results,
            "fairness": fairness_results
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
        for seed, lime_explainer in self.lime_explainers.items():
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
        for seed, lime_explainer in self.lime_explainers.items():
            lime_robs[seed] = {"jaccard": [], "spearman": []}
        shap_rob = {"jaccard": [], "spearman": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (l_f, _), (s_f, _) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

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
            fig.suptitle("Robustness Test Between Explainers", fontsize=16, fontweight="bold")
            
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
        for seed, lime_explainer in self.lime_explainers.items():
            sensitivity_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (_, l_s), (_, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            numeric_cols = inst_2d.select_dtypes(include=["float64", "int64"]).columns
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

            (_, l_s_n), (_, s_s_n) = self._get_lime_shap_attributions(inst_1d_noisy, inst_2d_noisy)

            for seed, lime_explainer in self.lime_explainers.items():
                sensitivity_data[f"LIME (Seed: {seed})"].append(np.linalg.norm(l_s[seed] - l_s_n[seed]))
            sensitivity_data["SHAP"].append(np.linalg.norm(s_s - s_s_n))

        if display:
            df = pd.DataFrame(sensitivity_data).melt(var_name="Explainer", value_name="Sensitivity")
            plt.figure(figsize=(14, 5))
            sns.boxplot(data=df, x="Explainer", y="Sensitivity", palette="Set2", showfliers=False, width=0.5)
            sns.stripplot(data=df, x="Explainer", y="Sensitivity", color=".2", size=5, alpha=0.6, jitter=True)
            plt.title("Sensitivity Test")
            plt.ylabel("L2 Norm of Score Differences")
            plt.tight_layout()
            plt.show()

        return sensitivity_data

    def fidelity_test(self, display=False, top_k=5):
        lime_fids = dict()
        for seed, lime_explainer in self.lime_explainers.items():
            lime_fids[f"LIME (Seed: {seed})"] = {"ABPC score": [], "LOCO score": [], "ABPC trending": [], "LOCO trending": []}
        shap_fid = {"ABPC score": [], "LOCO score": [], "ABPC trending": [], "LOCO trending": []}
        baseline_vector = self.X_train.median()

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (l_f, l_s), (s_f, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            exp_size = min(top_k, len(self.features) - 1)

            for seed, lime_explainer in self.lime_explainers.items():
                scores, trending = ABPC(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], baseline_vector)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC score"].append(scores)
                lime_fids[f"LIME (Seed: {seed})"]["ABPC trending"].append(trending)

            scores, trending = ABPC(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], baseline_vector)
            shap_fid["ABPC score"].append(scores)
            shap_fid["ABPC trending"].append(trending)

            for seed, lime_explainer in self.lime_explainers.items():
                scores, trending = LOCO(self.model, inst_2d, l_f[seed][:exp_size], l_s[seed][:exp_size], baseline_vector, self.X_train, self.y_train)
                lime_fids[f"LIME (Seed: {seed})"]["LOCO score"].append(scores)
                lime_fids[f"LIME (Seed: {seed})"]["LOCO trending"].append(trending)

            scores, trending = LOCO(self.model, inst_2d, s_f[:exp_size], s_s[:exp_size], baseline_vector, self.X_train, self.y_train)
            shap_fid["LOCO score"].append(scores)
            shap_fid["LOCO trending"].append(trending)

        if display:
            data_list = []
            for i in range(self.num_instance):
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "ABPC", "Score": lime_fid["ABPC score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "ABPC", "Score": shap_fid["ABPC score"][i]})
                for label, lime_fid in lime_fids.items():
                    data_list.append({"Explainer": label, "Metric": "LOCO", "Score": lime_fid["LOCO score"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "LOCO", "Score": shap_fid["LOCO score"][i]})
                
            df_box = pd.DataFrame(data_list)
            
            mean_lime_abpc = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_abpc[label] = np.mean(lime_fid["ABPC trending"], axis=0)
            mean_shap_abpc = np.mean(shap_fid["ABPC trending"], axis=0)
            mean_lime_loco = dict()
            for label, lime_fid in lime_fids.items():
                mean_lime_loco[label] = np.mean(lime_fid["LOCO trending"], axis=0)
            mean_shap_loco = np.mean(shap_fid["LOCO trending"], axis=0)

            trend_data = []
            for k_idx in range(exp_size):
                k_val = k_idx + 1
                for label, lime_fid in lime_fids.items():
                    trend_data.append({"Explainer": label, "Metric": "ABPC", "k": k_val, "Mean Score": mean_lime_abpc[label][k_idx]})
                    trend_data.append({"Explainer": label, "Metric": "LOCO", "k": k_val, "Mean Score": mean_lime_loco[label][k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "ABPC", "k": k_val, "Mean Score": mean_shap_abpc[k_idx]})
                trend_data.append({"Explainer": "SHAP", "Metric": "LOCO", "k": k_val, "Mean Score": mean_shap_loco[k_idx]})
                
            df_trend = pd.DataFrame(trend_data)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle("Fidelity/Faithfulness Evaluation", fontsize=16, fontweight="bold")
            
            df_abpc_box = df_box[df_box["Metric"] == "ABPC"]
            df_loco_box = df_box[df_box["Metric"] == "LOCO"]
            
            sns.boxplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_abpc_box, x="Explainer", y="Score", ax=axes[0, 0], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 0].set_title("ABPC Distribution (Overall)", fontsize=13)
            axes[0, 0].set_ylabel("Overall ABPC Score", fontsize=12)
            axes[0, 0].set_xlabel("Explainer Method", fontsize=12)
            
            sns.boxplot(data=df_loco_box, x="Explainer", y="Score", ax=axes[0, 1], palette="Set2", showfliers=False, width=0.5, legend=False)
            sns.stripplot(data=df_loco_box, x="Explainer", y="Score", ax=axes[0, 1], color=".2", size=4, alpha=0.6, jitter=True)
            axes[0, 1].set_title("LOCO Distribution (Overall)", fontsize=13)
            axes[0, 1].set_ylabel("Overall LOCO Score", fontsize=12)
            axes[0, 1].set_xlabel("Explainer Method", fontsize=12)
        
            df_abpc_trend = df_trend[df_trend["Metric"] == "ABPC"]
            df_loco_trend = df_trend[df_trend["Metric"] == "LOCO"]
            
            sns.lineplot(data=df_abpc_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 0], palette="Set2")
            axes[1, 0].set_title("ABPC Trend (Mean over Instances)", fontsize=13)
            axes[1, 0].set_ylabel("Mean Cumulative ABPC", fontsize=12)
            axes[1, 0].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 0].set_xticks(range(1, exp_size + 1))
            
            sns.lineplot(data=df_loco_trend, x="k", y="Mean Score", hue="Explainer", marker="o", ax=axes[1, 1], palette="Set2")
            axes[1, 1].set_title("LOCO Trend (Mean over Instances)", fontsize=13)
            axes[1, 1].set_ylabel("Mean Cumulative LOCO", fontsize=12)
            axes[1, 1].set_xlabel("Number of Masked Features (k)", fontsize=12)
            axes[1, 1].set_xticks(range(1, exp_size + 1))
            
            plt.tight_layout()
            plt.show()
        
        return lime_fids, shap_fid

    def fairness_test(self, sensitive_features, display=False):
        fairness_data = {"SHAP": []}
        for seed, lime_explainer in self.lime_explainers.items():
            fairness_data[f"LIME (Seed: {seed})"] = []

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (_, l_s), (_, s_s) = self._get_lime_shap_attributions(inst_1d, inst_2d)

            for seed, lime_explainer in self.lime_explainers.items():
                fairness_data[f"LIME (Seed: {seed})"].append(fairness_metric(l_s[seed], sensitive_features))
            fairness_data["SHAP"].append(fairness_metric(s_s, sensitive_features))

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

    def do_all_tests(self, sensitive_features, noise_std=0.01, display=False, top_k=5):
        consistency_results = self.consistency_test(display=display, top_k=top_k)
        robustness_results = self.robustness_test(noise_std=noise_std, display=display, top_k=top_k)
        sensitivity_results = self.sensitivity_test(noise_std=noise_std, display=display)
        fidelity_results = self.fidelity_test(display=display, top_k=top_k)
        fairness_results = self.fairness_test(sensitive_features=sensitive_features, display=display)
    
        return {
            "consistency": consistency_results,
            "robustness": robustness_results,
            "sensitivity": sensitivity_results,
            "fidelity": fidelity_results,
            "fairness": fairness_results
        }

class ExplainerUser:
    pass