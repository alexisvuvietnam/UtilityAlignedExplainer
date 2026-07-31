from explainer import *
import matplotlib.pyplot as plt
import lime
import seaborn as sns
import shap
from utils import *

class ExplainerTester:

    def __init__(self, model, features, actions, causal_model, utility_matrix, X_train, X_test, lime_explainer, shap_explainer, n_samples=100):
        self.model = model
        self.features = features
        self.actions = actions
        self.causal_model = causal_model
        self.utility_matrix = utility_matrix
        self.X_train = X_train
        self.X_test = X_test
        self.lime_explainer = lime_explainer
        self.shap_explainer = shap_explainer
        self.causal_explainer = PrototypeTabularExplainer(self.model, self.X_train, self.features, self.actions, self.causal_model, self.utility_matrix)
        self.num_instance = min(len(self.X_test), n_samples)

    ### XAI EMPIRICAL TESTS FROM HERE
    def primary_test(self, display=False):
        base_instance_1d = self.X_test.iloc[0]
        base_instance_2d = self.X_test.iloc[[0]]
        base_causal_explanation = self.causal_explainer.explain_instance(base_instance_2d)
        base_lime_explanation = lime_explanation_form(self.model, self.lime_explainer, base_instance_1d)
        base_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, base_instance_2d)
        
        base_causal_sample = [v["features"] for v in base_causal_explanation]
        base_lime_sample = [v["feature"] for v in base_lime_explanation]
        base_shap_sample = [v["feature"] for v in base_shap_explanation]
        exp_size = min(len(self.features) - 1, len(base_causal_explanation) - 1)
        
        causal_data = {"jaccard": [], "spearman": []}
        lime_data = {"jaccard": [], "spearman": []}
        shap_data = {"jaccard": [], "spearman": []}
        
        for i in range(1, self.num_instance + 1):
            example_instance_1d = self.X_test.iloc[i]
            example_instance_2d = self.X_test.iloc[[i]]
            new_causal_explanation = self.causal_explainer.explain_instance(example_instance_2d)
            new_lime_explanation = lime_explanation_form(self.model, self.lime_explainer, example_instance_1d)
            new_shap_explanation = shap_explanation_form(self.model, self.shap_explainer, example_instance_2d)
            
            new_causal_sample = [v["features"] for v in new_causal_explanation]
            new_lime_sample = [v["feature"] for v in new_lime_explanation]
            new_shap_sample = [v["feature"] for v in new_shap_explanation]
            
            causal_data["jaccard"].append(jaccard_similarity(base_causal_sample[:exp_size], new_causal_sample[:exp_size]))
            lime_data["jaccard"].append(jaccard_similarity(base_lime_sample[:exp_size], new_lime_sample[:exp_size]))
            shap_data["jaccard"].append(jaccard_similarity(base_shap_sample[:exp_size], new_shap_sample[:exp_size]))
            
            causal_data["spearman"].append(spearman_similarity(base_causal_sample, new_causal_sample))
            lime_data["spearman"].append(spearman_similarity(base_lime_sample, new_lime_sample))
            shap_data["spearman"].append(spearman_similarity(base_shap_sample, new_shap_sample))

        if display:
            data_list = []
            for i in range(self.num_instance):
                data_list.append({"Explainer": "Prototype", "Metric": "Jaccard", "Score": causal_data["jaccard"][i]})
                data_list.append({"Explainer": "LIME", "Metric": "Jaccard", "Score": lime_data["jaccard"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Jaccard", "Score": shap_data["jaccard"][i]})
                
                data_list.append({"Explainer": "Prototype", "Metric": "Spearman", "Score": causal_data["spearman"][i]})
                data_list.append({"Explainer": "LIME", "Metric": "Spearman", "Score": lime_data["spearman"][i]})
                data_list.append({"Explainer": "SHAP", "Metric": "Spearman", "Score": shap_data["spearman"][i]})
                
            df = pd.DataFrame(data_list)
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Robustness Test Between Explainers", fontsize=16, fontweight="bold")
            
            sns.barplot(
                data=df[df["Metric"] == "Jaccard"], 
                x="Explainer", y="Score", 
                ax=axes[0], palette="Set2",
                errorbar="sd", capsize=0.1
            )
            axes[0].set_title("Jaccard Similarity\n(Higher is more robust)", fontsize=13)
            axes[0].set_ylabel("Jaccard Score", fontsize=12)
            axes[0].set_xlabel("Explainer Method", fontsize=12)
            axes[0].set_ylim(-0.05, 1.05)
            
            sns.barplot(
                data=df[df["Metric"] == "Spearman"], 
                x="Explainer", y="Score", 
                ax=axes[1], palette="Set2",
                errorbar="sd", capsize=0.1
            )
            axes[1].set_title("Spearman Rank Correlation\n(Closer to 1 is more robust)", fontsize=13)
            axes[1].set_ylabel("Spearman Score", fontsize=12)
            axes[1].set_xlabel("Explainer Method", fontsize=12)
            axes[1].set_ylim(-1.05, 1.05)
            
            plt.tight_layout()
            plt.show()

        return (causal_data, lime_data, shap_data)

    def _get_attributions(self, instance_1d, instance_2d):
        proto_attr_raw = self.causal_explainer.extract_attribution(instance_2d)
        proto_features = np.array([v["features"] for v in proto_attr_raw])
        proto_scores = np.array([v["utility score"] for v in proto_attr_raw])

        lime_attr_raw = lime_explanation_form(self.model, self.lime_explainer, instance_1d)
        lime_features = np.array([v["feature"] for v in lime_attr_raw])
        lime_scores = np.array([v["absolute score"] for v in lime_attr_raw])

        shap_attr_raw = shap_explanation_form(self.model, self.shap_explainer, instance_2d)
        shap_features = np.array([v["feature"] for v in shap_attr_raw])
        shap_scores = np.array([v["absolute score"] for v in shap_attr_raw])

        return (proto_features, proto_scores), (lime_features, lime_scores), (shap_features, shap_scores)

    def robustness_test(self, noise_std=0.01, display=False):
        causal_rob = {"jaccard": [], "spearman": []}
        lime_rob = {"jaccard": [], "spearman": []}
        shap_rob = {"jaccard": [], "spearman": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (p_f, _), (l_f, _), (s_f, _) = self._get_attributions(inst_1d, inst_2d)

            numeric_cols = inst_1d.select_dtypes(include=[np.number]).index
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

            (p_f_n, _), (l_f_n, _), (s_f_n, _) = self._get_attributions(inst_1d_noisy, inst_2d_noisy)

            exp_size = len(self.features) // 2
            causal_rob["jaccard"].append(jaccard_similarity(list(p_f)[:exp_size], list(p_f_n)[:exp_size]))
            lime_rob["jaccard"].append(jaccard_similarity(list(l_f)[:exp_size], list(l_f_n)[:exp_size]))
            shap_rob["jaccard"].append(jaccard_similarity(list(s_f)[:exp_size], list(s_f_n)[:exp_size]))

            causal_rob["spearman"].append(spearman_similarity(list(p_f), list(p_f_n)))
            lime_rob["spearman"].append(spearman_similarity(list(l_f), list(l_f_n)))
            shap_rob["spearman"].append(spearman_similarity(list(s_f), list(s_f_n)))

        if display: pass
        
        return causal_rob, lime_rob, shap_rob

    def sensitivity_test(self, noise_std=0.01, display=False):
        sensitivity_data = {"Prototype": [], "LIME": [], "SHAP": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i].copy()
            inst_2d = self.X_test.iloc[[i]].copy()

            (_, p_s), (_, l_s), (_, s_s) = self._get_attributions(inst_1d, inst_2d)

            numeric_cols = inst_1d.select_dtypes(include=[np.number]).index
            inst_1d_noisy = inst_1d.copy()
            inst_1d_noisy[numeric_cols] += np.random.normal(0, noise_std, len(numeric_cols))
            inst_2d_noisy = inst_1d_noisy.to_frame().T

            (_, p_s_n), (_, l_s_n), (_, s_s_n) = self._get_attributions(inst_1d_noisy, inst_2d_noisy)

            sensitivity_data["Prototype"].append(np.linalg.norm(p_s - p_s_n))
            sensitivity_data["LIME"].append(np.linalg.norm(l_s - l_s_n))
            sensitivity_data["SHAP"].append(np.linalg.norm(s_s - s_s_n))

        if display: pass

        return sensitivity_data

    def fidelity_test(self, masking_value=0, display=False):
        """
        Kiểm tra Fidelity và Faithfulness sử dụng ABPC (Area Between Perturbation Curves)
        được cấu thành từ MoRF và LeRF.
        Điểm ABPC càng cao (hoặc MoRF cao, LeRF thấp) thì explainer càng trung thực.
        """
        fidelity_data = {"Prototype": [], "LIME": [], "SHAP": []}

        for i in range(self.num_instance):
            inst_1d = self.X_test.iloc[i]
            inst_2d = self.X_test.iloc[[i]]

            (p_f, p_s), (l_f, l_s), (s_f, s_s) = self._get_attributions(inst_1d, inst_2d)

            proto_abpc = ABPC(self.model, inst_2d, p_f, p_s, masking_value)[cite: 4]
            fidelity_data["Prototype"].append(proto_abpc)

            lime_abpc = ABPC(self.model, inst_2d, l_f, l_s, masking_value)[cite: 4]
            fidelity_data["LIME"].append(lime_abpc)

            shap_abpc = ABPC(self.model, inst_2d, s_f, s_s, masking_value)[cite: 4]
            fidelity_data["SHAP"].append(shap_abpc)

        if display:
            df = pd.DataFrame(fidelity_data).melt(var_name="Explainer", value_name="ABPC Score")
            plt.figure(figsize=(8, 5))
            sns.barplot(data=df, x="Explainer", y="ABPC Score", errorbar="sd", capsize=0.1, palette="Set2")
            plt.title("Fidelity/Faithfulness Test (ABPC)")
            plt.ylabel("Area Between Perturbation Curves (Higher is better)")
            plt.tight_layout()
            plt.show()

        return fidelity_data

    def causality_test(self, display=False):
        pass

    def utility_test(self, display=False):
        pass

class PrototypeExplainerUser:
    pass