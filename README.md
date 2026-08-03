# A Causal Decision-theoric Framework for Utility-aligned AI (local) explanation (over tabular data)

The structure of the framework:
- `utils.py`: This file contains mathematical tools to build the explainer (causal graph for causal model, Shannon and Gini's formulas for information quantity, etc...)
- `explainer.py`: This file contains the explainer pursuing the aim of the internship subject. So far, an explainer for tabular data is initially completed.
- `decision_maker.py`: This file plays a role as a "user" of explainers. There are two "agents" (to make everything funnier, not in the use of any agentic nor multi-agent system): an "agent" who tests the different XAI methods and the other plays a role of a decision-maker.
- `empirical_work_tabular.py`: This file represents the empirical studies, between LIME, SHAP and the initially created Utility-aligned Explainer.

The aim of this empirical study: Measure the consistency, robustness, sensitivity, fidelity/faithfulness, causal validity and causal decision utility of explanations created from LIME, SHAP and Utility-aligned Explainer.
