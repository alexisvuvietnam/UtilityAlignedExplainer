import numpy as np
from scipy.stats import spearmanr

def _get_target_prob(model, instance, target_class):
    probs = model.predict_proba(instance)
    return probs[0, target_class] if probs.ndim == 2 else probs[target_class]

def MoRF(model, x_instance, features, attributions, masking_value):
    rank = np.argsort(attributions)[::-1]
    ranked_features = features[rank]
    
    res = model.predict(x_instance)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_instance, target_class)
    
    instance = x_instance.copy()
    morf_score = 0.0
    
    for k in range(len(features)):
        feature_to_mask = ranked_features[k]
        instance[feature_to_mask] = masking_value
        
        new_prob = _get_target_prob(model, instance, target_class)
        morf_score += (original_prob - new_prob)
        
    return morf_score / (len(features) + 1)
    
def LeRF(model, x_instance, features, attributions, masking_value):
    rank = np.argsort(attributions)
    ranked_features = features[rank]
    
    res = model.predict(x_instance)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_instance, target_class)
    
    instance = x_instance.copy()
    lerf_score = 0.0
    
    for k in range(len(features)):
        feature_to_mask = ranked_features[k]
        instance[feature_to_mask] = masking_value
        
        new_prob = _get_target_prob(model, instance, target_class)
        lerf_score += (original_prob - new_prob)
        
    return lerf_score / (len(features) + 1)

def ABPC(model, x_instance, features, attributions, masking_value):
    return MoRF(model, x_instance, features, attributions, masking_value) - LeRF(model, x_instance, features, attributions, masking_value)

# Agreement-Disagreement
def normalize_attribution(values, eps = 1e-12):
    values = np.asarray(values, dtype=float)
    return values / (np.abs(values).sum() + eps)

def top_k_jaccard(a, b, k: int = 5):
    top_a = set(np.argsort(np.abs(a))[-k:].flatten())
    top_b = set(np.argsort(np.abs(b))[-k:].flatten())
    return len(top_a & top_b) / len(top_a | top_b)

def weighted_sign_contradiction(a, b, threshold = 0.01):
    a = normalize_attribution(a)
    b = normalize_attribution(b)
    relevant = (np.abs(a) >= threshold) & (np.abs(b) >= threshold)
    if not np.any(relevant):
        return np.nan
    weights = np.minimum(np.abs(a), np.abs(b))
    contradiction = (a * b) < 0
    return float(np.sum(weights[relevant] * contradiction[relevant]) / np.sum(weights[relevant]))

def signed_spearman(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else np.nan

def absolute_spearman(a, b):
    value = spearmanr(np.abs(a), np.abs(b)).statistic
    return float(value) if np.isfinite(value) else np.nan

# Stability and Consistency
def variant_robustness(model, explainer, X_test, random_state=42):
    pass