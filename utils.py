import itertools
import numpy as np
import pandas as pd
import pyagrum as gum
from sklearn.base import BaseEstimator, clone
import xgboost as xgb

def shannon_surprisal(old_prob, new_prob, base=np.e):
    old_log = 0 if old_prob == 0 else np.log(old_prob)/np.log(base)
    new_log = 0 if new_prob == 0 else np.log(new_prob)/np.log(base)
    return new_log - old_log

def shannon_entropy(probs, base=np.e):
    probs = np.asarray(probs, dtype=float)
    assert np.isclose(np.sum(probs), 1.0), "This probability vector is not valid"
    nz_probs = probs[probs > 0]
    log_probs = np.log(nz_probs) / np.log(base)
    return -np.sum(nz_probs * log_probs)

def hick_entropy(probs, base=np.e):
    probs = np.asarray(probs, dtype=float)
    assert np.isclose(np.sum(probs), 1.0), "This probability vector is not valid"
    incremental_probs = probs + 1
    log_probs = np.log(incremental_probs) / np.log(base)
    return np.sum(probs * log_probs) + shannon_entropy(probs, base=base)

def bias_entropy(old_probs, new_probs, base=np.e):
    old_probs = np.asarray(old_probs, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)
    assert np.isclose(np.sum(old_probs), 1.0), "This probability vector is not valid"
    assert np.isclose(np.sum(new_probs), 1.0), "This probability vector is not valid"
    epsilon = 1e-15
    p = new_probs.copy()
    q = old_probs.copy()
    p = np.clip(p, epsilon, 1.0)
    #q = np.clip(q, epsilon, 1.0)
    #old_log_probs = np.log(q) / np.log(base)
    new_log_probs = np.log(p) / np.log(base)
    return - np.sum(q * new_log_probs) #+ np.sum(q * old_log_probs)

def kullback_leibler_divergence(old_probs, new_probs, base=np.e):
    old_probs = np.asarray(old_probs, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)
    assert np.isclose(np.sum(old_probs), 1.0), "This probability vector is not valid"
    assert np.isclose(np.sum(new_probs), 1.0), "This probability vector is not valid"
    mask = new_probs > 0
    p = new_probs[mask]
    q = old_probs[mask]
    epsilon = 1e-15
    q = np.clip(q, epsilon, 1.0)
    old_log_probs = np.log(q) / np.log(base)
    new_log_probs = np.log(p) / np.log(base)
    return - np.sum(q * new_log_probs) + np.sum(q * old_log_probs)

def jeffrey_divergence(old_probs, new_probs, base=np.e):
    old_probs = np.asarray(old_probs, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)
    assert np.isclose(np.sum(old_probs), 1.0), "This probability vector is not valid"
    assert np.isclose(np.sum(new_probs), 1.0), "This probability vector is not valid"
    epsilon = 1e-15
    p = new_probs.copy()
    q = old_probs.copy()
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    old_log_probs = np.log(q) / np.log(base)
    new_log_probs = np.log(p) / np.log(base)
    return np.sum((p - q) * (new_log_probs - old_log_probs))

def gini_impurity(probs):
    probs = np.asarray(probs, dtype=float)
    assert np.isclose(np.sum(probs), 1.0), "This probability vector is not valid"
    return 1 - np.sum(probs ** 2)

def entropy_by_hick(allow_rational_time, observation_time, **kwargs):
    assert observation_time > 0, "Are you an alien?"
    if "reflexion_time" in kwargs:
        assert kwargs["reflexion_time"] >= 0, "Are you an alien?"
        return (allow_rational_time - kwargs["reflexion_time"]) / observation_time
    return allow_rational_time / observation_time

def entropy_by_weber_fechner(saliency_size, observation_size, base=np.e):
    assert observation_size > 0, "Are you blind?"
    assert saliency_size > 0, "Trivial explanation"
    if base == 10:
        return np.log10(observation_size) - np.log10(saliency_size)    
    return np.log(observation_size) - np.log(saliency_size)

def get_combinations_up_to_k(data, k):
    result = []
    for r in range(1, k + 1):
        result.extend(itertools.combinations(data, r))
    return result

def _get_target_prob(model, instance, target_class):
    probs = model.predict_proba(instance)
    return probs[0, target_class] if probs.ndim == 2 else probs[target_class]

def MoRF(model, x_instance, features, attributions, baseline_vector):
    rank = np.argsort(attributions)[::-1]
    ranked_features = features[rank]
    
    res = model.predict(x_instance)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_instance, target_class)
    
    instance = x_instance.copy()
    morf_score = 0.0
    morf_cumul = []
    
    for k in range(len(features)):
        feature_to_mask = ranked_features[k]
        instance[feature_to_mask] = baseline_vector[feature_to_mask]
        
        new_prob = _get_target_prob(model, instance, target_class)
        morf_score += (original_prob - new_prob)
        morf_cumul.append(original_prob - new_prob)

    return morf_score / (len(features) + 1), np.array(morf_cumul)
    
def LeRF(model, x_instance, features, attributions, baseline_vector):
    rank = np.argsort(attributions)
    ranked_features = features[rank]
    
    res = model.predict(x_instance)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_instance, target_class)
    
    instance = x_instance.copy()
    lerf_score = 0.0
    lerf_cumul = []
    
    for k in range(len(features)):
        feature_to_mask = ranked_features[k]
        instance[feature_to_mask] = baseline_vector[feature_to_mask]
        
        new_prob = _get_target_prob(model, instance, target_class)
        lerf_score += (original_prob - new_prob)
        lerf_cumul.append(original_prob - new_prob)

    return lerf_score / (len(features) + 1), np.array(lerf_cumul)

def ABPC(model, x_instance, features, attributions, baseline_vector):
    mo, morf_cumul = MoRF(model, x_instance, features, attributions, baseline_vector)
    le, lerf_cumul = LeRF(model, x_instance, features, attributions, baseline_vector)
    return mo - le, morf_cumul - lerf_cumul

def RoAR(model, x_instance, features, attributions, X_train, y_train):
    rank = np.argsort(attributions)[::-1]
    ranked_features = np.array(features)[rank]
    
    x_inst_2d = x_instance.to_frame().T if isinstance(x_instance, pd.Series) else (
        x_instance.reshape(1, -1) if isinstance(x_instance, np.ndarray) and x_instance.ndim == 1 else x_instance
    )
    
    res = model.predict(x_inst_2d)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_inst_2d, target_class)
    
    roar_score = 0.0
    roar_cumul = []
    
    for k in range(len(features)):
        features_to_drop = ranked_features[:k+1]
        
        if isinstance(X_train, pd.DataFrame):
            X_train_k = X_train.drop(columns=features_to_drop)
            instance_k = x_instance.drop(columns=features_to_drop) if isinstance(x_instance, pd.DataFrame) else x_instance.drop(labels=features_to_drop)
        else:
            X_train_k = np.delete(X_train, features_to_drop, axis=1)
            instance_k = np.delete(x_instance, features_to_drop, axis=1 if x_instance.ndim == 2 else 0)

        if isinstance(model, BaseEstimator):
            cloned_model = clone(model)
        elif isinstance(model, xgb.XGBClassifier):
            cloned_model = xgb.XGBClassifier()
            cloned_model.set_params(**model.get_params())
        else:
            raise ValueError("Unsupported model type. Please provide a scikit-learn estimator or an XGBoost classifier.")

        cloned_model.fit(X_train_k, y_train)    
        
        inst_2d = instance_k.to_frame().T if isinstance(instance_k, pd.Series) else (
            instance_k.reshape(1, -1) if isinstance(instance_k, np.ndarray) and instance_k.ndim == 1 else instance_k
        )
        new_prob = _get_target_prob(cloned_model, inst_2d, target_class)

        roar_score += (original_prob - new_prob)
        roar_cumul.append(original_prob - new_prob)

    return roar_score / (len(features) + 1), np.array(roar_cumul)

def local_MoRF(model, x_instance, features, attributions, baseline_vector, k):
    rank = np.argsort(attributions)[::-1]
    ranked_features = features[rank][:k]
    
    res = model.predict(x_instance)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_instance, target_class)
    
    instance = x_instance.copy()
    
    for feature_to_mask in ranked_features:
        instance[feature_to_mask] = baseline_vector[feature_to_mask]
        
    new_prob = _get_target_prob(model, instance, target_class)
        
    return original_prob - new_prob

def local_RoAR(model, x_instance, features, attributions, k, X_train, y_train):
    rank = np.argsort(attributions)[::-1]
    ranked_features = np.array(features)[rank][:k]
    
    x_inst_2d = x_instance.to_frame().T if isinstance(x_instance, pd.Series) else (
        x_instance.reshape(1, -1) if isinstance(x_instance, np.ndarray) and x_instance.ndim == 1 else x_instance
    )
    
    res = model.predict(x_inst_2d)
    target_class = res[0] if isinstance(res, (np.ndarray, list)) else res
    original_prob = _get_target_prob(model, x_inst_2d, target_class)
    
    if isinstance(X_train, pd.DataFrame):
        X_train_clone = X_train.drop(columns=ranked_features)
        instance = x_instance.drop(columns=ranked_features) if isinstance(x_instance, pd.DataFrame) else x_instance.drop(labels=ranked_features)
    else:
        X_train_clone = np.delete(X_train, ranked_features, axis=1)
        instance = np.delete(x_instance, ranked_features, axis=1 if x_instance.ndim == 2 else 0)

    if isinstance(model, BaseEstimator):
        cloned_model = clone(model)
    elif isinstance(model, xgb.XGBClassifier):
        cloned_model = xgb.XGBClassifier()
        cloned_model.set_params(**model.get_params())
    else:
        raise ValueError("Unsupported model type. Please provide a scikit-learn estimator or an XGBoost classifier.")

    cloned_model.fit(X_train_clone, y_train)    
    
    inst_2d = instance.to_frame().T if isinstance(instance, pd.Series) else (
        instance.reshape(1, -1) if isinstance(instance, np.ndarray) and instance.ndim == 1 else instance
    )
    new_prob = _get_target_prob(cloned_model, inst_2d, target_class)
        
    return original_prob - new_prob

def jaccard_similarity(A, B):
    return len(set(A) & set(B)) / max(len(set(A) | set(B)), 1e-16)

def jaccard_distance(A, B):
    return 1 - jaccard_similarity(A, B)

def spearman_similarity(A, B):
    C = set(A) & set(B)
    assert C != set(), "There is no similarity to compare"
    return 1 - (6 * sum((A.index(x) - B.index(x))**2 for x in C)) / max(len(A) * (len(A)**2 - 1), 1e-16)

def fairness_metric(explanation, sensitive_features):
    if set(explanation) & set(sensitive_features) == set():
        return 0.0
    return len(set(explanation) & set(sensitive_features)) / len(set(sensitive_features))

class CausalModel:

    def __init__(self, features, actions):
        self.regimes = set(map(lambda x: f"F_{x}", features))
        self.treatments = set(features)
        self.outcomes = set(actions)
        self.nodes = self.treatments | self.regimes | self.outcomes
        #self.edges = []
        self.graph = gum.DAG()
        self.id = set(enumerate(self.nodes))
        self.treatmentId = [i for (i, name) in self.id if name in self.treatments]
        self.outcomeId = [i for (i, name) in self.id if name in self.outcomes]
        for i, name in self.id:
            self.graph.addNodeWithId(i)
            self.graph.setName(i, name)
        for i in self.treatmentId:
            name = self.graph.nameFromId(i)
            j = self.graph.idFromName(f"F_{name}")
            self.graph.addArc(j, i)

    def add_causal_edge(self, feature_from, feature_to):
        assert feature_from in self.nodes and feature_to in self.nodes, "Those things do not exist"
        i = self.graph.idFromName(feature_from)
        j = self.graph.idFromName(feature_to)
        assert i is not None and j is not None, "Those things do not exist"
        self.graph.addArc(i, j)

    def causal_validity(self, feature):
        assert feature in self.treatments, "This feature does not exist"
        featureId = self.graph.idFromName(feature)
        regimeId = self.graph.idFromName(f"F_{feature}")
        remained_treatmentId = self.treatmentId[:]
        remained_treatmentId.remove(featureId)
        #return self.graph.dSeparation(regimeId, remained_treatmentId) and self.graph.dSeparation(regimeId, self.outcomeId, self.treatmentId)
        return self.graph.dSeparation(regimeId, remained_treatmentId) and not self.graph.dSeparation(regimeId, self.outcomeId, featureId)
        #return self.graph.dSeparation(regimeId, remained_treatmentId) and not self.graph.dSeparation(regimeId, self.outcomeId)

    def extract_parents(self, feature):
        assert feature in self.treatments, "This feature does not exist"
        parents = set()
        featureId = self.graph.idFromName(feature)
        parentIds = self.graph.parents(featureId)
        for i in parentIds:
            parent = self.graph.nameFromId(i)
            if parent not in self.regimes:
                parents = parents | {parent}
        return parents

    def possible_actions(self, explanation_signals):
        assert (self.causal_validity(e) for e in explanation_signals), "This signal is in valid"
        action_sets = set()
        for feature in explanation_signals:
            featureId = self.graph.idFromName(feature)
            regimeId = self.graph.idFromName(f"F_{feature}")
            for o in self.outcomeId:
                action = self.graph.nameFromId(o)
                if not self.graph.dSeparation(regimeId, o, featureId):
                    action_sets = action_sets | {action}
        return action_sets

    def causal_validity_rate(self, features_list):
        assert set(features_list) & self.treatments != set() and set(features_list) & self.outcomes == set(), "Invalid input"
        causal_valid = 0
        for feature in features_list:
            causal_valid += self.causal_validity(feature)
        return causal_valid / len(features_list)

    def getDAG(self):
        return self.graph