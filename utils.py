import itertools
import numpy as np
import pyagrum as gum
from scipy.stats import spearmanr

def shannon_entropy(probs, base=np.e):
    probs = np.asarray(probs, dtype=float)
    assert np.isclose(np.sum(probs), 1.0), "This probability vector is not valid"
    nz_probs = probs[probs > 0]
    log_probs = np.log(nz_probs) / np.log(base)
    return -np.sum(nz_probs * log_probs)

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

def jaccard_similarity(A, B):
    return len(set(A) & set(B)) / max(len(set(A) | set(B)), 1e-16)

def spearman_similarity(A, B):
    C = set(A) & set(B)
    assert C != set(), "There is no similarity to compare"
    return 1 - (6 * sum((A.index(x) - B.index(x))**2 for x in C)) / max(len(A) * (len(A)**2 - 1), 1e-16)

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

    def backdoor_satisfaction(self, feature):
        assert feature in self.treatments, "This feature does not exist"
        featureId = self.graph.idFromName(feature)
        regimeId = self.graph.idFromName(f"F_{feature}")
        remained_treatmentId = self.treatmentId[:]
        remained_treatmentId.remove(featureId)
        return self.graph.dSeparation(regimeId, remained_treatmentId) and self.graph.dSeparation(regimeId, self.outcomeId, self.treatmentId)

    def causal_consistency(self, features_list):
        assert set(features_list) & self.treatments != set() and set(features_list) & self.outcomes == set(), "Invalid input"
        causal_valid = 0
        for feature in features_list:
            causal_valid += self.backdoor_satisfaction(feature)
        return causal_valid / len(features_list)

    def getDAG(self):
        return self.graph