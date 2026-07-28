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

class CausalModel:

    def __init__(self, features, actions):
        self.regimes = set(map(lambda x: f"F_{x}", features))
        self.treatments = features
        self.outcomes = actions
        self.nodes = self.treatments + self.regimes + self.outcomes
        self.edges = []
        self.graph = gum.DAG()
        self.id = set(enumerate(self.nodes))
        self.treatmentId = [i for (i, name) in self.id if name in self.treatments]
        self.outcomeId = [i for (i, name) in self.id if name in self.outcomes]
        for i, name in self.id:
            self.graph.addNodeWithId(i)
            self.graph.setName(i, name)

    def add_causal_edge(self, feature_from, feature_to):
        assert feature_from in self.nodes and feature_to in self.nodes, "Those things do not exist"
        i = self.graph.idFromName(feature_from)
        j = self.graph.idFromName(feature_to)
        assert i is not None and j is not None, "Those things do not exist"
        self.graph.addArc(tail=i, head=j)

    def backdoor_satisfaction(self, feature):
        assert feature in self.treatments, "This feature does not exist"
        featureId = self.graph.idFromName(feature)
        regimeId = self.graph.idFromName(f"F_{feature}")
        remained_treatmentId = self.treatmentId[:]
        remained_treatmentId.remove(featureId)
        return self.graph.dSeparation(regimeId, remained_treatmentId) and self.graph.dSeparation(regimeId, self.outcomeId, Z=self.treatmentId)

    def getDAG(self):
        return self.graph