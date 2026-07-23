import numpy as np

def shannon_entropy(probs, base=np.e):
    probs = np.asarray(probs, dtype=float)
    assert np.isclose(np.sum(probs), 1.0), "This probability vector is not valid"
    nz_probs = probs[probs > 0]
    log_probs = np.log(nz_probs) / np.log(base)
    return -np.sum(nz_probs * log_probs)

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

def entropy_by_miller(chunk_size, base=np.e):
    assert chunk_size > 0, "Are you blind?"
    if base == 2:
        return np.log2(7) + np.log2(chunk_size)  
    elif base == 10:
        return np.log10(7) + np.log10(chunk_size)
    return np.log(7) + np.log(chunk_size)