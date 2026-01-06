from .Ours import DiT
from .Ours_CM import DiT_CM
from .Ours_CATS import DiT_CATS
from .Ours_TID import DiT_TID



_model_registry = {
    "Ours": DiT,
    "Ours_CM": DiT_CM,      # Channel Mixing version for multivariate
    "Ours_CATS": DiT_CATS,  # CATS (Auxiliary Time Series) for multivariate
    "Ours_TID": DiT_TID,    # TID (Time Image Decomposition) - Dual-Axis Attention
}

def get_model(name, args):
    if name not in _model_registry:
        raise ValueError(f"Unknown model: {name}. Available models: {list(_model_registry.keys())}")
    return _model_registry[name](args)