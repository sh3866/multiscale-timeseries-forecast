from .Ours import DiT
from .Ours_new import DiT as DiT_new
from .TimeMixer import Model as TimeMixer
from .TimeMixer import MADiffusionModelWrapper as TimeMixer_MA



_model_registry = {
    "Ours": DiT,
    "Ours_new": DiT_new,    # New experimental version
    "TimeMixer": TimeMixer, # Original TimeMixer baseline
    "TimeMixer_MA": TimeMixer_MA,  # TimeMixer with MA-Diffusion wrapper
}

def get_model(name, args):
    if name not in _model_registry:
        raise ValueError(f"Unknown model: {name}. Available models: {list(_model_registry.keys())}")
    return _model_registry[name](args)