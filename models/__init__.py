from .Ours import DiT
from .TimeMixer import TimeMixer
from .MLP import MLPForecast
from .RNN import RNNForecast

_model_registry = {
    "Ours": DiT,
    "TimeMixer": TimeMixer,
    "MLP": MLPForecast,
    "RNN": RNNForecast,
}

def get_model(name, args):
    if name not in _model_registry:
        raise ValueError(f"Unknown model: {name}. Available models: {list(_model_registry.keys())}")
    return _model_registry[name](args)