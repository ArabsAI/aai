from typing import Any

_ATTENTIONS: dict[str, Any] = {}


def register_attention_fn(fn):
    _ATTENTIONS[fn.__name__.lower()] = fn
    return fn


from aai.modeling.modules.attentions.attention import *  # isort: skip
from aai.modeling.modules.attentions.ring_attention import *  # isort: skip
from aai.modeling.modules.attentions.flash_attention import *  # isort: skip


def get_attention_fn(name: str):
    assert (
        name in _ATTENTIONS
    ), f"attention fn {name=} is not supported. Available attentions: {_ATTENTIONS.keys()}"
    return _ATTENTIONS[name.lower()]
