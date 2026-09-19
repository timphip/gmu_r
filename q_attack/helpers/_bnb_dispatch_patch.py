"""Shared monkey-patch for the transformers==4.40 + accelerate==1.11 + bnb
ordering bug.

Importing this module (once, from anywhere) installs the patch as a side
effect, so both `q_attack.helpers.model_func` and the safecoder evaluation
scripts get the fix without each having to duplicate the patch logic.

Why this exists: see `q_attack/helpers/model_func.py` for the full diagnosis.
"""

try:
    import sys
    import accelerate.big_modeling as _abm

    _orig_dispatch_model = _abm.dispatch_model

    def _patched_dispatch_model(model, device_map, **kwargs):
        # Force `force_hooks=True` whenever the model contains any bnb
        # `Linear8bitLt` / `Linear4bit` module, so accelerate skips the
        # single-device `model.to(device)` path that raises on bnb models.
        if not kwargs.get("force_hooks", False):
            try:
                from bitsandbytes.nn import Linear4bit, Linear8bitLt

                for _m in model.modules():
                    if isinstance(_m, (Linear4bit, Linear8bitLt)):
                        kwargs["force_hooks"] = True
                        break
            except ImportError:
                pass
        return _orig_dispatch_model(model, device_map, **kwargs)

    _abm.dispatch_model = _patched_dispatch_model

    # Replace every cached reference to the original function (notably
    # `transformers.modeling_utils.dispatch_model`, which captured a stale
    # reference via `from accelerate import dispatch_model` at import time).
    for _name, _mod in list(sys.modules.items()):
        if _mod is None:
            continue
        if getattr(_mod, "dispatch_model", None) is _orig_dispatch_model:
            try:
                setattr(_mod, "dispatch_model", _patched_dispatch_model)
            except (AttributeError, TypeError):
                pass
except ImportError:
    pass
