"""Warning filters with no heavy deps — import and call before finetune/groundingdino/timm."""

from __future__ import annotations

import warnings


def silence_known_training_warnings() -> None:
    """Suppress noisy third-party deprecations during GroundingDINO wrapper train/eval.

    ``warnings.filterwarnings(..., message=...)`` uses ``re.match`` (must match from the
    start of the warning text).
    """
    rules: list[tuple[str, type[Warning]]] = [
        (r"Importing from timm\.models\.layers.*", FutureWarning),
        (r"torch\.meshgrid:.*indexing.*", UserWarning),
        (r"You are using `torch\.load` with `weights_only=False`.*", FutureWarning),
        (r"The `device` argument is deprecated.*", FutureWarning),
        (r"The `device` argument is deprecated.*", UserWarning),
        (r"torch\.utils\.checkpoint:.*use_reentrant.*", UserWarning),
        (r"None of the inputs have requires_grad=True\..*", UserWarning),
        (r"`torch\.cuda\.amp\.autocast\(.*\)` is deprecated\..*", FutureWarning),
        (r"`torch\.cpu\.amp\.autocast\(.*\)` is deprecated\..*", FutureWarning),
    ]
    for pattern, category in rules:
        warnings.filterwarnings("ignore", message=pattern, category=category)

    # timm emits from this submodule; module filter uses ``re.match`` on the warning module.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.models\.layers.*")
    # dynamo wraps checkpoint and repeats checkpoint UserWarnings here
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo\.eval_frame")
