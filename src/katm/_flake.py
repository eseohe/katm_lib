"""Lazy access to the optional ``flake`` library (FLaKE keyphrase extraction).

KATM only needs it for ``keyphrase_scope="flake"`` and for streaming content-word
extraction, so it is imported on first use rather than at ``import katm``.
"""


def _missing(err: ImportError):
    return ImportError(
        "this option needs the 'flake' library (FLaKE keyphrase extraction). "
        "Install it, e.g. `pip install -e /path/to/flake`."
    ).with_traceback(err.__traceback__)


def flake_core():
    """The ``flake.core`` module (extract_flake_keyphrases, select_facility_location)."""
    try:
        from flake import core
    except ImportError as e:
        raise _missing(e) from e
    return core


def flake_stream():
    """The ``flake.stream`` module (streaming extractor, corpus readers)."""
    try:
        from flake import stream
    except ImportError as e:
        raise _missing(e) from e
    return stream
