"""Every look in one place: the style libraries (glitch, carnival, edm), each a family of looks for a whole film.

reel.json "look" names one (a bare key, unique across the libraries), top level or per format under plan.<fmt>.
A look gives the planner its transitions, subject effects and accents (plan.glitch_look) and, when it has one, the
overlay it suits ("overlay": MilkDrop presets, flickers, strength), used as the default under reel.json's own
"overlay" key: anything reel.json names wins, and "overlay": false turns it off.

    LIBRARIES   {library: {"title", "prefix", "looks"}}      LOOKS   {look: dict(..., library=library)}
"""
from . import carnival, edm, glitch

LIBRARIES = {
    "glitch": dict(title="Glitch", prefix="g_", looks=glitch.LOOKS),
    "carnival": dict(title="Carnival", prefix="c_", looks=carnival.LOOKS),
    "edm": dict(title="EDM", prefix=None, looks=edm.LOOKS),
}


def _merge():
    out = {}
    for lib, d in LIBRARIES.items():
        for k, v in d["looks"].items():
            if k in out:
                raise ValueError(f"look {k!r} is in both {out[k]['library']} and {lib}: look names must be unique")
            out[k] = dict(v, library=lib)
    return out


LOOKS = _merge()


def look_name(config: dict, fmt: str | None = None):
    """The look a project uses in a format: plan.<fmt>.look, else the top-level "look"."""
    per = ((config.get("plan") or {}).get(fmt) or {}).get("look") if fmt else None
    return per or config.get("look")


def overlay(config: dict, fmt: str | None = None):
    """The overlay settings for a project (and format): the look's "overlay" under reel.json's own "overlay".
    None / False: no overlay."""
    own = config.get("overlay")
    if own is False:
        return None
    lk = LOOKS.get(look_name(config, fmt) or "")
    base = (lk or {}).get("overlay")
    if not base:
        return own or None
    return {**base, **(own if isinstance(own, dict) else {})}
