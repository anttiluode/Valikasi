"""Colour lexicon used by the filter. Frozen before the first real run (PREREG.md).

An episode's note is "filtered-clean" if it contains no word from ANY colour's
list (names, cue objects, synonyms) and none of the generic colour words.
"""
import re

COLOURS = ["red", "blue", "green", "yellow"]

CUES = {
    "red": [
        "red", "crimson", "scarlet", "ruby", "tomato", "tomatoes", "fire engine",
        "rose", "cherry", "cherries", "blood", "strawberry", "strawberries",
        "cardinal", "brick", "lipstick", "ladybug", "poppy", "maroon",
        "burgundy", "rust", "ripe", "flame", "flames", "fire", "apple", "apples",
    ],
    "blue": [
        "blue", "navy", "azure", "sapphire", "sea", "ocean", "sky", "cobalt",
        "denim", "teal", "turquoise", "cyan", "indigo", "blueberry",
        "blueberries", "bluebird", "wave", "waves", "lagoon", "clear", "water",
        "beach", "sailing", "sail", "horizon",
    ],
    "green": [
        "green", "emerald", "grass", "forest", "forests", "frog", "frogs", "leaf",
        "leaves", "lime", "jade", "olive", "moss", "mossy", "mint", "fern",
        "ferns", "clover", "pine", "spinach", "lettuce", "cucumber", "meadow",
        "fresh", "garden", "plant", "plants", "herb", "herbs", "salad",
    ],
    "yellow": [
        "yellow", "banana", "bananas", "lemon", "lemons", "sun", "sunny",
        "sunshine", "gold", "golden", "canary", "mustard", "daffodil",
        "sunflower", "sunflowers", "honey", "butter", "amber", "corn", "bright",
        "warm", "cheerful", "sunrise", "sunset",
    ],
}

GENERIC = ["color", "colour", "colors", "colours", "colorful", "colourful",
           "hue", "hues", "shade", "shades", "tint", "tints", "pigment"]

_ALL = set(GENERIC)
for _c, _ws in CUES.items():
    for _w in _ws:
        _ALL.update(_w.split())
        _ALL.add(_w)

_WORD = re.compile(r"[a-z]+")


def _forms(w):
    yield w
    for suf in ("es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            yield w[: -len(suf)]


def hits(text):
    """Return the sorted set of lexicon words found in text."""
    found = set()
    for w in _WORD.findall(text.lower()):
        if any(f in _ALL for f in _forms(w)):
            found.add(w)
    return sorted(found)


def is_clean(text):
    return len(hits(text)) == 0


def colour_names_only(text):
    names = set(COLOURS)
    return sorted({w for w in _WORD.findall(text.lower()) if w in names})
