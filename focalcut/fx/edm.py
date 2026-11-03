"""EDM styles: a main stage's light show -- the look the Electric Nation reels were built on -- as looks for a whole film.

Where glitch.py dresses a film in digital damage and carnival.py in colour and confetti, an EDM look is the overlay
(reel/overlay.py): real MilkDrop presets round the edges of the frame (space tunnels, lasers, fire, ice), white and
dark flickers sharp on the beat into every drop and on the hardest hits, kick blinks and RGB splits. The cuts stay
the stock grammar's (flash on the hits); no subject effects of its own.

LOOKS: each one's "overlay" is the default under reel.json's own "overlay" (fx/looks.py); a look with no presets
named uses overlay.py's own (MILKDROP, MILKDROP_CALM), which are this library's.
"""

LOOKS = {
    "mainstage": dict(
        title="Main Stage", mood="an EDM festival's main stage: MilkDrop tunnels, lasers, fire and ice round the edges, "
        "white and dark flickers into every drop",
        base=[], subject=[], transitions=[], accents=[],
        use="the overlay all film long, stronger at night; three white flashes and a blackout into each drop, hard "
            "single hits late in the night; stock cuts",
        overlay=dict(engine="milkdrop", flicker=dict(auto=True, at=[]))),
    "lasershow": dict(
        title="Laser Show", mood="focalcut's own light show (no MilkDrop): tunnels, lasers, starfields and flame in neon "
        "and synthwave palettes, kick blinks and drop bursts",
        base=[], subject=[], transitions=[], accents=[],
        use="the built-in overlay all film long; a new preset every 8 bars; flickers into the drops",
        overlay=dict(engine="builtin", flicker=dict(auto=True, at=[]))),
}
