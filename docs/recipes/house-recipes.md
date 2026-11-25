# House recipes: your film styles, outside focalcut

focalcut is a general cutting engine: it knows how to find moments, keep the camera rules, time cuts to music and
render, for any kind of film. *What a particular kind of film should be* -- a church's beach baptisms as a quiet
documentary, a festival as a fast night-time reel -- is a **recipe**: a reel.json-shaped file of settings, kept with
the people who make those films, not in this repository.

```json
// <shoot>/focalcut/reel/reel.json: only this shoot's own settings
{"recipe": "church-baptism-documentary", "song": "...", "name": "...", "lut": {...}, "reject": {...}}
```

- `"recipe"`: a name or a list of names (later ones win), or a path. A name is looked up in `$FOCALCUT_RECIPES`
  (folders, `:`-separated -- point it at your team's own repo), the shoot's `focalcut/recipes/`, then
  `~/.config/focalcut/recipes/`.
- A recipe may name its own `"recipe"` to build on (a film style on a house style: the crew's cameras and how each
  is matched and steadied, under the structure of the film).
- Settings merge all the way down: the project's `"steady": {"swing": 0.25}` changes only `swing`.
- Keys starting with `_` are notes (`"_about"`), not settings.
- `focalcut` never writes a recipe into reel.json: saving a project writes only the settings it has of its own.

Everything a recipe can say is a reel.json key -- the documentary engine (`"mode": "documentary"`, `"documentary"`),
events, the subject and holding rules, lineups, dust, looks, camera rules -- see docs/recipes/reference.md and
`focalcut/reel/project.py`.
