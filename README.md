# RoPE Demo: Position as Rotation

An interactive static demo explaining rotary positional embeddings (RoPE) for a
class extra-credit submission.

## Run Locally

Open `index.html` directly in a browser, or serve the folder with a simple local
server. If port `8000` is already in use, switch to any free port such as
`8010`.

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Files

- `index.html`: demo structure and teaching flow
- `styles.css`: visual design and responsive layout
- `app.js`: RoPE math, controls, animation, and canvas visualizations

## Demo Goal

The demo has three parts:

- Compare concrete position-vector methods on the sentence `I like black coffee`.
- Visualize RoPE as 2D rotations of query/key feature pairs, including a
  "shift the whole sentence" interaction for `The dog chased the cat`.
- Train a tiny attention scorer on the task "attend to position `m - 2`" and
  compare how different position methods extrapolate as you change training
  max length and testing length.

The training task is intentionally small: it isolates the positional part of a
transformer attention layer rather than pretending to train a full language
model in the browser.

The learned absolute embedding view intentionally grays out positions beyond
the training max length because those positions have no learned lookup row in
the toy setup. RoPE keeps computing `angle = position * theta`.
