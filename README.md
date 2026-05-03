# RoPE Playground: Spin the Tokens

A concise, single-page interactive demo for explaining rotary position embeddings:

> RoPE does not add a position vector to tokens. It rotates query/key vectors by position, making attention scores naturally depend on relative distance.

## Files

- `index.html` - page structure and six classroom-focused tabs
- `style.css` - responsive layout, token cards, vector plots, clocks, heatmaps, and training cards
- `script.js` - RoPE math, SVG drawing, sliders, dot products, toy attention modes, and training replay data
- `scripts/train_tiny_order.py` - local TensorFlow script for the A-before-B training task
- `training_replay_data.json` - generated real tiny-training replay data
- `training_replay_data.js` - same replay data wrapped for direct browser loading from `file://`

## Run

Open `index.html` directly in a browser, or serve the folder:

```bash
python3 -m http.server 8030
```

Then visit `http://127.0.0.1:8030/`.

## Demo Flow

1. Why position information is needed
2. Position as a 2D rotation for query/key vectors
3. Relative distance emerging from the dot product
4. Multi-frequency rotations across four 2D pairs with a draggable position control
5. A toy attention heatmap plus long-distance decay curve
6. A precomputed tiny Transformer training replay for an order-sensitive task

The heatmap is intentionally labeled as a toy visualization because real attention also depends on learned content vectors, not only distance.

## Regenerate Training Replay

```bash
python3 scripts/train_tiny_order.py
```

The script trains four tiny attention models on the A-before-B order task and rewrites `training_replay_data.json` plus `training_replay_data.js`.
