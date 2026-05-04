# RoPE Playground: Spin the Tokens

**RoPE Playground** is a single-page interactive demo that explains rotary position embeddings through visual, hands-on examples.

The main idea of the demo is:

> RoPE does not add a separate position vector to tokens. Instead, it rotates query and key vectors based on position, so attention scores can naturally depend on relative distance.

This demo is designed for students who are learning positional embeddings and Transformer attention for the first time.

## Files

- `index.html` — Main page structure with six classroom-focused demo tabs
- `style.css` — Styling for the layout, token cards, vector plots, clocks, heatmaps, and training cards
- `script.js` — RoPE math, SVG visualizations, sliders, dot products, toy attention modes, and interaction logic
- `training_replay_data.json` — Precomputed tiny Transformer training replay data
- `training_replay_data.js` — Same replay data wrapped for direct browser loading
- `scripts/train_tiny_order.py` — Local TensorFlow script used to regenerate the A-before-B training replay

## How to Run

You can open the demo directly in a browser:

```bash
open index.html