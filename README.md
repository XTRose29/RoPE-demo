# RoPE Playground: Spin the Tokens

A concise, single-page interactive demo for explaining rotary position embeddings:

> RoPE does not add a position vector to tokens. It rotates query/key vectors by position, making attention scores naturally depend on relative distance.

## Files

- `index.html` - page structure and four classroom-focused tabs
- `style.css` - responsive layout, token cards, vector plots, clocks, and heatmap styling
- `script.js` - RoPE math, SVG drawing, sliders, dot products, and toy attention modes

## Run

Open `index.html` directly in a browser, or serve the folder:

```bash
python3 -m http.server 8030
```

Then visit `http://127.0.0.1:8030/netID_rope_demo/`.

## Demo Flow

1. Why position information is needed
2. Position as a 2D rotation for query/key vectors
3. Relative distance emerging from the dot product
4. Multi-frequency rotations plus a toy attention heatmap

The heatmap is intentionally labeled as a toy visualization because real attention also depends on learned content vectors, not only distance.
