# Fruities Web UI

This repository contains a simple cooperative orchard environment.

A lightweight debugging web interface can be launched via:

```bash
python -m orchardcoop.webui
```

The interface runs a tiny Flask server and renders the grid in your browser.
Each step is advanced every 0.5 seconds so you can observe the game state.
The debug viewer now uses simple 30x30 sprites for each object to make the
visualisation more pleasant.
This feature is intended purely for debugging and is not used by default.

Dependencies for the web interface are minimal. Install Flask before running:

```bash
pip install Flask
```
