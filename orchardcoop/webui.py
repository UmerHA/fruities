from __future__ import annotations

import json
from flask import Flask, jsonify
from .orchard_env import OrchardCoop
from .policies import RandomPolicy

HTML = """
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>OrchardCoop Debug UI</title>
<style>
canvas { border:1px solid #999; }
</style>
</head>
<body>
<canvas id=\"c\"></canvas>
<script>
const cell = 30;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let grid = 25;

const images = {
  apple: new Image(),
  mushroom: new Image(),
  sapling: new Image(),
  agent: new Image(),
  grass: new Image(),
};
images.apple.src = '/static/apple.png';
images.mushroom.src = '/static/mushroom.png';
images.sapling.src = '/static/sapling.png';
images.agent.src = '/static/agent.png';
images.grass.src = '/static/grass.png';

function draw(state){
  grid = state.grid_sz;
  canvas.width = grid * cell;
  canvas.height = grid * cell;

  for(let i=0;i<grid;i++){
    for(let j=0;j<grid;j++){
      ctx.drawImage(images.grass, j*cell, i*cell, cell, cell);
    }
  }

  for(const a of state.apples){
    ctx.drawImage(images.apple, a[1]*cell, a[0]*cell, cell, cell);
  }
  for(const m of state.mushrooms){
    ctx.drawImage(images.mushroom, m[1]*cell, m[0]*cell, cell, cell);
  }
  for(const s of state.saplings){
    ctx.drawImage(images.sapling, s[1]*cell, s[0]*cell, cell, cell);
  }
  for(const [agent,pos] of Object.entries(state.agents)){
    ctx.drawImage(images.agent, pos[1]*cell, pos[0]*cell, cell, cell);
  }
}

function step(){
  fetch('/step').then(r=>r.json()).then(s=>{draw(s);});
}
setInterval(step, 500);
</script>
</body>
</html>
"""


def _serialize(env: OrchardCoop) -> dict:
    apples = list(map(lambda x: [int(x[0]), int(x[1])], zip(*env.apples.nonzero())))
    mushrooms = list(map(lambda x: [int(x[0]), int(x[1])], zip(*env.mushrooms.nonzero())))
    saplings = list(map(lambda x: [int(x[0]), int(x[1])], zip(*env.saplings.nonzero())))
    agents = {a: env.agent_pos[a].tolist() for a in env.agents}
    return {
        'grid_sz': env.grid_sz,
        'apples': apples,
        'mushrooms': mushrooms,
        'saplings': saplings,
        'agents': agents,
    }


def run_webui(step_time: float = 0.5, grid_sz: int = 25, n_agents: int = 5):
    """Run a minimal web UI for debugging purposes."""
    app = Flask(__name__)
    env = OrchardCoop(grid_sz=grid_sz, n_agents=n_agents)
    policy = RandomPolicy(env)
    obs = env.reset()

    @app.route('/')
    def index():
        return HTML

    @app.route('/step')
    def step():
        nonlocal obs
        actions = policy.act(obs)
        obs, _, terms, truncs, _ = env.step(actions)
        if all(truncs.values()):
            obs = env.reset()
        return jsonify(_serialize(env))

    app.run(debug=False)


if __name__ == '__main__':
    run_webui()
