from __future__ import annotations

import json
from flask import Flask, jsonify
from pathlib import Path
from .orchard_env import OrchardCoop
from .policies import RandomPolicy

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OrchardCoop Debug UI</title>
<style>
canvas { border:1px solid #999; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
const cell = 20;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let grid = 25;

function draw(state){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  grid = state.grid_sz;
  canvas.width = grid * cell;
  canvas.height = grid * cell;

  for(const a of state.apples){
    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.arc((a[1]+0.5)*cell, (a[0]+0.5)*cell, cell/3, 0, 2*Math.PI);
    ctx.fill();
  }
  for(const m of state.mushrooms){
    ctx.fillStyle = 'brown';
    ctx.beginPath();
    ctx.arc((m[1]+0.5)*cell, (m[0]+0.5)*cell, cell/3, 0, 2*Math.PI);
    ctx.fill();
  }
  for(const s of state.saplings){
    ctx.fillStyle = 'green';
    ctx.fillRect(s[1]*cell+cell/4, s[0]*cell+cell/4, cell/2, cell/2);
  }
  for(const [agent,pos] of Object.entries(state.agents)){
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc((pos[1]+0.5)*cell, (pos[0]+0.5)*cell, cell/2.5, 0, 2*Math.PI);
    ctx.fill();
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
