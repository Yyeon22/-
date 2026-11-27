"""
웹 화면(브라우저)에서 게임을 그리는 HTML/JavaScript를 만든다. Streamlit 앱(app.py)에서 사용.

키보드로 실시간 플레이해야 해서 게임이 브라우저 안에서 직접 돌아간다.
game.py의 규칙과 brain.py의 신경망 계산을 JavaScript로 옮겨 적었고,
AI의 뇌(가중치)는 파이썬으로 학습시킨 results/run/best_brain.npz를 그대로 쓴다.
"""
import json

from brain import HIT_DIST, LOOK_AHEAD, SENSOR_OFFSETS
from game import HEIGHT, PLAYER_H, PLAYER_SPEED, PLAYER_W, POOP_SIZE, WIDTH

CONSTS = dict(W=WIDTH, H=HEIGHT, PW=PLAYER_W, PH=PLAYER_H, SPEED=PLAYER_SPEED,
              PS=POOP_SIZE, OFFSETS=SENSOR_OFFSETS, HIT=HIT_DIST, LOOK=LOOK_AHEAD)

# 두 화면이 같이 쓰는 그림 그리기 코드
COMMON = r"""
const C = __CONSTS__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
function oval(x, y, w, h, color) {
  ctx.fillStyle = color; ctx.beginPath();
  ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2); ctx.fill();
}
function drawPoop(x, y) {
  const s = C.PS;
  oval(x, y + s * 0.55, s, s * 0.45, '#78481e');
  oval(x + s * 0.15, y + s * 0.3, s * 0.7, s * 0.4, '#965f2d');
  oval(x + s * 0.3, y + s * 0.08, s * 0.4, s * 0.32, '#78481e');
}
function drawPlayer(x, y, color, alpha) {
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x + C.PW / 2, y + 12, 11, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x + 8, y + 22, C.PW - 16, C.PH - 22, 6); else ctx.rect(x + 8, y + 22, C.PW - 16, C.PH - 22);
  ctx.fill();
  ctx.fillStyle = '#000';
  ctx.beginPath(); ctx.arc(x + C.PW / 2 - 4, y + 11, 2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(x + C.PW / 2 + 4, y + 11, 2, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 1;
}
function drawSensors(px, py, sensors) {
  const cx = px + C.PW / 2;
  C.OFFSETS.forEach((off, i) => {
    const d = sensors[i];
    ctx.fillStyle = `rgb(${Math.round(255 * d)},${Math.round(200 * (1 - d))},60)`;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx + off - 12, py - 26, 24, 10, 3); else ctx.rect(cx + off - 12, py - 26, 24, 10);
    ctx.fill();
  });
}
function background() {
  ctx.fillStyle = '#c8e6ff'; ctx.fillRect(0, 0, C.W, C.H);
  ctx.fillStyle = '#78b45a'; ctx.fillRect(0, C.H - 10, C.W, 10);
}
function hud(lines) {
  ctx.fillStyle = '#000'; ctx.font = 'bold 18px sans-serif';
  lines.forEach((t, i) => ctx.fillText(t, 10, 26 + 24 * i));
}
function banner(text, sub) {
  ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillRect(0, C.H / 2 - 60, C.W, 100);
  ctx.fillStyle = '#c81e1e'; ctx.font = 'bold 40px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(text, C.W / 2, C.H / 2 - 10);
  ctx.fillStyle = '#000'; ctx.font = '18px sans-serif'; ctx.fillText(sub, C.W / 2, C.H / 2 + 24);
  ctx.textAlign = 'left';
}
"""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
body { margin: 0; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }
canvas { width: 100%; max-width: 480px; border-radius: 8px; outline: none; touch-action: none; }
.bar { display: flex; gap: 6px; margin: 8px 0; flex-wrap: wrap; justify-content: center; }
button { font-size: 16px; padding: 6px 14px; border-radius: 6px; border: 1px solid #aaa; background: #fff; cursor: pointer; }
button.on { background: #ffe08a; }
</style></head><body>
__BODY__
<script>
__COMMON__
__SCRIPT__
</script></body></html>"""


def _page(body, script):
    common = COMMON.replace("__CONSTS__", json.dumps(CONSTS))
    return PAGE.replace("__BODY__", body).replace("__COMMON__", common).replace("__SCRIPT__", script)


LIVE_SCRIPT = r"""
const WEIGHTS = __WEIGHTS__;
let vs = true, game = null, started = false, keys = {left: false, right: false};

// ---- game.py 규칙을 그대로 옮긴 것 ----
function box(x, y, w, h) { return [Math.trunc(x), Math.trunc(y), Math.trunc(w), Math.trunc(h)]; }
function overlaps(a, b) { return a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]; }
function newPlayer() { return {x: C.W / 2 - C.PW / 2, y: C.H - C.PH - 10, alive: true, score: 0}; }
function newGame() { return {players: vs ? [newPlayer(), newPlayer()] : [newPlayer()], poops: [], frame: 0}; }
function step(g, actions) {
  g.frame++;
  const interval = Math.max(6, 30 - Math.floor(g.frame / 300));
  if (g.frame % interval === 0)
    g.poops.push({x: Math.random() * (C.W - C.PS), y: -C.PS, speed: 4 + g.frame / 900 + Math.random() * 2});
  g.poops.forEach(p => p.y += p.speed);
  g.poops = g.poops.filter(p => p.y < C.H);
  const boxes = g.poops.map(p => box(p.x + 4, p.y + 6, C.PS - 8, C.PS - 8));
  g.players.forEach((pl, n) => {
    if (!pl.alive) return;
    if (actions[n] === 1) pl.x -= C.SPEED; else if (actions[n] === 2) pl.x += C.SPEED;
    pl.x = Math.max(0, Math.min(C.W - C.PW, pl.x));
    const me = box(pl.x + 6, pl.y + 8, C.PW - 12, C.PH - 8);
    if (boxes.some(b => overlaps(me, b))) pl.alive = false; else pl.score++;
  });
}

// ---- brain.py의 눈(센서)과 뇌(신경망)를 그대로 옮긴 것 ----
function observe(g, pl) {
  const px = pl.x + C.PW / 2, obs = [px / C.W * 2 - 1];
  for (const off of C.OFFSETS) {
    const sx = px + off;
    if (sx < C.PW / 2 || sx > C.W - C.PW / 2) { obs.push(1); continue; }
    let soonest = C.LOOK;
    for (const p of g.poops) {
      if (p.y > pl.y + 10) continue;
      if (Math.abs(p.x + C.PS / 2 - sx) < C.HIT) soonest = Math.min(soonest, Math.max(0, (pl.y - p.y) / p.speed));
    }
    obs.push(1 - soonest / C.LOOK);
  }
  return obs;
}
function act(obs) {
  const W = WEIGHTS, hidden = W.b1.map((b, j) => Math.tanh(obs.reduce((s, x, i) => s + x * W.w1[i][j], b)));
  const out = W.b2.map((b, k) => hidden.reduce((s, h, j) => s + h * W.w2[j][k], b));
  let best = 0; out.forEach((v, k) => { if (v > out[best]) best = k; });
  return best;
}

// ---- 화면과 조작 ----
function render() {
  background();
  game.poops.forEach(p => drawPoop(p.x, p.y));
  const ai = vs ? game.players[0] : null, me = vs ? game.players[1] : game.players[0];
  if (ai && ai.alive) { drawSensors(ai.x, ai.y, observe(game, ai).slice(1)); drawPlayer(ai.x, ai.y, '#e63c3c', 1); }
  if (me.alive) drawPlayer(me.x, me.y, '#ffc83c', 1);
  const lines = [];
  if (ai) lines.push(`AI ${(ai.score / 60).toFixed(1)}초`);
  lines.push(`나 ${(me.score / 60).toFixed(1)}초`);
  hud(lines);
  if (!started) banner('클릭해서 시작', '← → 키 또는 아래 버튼으로 이동');
  else if (game.players.every(p => !p.alive)) {
    const msg = !ai ? '게임 오버' : me.score > ai.score ? '내가 이겼다!' : me.score < ai.score ? 'AI 승리!' : '무승부';
    banner(msg, '클릭 또는 R 키로 다시 시작');
  }
}
let last = 0, acc = 0;
function loop(t) {
  acc += Math.min(100, t - (last || t)); last = t;
  while (acc >= 1000 / 60) {            // 컴퓨터 화면 주사율과 상관없이 1초에 60번 진행
    acc -= 1000 / 60;
    if (started && !game.players.every(p => !p.alive)) {
      const mine = keys.left && !keys.right ? 1 : keys.right && !keys.left ? 2 : 0;
      const actions = vs ? [game.players[0].alive ? act(observe(game, game.players[0])) : 0, mine] : [mine];
      step(game, actions);
    }
  }
  render(); requestAnimationFrame(loop);
}
function restart() { game = newGame(); started = true; cv.focus(); }
cv.addEventListener('pointerdown', () => { if (!started || game.players.every(p => !p.alive)) restart(); cv.focus(); });
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') keys.left = true;
  if (e.key === 'ArrowRight') keys.right = true;
  if (e.key === 'r' || e.key === 'R') restart();
  if (e.key.startsWith('Arrow')) e.preventDefault();
});
window.addEventListener('keyup', e => {
  if (e.key === 'ArrowLeft') keys.left = false;
  if (e.key === 'ArrowRight') keys.right = false;
});
function hold(id, k) {
  const b = document.getElementById(id);
  b.addEventListener('pointerdown', e => { keys[k] = true; if (!started) restart(); e.preventDefault(); });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => b.addEventListener(ev, () => keys[k] = false));
}
hold('left', 'left'); hold('right', 'right');
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => {
  vs = b.dataset.mode === 'vs'; started = false; game = newGame();
  document.querySelectorAll('[data-mode]').forEach(x => x.classList.toggle('on', x === b));
});
game = newGame();
requestAnimationFrame(loop);
"""

LIVE_BODY = """<div class="bar"><button data-mode="vs" class="on">🤖 AI와 대결</button><button data-mode="solo">🙂 혼자 하기</button></div>
<canvas id="cv" width="480" height="640" tabindex="0"></canvas>
<div class="bar"><button id="left" style="font-size:22px;padding:10px 40px">◀</button>
<button id="right" style="font-size:22px;padding:10px 40px">▶</button></div>"""


def live_html(brain):
    weights = {k: v.tolist() for k, v in brain.w.items()}
    return _page(LIVE_BODY, LIVE_SCRIPT.replace("__WEIGHTS__", json.dumps(weights)))
