"""
웹 화면(브라우저)에서 게임을 그리는 HTML/JavaScript를 만든다. Streamlit 앱(app.py)에서 사용.

키보드로 실시간 플레이해야 해서 게임이 브라우저 안에서 직접 돌아간다.
game.py의 규칙과 brain.py의 신경망 계산을 JavaScript로 옮겨 적었고,
AI의 뇌(가중치 + 센서 설정)는 파이썬으로 학습시킨 .npz 파일을 그대로 쓴다.

    live_html(brain)          게임 화면: AI와 대결 / 혼자 하기 (혼자 하기에만 아이템)
    race_html([(이름, brain)])  AI 성장 레이스: 여러 세대의 AI가 같은 똥을 동시에 피한다
"""
import json

import game as G

CONSTS = dict(W=G.WIDTH, H=G.HEIGHT, PW=G.PLAYER_W, PH=G.PLAYER_H, SPEED=G.PLAYER_SPEED,
              STAGE=G.STAGE_FRAMES, PS=G.POOP_SIZE, GROW=G.POOP_GROW, PMAX=G.POOP_MAX,
              JITTER=G.POOP_SIZE_JITTER, SP0=G.SPAWN_START, SPSTEP=G.SPAWN_STEP, SPMIN=G.SPAWN_MIN,
              V0=G.SPEED_START, VSTEP=G.SPEED_STEP, VRAND=G.SPEED_RANDOM)

# 레이스에서 세대마다 쓰는 색 (오래된 세대 → 최근 세대)
RACE_COLORS = ["#9e9e9e", "#8d6e63", "#43a047", "#00acc1", "#1e88e5", "#8e24aa", "#fb8c00", "#e53935"]

# 두 화면이 같이 쓰는 코드: 그림 그리기, 게임 규칙, AI의 눈과 뇌, 1초에 60번 도는 루프
COMMON = r"""
const C = __CONSTS__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');

// ---------------- 그림 그리기 ----------------
function oval(x, y, w, h, color) {
  ctx.fillStyle = color; ctx.beginPath();
  ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2); ctx.fill();
}
function rrect(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
  ctx.fill();
}
function drawPoop(x, y, s) {
  oval(x, y + s * 0.55, s, s * 0.45, '#78481e');
  oval(x + s * 0.15, y + s * 0.3, s * 0.7, s * 0.4, '#965f2d');
  oval(x + s * 0.3, y + s * 0.08, s * 0.4, s * 0.32, '#78481e');
}
function drawPlayer(x, y, color, alpha) {
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x + C.PW / 2, y + 12, 11, 0, Math.PI * 2); ctx.fill();
  rrect(x + 8, y + 22, C.PW - 16, C.PH - 22, 6);
  ctx.fillStyle = '#000';
  ctx.beginPath(); ctx.arc(x + C.PW / 2 - 4, y + 11, 2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(x + C.PW / 2 + 4, y + 11, 2, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 1;
}
function drawSensors(px, py, offsets, sensors) {
  const cx = px + C.PW / 2;
  offsets.forEach((off, i) => {
    const d = sensors[i];
    ctx.fillStyle = `rgb(${Math.round(255 * d)},${Math.round(200 * (1 - d))},60)`;
    rrect(cx + off - 11, py - 26, 22, 10, 3);
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
  ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fillRect(0, C.H / 2 - 60, C.W, 100);
  ctx.fillStyle = '#c81e1e'; ctx.font = 'bold 40px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(text, C.W / 2, C.H / 2 - 10);
  ctx.fillStyle = '#000'; ctx.font = '18px sans-serif'; ctx.fillText(sub, C.W / 2, C.H / 2 + 24);
  ctx.textAlign = 'left';
}
function stageFlash(g) {      // 스테이지가 오르면 1초간 알림
  if (g.frame >= C.STAGE && g.frame % C.STAGE < 60) {
    ctx.fillStyle = '#c81e1e'; ctx.font = 'bold 44px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`STAGE ${stageOf(g)}!`, C.W / 2, 140); ctx.textAlign = 'left';
  }
}
const sec = frames => (frames / 60).toFixed(1) + '초';

// ---------------- game.py 규칙을 그대로 옮긴 것 ----------------
// (g.items가 true일 때만 아이템이 나온다 = 웹 '혼자 하기' 전용. 꺼져 있으면 game.py와 완전히 같다)
const ITEM_EVERY = 480, ITEM_SIZE = 32, ITEM_TIME = 180;      // 8초마다 아이템, 효과 3초
function box(x, y, w, h) { return [Math.trunc(x), Math.trunc(y), Math.trunc(w), Math.trunc(h)]; }
function overlaps(a, b) { return a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]; }
function newPlayer() { return {x: C.W / 2 - C.PW / 2, y: C.H - C.PH - 10, alive: true, score: 0, last: 0}; }
function makeGame(n, items) {
  return {players: Array.from({length: n}, newPlayer), poops: [], frame: 0,
          items: !!items, drops: [], fx: {shield: 0, slow: 0}, events: []};
}
function stageOf(g) { return Math.floor(g.frame / C.STAGE) + 1; }
function hitDist(size) { return ((C.PW - 12) + (size - 8)) / 2; }
function allDead(g) { return g.players.every(p => !p.alive); }
function step(g, actions) {
  g.frame++;
  if (g.frame % C.STAGE === 0) g.events.push('stage');
  const st = stageOf(g) - 1;          // 스테이지가 오를수록 자주, 크게, 빠르게
  const interval = Math.max(C.SPMIN, C.SP0 - C.SPSTEP * st);
  if (g.frame % interval === 0) {
    const size = Math.min(C.PMAX, C.PS + C.GROW * st) * (1 - C.JITTER + Math.random() * 2 * C.JITTER);
    g.poops.push({x: Math.random() * (C.W - size), y: -size, size,
                  speed: C.V0 + C.VSTEP * st + Math.random() * C.VRAND});
  }
  if (g.items && g.frame % ITEM_EVERY === 0)
    g.drops.push({x: Math.random() * (C.W - ITEM_SIZE), y: -ITEM_SIZE, kind: Math.random() < 0.5 ? 'shield' : 'slow'});
  g.fx.shield = Math.max(0, g.fx.shield - 1); g.fx.slow = Math.max(0, g.fx.slow - 1);
  const slow = g.fx.slow > 0 ? 0.5 : 1;
  g.poops.forEach(p => p.y += p.speed * slow);
  g.poops = g.poops.filter(p => p.y < C.H);
  g.drops.forEach(d => d.y += 3);
  g.drops = g.drops.filter(d => d.y < C.H);
  const boxes = g.poops.map(p => box(p.x + 4, p.y + 6, p.size - 8, p.size - 8));
  g.players.forEach((pl, n) => {
    if (!pl.alive) return;
    pl.last = actions[n];
    if (actions[n] === 1) pl.x -= C.SPEED; else if (actions[n] === 2) pl.x += C.SPEED;
    pl.x = Math.max(0, Math.min(C.W - C.PW, pl.x));
    const me = box(pl.x + 6, pl.y + 8, C.PW - 12, C.PH - 8);
    g.drops = g.drops.filter(d => {
      if (!overlaps(me, box(d.x, d.y, ITEM_SIZE, ITEM_SIZE))) return true;
      g.fx[d.kind] = ITEM_TIME; g.events.push('item'); return false;
    });
    if (g.fx.shield === 0 && boxes.some(b => overlaps(me, b))) { pl.alive = false; g.events.push('dead'); }
    else pl.score++;
  });
}
function drawItems(g) {
  ctx.font = `${ITEM_SIZE - 4}px sans-serif`; ctx.textBaseline = 'top';
  g.drops.forEach(d => ctx.fillText(d.kind === 'shield' ? '🛡️' : '⏳', d.x, d.y));
  ctx.textBaseline = 'alphabetic';
}

// ---------------- brain.py의 눈(센서)과 뇌(신경망)를 그대로 옮긴 것 ----------------
// eye = {offsets: 센서 위치들, look: 몇 프레임 뒤까지 볼지, soft: 부드러운 센서 폭, memory: 직전 행동 기억}
function coverage(dist, hd, soft) {
  return soft === 0 ? (dist < hd ? 1 : 0) : Math.min(1, Math.max(0, (hd + soft - dist) / (2 * soft)));
}
function observe(g, pl, eye) {
  const px = pl.x + C.PW / 2, obs = [px / C.W * 2 - 1];
  for (const off of eye.offsets) {
    const sx = px + off;
    if (sx < C.PW / 2 || sx > C.W - C.PW / 2) { obs.push(1); continue; }
    let danger = 0;
    for (const p of g.poops) {
      if (p.y > pl.y + 10) continue;
      const urgency = Math.max(0, 1 - Math.max(0, (pl.y - p.y) / p.speed) / eye.look);
      danger = Math.max(danger, urgency * coverage(Math.abs(p.x + p.size / 2 - sx), hitDist(p.size), eye.soft));
    }
    obs.push(danger);
  }
  if (eye.memory) obs.push(pl.last === 1 ? 1 : 0, pl.last === 2 ? 1 : 0);
  return obs;
}
function act(W, obs) {
  const hidden = W.b1.map((b, j) => Math.tanh(obs.reduce((s, x, i) => s + x * W.w1[i][j], b)));
  const out = W.b2.map((b, k) => hidden.reduce((s, h, j) => s + h * W.w2[j][k], b));
  let best = 0; out.forEach((v, k) => { if (v > out[best]) best = k; });
  return best;
}

// 컴퓨터 화면 주사율과 상관없이 1초에 60번(× 배속) 게임을 진행한다
function runLoop(tick, draw, speed) {
  let last = 0, acc = 0;
  function frame(t) {
    acc += Math.min(100, t - (last || t)); last = t;
    while (acc >= 1000 / 60) { acc -= 1000 / 60; for (let i = 0; i < speed(); i++) tick(); }
    draw(); requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
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


def _brain_js(brain):
    """뇌 하나를 JavaScript에 넘길 모양으로: 가중치 + 눈 설정"""
    return {"w": {k: v.tolist() for k, v in brain.w.items()},
            "eye": {"offsets": list(brain.eye.offsets), "look": brain.eye.look,
                    "soft": brain.eye.soft, "memory": brain.eye.memory}}


# ======================= 게임 화면 (AI와 대결 / 혼자 하기) =======================
LIVE_SCRIPT = r"""
const AI = __AI__;
const BEST_KEY = 'poop-dodge-best';
let vs = true, game = null, started = false, keys = {left: false, right: false}, dragX = null;
let best = 0, newRecord = false, muted = false, audio = null;
try { best = Number(localStorage.getItem(BEST_KEY)) || 0; } catch (e) {}   // 저장이 막힌 곳이면 그냥 0

// ---- 효과음: 소리 파일 없이 브라우저가 직접 만드는 '삑' 소리 ----
const SOUNDS = {stage: [[660, 0], [880, 0.09]], item: [[990, 0], [1320, 0.07], [1760, 0.14]],
                dead: [[330, 0], [220, 0.12], [140, 0.24]], record: [[523, 0], [659, 0.1], [784, 0.2], [1047, 0.3]]};
function sfx(kind) {
  if (muted || !audio) return;
  const t0 = audio.currentTime;
  SOUNDS[kind].forEach(([freq, t]) => {
    const o = audio.createOscillator(), v = audio.createGain();
    o.type = 'square'; o.frequency.value = freq;
    v.gain.setValueAtTime(0.06, t0 + t); v.gain.exponentialRampToValueAtTime(0.001, t0 + t + 0.12);
    o.connect(v).connect(audio.destination); o.start(t0 + t); o.stop(t0 + t + 0.13);
  });
}
function unlockAudio() {      // 브라우저는 클릭한 뒤에만 소리를 낼 수 있게 해 준다
  try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } catch (e) {}
}

function newGame() { return makeGame(vs ? 2 : 1, !vs); }       // 아이템은 혼자 하기에서만
const ai = () => vs ? game.players[0] : null;
const me = () => vs ? game.players[1] : game.players[0];

function humanAction() {
  if (keys.left && !keys.right) return 1;
  if (keys.right && !keys.left) return 2;
  if (dragX !== null) {          // 손가락(마우스)이 있는 곳으로 따라간다
    const d = dragX - (me().x + C.PW / 2);
    return d > C.SPEED / 2 ? 2 : d < -C.SPEED / 2 ? 1 : 0;
  }
  return 0;
}
function tick() {
  if (!started || allDead(game)) return;
  const a = ai(), actions = vs ? [a.alive ? act(AI.w, observe(game, a, AI.eye)) : 0, humanAction()] : [humanAction()];
  const wasAlive = me().alive;
  step(game, actions);
  const ev = game.events; game.events = [];
  if (ev.includes('stage')) sfx('stage');
  if (ev.includes('item')) sfx('item');
  if (wasAlive && !me().alive) sfx('dead');          // 내가 방금 죽음
  if (allDead(game) && !vs && me().score > best) {
    best = me().score; newRecord = true; sfx('record');
    try { localStorage.setItem(BEST_KEY, best); } catch (e) {}
  }
}
function render() {
  background();
  game.poops.forEach(p => drawPoop(p.x, p.y, p.size));
  drawItems(game);
  const a = ai(), m = me();
  if (a && a.alive) { drawSensors(a.x, a.y, AI.eye.offsets, observe(game, a, AI.eye).slice(1)); drawPlayer(a.x, a.y, '#e63c3c', 1); }
  if (m.alive) {
    if (game.fx.shield > 0) {                       // 무적: 방패 테두리 (끝날 때쯤 깜빡임)
      if (game.fx.shield > 45 || game.fx.shield % 10 < 5) {
        ctx.strokeStyle = '#2196f3'; ctx.lineWidth = 3; ctx.beginPath();
        ctx.arc(m.x + C.PW / 2, m.y + C.PH / 2, 30, 0, Math.PI * 2); ctx.stroke();
      }
    }
    drawPlayer(m.x, m.y, '#ffc83c', 1);
  }
  const lines = [`스테이지 ${stageOf(game)}`];
  if (a) lines.push(`AI ${sec(a.score)}`);
  lines.push(`나 ${sec(m.score)}`);
  if (!vs) lines.push(`최고 기록 ${sec(best)}`);
  if (game.fx.shield > 0) lines.push(`🛡️ 무적 ${Math.ceil(game.fx.shield / 60)}`);
  if (game.fx.slow > 0) lines.push(`⏳ 느리게 ${Math.ceil(game.fx.slow / 60)}`);
  hud(lines);
  if (started && !allDead(game)) stageFlash(game);
  if (!started) banner('클릭해서 시작', vs ? '← → 키, 버튼, 또는 화면을 끌어서 이동' : '🛡️ 무적, ⏳ 느리게 아이템을 먹어 보세요');
  else if (allDead(game)) {
    if (newRecord) banner('🎉 신기록!', `${sec(best)} · 클릭 또는 R 키로 다시 시작`);
    else banner(!a ? '게임 오버' : m.score > a.score ? '내가 이겼다!' : m.score < a.score ? 'AI 승리!' : '무승부',
                '클릭 또는 R 키로 다시 시작');
  }
}
function restart() { unlockAudio(); game = newGame(); started = true; newRecord = false; cv.focus(); }

// ---- 조작: 키보드 / 아래 버튼 / 화면 끌기(모바일) ----
const toCanvasX = e => { const r = cv.getBoundingClientRect(); return (e.clientX - r.left) * C.W / r.width; };
cv.addEventListener('pointerdown', e => {
  if (!started || allDead(game)) restart();
  dragX = toCanvasX(e); cv.setPointerCapture(e.pointerId); cv.focus();
});
cv.addEventListener('pointermove', e => { if (dragX !== null) dragX = toCanvasX(e); });
['pointerup', 'pointercancel'].forEach(ev => cv.addEventListener(ev, () => dragX = null));
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
document.getElementById('mute').onclick = e => { muted = !muted; e.target.textContent = muted ? '🔇' : '🔊'; };
game = newGame();
runLoop(tick, render, () => 1);
"""

LIVE_BODY = """<div class="bar"><button data-mode="vs" class="on">🤖 AI와 대결</button><button data-mode="solo">🙂 혼자 하기</button>
<button id="mute" title="소리 켜기/끄기">🔊</button></div>
<canvas id="cv" width="480" height="640" tabindex="0"></canvas>
<div class="bar"><button id="left" style="font-size:22px;padding:10px 40px">◀</button>
<button id="right" style="font-size:22px;padding:10px 40px">▶</button></div>"""


def live_html(brain):
    return _page(LIVE_BODY, LIVE_SCRIPT.replace("__AI__", json.dumps(_brain_js(brain))))


# ======================= AI 성장 레이스 =======================
RACE_SCRIPT = r"""
const RACERS = __RACERS__;      // [{label, color, w, eye}]  오래된 세대 → 최근 세대
let game = makeGame(RACERS.length), started = false, speed = 1;

function tick() {
  if (!started || allDead(game)) return;
  step(game, RACERS.map((r, i) => game.players[i].alive ? act(r.w, observe(game, game.players[i], r.eye)) : 0));
  game.events = [];
}
function render() {
  background();
  game.poops.forEach(p => drawPoop(p.x, p.y, p.size));
  RACERS.forEach((r, i) => { const p = game.players[i]; if (p.alive) drawPlayer(p.x, p.y, r.color, 0.9); });
  ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
  RACERS.forEach((r, i) => {          // 머리 위 이름표: 다른 AI와 떨어져 있을 때만 (뭉쳐 있으면 순위표 색으로 구분)
    const p = game.players[i]; if (!p.alive) return;
    const crowded = game.players.some((q, j) => j !== i && q.alive && Math.abs(q.x - p.x) < 40);
    if (!crowded) { ctx.fillStyle = r.color; ctx.fillText(r.label, p.x + C.PW / 2, p.y - 6); }
  });
  ctx.textAlign = 'left';
  hud([`스테이지 ${stageOf(game)}`, `${sec(game.frame)}`, `배속 x${speed}`]);

  // 오른쪽 위 순위표: 오래 버틴 순
  const order = RACERS.map((r, i) => i).sort((a, b) => game.players[b].score - game.players[a].score);
  const bx = C.W - 170, by = 10, rowH = 22;
  ctx.fillStyle = 'rgba(255,255,255,0.85)'; rrect(bx, by, 160, 12 + rowH * RACERS.length, 6);
  ctx.font = 'bold 14px sans-serif';
  order.forEach((i, rank) => {
    const r = RACERS[i], p = game.players[i], y = by + 22 + rowH * rank;
    ctx.fillStyle = r.color; rrect(bx + 8, y - 11, 12, 12, 3);
    ctx.fillStyle = p.alive ? '#000' : '#999';
    ctx.fillText(`${rank + 1}. ${r.label}`, bx + 26, y);
    ctx.textAlign = 'right'; ctx.fillText(p.alive ? sec(p.score) : `✖ ${sec(p.score)}`, bx + 152, y); ctx.textAlign = 'left';
  });
  if (started && !allDead(game)) stageFlash(game);
  if (!started) banner('클릭해서 레이스 시작', '여러 세대의 AI가 똑같은 똥을 동시에 피합니다');
  else if (allDead(game)) banner(`🏆 ${RACERS[order[0]].label} 우승!`, `${sec(game.players[order[0]].score)} 버팀 · 클릭해서 다시`);
}
function restart() { game = makeGame(RACERS.length); started = true; }
cv.addEventListener('pointerdown', () => { if (!started || allDead(game)) restart(); });
document.getElementById('again').onclick = restart;
document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => {
  speed = Number(b.dataset.speed);
  document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x === b));
});
runLoop(tick, render, () => speed);
"""

RACE_BODY = """<div class="bar"><button id="again">🔄 새 레이스</button>
<button data-speed="1" class="on">x1</button><button data-speed="2">x2</button>
<button data-speed="4">x4</button><button data-speed="8">x8</button></div>
<canvas id="cv" width="480" height="640"></canvas>"""


def race_html(entries):
    """entries: [(이름표, Brain), ...] 오래된 세대부터"""
    n, k = len(entries), len(RACE_COLORS)
    if n == 1:
        colors = [RACE_COLORS[-1]]
    elif n <= k:                                     # 팔레트에서 고르게 뽑기 (마지막 세대 = 빨강)
        colors = [RACE_COLORS[round(i * (k - 1) / (n - 1))] for i in range(n)]
    else:
        colors = [RACE_COLORS[i % k] for i in range(n)]
    racers = [dict(label=label, color=color, **_brain_js(brain))
              for (label, brain), color in zip(entries, colors)]
    return _page(RACE_BODY, RACE_SCRIPT.replace("__RACERS__", json.dumps(racers)))
