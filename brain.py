"""
AI의 '눈'(observe)과 '뇌'(Brain).

뇌는 numpy로 직접 만든 아주 작은 신경망이다.
    입력(눈으로 본 것) → 은닉층 → 출력(정지/왼쪽/오른쪽 점수)
점수가 가장 큰 행동을 고른다.
학습은 경사하강법이 아니라 '유전 알고리즘'으로 한다 (train.py).
"""
from dataclasses import dataclass

import numpy as np

from game import LEFT, PLAYER_W, POOP_SIZE, RIGHT, WIDTH, poop_hit_dist

#   AI의 눈 = 내 위치를 기준으로 좌우에 늘어선 센서 9개
#
#        ▼        ▼
#   [-4][-3][-2][-1][ 0][+1][+2][+3][+4]     ← 칸마다 '위험도' 0~1
#                    🙂
#
#   위험도 1 = 이 칸에 곧 똥이 떨어짐 (또는 벽 밖이라 갈 수 없음)
#   위험도 0 = 이 칸은 당분간 안전
#   (부드러운 센서) 똥 판정 경계에서 위험도가 0↔1로 뚝 끊기지 않고 soft 픽셀에 걸쳐 서서히 변한다.
#      → 7px 움직일 때마다 센서가 깜빡여서 AI가 좌우로 떨다 죽는 문제를 막는다.
#   (기억) 직전에 누른 방향(왼쪽/오른쪽)을 입력으로 같이 준다 → 한 방향으로 꾸준히 움직이는 법을 배울 수 있다.
#
#   눈 설정(Eye)은 뇌마다 다를 수 있고 뇌 파일(.npz)에 같이 저장된다.
#   OLD_EYE는 설정이 저장되지 않은 옛날 뇌(9센서 ±120px, 딱딱한 센서, 기억 없음)를 불러올 때 쓴다.
SENSOR_OFFSETS = [-120, -90, -60, -30, 0, 30, 60, 90, 120]   # 옛 센서 위치 (픽셀)
HIT_DIST = poop_hit_dist(POOP_SIZE)   # 1스테이지 똥 기준 충돌 거리 (똥이 커지면 똥마다 다시 계산)
LOOK_AHEAD = 60         # 몇 프레임 뒤까지 내다볼지 (60프레임 = 1초)

N_HIDDEN = 12
N_OUTPUTS = 3                          # 정지, 왼쪽, 오른쪽


@dataclass(frozen=True)
class Eye:
    offsets: tuple = tuple(SENSOR_OFFSETS)   # 센서 위치들 (내 몸 가운데 기준, 픽셀)
    look: int = LOOK_AHEAD                   # 몇 프레임 뒤까지 볼지
    soft: int = 0                            # 0 = 딱딱한 센서(옛날 방식), 15 = 경계 ±15px에서 서서히 변함
    memory: bool = False                     # 직전 행동을 입력으로 받을지

    @property
    def n_inputs(self):
        return 1 + len(self.offsets) + (2 if self.memory else 0)


OLD_EYE = Eye()


def make_offsets(n_sensors, spacing):
    """내 위치를 가운데로 좌우 대칭인 센서 위치 목록. 예) make_offsets(9, 30) → -120 ~ +120"""
    half = (n_sensors - 1) / 2
    return tuple(round((i - half) * spacing) for i in range(n_sensors))


def coverage(dist, hit_dist, soft):
    """센서가 똥 판정 범위 안에 얼마나 들어가 있나 (0~1). soft=0이면 들어가면 1, 아니면 0"""
    if soft == 0:
        return (dist < hit_dist).astype(float) if isinstance(dist, np.ndarray) else float(dist < hit_dist)
    return np.clip((hit_dist + soft - dist) / (2 * soft), 0, 1)


def observe(game, player, eye=OLD_EYE):
    """플레이어 입장에서 본 세상을 숫자 목록으로 바꾼다."""
    px = player.x + PLAYER_W / 2                       # 내 몸 가운데
    obs = [px / WIDTH * 2 - 1]                         # 왼쪽 벽 -1, 가운데 0, 오른쪽 벽 +1

    for offset in eye.offsets:
        sx = px + offset
        if sx < PLAYER_W / 2 or sx > WIDTH - PLAYER_W / 2:
            obs.append(1.0)                            # 벽 밖 = 갈 수 없음 = 최대 위험
            continue
        danger = 0.0
        for p in game.poops:
            if p.y > player.y + 10:                    # 이미 지나간 똥은 무시
                continue
            frames_left = max(0.0, (player.y - p.y) / p.speed)      # 이 똥이 내 높이에 오기까지
            urgency = max(0.0, 1 - frames_left / eye.look)          # 곧 올수록 1에 가깝다
            # 큰 똥일수록 넓은 범위가 위험
            cover = coverage(abs(p.x + p.size / 2 - sx), poop_hit_dist(p.size), eye.soft)
            danger = max(danger, float(urgency * cover))
        obs.append(danger)
    if eye.memory:
        obs += [float(player.last_action == LEFT), float(player.last_action == RIGHT)]
    return np.array(obs)


def observe_many(game, players, eye=OLD_EYE):
    """observe()와 똑같은 계산을 여러 플레이어에 대해 numpy로 한꺼번에 한다 (학습 속도용).
    결과: (플레이어 수, 입력 수) 배열"""
    pyl = players[0].y                                  # 플레이어는 모두 같은 높이에 있다
    px = np.array([p.x for p in players]) + PLAYER_W / 2
    sx = px[:, None] + np.array(eye.offsets)[None, :]             # (플레이어, 센서)
    danger = np.zeros(sx.shape)
    ahead = [p for p in game.poops if p.y <= pyl + 10]           # 아직 안 지나간 똥만
    if ahead:
        cx = np.array([p.x + p.size / 2 for p in ahead])
        hit_dist = np.array([poop_hit_dist(p.size) for p in ahead])
        frames_left = np.array([max(0.0, (pyl - p.y) / p.speed) for p in ahead])
        urgency = np.maximum(0.0, 1 - frames_left / eye.look)
        cover = coverage(np.abs(cx[None, None, :] - sx[:, :, None]), hit_dist, eye.soft)  # (플레이어, 센서, 똥)
        danger = (cover * urgency).max(axis=2)
    danger[(sx < PLAYER_W / 2) | (sx > WIDTH - PLAYER_W / 2)] = 1.0  # 벽 밖
    cols = [px / WIDTH * 2 - 1, danger]
    if eye.memory:
        last = np.array([p.last_action for p in players])
        cols += [last == LEFT, last == RIGHT]
    return np.column_stack(cols).astype(float)


def stack_weights(brains):
    """뇌 여러 개의 가중치를 쌓아 act_many()에서 한 번에 계산할 수 있게 한다."""
    return {k: np.stack([b.w[k] for b in brains]) for k in brains[0].w}


def act_many(stacked, obs):
    """Brain.act()를 여러 뇌에 대해 한꺼번에. stacked는 stack_weights()의 결과 (obs와 같은 순서)"""
    hidden = np.tanh(np.einsum("mi,mij->mj", obs, stacked["w1"]) + stacked["b1"])
    out = np.einsum("mj,mjk->mk", hidden, stacked["w2"]) + stacked["b2"]
    return out.argmax(axis=1)


class Brain:
    def __init__(self, rng, weights=None, eye=OLD_EYE, n_hidden=N_HIDDEN):
        self.eye = eye                               # 이 뇌의 눈 설정
        if weights is None:
            # 처음엔 완전히 무작위 = 아무것도 모르는 뇌
            weights = {
                "w1": rng.normal(0, 1, (eye.n_inputs, n_hidden)),
                "b1": rng.normal(0, 1, n_hidden),
                "w2": rng.normal(0, 1, (n_hidden, N_OUTPUTS)),
                "b2": rng.normal(0, 1, N_OUTPUTS),
            }
        self.w = weights

    def observe(self, game, player):
        return observe(game, player, self.eye)

    def act(self, obs):
        hidden = np.tanh(obs @ self.w["w1"] + self.w["b1"])
        out = hidden @ self.w["w2"] + self.w["b2"]
        return int(np.argmax(out))          # 0=정지, 1=왼쪽, 2=오른쪽

    def describe(self):
        e = self.eye
        extra = ((f", {e.look}프레임 앞" if e.look != LOOK_AHEAD else "") +
                 (f", 부드러운 센서 ±{e.soft}px" if e.soft else "") + (", 직전 행동 기억" if e.memory else ""))
        return f"센서 {len(e.offsets)}개(±{max(e.offsets)}px), 뉴런 {self.w['w1'].shape[1]}개{extra}"

    # ---------- 유전 알고리즘에 쓰는 기능 ----------
    def copy(self):
        # 주의: .copy()를 안 하면 부모와 자식이 '같은 배열'을 공유해서
        # 자식을 돌연변이시키면 부모까지 바뀌어 버린다!
        return Brain(None, {k: v.copy() for k, v in self.w.items()}, self.eye)

    def crossover(self, other, rng):
        """두 부모의 가중치를 하나씩 무작위로 골라 섞은 자식"""
        child = {}
        for k in self.w:
            pick = rng.random(self.w[k].shape) < 0.5
            child[k] = np.where(pick, self.w[k], other.w[k])
        return Brain(None, child, self.eye)

    def mutate(self, rng, rate, scale=0.5):
        """가중치 중 rate 비율만큼 작은 무작위 값을 더한다"""
        for k in self.w:
            mask = rng.random(self.w[k].shape) < rate
            self.w[k] = self.w[k] + mask * rng.normal(0, scale, self.w[k].shape)

    def save(self, path):
        e = self.eye
        np.savez(path, **self.w, offsets=np.array(e.offsets), look_ahead=e.look, soft=e.soft, memory=e.memory)

    @staticmethod
    def load(path):
        data = np.load(path)
        weights = {k: data[k] for k in ("w1", "b1", "w2", "b2")}
        if "offsets" not in data.files:              # 옛날 뇌 = 기본 눈
            return Brain(None, weights, OLD_EYE)
        eye = Eye(tuple(int(o) for o in data["offsets"]), int(data["look_ahead"]),
                  int(data["soft"]) if "soft" in data.files else 0,
                  bool(data["memory"]) if "memory" in data.files else False)
        return Brain(None, weights, eye)
