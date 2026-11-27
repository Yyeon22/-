"""
AI의 '눈'(observe)과 '뇌'(Brain).

뇌는 numpy로 직접 만든 아주 작은 신경망이다.
    입력(눈으로 본 것) → 은닉층 → 출력(정지/왼쪽/오른쪽 점수)
점수가 가장 큰 행동을 고른다.
학습은 경사하강법이 아니라 '유전 알고리즘'으로 한다 (train.py).
"""
import numpy as np

from game import PLAYER_W, POOP_SIZE, WIDTH, poop_hit_dist

#   AI의 눈 = 내 위치를 기준으로 좌우에 늘어선 센서 9개
#
#        ▼        ▼
#   [-4][-3][-2][-1][ 0][+1][+2][+3][+4]     ← 칸마다 '위험도' 0~1
#                    🙂
#
#   위험도 1 = 이 칸에 곧 똥이 떨어짐 (또는 벽 밖이라 갈 수 없음)
#   위험도 0 = 이 칸은 당분간 안전
SENSOR_OFFSETS = [-120, -90, -60, -30, 0, 30, 60, 90, 120]   # 센서 위치 (픽셀)
HIT_DIST = poop_hit_dist(POOP_SIZE)   # 1스테이지 똥 기준 충돌 거리 (똥이 커지면 똥마다 다시 계산)
LOOK_AHEAD = 60         # 몇 프레임 뒤까지 내다볼지 (60프레임 = 1초)

N_INPUTS = 1 + len(SENSOR_OFFSETS)     # 내 위치 1개 + 센서 9개
N_HIDDEN = 12
N_OUTPUTS = 3                          # 정지, 왼쪽, 오른쪽


def observe(game, player):
    """플레이어 입장에서 본 세상을 숫자 목록으로 바꾼다."""
    px = player.x + PLAYER_W / 2                       # 내 몸 가운데
    obs = [px / WIDTH * 2 - 1]                         # 왼쪽 벽 -1, 가운데 0, 오른쪽 벽 +1

    for offset in SENSOR_OFFSETS:
        sx = px + offset
        if sx < PLAYER_W / 2 or sx > WIDTH - PLAYER_W / 2:
            obs.append(1.0)                            # 벽 밖 = 갈 수 없음 = 최대 위험
            continue
        soonest = LOOK_AHEAD                           # 이 칸에 똥이 도착하기까지 남은 프레임
        for p in game.poops:
            if p.y > player.y + 10:                    # 이미 지나간 똥은 무시
                continue
            if abs(p.x + p.size / 2 - sx) < poop_hit_dist(p.size):   # 큰 똥일수록 넓은 범위가 위험
                frames_left = max(0.0, (player.y - p.y) / p.speed)
                soonest = min(soonest, frames_left)
        obs.append(1 - soonest / LOOK_AHEAD)
    return np.array(obs)


def observe_many(game, players):
    """observe()와 똑같은 계산을 여러 플레이어에 대해 numpy로 한꺼번에 한다 (학습 속도용).
    결과: (플레이어 수, N_INPUTS) 배열"""
    pyl = players[0].y                                  # 플레이어는 모두 같은 높이에 있다
    px = np.array([p.x for p in players]) + PLAYER_W / 2
    sx = px[:, None] + np.array(SENSOR_OFFSETS)[None, :]          # (플레이어, 센서)
    soonest = np.full(sx.shape, float(LOOK_AHEAD))
    ahead = [p for p in game.poops if p.y <= pyl + 10]           # 아직 안 지나간 똥만
    if ahead:
        cx = np.array([p.x + p.size / 2 for p in ahead])
        hit_dist = np.array([poop_hit_dist(p.size) for p in ahead])
        frames_left = np.array([max(0.0, (pyl - p.y) / p.speed) for p in ahead])
        hit = np.abs(cx[None, None, :] - sx[:, :, None]) < hit_dist  # (플레이어, 센서, 똥)
        soonest = np.minimum(soonest, np.where(hit, frames_left, np.inf).min(axis=2))
    danger = 1 - soonest / LOOK_AHEAD
    danger[(sx < PLAYER_W / 2) | (sx > WIDTH - PLAYER_W / 2)] = 1.0  # 벽 밖
    return np.column_stack([px / WIDTH * 2 - 1, danger])


def stack_weights(brains):
    """뇌 여러 개의 가중치를 쌓아 act_many()에서 한 번에 계산할 수 있게 한다."""
    return {k: np.stack([b.w[k] for b in brains]) for k in brains[0].w}


def act_many(stacked, obs):
    """Brain.act()를 여러 뇌에 대해 한꺼번에. stacked는 stack_weights()의 결과 (obs와 같은 순서)"""
    hidden = np.tanh(np.einsum("mi,mij->mj", obs, stacked["w1"]) + stacked["b1"])
    out = np.einsum("mj,mjk->mk", hidden, stacked["w2"]) + stacked["b2"]
    return out.argmax(axis=1)


class Brain:
    def __init__(self, rng, weights=None):
        if weights is None:
            # 처음엔 완전히 무작위 = 아무것도 모르는 뇌
            weights = {
                "w1": rng.normal(0, 1, (N_INPUTS, N_HIDDEN)),
                "b1": rng.normal(0, 1, N_HIDDEN),
                "w2": rng.normal(0, 1, (N_HIDDEN, N_OUTPUTS)),
                "b2": rng.normal(0, 1, N_OUTPUTS),
            }
        self.w = weights

    def act(self, obs):
        hidden = np.tanh(obs @ self.w["w1"] + self.w["b1"])
        out = hidden @ self.w["w2"] + self.w["b2"]
        return int(np.argmax(out))          # 0=정지, 1=왼쪽, 2=오른쪽

    # ---------- 유전 알고리즘에 쓰는 기능 ----------
    def copy(self):
        # 주의: .copy()를 안 하면 부모와 자식이 '같은 배열'을 공유해서
        # 자식을 돌연변이시키면 부모까지 바뀌어 버린다!
        return Brain(None, {k: v.copy() for k, v in self.w.items()})

    def crossover(self, other, rng):
        """두 부모의 가중치를 하나씩 무작위로 골라 섞은 자식"""
        child = {}
        for k in self.w:
            pick = rng.random(self.w[k].shape) < 0.5
            child[k] = np.where(pick, self.w[k], other.w[k])
        return Brain(None, child)

    def mutate(self, rng, rate, scale=0.5):
        """가중치 중 rate 비율만큼 작은 무작위 값을 더한다"""
        for k in self.w:
            mask = rng.random(self.w[k].shape) < rate
            self.w[k] = self.w[k] + mask * rng.normal(0, scale, self.w[k].shape)

    def save(self, path):
        np.savez(path, **self.w)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Brain(None, {k: data[k] for k in data.files})
