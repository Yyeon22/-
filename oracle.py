"""
AI는 얼마나 잘하고 있는 걸까? 세 선수를 같은 판에서 비교한다.

1) 신 (god): 앞으로 떨어질 똥까지 '미래를 다 안다' = 이 게임의 이론적 한계.
   똥은 플레이어가 어떻게 움직이든 seed만으로 똑같이 떨어진다.
   그래서 프레임마다 "지금 내가 서 있을 수 있는 모든 x 위치"를 계속 따라가면
   (왼쪽/정지/오른쪽 세 갈래로 퍼뜨리고, 똥에 맞는 위치는 지우고)
   살 수 있는 위치가 하나도 안 남는 순간이 '아무리 잘해도 죽는 순간'이 된다.
2) 계획자 (planner): AI와 똑같이 '지금 화면에 보이는 똥'만 안다 (새로 나올 똥은 모름).
   매 프레임 세 행동(정지/왼쪽/오른쪽)마다 위와 같은 계산으로 1.5초 앞을 내다보고
   가장 오래, 가장 넓게 살아남는 행동을 고른다. = 같은 정보로 얼마나 잘할 수 있나.
3) AI: 유전 알고리즘으로 학습한 신경망.

    python oracle.py                       # 20판: 신 vs AI
    python oracle.py --planner             # 계획자도 같이 (느림: 판마다 수십 초)
    python oracle.py --games 50 --brain results/exp_old/best_brain.npz
결과는 results/oracle.csv 에 저장된다 (웹 '학습 그래프' 탭, plot.py에서 사용).
"""
import argparse
import csv
import os

import numpy as np

from brain import Brain
from game import LEFT, PLAYER_SPEED, PLAYER_W, RIGHT, STAY, WIDTH, Game, Player

TEST_SEED = 10_000        # watch.py --test 와 같은 판
MAX_FRAMES = 20_000
PLAN_FRAMES = 90          # 계획자가 내다보는 길이 (1.5초)

HX, HY, HW, HH = Player().hitbox()          # 판정 상자 (x만 바뀐다)
DX = HX - int(Player().x)                   # 캐릭터 x → 판정 상자 x 차이
MOVES = np.array([-PLAYER_SPEED, 0, PLAYER_SPEED])
X_MAX = WIDTH - PLAYER_W


def spread(xs):
    """한 프레임 뒤에 있을 수 있는 위치들 (왼쪽/정지/오른쪽)"""
    return np.unique(np.clip((xs[:, None] + MOVES).ravel(), 0, X_MAX))


def survivors(xs, boxes):
    """똥 판정 상자들(boxes)에 안 맞는 위치만 남긴다. game.py의 Player.hitbox()/overlaps()와 같은 판정."""
    boxes = [b for b in boxes if HY < b[1] + b[3] and b[1] < HY + HH]   # 내 높이와 겹치는 똥만
    if not boxes:
        return xs
    b = np.array(boxes)
    ax = xs + DX
    hit = (ax[:, None] < b[:, 0] + b[:, 2]) & (b[:, 0] < ax[:, None] + HW)
    return xs[~hit.any(axis=1)]


def god_score(seed, max_frames=MAX_FRAMES):
    """seed 판에서 미래를 다 알 때 버틸 수 있는 최대 프레임 수"""
    game = Game(n_players=0, seed=seed)            # 똥만 떨어지는 판
    xs = np.array([int(Player().x)])               # 지금 서 있을 수 있는 x 위치들
    while game.frame < max_frames:
        game.step([])
        xs = survivors(spread(xs), [p.hitbox() for p in game.poops])
        if len(xs) == 0:
            return game.frame - 1                  # 이번 프레임에 모든 위치가 죽음
    return max_frames


def plan(game, x, horizon=PLAN_FRAMES):
    """화면에 보이는 똥만 보고 고른 행동. 보이는 똥은 지금 속도 그대로 떨어진다고 계산한다."""
    poops = [(p.x, p.y, p.size, p.speed) for p in game.poops]
    best, best_key = STAY, None
    for action, first in ((STAY, 0), (LEFT, -PLAYER_SPEED), (RIGHT, PLAYER_SPEED)):
        xs = np.array([min(max(x + first, 0), X_MAX)])
        lasted = 0
        for t in range(1, horizon + 1):
            if t > 1:
                xs = spread(xs)
            boxes = [(int(px + 4), int(py + v * t + 6), int(s - 8), int(s - 8)) for px, py, s, v in poops]
            xs = survivors(xs, boxes)
            if len(xs) == 0:
                break
            lasted = t
        key = (lasted, len(xs))                    # 오래 살수록, 갈 수 있는 곳이 많을수록 좋다
        if best_key is None or key > best_key:
            best, best_key = action, key
    return best


def play(seed, choose, max_frames=MAX_FRAMES):
    game = Game(1, seed=seed)
    while not game.all_dead() and game.frame < max_frames:
        game.step([choose(game, game.players[0])])
    return game.players[0].score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--brain", default="results/run/best_brain.npz")
    ap.add_argument("--planner", action="store_true", help="'보이는 것만 아는 계획자'도 측정 (느림)")
    args = ap.parse_args()

    brain = Brain.load(args.brain) if os.path.exists(args.brain) else None
    rows = []
    for i in range(args.games):
        seed = TEST_SEED + i
        god = god_score(seed)
        planner = play(seed, lambda g, p: plan(g, p.x)) if args.planner else ""
        ai = play(seed, lambda g, p: brain.act(brain.observe(g, p))) if brain else ""
        rows.append((seed, god, planner, ai))
        text = f"{i + 1:2d}판: 신 {god / 60:5.1f}초"
        if args.planner:
            text += f" | 계획자 {planner / 60:5.1f}초"
        if brain:
            text += f" | AI {ai / 60:5.1f}초"
        print(text, flush=True)

    print()
    for col, name in ((1, "신 (미래를 다 앎)"), (2, "계획자 (보이는 것만)"), (3, "학습된 AI")):
        vals = [r[col] for r in rows if r[col] != ""]
        if vals:
            print(f"{name:14s} 평균 {np.mean(vals) / 60:6.1f}초")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "oracle.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "god", "planner", "ai"])
        w.writerows(rows)
    print("저장: results/oracle.csv")


if __name__ == "__main__":
    main()
