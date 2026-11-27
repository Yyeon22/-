"""
유전 알고리즘으로 똥피하기 AI 학습시키기.

한 세대 = AI 여러 명이 '똑같은 똥'을 동시에 피하는 게임.
모두 죽으면 오래 버틴 AI들을 부모로 골라 다음 세대를 만든다.
    1) 엘리트: 상위 몇 명은 그대로 다음 세대로 (좋은 뇌를 잃지 않게)
    2) 교배:   상위권 부모 두 명의 가중치를 섞는다
    3) 돌연변이: 가중치 일부를 살짝 바꾼다 (새로운 시도)

사용 예:
    python train.py                       # 빠른 학습 (화면 없음)
    python train.py --show                # 학습 과정을 화면으로 보기
    python train.py --mutation-rate 0.3 --name mut30   # 실험용
"""
import argparse
import csv
import os

import numpy as np

from brain import Brain, observe
from game import Game


def play_generation(brains, game_seed, max_frames, show=None):
    """모든 뇌가 같은 게임을 한 판 하고, 각자의 점수(생존 프레임)를 돌려준다."""
    game = Game(n_players=len(brains), seed=game_seed)
    while not game.all_dead() and game.frame < max_frames:
        actions = [
            brain.act(observe(game, player)) if player.alive else 0
            for brain, player in zip(brains, game.players)
        ]
        game.step(actions)
        if show is not None and not show(game):
            raise KeyboardInterrupt
    return np.array([p.score for p in game.players], dtype=float)


def next_generation(brains, fitness, rng, n_elite, parent_ratio, mutation_rate):
    order = np.argsort(fitness)[::-1]                    # 점수 높은 순
    ranked = [brains[i] for i in order]
    n_parents = max(2, int(len(brains) * parent_ratio))
    parents = ranked[:n_parents]

    new = [b.copy() for b in ranked[:n_elite]]           # 엘리트는 그대로
    while len(new) < len(brains):
        a, b = rng.choice(n_parents, 2, replace=False)
        child = parents[a].crossover(parents[b], rng)
        child.mutate(rng, mutation_rate)
        new.append(child)
    return new, ranked[0]


def make_viewer():
    """--show 옵션일 때 쓰는 화면. ↑↓로 속도 조절, ESC로 중단(저장됨)."""
    import pygame

    from game import WIDTH, HEIGHT, draw_player, draw_poop, SKY, GROUND

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Poop Dodge - AI training")
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    state = {"speed": 1, "gen": 0, "best_ever": 0}

    def show(game):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_UP:
                    state["speed"] = min(64, state["speed"] * 2)
                if event.key == pygame.K_DOWN:
                    state["speed"] = max(1, state["speed"] // 2)
        if game.frame % state["speed"]:
            return True                                  # 빨리감기: 몇 프레임마다 한 번만 그림

        screen.fill(SKY)
        pygame.draw.rect(screen, GROUND, (0, HEIGHT - 10, WIDTH, 10))
        for poop in game.poops:
            draw_poop(screen, poop.x, poop.y)
        alive = [p for p in game.players if p.alive]
        for p in alive[1:]:
            draw_player(screen, p, color=(255, 200, 60), alpha=70)   # 나머지는 반투명
        if game.players[0].alive:
            draw_player(screen, game.players[0], color=(230, 60, 60))  # 0번 = 지난 세대 1등
        lines = [
            f"GEN {state['gen']}   ALIVE {len(alive)}/{len(game.players)}",
            f"SCORE {game.frame}   BEST EVER {state['best_ever']}",
            f"SPEED x{state['speed']}  (UP/DOWN)",
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (0, 0, 0)), (10, 10 + 24 * i))
        pygame.display.flip()
        clock.tick(60)
        return True

    return show, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--parent-ratio", type=float, default=0.2)
    ap.add_argument("--mutation-rate", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=6000, help="한 판 최대 길이 (60프레임 = 1초)")
    ap.add_argument("--games", type=int, default=3, help="실력을 잴 때 몇 판의 평균을 낼지")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="run", help="결과 폴더 이름 (results/이름)")
    ap.add_argument("--show", action="store_true", help="학습 과정을 화면으로 보기")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join("results", args.name)
    os.makedirs(out_dir, exist_ok=True)

    show, view_state = make_viewer() if args.show else (None, None)

    brains = [Brain(rng) for _ in range(args.population)]
    best_ever, log = -1, []
    try:
        for gen in range(1, args.generations + 1):
            if view_state:
                view_state["gen"] = gen
            # 세대마다 똥 패턴을 바꾼다 (한 패턴만 외워버리는 것을 막기 위해)
            # 한 판은 운이 크게 작용하므로 여러 판의 '평균'으로 실력을 잰다
            fitness = np.mean([
                play_generation(brains, args.seed * 100000 + gen * 10 + g,
                                args.max_frames, show)
                for g in range(args.games)
            ], axis=0)

            best_i = int(np.argmax(fitness))
            if fitness[best_i] > best_ever:
                best_ever = int(fitness[best_i])
                brains[best_i].save(os.path.join(out_dir, "best_brain.npz"))
            if view_state:
                view_state["best_ever"] = best_ever

            log.append((gen, int(fitness.max()), round(float(fitness.mean()), 1),
                        int(np.median(fitness))))
            print(f"세대 {gen:3d} | 1등 {int(fitness.max()):5d} | 평균 {fitness.mean():7.1f} "
                  f"| 중앙값 {int(np.median(fitness)):5d} | 최고기록 {best_ever}")

            brains, _ = next_generation(brains, fitness, rng, args.elite,
                                        args.parent_ratio, args.mutation_rate)
    except KeyboardInterrupt:
        print("중단됨 - 지금까지 결과를 저장합니다.")

    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best", "average", "median"])
        w.writerows(log)
    print(f"저장 완료: {out_dir}/log.csv, {out_dir}/best_brain.npz")


if __name__ == "__main__":
    main()
