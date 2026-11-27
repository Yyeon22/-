"""
학습된 AI 보기 / AI와 대결하기 / 실력 측정.

    python watch.py                      # 저장된 최고 AI가 플레이 (센서도 보여줌)
    python watch.py --vs                 # AI(빨강) vs 나(노랑): 같은 똥을 동시에 피하기
    python watch.py --test 20            # 화면 없이 20판 해서 평균 점수 측정
    python watch.py --brain results/mut30/best_brain.npz   # 다른 AI 불러오기
조작: ← → 이동(대결 모드) / R 다시 시작 / ESC 종료
"""
import argparse
import sys

import numpy as np

from brain import SENSOR_OFFSETS, Brain, observe
from game import (FPS, HEIGHT, LEFT, PLAYER_W, RIGHT, STAY, WIDTH, Game,
                  GROUND, SKY, draw_player, draw_poop)


def test(brain, n_games):
    scores = []
    for seed in range(n_games):
        game = Game(1, seed=10_000 + seed)      # 학습 때 안 본 새로운 똥 패턴
        while not game.all_dead() and game.frame < 20_000:
            game.step([brain.act(observe(game, game.players[0]))])
        scores.append(game.players[0].score)
        print(f"{seed + 1:2d}판: {scores[-1]}")
    print(f"평균 {np.mean(scores):.0f} / 최고 {max(scores)} / 최저 {min(scores)}")


def draw_sensors(screen, pygame, player, obs):
    """AI가 '보고 있는' 위험도를 캐릭터 위에 색깔 칸으로 표시 (초록=안전, 빨강=위험)"""
    cx = player.x + PLAYER_W / 2
    for offset, danger in zip(SENSOR_OFFSETS, obs[1:]):
        color = (int(255 * danger), int(200 * (1 - danger)), 60)
        pygame.draw.rect(screen, color, (cx + offset - 12, player.y - 26, 24, 10), border_radius=3)


def run(brain, vs):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Poop Dodge - AI vs YOU" if vs else "Poop Dodge - AI")
    font = pygame.font.SysFont(None, 30)
    big = pygame.font.SysFont(None, 56)
    clock = pygame.time.Clock()

    def new_game():
        return Game(n_players=2 if vs else 1)

    game = new_game()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game = new_game()

        ai = game.players[0]
        obs = observe(game, ai)
        actions = [brain.act(obs) if ai.alive else STAY]
        if vs:
            keys = pygame.key.get_pressed()
            me = RIGHT if keys[pygame.K_RIGHT] else LEFT if keys[pygame.K_LEFT] else STAY
            actions.append(me)
        if not game.all_dead():
            game.step(actions)

        screen.fill(SKY)
        pygame.draw.rect(screen, GROUND, (0, HEIGHT - 10, WIDTH, 10))
        for poop in game.poops:
            draw_poop(screen, poop.x, poop.y)
        if ai.alive:
            draw_sensors(screen, pygame, ai, obs)
            draw_player(screen, ai, color=(230, 60, 60))
        texts = [f"AI {ai.score}"]
        if vs:
            me = game.players[1]
            if me.alive:
                draw_player(screen, me)
            texts.append(f"YOU {me.score}")
        for i, t in enumerate(texts):
            screen.blit(font.render(t, True, (0, 0, 0)), (10, 10 + 26 * i))

        if game.all_dead():
            if vs:
                me = game.players[1]
                msg = "YOU WIN!" if me.score > ai.score else "AI WINS!" if ai.score > me.score else "DRAW"
            else:
                msg = "GAME OVER"
            surf = big.render(msg, True, (200, 30, 30))
            screen.blit(surf, surf.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 20)))
            hint = font.render("Press R to restart", True, (0, 0, 0))
            screen.blit(hint, hint.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 25)))
        pygame.display.flip()
        clock.tick(FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="results/run/best_brain.npz")
    ap.add_argument("--vs", action="store_true", help="AI와 대결")
    ap.add_argument("--test", type=int, default=0, help="화면 없이 N판 실력 측정")
    args = ap.parse_args()

    brain = Brain.load(args.brain)
    if args.test:
        test(brain, args.test)
    else:
        run(brain, args.vs)


if __name__ == "__main__":
    main()
