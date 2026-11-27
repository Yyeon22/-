"""
사람이 직접 하는 똥피하기.
조작: ← → 이동 / R 다시 시작 / ESC 종료
"""
import sys

import pygame

from draw import draw
from game import FPS, HEIGHT, LEFT, RIGHT, STAY, WIDTH, Game


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Poop Dodge")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    big_font = pygame.font.SysFont(None, 64)

    game = Game(n_players=1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r and game.all_dead():
                    game = Game(n_players=1)

        # 키보드 → 행동 번호
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            action = LEFT
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            action = RIGHT
        else:
            action = STAY

        if not game.all_dead():
            game.step([action])

        draw(screen, game, font)
        if game.all_dead():
            text = big_font.render("GAME OVER", True, (200, 30, 30))
            screen.blit(text, text.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 20)))
            hint = font.render("Press R to restart", True, (0, 0, 0))
            screen.blit(hint, hint.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 30)))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
