"""pygame으로 게임 화면 그리기 (내 컴퓨터에서 실행하는 play.py / train.py / watch.py용)"""
import pygame

from game import HEIGHT, PLAYER_H, PLAYER_W, POOP_SIZE, WIDTH

SKY = (200, 230, 255)
GROUND = (120, 180, 90)
BROWN = (120, 72, 30)
BROWN_LIGHT = (150, 95, 45)


def draw_poop(screen, x, y, size=POOP_SIZE):
    s = size
    # 아래에서 위로 점점 작아지는 세 덩어리
    pygame.draw.ellipse(screen, BROWN, (x, y + s * 0.55, s, s * 0.45))
    pygame.draw.ellipse(screen, BROWN_LIGHT, (x + s * 0.15, y + s * 0.3, s * 0.7, s * 0.4))
    pygame.draw.ellipse(screen, BROWN, (x + s * 0.3, y + s * 0.08, s * 0.4, s * 0.32))


def draw_player(screen, player, color=(255, 200, 60), alpha=255):
    surf = pygame.Surface((PLAYER_W, PLAYER_H), pygame.SRCALPHA)
    c = (*color, alpha)
    pygame.draw.circle(surf, c, (PLAYER_W // 2, 12), 11)                     # 머리
    pygame.draw.rect(surf, c, (8, 22, PLAYER_W - 16, PLAYER_H - 22), border_radius=6)  # 몸
    pygame.draw.circle(surf, (0, 0, 0, alpha), (PLAYER_W // 2 - 4, 11), 2)   # 눈
    pygame.draw.circle(surf, (0, 0, 0, alpha), (PLAYER_W // 2 + 4, 11), 2)
    screen.blit(surf, (player.x, player.y))


def draw(screen, game, font):
    screen.fill(SKY)
    pygame.draw.rect(screen, GROUND, (0, HEIGHT - 10, WIDTH, 10))
    for poop in game.poops:
        draw_poop(screen, poop.x, poop.y, poop.size)
    for player in game.players:
        if player.alive:
            draw_player(screen, player)
    best = max(p.score for p in game.players)
    screen.blit(font.render(f"STAGE {game.stage()}   SCORE {best}", True, (0, 0, 0)), (10, 10))
