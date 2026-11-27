"""
똥피하기 게임의 '규칙'만 담은 파일.

화면 그리기(draw)와 규칙(Game)을 나눠 둔 이유:
나중에 AI 100명이 동시에 플레이할 때는 화면을 그리지 않고
규칙만 빠르게 돌려서 학습시키기 위해서다.
"""
import random

import pygame

# ---------------- 게임 설정 ----------------
WIDTH, HEIGHT = 480, 640       # 화면 크기
PLAYER_W, PLAYER_H = 36, 44    # 캐릭터 크기
PLAYER_SPEED = 7               # 한 프레임에 움직이는 거리
POOP_SIZE = 30                 # 똥 크기
FPS = 60

# 행동 번호 (AI도 이 번호로 행동을 고른다)
STAY, LEFT, RIGHT = 0, 1, 2


class Player:
    def __init__(self):
        self.x = WIDTH / 2 - PLAYER_W / 2   # 왼쪽 위 모서리 x
        self.y = HEIGHT - PLAYER_H - 10
        self.alive = True
        self.score = 0                       # 살아남은 프레임 수

    def move(self, action):
        if action == LEFT:
            self.x -= PLAYER_SPEED
        elif action == RIGHT:
            self.x += PLAYER_SPEED
        # 화면 밖으로 나가지 못하게 막기
        self.x = max(0, min(WIDTH - PLAYER_W, self.x))

    def hitbox(self):
        # 그림보다 판정을 조금 작게 해서 억울한 죽음을 줄인다
        return pygame.Rect(self.x + 6, self.y + 8, PLAYER_W - 12, PLAYER_H - 8)


class Poop:
    def __init__(self, x, speed):
        self.x = x
        self.y = -POOP_SIZE
        self.speed = speed

    def update(self):
        self.y += self.speed

    def hitbox(self):
        return pygame.Rect(self.x + 4, self.y + 6, POOP_SIZE - 8, POOP_SIZE - 8)


class Game:
    """플레이어 여러 명이 '같은 똥'을 피하는 게임. 사람 혼자면 n_players=1."""

    def __init__(self, n_players=1, seed=None):
        self.rng = random.Random(seed)   # seed가 같으면 똥이 똑같이 떨어진다 (공정한 비교용)
        self.players = [Player() for _ in range(n_players)]
        self.poops = []
        self.frame = 0

    # 시간이 지날수록 어려워진다
    def spawn_interval(self):
        return max(6, 30 - self.frame // 300)        # 똥이 나오는 간격(프레임)

    def poop_speed(self):
        return 4 + self.frame / 900 + self.rng.random() * 2

    def step(self, actions):
        """actions: 플레이어마다 행동 번호 하나씩 (STAY/LEFT/RIGHT)"""
        self.frame += 1

        # 1) 새 똥 만들기
        if self.frame % self.spawn_interval() == 0:
            x = self.rng.uniform(0, WIDTH - POOP_SIZE)
            self.poops.append(Poop(x, self.poop_speed()))

        # 2) 똥 떨어뜨리기, 바닥에 닿은 똥 지우기
        for poop in self.poops:
            poop.update()
        self.poops = [p for p in self.poops if p.y < HEIGHT]

        # 3) 플레이어 움직이기 + 충돌 검사
        for player, action in zip(self.players, actions):
            if not player.alive:
                continue
            player.move(action)
            box = player.hitbox()
            if any(box.colliderect(p.hitbox()) for p in self.poops):
                player.alive = False
            else:
                player.score += 1

    def all_dead(self):
        return not any(p.alive for p in self.players)


# ---------------- 그림 그리기 (직접 그린 도형) ----------------
SKY = (200, 230, 255)
GROUND = (120, 180, 90)
BROWN = (120, 72, 30)
BROWN_LIGHT = (150, 95, 45)


def draw_poop(screen, x, y):
    s = POOP_SIZE
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
        draw_poop(screen, poop.x, poop.y)
    for player in game.players:
        if player.alive:
            draw_player(screen, player)
    best = max(p.score for p in game.players)
    screen.blit(font.render(f"SCORE {best}", True, (0, 0, 0)), (10, 10))
