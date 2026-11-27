"""
똥피하기 게임의 '규칙'만 담은 파일. (pygame 없이 동작한다)

화면 그리기(draw.py)와 규칙(Game)을 나눠 둔 이유:
1) AI 100명이 동시에 플레이할 때는 화면 없이 규칙만 빠르게 돌려서 학습시키기 위해
2) 웹(Streamlit) 서버처럼 화면이 없는 곳에서도 그대로 쓰기 위해
"""
import random

# ---------------- 게임 설정 ----------------
WIDTH, HEIGHT = 480, 640       # 화면 크기
PLAYER_W, PLAYER_H = 36, 44    # 캐릭터 크기
PLAYER_SPEED = 7               # 한 프레임에 움직이는 거리
POOP_SIZE = 30                 # 똥 크기
FPS = 60

# 행동 번호 (AI도 이 번호로 행동을 고른다)
STAY, LEFT, RIGHT = 0, 1, 2


def box(x, y, w, h):
    """판정 상자 (x, y, 너비, 높이). 소수점은 버린다."""
    return int(x), int(y), int(w), int(h)


def overlaps(a, b):
    """두 상자가 겹치면 True (테두리가 맞닿기만 한 것은 충돌이 아님)"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


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
        return box(self.x + 6, self.y + 8, PLAYER_W - 12, PLAYER_H - 8)


class Poop:
    def __init__(self, x, speed):
        self.x = x
        self.y = -POOP_SIZE
        self.speed = speed

    def update(self):
        self.y += self.speed

    def hitbox(self):
        return box(self.x + 4, self.y + 6, POOP_SIZE - 8, POOP_SIZE - 8)


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
        poop_boxes = [p.hitbox() for p in self.poops]
        for player, action in zip(self.players, actions):
            if not player.alive:
                continue
            player.move(action)
            me = player.hitbox()
            if any(overlaps(me, b) for b in poop_boxes):
                player.alive = False
            else:
                player.score += 1

    def all_dead(self):
        return not any(p.alive for p in self.players)
