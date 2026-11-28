# 똥피하기 AI 🙂💩

직접 만든 똥피하기 게임을, 규칙을 하나도 모르는 AI가 **유전 알고리즘**으로 스스로 배워서 깨는 프로젝트.
신경망과 유전 알고리즘은 라이브러리 없이 numpy로 직접 구현했다.

## 🌐 웹 버전 (Streamlit) — 설치 없이 브라우저로 하기
```bash
pip install -r requirements.txt
streamlit run app.py
```
탭 3개:
- **🎮 게임** — 혼자 하기 / 학습된 AI와 대결. 키보드, 버튼, 화면 끌기(모바일)로 조작. 효과음(🔊로 끄기).
  혼자 하기에는 8초마다 아이템(🛡️ 3초 무적, ⏳ 3초 느리게)이 나오고 최고 기록이 브라우저에 저장된다.
  AI 대결은 공정하게 아이템 없이 파이썬과 똑같은 규칙.
- **🧬 AI 성장 레이스** — 학습 중간중간 저장한 세대별 AI(`results/run/gen_XXXX.npz`)가 같은 똥을 동시에 피한다. 배속 x1~x8.
- **📈 학습 그래프** — 세대별 실력, 센서 실험 비교(`results/exp_*`), 이론적 한계와 비교(`results/oracle.csv`).

10초마다 스테이지가 올라가며 똥이 커지고(30 → 최대 70), 빨라지고, 더 자주 떨어진다. 설정값은 `game.py` 맨 위에 있다.
실시간 조작을 위해 게임 규칙과 학습된 AI의 가중치를 JavaScript로 옮겨 브라우저에서 돌린다.

## 내 컴퓨터용 (pygame) 설치
```bash
pip install pygame numpy matplotlib
```

## 파일
| 파일 | 역할 |
|---|---|
| `game.py` | 게임 규칙 (여러 명이 같은 똥을 동시에 피할 수 있음, pygame 불필요) |
| `draw.py` | pygame 화면 그리기 |
| `app.py` | Streamlit 웹 앱 |
| `web_view.py` | 웹 게임 화면 (HTML/JavaScript) |
| `play.py` | 사람이 직접 플레이 |
| `brain.py` | AI의 눈(센서)과 뇌(작은 신경망). 센서 개수·간격은 뇌 파일에 같이 저장된다 |
| `train.py` | 유전 알고리즘으로 학습 (세대 상위권은 처음 보는 판으로 다시 시험해서 진짜 잘하는 뇌만 저장) |
| `oracle.py` | 미래를 다 아는 '신'이 몇 초 버티는지 = 게임의 이론적 한계 계산 |
| `watch.py` | 학습된 AI 보기 / AI와 대결 / 실력 측정 |
| `plot.py` | 학습 그래프 그리기 |

## 사용법
```bash
python play.py                         # 내가 직접 플레이
python train.py --show                 # 학습 과정을 화면으로 보기 (↑↓ 속도 조절)
python train.py                        # 화면 없이 빠르게 학습 → results/run/
python train.py --generations 300 --population 200 --games 6 --val-games 12 --val-top 5 --max-frames 9000
                                       # 강하게 학습 (지금 results/run/의 AI, CPU 여러 개로 동시에 돌림, 약 1시간)
python train.py --sensors 15 --hidden 16 --soft 0 --no-memory   # AI 눈/뇌 바꾸기 (기본: 9센서, 12뉴런, 부드러운 센서, 기억)
python train.py --init results/run/best_brain.npz --name more   # 저장된 AI에서 이어서 학습
python watch.py                        # 학습된 AI 보기
python watch.py --vs                   # AI(빨강) vs 나(노랑)
python watch.py --test 20              # 처음 보는 20판으로 실력 측정
python oracle.py --games 20 --planner   # 신 / 계획자 / 학습된 AI 비교 → results/oracle.csv
python watch.py --compare exp_old exp_both --test 30   # 실험끼리 같은 30판으로 비교 → results/experiments.csv
python plot.py run                     # 그래프 → results/graph.png
```

실험 예시 (돌연변이율 비교):
```bash
python train.py --mutation-rate 0.01 --name mut01
python train.py --mutation-rate 0.1  --name mut10
python train.py --mutation-rate 0.3  --name mut30
python plot.py mut01 mut10 mut30
```

## 탐구 결과: AI를 더 강하게 만들 수 있을까?
세 선수를 처음 보는 같은 20판에서 비교했다 (`python oracle.py --games 20 --planner`).

| 선수 | 아는 것 | 평균 생존 |
|---|---|---|
| 신 | 앞으로 떨어질 똥까지 전부 (= 이 게임의 이론적 한계) | 약 241초 |
| 계획자 | AI와 똑같이 화면에 보이는 똥만, 매 프레임 1.5초 앞을 계산 | 약 91초 |
| 학습된 AI | 화면에 보이는 똥 (센서 9개) | 약 38초 |

1. **센서를 넓히면 오히려 나빠졌다.** 9개(±120px) 38.7초 → 15개 36.9초 → 21개 25.6초.
   입력이 늘면 유전 알고리즘이 맞춰야 할 가중치가 많아져 학습이 느려진다.
2. **AI가 죽는 장면을 분석했다.** 죽기 0.5초 전 위치에서는 30판 모두 피할 길이 있었다 → 함정에 갇힌 게 아니라 순간 실수.
   실제 장면은 '좌우로 떨기'였다: 똥 경계가 센서 위에 걸리면 7px 움직일 때마다 센서 값이 0↔1로 깜빡이고,
   AI는 왼쪽·오른쪽을 번갈아 누르며 제자리에서 떨다 맞았다.
3. **부드러운 센서 + 직전 행동 기억**을 넣었더니 가장 좋았지만(40.0초) 차이는 작았다.
   이 설정으로 300세대 학습한 지금 AI는 이전 AI와 60판 평균 41.5초 vs 41.0초로 사실상 같다
   (아주 일찍 죽는 판은 조금 줄었다: 최저 7.8초 vs 2.2초).
4. **결론**: 같은 정보만 가진 계획자가 AI보다 2배 넘게 버틴다. AI가 못 보는 게 아니라,
   작은 신경망 + 유전 알고리즘이 '보이는 정보를 잘 쓰는 법'을 다 배우지 못한 것이 한계다.
   (다음 탐구 거리: 계획자의 행동을 신경망이 따라 배우는 '모방 학습'과 비교하기)

실험 기록: `results/exp_*/` (모두 같은 seed, 100세대, 150마리), 비교표 `results/experiments.csv`.
