# 똥피하기 AI 🙂💩

직접 만든 똥피하기 게임을, 규칙을 하나도 모르는 AI가 **유전 알고리즘**으로 스스로 배워서 깨는 프로젝트.
신경망과 유전 알고리즘은 라이브러리 없이 numpy로 직접 구현했다.

## 🌐 웹 버전 (Streamlit) — 설치 없이 브라우저로 하기
```bash
pip install -r requirements.txt
streamlit run app.py
```
혼자 하기 / 학습된 AI와 대결하기.
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
| `brain.py` | AI의 눈(센서 9개)과 뇌(작은 신경망) |
| `train.py` | 유전 알고리즘으로 학습 |
| `watch.py` | 학습된 AI 보기 / AI와 대결 / 실력 측정 |
| `plot.py` | 학습 그래프 그리기 |

## 사용법
```bash
python play.py                         # 내가 직접 플레이
python train.py --show                 # 학습 과정을 화면으로 보기 (↑↓ 속도 조절)
python train.py                        # 화면 없이 빠르게 학습 → results/run/
python watch.py                        # 학습된 AI 보기
python watch.py --vs                   # AI(빨강) vs 나(노랑)
python watch.py --test 20              # 처음 보는 20판으로 실력 측정
python plot.py run                     # 그래프 → results/graph.png
```

실험 예시 (돌연변이율 비교):
```bash
python train.py --mutation-rate 0.01 --name mut01
python train.py --mutation-rate 0.1  --name mut10
python train.py --mutation-rate 0.3  --name mut30
python plot.py mut01 mut10 mut30
```
