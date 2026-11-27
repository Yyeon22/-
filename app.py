"""
똥피하기 AI - 웹 버전 (Streamlit)

    streamlit run app.py

게임 규칙(game.py), AI(brain.py), 유전 알고리즘(train.py)은 파이썬에서 그대로 돌리고,
브라우저는 그 결과를 그림으로 보여주기만 한다 (web_view.py).
"""
import csv

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from brain import Brain, observe
from game import Game
from train import next_generation, play_generation
from web_view import live_html, replay_html

BEST_BRAIN = "results/run/best_brain.npz"
RANDOM_BASELINE = 371
VIEW_HEIGHT = 760

st.set_page_config(page_title="똥피하기 AI", page_icon="💩", layout="wide")


@st.cache_resource
def best_brain():
    return Brain.load(BEST_BRAIN)


def record(brains, seed, max_frames, with_sensors=False):
    """게임 한 판을 진행하면서 매 프레임의 위치를 기록한다 (브라우저에서 재생용)."""
    game = Game(n_players=len(brains), seed=seed)
    frames = []
    while not game.all_dead() and game.frame < max_frames:
        actions = [b.act(observe(game, p)) if p.alive else 0 for b, p in zip(brains, game.players)]
        game.step(actions)
        frame = {"p": [int(v) for p in game.poops for v in (p.x, p.y)],
                 "a": [int(p.x) if p.alive else -1 for p in game.players]}
        if with_sensors and game.players[0].alive:
            frame["s"] = [round(float(v), 2) for v in observe(game, game.players[0])[1:]]
        frames.append(frame)
    return frames, [p.score for p in game.players]


@st.cache_data(show_spinner="AI가 플레이하는 중...")
def replay_best(seed, max_frames):
    return record([best_brain()], seed, max_frames, with_sensors=True)


@st.cache_data(show_spinner="1세대 AI 100명이 플레이하는 중...")
def replay_first_generation(seed):
    rng = np.random.default_rng(seed)
    return record([Brain(rng) for _ in range(100)], seed, 6000)


@st.cache_data(show_spinner="처음 보는 똥 패턴으로 시험 중...")
def test_best(n_games):
    scores = []
    for s in range(n_games):
        game = Game(1, seed=10_000 + s)
        while not game.all_dead() and game.frame < 20_000:
            game.step([best_brain().act(observe(game, game.players[0]))])
        scores.append(game.players[0].score)
    return scores


def seconds(frames):
    return f"{frames / 60:.1f}초"


# ---------------- 화면 ----------------
st.title("💩 똥피하기 AI")
st.caption("규칙을 하나도 모르는 AI가 유전 알고리즘으로 스스로 똥 피하는 법을 배운다. "
           "신경망과 유전 알고리즘은 numpy로 직접 구현했다.")

tab_ai, tab_train, tab_vs, tab_result = st.tabs(
    ["🤖 학습된 AI 보기", "🧬 학습 과정", "🎮 AI와 대결", "📊 결과"])

with tab_ai:
    left, right = st.columns([3, 2])
    with right:
        st.subheader("30세대 학습한 AI")
        st.markdown(
            "- **빨간 캐릭터**가 AI입니다.\n"
            "- 머리 위 **9칸**은 AI의 눈(센서)입니다. 그 칸에 똥이 곧 떨어지면 **빨강**, 안전하면 **초록**.\n"
            "- AI는 이 10개의 숫자(센서 9개 + 내 위치)만 보고 정지/왼쪽/오른쪽을 고릅니다.")
        seed = st.number_input("똥 패턴 번호 (바꾸면 다른 판)", 0, 99_999, 10_001)
        limit = st.slider("최대 재생 시간(초)", 30, 120, 60, step=10)
        frames, scores = replay_best(int(seed), limit * 60)
        if len(frames) >= limit * 60:
            st.success(f"{limit}초 동안 한 번도 안 맞았어요!")
        else:
            st.info(f"이번 판 기록: {seconds(scores[0])}")
    with left:
        components.html(replay_html(frames, "학습된 AI", "AI가 맞았다!" if len(frames) < limit * 60 else "시간 종료"),
                        height=VIEW_HEIGHT)

with tab_train:
    st.subheader("① 1세대: 아무것도 모르는 AI 100명")
    st.write("뇌(가중치)가 완전히 무작위라서 대부분 몇 초 만에 맞습니다. "
             "빨간 캐릭터는 그중 한 명, 노란 반투명 캐릭터들이 나머지입니다.")
    c1, c2 = st.columns([3, 2])
    with c1:
        frames, scores = replay_first_generation(0)
        components.html(replay_html(frames, "1세대 AI 100명", "전원 탈락"), height=VIEW_HEIGHT)
    with c2:
        st.metric("1세대 평균 생존 시간", seconds(np.mean(scores)))
        st.metric("1세대 1등", seconds(max(scores)))
        st.caption(f"참고: 무작위로 움직이면 평균 {seconds(RANDOM_BASELINE)}")

    st.divider()
    st.subheader("② 직접 학습시켜 보기")
    st.write("값을 바꿔 학습시키면 세대마다 점수가 어떻게 오르는지 그래프로 볼 수 있습니다. "
             "(웹에서는 빠르게 보도록 한 판 최대 50초로 제한, 부모는 상위 10%에서 고름)")
    s1, s2, s3, s4 = st.columns(4)
    n_gen = s1.slider("세대 수", 5, 40, 15)
    pop = s2.slider("한 세대 인원", 20, 100, 50, step=10)
    mut = s3.select_slider("돌연변이율", [0.01, 0.03, 0.1, 0.2, 0.3, 0.5], 0.3)
    games = s4.slider("평가할 판 수", 1, 3, 2)

    def draw_log(target, log):
        target.line_chart(pd.DataFrame(log).set_index("세대"), x_label="세대",
                          y_label="생존 시간 (60 = 1초)", color=["#dd3333", "#3366cc"])

    if st.button("🧬 학습 시작", type="primary"):
        rng = np.random.default_rng(1)
        brains = [Brain(rng) for _ in range(pop)]
        progress = st.progress(0.0, "학습 중...")
        chart = st.empty()
        log = []
        for gen in range(1, n_gen + 1):
            fitness = np.mean([play_generation(brains, 778_000 + gen * 10 + g, 3000)
                               for g in range(games)], axis=0)
            log.append({"세대": gen, "1등": fitness.max(), "평균": fitness.mean()})
            draw_log(chart, log)
            progress.progress(gen / n_gen, f"{gen}세대 완료 · 평균 {seconds(fitness.mean())}")
            if gen < n_gen:
                brains, _ = next_generation(brains, fitness, rng, max(1, pop // 25), 0.1, mut)
        progress.empty()
        chart.empty()
        st.session_state["trained"] = {"brains": brains, "log": log}

    if "trained" in st.session_state:
        t = st.session_state["trained"]
        draw_log(st, t["log"])
        first, last = t["log"][0]["평균"], t["log"][-1]["평균"]
        st.success(f"평균 생존 시간: 1세대 {seconds(first)} → {len(t['log'])}세대 {seconds(last)}")
        if st.button("▶ 마지막 세대 전원이 플레이하는 모습 보기"):
            frames, _ = record(t["brains"], 424_242, 3600)
            components.html(replay_html(frames, f"{len(t['log'])}세대 AI {len(t['brains'])}명", "전원 탈락"),
                            height=VIEW_HEIGHT)

with tab_vs:
    c1, c2 = st.columns([3, 2])
    with c1:
        components.html(live_html(best_brain()), height=VIEW_HEIGHT + 60)
    with c2:
        st.subheader("AI(빨강) vs 나(노랑)")
        st.markdown(
            "1. 게임 화면을 **한 번 클릭**하면 시작합니다.\n"
            "2. 키보드 **← →** 또는 아래 **◀ ▶ 버튼**으로 움직입니다.\n"
            "3. 둘 다 같은 똥을 피하고, 더 오래 버틴 쪽이 이깁니다.\n"
            "4. **R** 또는 클릭으로 다시 시작합니다.")
        st.caption("이 탭은 실시간 조작을 위해 브라우저 안에서 돌아갑니다. "
                   "game.py의 규칙과 학습된 AI의 신경망 가중치를 JavaScript로 옮겨 그대로 계산합니다.")

with tab_result:
    with open("results/run/log.csv") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame({"세대": [int(r["generation"]) for r in rows],
                       "1등": [float(r["best"]) for r in rows],
                       "평균": [float(r["average"]) for r in rows]})
    df["무작위로 움직이기"] = RANDOM_BASELINE
    st.subheader("30세대 학습 기록 (AI 100명, 3판 평균)")
    st.line_chart(df.set_index("세대"), x_label="세대", y_label="생존 시간 (60 = 1초)",
                  color=["#dd3333", "#3366cc", "#999999"])

    st.subheader("처음 보는 똥 패턴으로 시험")
    if st.button("최고 AI 10판 시험하기"):
        scores = test_best(10)
        m1, m2, m3 = st.columns(3)
        m1.metric("평균", seconds(np.mean(scores)), f"무작위의 {np.mean(scores) / RANDOM_BASELINE:.0f}배")
        m2.metric("최고", seconds(max(scores)))
        m3.metric("최저", seconds(min(scores)))
        st.bar_chart(pd.DataFrame({"판": [f"{i + 1}판" for i in range(10)], "생존 시간(초)": [s / 60 for s in scores]})
                     .set_index("판"))
