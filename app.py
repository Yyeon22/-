"""
똥피하기 - 웹 버전 (Streamlit)

    streamlit run app.py

🎮 게임: 혼자 하기 / 학습된 AI와 대결
🧬 AI 성장 레이스: 1세대 ~ 마지막 세대 AI가 같은 똥을 동시에 피한다
📈 학습 그래프: 세대별 실력, 센서 실험 비교, 이론적 한계와 비교
"""
import glob
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from brain import Brain
from web_view import live_html, race_html

RUN_DIR = "results/run"
RANDOM_BASELINE = 253     # 무작위로 움직일 때 평균 생존 프레임 (plot.py와 같은 값)

st.set_page_config(page_title="똥피하기", page_icon="💩", layout="wide")


@st.cache_resource
def load_brain(path):
    return Brain.load(path)


def snapshots():
    """results/run/gen_XXXX.npz → [(세대 번호, 경로)] 세대 순"""
    found = []
    for path in glob.glob(os.path.join(RUN_DIR, "gen_*.npz")):
        m = re.search(r"gen_(\d+)\.npz$", path)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def lines(df, colors=None, x_title="세대", y_title="초"):
    """확대/이동이 안 되는 고정 선 그래프 (발표 중 스크롤하다 그래프가 움직이지 않게)"""
    data = df.reset_index(names="x").melt("x", var_name="항목", value_name="y")
    color = alt.Color("항목:N", sort=list(df.columns), legend=alt.Legend(orient="bottom", title=None, labelLimit=0, columns=2),
                      scale=alt.Scale(domain=list(df.columns), range=colors) if colors else alt.Undefined)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X("x:Q", title=x_title), y=alt.Y("y:Q", title=y_title), color=color,
        tooltip=[alt.Tooltip("항목:N"), alt.Tooltip("x:Q", title=x_title), alt.Tooltip("y:Q", title=y_title, format=".1f")])
    st.altair_chart(chart, use_container_width=True)


def read_log(folder):
    df = pd.read_csv(os.path.join(folder, "log.csv")).set_index("generation")
    return df / 60                                  # 프레임 → 초


st.title("💩 똥피하기 AI")
tab_game, tab_race, tab_graph = st.tabs(["🎮 게임", "🧬 AI 성장 레이스", "📈 학습 그래프"])

# ---------------- 게임 ----------------
with tab_game:
    game, guide = st.columns([3, 2])
    with game:
        st.iframe(live_html(load_brain(os.path.join(RUN_DIR, "best_brain.npz"))), height=840)
    with guide:
        st.subheader("하는 법")
        st.markdown(
            "1. 위에서 **AI와 대결** 또는 **혼자 하기**를 고릅니다.\n"
            "2. 게임 화면을 **한 번 클릭**하면 시작합니다.\n"
            "3. 키보드 **← →**, 아래 **◀ ▶ 버튼**, 또는 화면을 **누른 채 끌어서** 움직입니다 (모바일 OK).\n"
            "4. **R** 또는 클릭으로 다시 시작합니다. 🔊 버튼으로 소리를 끌 수 있습니다.\n"
            "5. **10초마다 스테이지**가 올라가며 똥이 **커지고, 빨라지고, 더 자주** 떨어집니다.")
        st.markdown(
            "**AI와 대결**: 빨간 캐릭터가 학습된 AI, 노란 캐릭터가 나입니다. "
            "같은 똥을 피해서 더 오래 버틴 쪽이 이깁니다. "
            "AI 머리 위 칸들은 AI의 눈(센서)으로, 똥이 곧 떨어질 칸이 빨갛게 표시됩니다.")
        st.markdown(
            "**혼자 하기**: 8초마다 아이템이 떨어집니다. "
            "🛡️ = 3초 무적, ⏳ = 3초 동안 똥이 느려짐. 최고 기록이 브라우저에 저장됩니다. "
            "(AI 대결에는 공정하게 아이템이 없습니다.)")

# ---------------- AI 성장 레이스 ----------------
with tab_race:
    snaps = snapshots()
    if not snaps:
        st.info("세대별 AI가 없습니다. `python train.py`로 학습하면 results/run/gen_XXXX.npz가 생깁니다.")
    else:
        left, right = st.columns([3, 2])
        with right:
            st.subheader("세대마다 얼마나 똑똑해졌을까?")
            st.markdown(
                "유전 알고리즘 학습 중간중간의 '그 세대 최고 AI'를 저장해 두었다가, "
                "**똑같은 똥**을 동시에 피하게 했습니다. 오래 버틴 순으로 오른쪽 위에 순위가 나옵니다.\n\n"
                "후반 세대는 오래 버티니 위의 **x4, x8 배속** 버튼을 쓰세요. "
                "똥은 매번 무작위라 판마다 결과가 조금씩 다릅니다.")
            gens = [g for g, _ in snaps]
            picked = st.multiselect("레이스에 내보낼 세대", gens, default=gens,
                                    format_func=lambda g: f"{g}세대")
            st.caption("AI의 눈과 뇌: " + load_brain(snaps[-1][1]).describe())
        with left:
            entries = [(f"{g}세대", load_brain(p)) for g, p in snaps if g in picked]
            if entries:
                st.iframe(race_html(entries), height=740)
            else:
                st.warning("세대를 하나 이상 골라 주세요.")

# ---------------- 학습 그래프 ----------------
with tab_graph:
    if os.path.exists(os.path.join(RUN_DIR, "log.csv")):
        st.subheader("세대별 실력 (생존 시간, 초)")
        df = read_log(RUN_DIR).rename(columns={"best": "그 세대 1등", "average": "평균",
                                               "median": "중앙값", "validation": "처음 보는 판 시험"})
        df["무작위로 움직이기"] = RANDOM_BASELINE / 60
        lines(df[["그 세대 1등", "처음 보는 판 시험", "평균", "무작위로 움직이기"]],
              ["#e53935", "#43a047", "#1e88e5", "#9e9e9e"])
        st.caption("'처음 보는 판 시험' = 그 세대 상위 AI를 학습 때 안 본 똥 패턴으로 다시 시험한 점수. "
                   "운 좋게 쉬운 판을 만난 AI를 걸러낸다.")

    exp_path = "results/experiments.csv"
    if os.path.exists(exp_path):
        st.subheader("실험: AI의 눈과 뇌를 바꾸면 더 잘할까?")
        ex = pd.read_csv(exp_path)
        chart = alt.Chart(ex).mark_bar(color="#1e88e5").encode(
            x=alt.X("mean_sec:Q", title="처음 보는 판 평균 생존 시간 (초)"),
            y=alt.Y("description:N", sort=None, title=None, axis=alt.Axis(labelLimit=0)),
            tooltip=[alt.Tooltip("description:N", title="설정"), alt.Tooltip("mean_sec:Q", title="평균(초)"),
                     alt.Tooltip("median_sec:Q", title="중앙값(초)")])
        text = chart.mark_text(align="left", dx=4, color="#888").encode(text=alt.Text("mean_sec:Q", format=".1f"))
        st.altair_chart(chart + text, use_container_width=True)
        st.markdown(
            "모두 같은 조건(같은 seed, 100세대, 150마리)으로 학습하고 처음 보는 같은 30판으로 시험했다 "
            "(`python watch.py --compare ...`).\n\n"
            "- **센서를 넓히면 오히려 나빠졌다.** 입력이 늘면 유전 알고리즘이 맞춰야 할 가중치가 많아져 학습이 느려진다.\n"
            "- **AI가 죽는 장면을 분석하니 '좌우로 떨기'였다.** 똥 경계가 센서 위에 걸리면 7px 움직일 때마다 "
            "센서 값이 0↔1로 깜빡이고, AI는 왼쪽·오른쪽을 번갈아 누르며 제자리에서 떨다 맞았다. "
            "죽기 0.5초 전에는 30판 모두 피할 길이 있었다.\n"
            "- 그래서 **부드러운 센서**(경계에서 위험도가 서서히 변함)와 **직전 행동 기억**을 넣었다. "
            "둘 다 넣은 것이 가장 좋았지만 차이는 작았다 → 마지막 학습(300세대)은 이 설정으로 했다.")

    oracle_path = "results/oracle.csv"
    if os.path.exists(oracle_path):
        st.subheader("AI는 얼마나 잘하는 걸까? 세 선수 비교")
        oc = pd.read_csv(oracle_path)
        who = {"god": "신 (미래를 다 앎)", "planner": "계획자 (보이는 것만)", "ai": "학습된 AI"}
        who = {k: v for k, v in who.items() if k in oc and oc[k].notna().all()}
        cols = st.columns(len(who))
        for col, (k, label) in zip(cols, who.items()):
            col.metric(f"{label} 평균", f"{oc[k].mean() / 60:.1f}초")
        bars = pd.DataFrame([{"판": i + 1, "누가": label, "초": oc[k][i] / 60}
                             for k, label in who.items() for i in range(len(oc))])
        palette = {"신 (미래를 다 앎)": "#bdbdbd", "계획자 (보이는 것만)": "#1e88e5", "학습된 AI": "#e53935"}
        chart = alt.Chart(bars).mark_bar().encode(
            x=alt.X("판:O", title="판"), xOffset=alt.XOffset("누가:N", sort=list(who.values())),
            y=alt.Y("초:Q", title="버틴 시간 (초)"),
            color=alt.Color("누가:N", sort=list(who.values()),
                            scale=alt.Scale(domain=list(who.values()), range=[palette[v] for v in who.values()]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["판", "누가", alt.Tooltip("초:Q", format=".1f")])
        st.altair_chart(chart, use_container_width=True)
        st.markdown(
            "- **신**: 똥은 플레이어가 어떻게 움직이든 똑같이 떨어지므로, 매 프레임 '서 있을 수 있는 모든 위치'를 "
            "따라가면 아무리 잘해도 죽는 순간을 정확히 계산할 수 있다. 아직 안 나온 똥까지 아는 '천장'이다 "
            "(333초에서 계산을 멈춤).\n"
            "- **계획자**: AI와 똑같이 화면에 보이는 똥만 알고, 매 프레임 1.5초 앞을 계산해서 움직인다.\n"
            "- **결론**: 계획자가 AI보다 훨씬 오래 버틴다 → AI가 못 보는 게 아니라, "
            "작은 신경망 + 유전 알고리즘이 '보이는 정보를 잘 쓰는 법'을 다 배우지 못한 것이 한계다 (`oracle.py`).")
