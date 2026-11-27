"""
똥피하기 - 웹 버전 (Streamlit)

    streamlit run app.py

혼자 하기 / 학습된 AI와 대결하기.
"""
import streamlit as st

from brain import Brain
from web_view import live_html

st.set_page_config(page_title="똥피하기", page_icon="💩", layout="wide")


@st.cache_resource
def best_brain():
    return Brain.load("results/run/best_brain.npz")


st.title("💩 똥피하기")

game, guide = st.columns([3, 2])
with game:
    st.iframe(live_html(best_brain()), height=820)
with guide:
    st.subheader("하는 법")
    st.markdown(
        "1. 위에서 **혼자 하기** 또는 **AI와 대결**을 고릅니다.\n"
        "2. 게임 화면을 **한 번 클릭**하면 시작합니다.\n"
        "3. 키보드 **← →** 또는 아래 **◀ ▶ 버튼**으로 움직입니다.\n"
        "4. **R** 또는 클릭으로 다시 시작합니다.\n"
        "5. **10초마다 스테이지**가 올라가며 똥이 **커지고, 빨라지고, 더 자주** 떨어집니다.")
    st.markdown(
        "**AI와 대결**에서는 빨간 캐릭터가 학습된 AI, 노란 캐릭터가 나입니다. "
        "같은 똥을 피해서 더 오래 버틴 쪽이 이깁니다. "
        "AI 머리 위 9칸은 AI의 눈(센서)으로, 똥이 곧 떨어질 칸이 빨갛게 표시됩니다.")
