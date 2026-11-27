"""
학습 기록(log.csv)을 그래프로 그린다. 보고서/발표용.

    python plot.py run                         # results/run 하나 그리기
    python plot.py mut01 mut10 mut30           # 여러 실험 비교 (평균 점수끼리)
그림은 results/graph.png 로 저장된다.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

RANDOM_BASELINE = 253     # 무작위로 움직일 때 평균 생존 프레임 (스테이지 규칙에서 1000명 직접 측정한 값)


def read_log(name):
    with open(os.path.join("results", name, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    gens = [int(r["generation"]) for r in rows]
    val = [float(r["validation"]) for r in rows] if rows and "validation" in rows[0] else None
    return gens, [float(r["best"]) for r in rows], [float(r["average"]) for r in rows], val


def main():
    names = sys.argv[1:] or ["run"]
    plt.figure(figsize=(8, 5))
    if len(names) == 1:
        gens, best, avg, val = read_log(names[0])
        plt.plot(gens, best, label="1st place", color="#d33")
        plt.plot(gens, avg, label="Average", color="#36c")
        if val:
            plt.plot(gens, val, label="Top AI on unseen games", color="#2a2", alpha=0.8)
    else:
        for name in names:
            gens, _, avg, _ = read_log(name)
            plt.plot(gens, avg, label=name)
    plt.axhline(RANDOM_BASELINE, ls="--", color="gray", label="Random moves")
    plt.xlabel("Generation")
    plt.ylabel("Survival time (frames, 60 = 1 sec)")
    plt.title("Poop Dodge AI - learning progress")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join("results", "graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
