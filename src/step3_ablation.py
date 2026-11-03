"""3단계-③: '왜 CNN인가' — 24시간 센서 파형을 뺀 버전과 비교.

같은 구조·같은 학습 조건에서 파형(합성곱) 유무만 바꾸고, seed 3개로 반복해 운의 영향을 본다.
"""
import time
import pandas as pd
from sklearn.metrics import average_precision_score
from step3_models import train_cnn, predict, daily_topn, te, PROC

rows = []
for seed in (0, 1, 2):
    for use_seq in (True, False):
        t = time.time()
        net, best = train_cnn(use_seq, seed, verbose=False)
        s = predict(net, te)
        r = {"모델": "파형+특징 (1D-CNN)" if use_seq else "특징만 (MLP)", "seed": seed,
             "PR-AUC": average_precision_score(te.y, s)}
        for n in (3, 5):
            r[f"하루{n}대 잡은 고장"] = daily_topn(te, s, n)[0]
        rows.append(r)
        print(r, f"{time.time() - t:.0f}초", flush=True)

res = pd.DataFrame(rows)
res.to_csv(PROC / "step3_ablation.csv", index=False)
summary = res.drop(columns="seed").groupby("모델").agg(["mean", "min", "max"]).round(3)
print(summary.to_string())
