"""6단계: 대시보드용 데이터 미리 계산 (Streamlit Cloud에서는 모델 없이 이 파일만 읽음).

- 모델: MLflow champion (5단계 게이트 통과 운영 모델)
- 시험 기간(11~12월) 매일 07시 기준 위험 점수, 순위, 상태, SHAP 근거 3줄
- 점검 추천 정책: 상위 5대 + 점수 0.9 이상 전부 (4단계 놓친 고장 분석에서 나온 제안)
"""
import json
import os
import numpy as np
import pandas as pd
import shap
import torch
import mlflow
from torch import nn
from step3_models import tensors, predict, FEATS, SENSORS, tr, te, ROOT

os.chdir(ROOT)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
NAME = "pdm-failure-24h"
OUT = ROOT / "app/data"
OUT.mkdir(parents=True, exist_ok=True)
WARN, ALARM, TOP_N = 0.5, 0.9, 5

mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, "champion")
net = mlflow.pytorch.load_model(f"models:/{NAME}@champion").eval()
run = mlflow.get_run(mv.run_id)

# ---------- 매일 07시 점수·순위·상태 ----------
d = te[te.datetime.dt.hour == 7].copy()
d["score"] = predict(net, d)
d["date"] = d.datetime.dt.normalize()
d["rank"] = d.groupby("date").score.rank(ascending=False, method="first").astype(int)
d["status"] = np.select([d.score >= ALARM, d.score >= WARN], ["위험", "주의"], "정상")
d["recommend"] = (d["rank"] <= TOP_N) | (d.score >= ALARM)


class Logit(nn.Module):  # 운영 모델은 파형을 안 쓰므로 특징 입력만 설명
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, tab):
        return self.net(None, tab).unsqueeze(1)


bg = np.random.default_rng(0).choice(np.flatnonzero(tr.y.values == 0), 200, replace=False)
assert not net.use_seq
explainer = shap.GradientExplainer(Logit(net), tensors(tr, bg)[1])
tab = tensors(d, np.arange(len(d)))[1]
sv = np.asarray(explainer.shap_values(tab, nsamples=200, rseed=0)).reshape(tab.shape)

fi = {f: i for i, f in enumerate(FEATS)}
groups = {s: [fi[f] for f in FEATS if f.startswith(s + "_")] for s in SENSORS}
groups |= {f.split("_")[0]: [fi[f]] for f in FEATS if f.startswith(("error", "comp"))}
groups |= {"age": [fi["age"]], "model": [fi[f] for f in FEATS if f.startswith("model")]}
G = pd.DataFrame({g: sv[:, idx].sum(1) for g, idx in groups.items()}, index=d.index)

normal = tr[tr.y == 0][FEATS].median()
KO = {"volt": "전압", "rotate": "회전속도", "pressure": "압력", "vibration": "진동"}


def reason(g, r):
    if g in SENSORS:
        v, n = r[g + "_mean24h"], normal[g + "_mean24h"]
        return f"{KO[g]} 24h 평균 {v:.1f} · 정상 {n:.1f} 대비 {(v / n - 1):+.0%}"
    if g.startswith("error"):
        return f"최근 24h {g} {int(r[g + '_24h'])}회 발생"
    if g.startswith("comp"):
        return f"{g} 교체 후 {r[g + '_since_h'] / 24:.0f}일 경과 · 정상 {normal[g + '_since_h'] / 24:.0f}일"
    if g == "age":
        return f"연식 {int(r.age)}년"
    return "기계 모델 특성"


reasons, contribs = [], []
for ix, r in d.iterrows():
    top = G.loc[ix][G.loc[ix] > 0].nlargest(3)
    reasons.append([reason(g, r) for g in top.index] + [None] * (3 - len(top)))
    contribs.append(list(top.round(3).values) + [np.nan] * (3 - len(top)))
d[["reason1", "reason2", "reason3"]] = reasons
d[["contrib1", "contrib2", "contrib3"]] = contribs
d["model"] = d[["model1", "model2", "model3", "model4"]].idxmax(axis=1)
for c in ("comp1", "comp2", "comp3", "comp4"):
    d[f"{c}_since_d"] = (d[f"{c}_since_h"] / 24).round(1)
keep = ["date", "machineID", "model", "age", "score", "rank", "status", "recommend",
        "reason1", "reason2", "reason3", "contrib1", "contrib2", "contrib3",
        *[f"{c}_since_d" for c in ("comp1", "comp2", "comp3", "comp4")],
        *[f"error{i}_24h" for i in range(1, 6)], "y", "fail_comps"]
d[keep].rename(columns={"y": "actual_fail_24h"}).to_parquet(OUT / "daily.parquet", index=False)

# ---------- 센서(시간별)와 정상 범위 ----------
te[["machineID", "datetime", *SENSORS]].astype({s: "float32" for s in SENSORS}).to_parquet(OUT / "sensors.parquet", index=False)
q = tr.loc[tr.y == 0, SENSORS].quantile([0.05, 0.5, 0.95])

# ---------- 정비·에러·고장 이력 (시험 기간과 그 전 60일) ----------
RAW = ROOT / "데이터/raw"
since = te.datetime.min() - pd.Timedelta(days=60)
maint = pd.read_csv(ROOT / "데이터/processed/maint_typed.csv", parse_dates=["datetime"])
errors = pd.read_csv(RAW / "PdM_errors.csv", parse_dates=["datetime"])
fails = pd.read_csv(RAW / "PdM_failures.csv", parse_dates=["datetime"])
ev = pd.concat([
    maint.assign(event=maint.type.map({"preventive": "예방정비", "corrective": "고장 교체"}), detail=maint.comp),
    errors.assign(event="에러", detail=errors.errorID),
    fails.assign(event="고장", detail=fails.failure)])
ev[ev.datetime >= since][["datetime", "machineID", "event", "detail"]].sort_values("datetime").to_parquet(OUT / "events.parquet", index=False)

drift = pd.read_csv(ROOT / "결과/step5_drift.csv", index_col=0)
meta = {"model_name": NAME, "model_version": mv.version, "model_desc": run.data.params.get("desc"),
        "golden_prauc": round(run.data.metrics["golden_prauc"], 3),
        "golden_recall5": round(run.data.metrics["golden_recall5"], 3),
        "drift_psi_max": float(drift["평소_최근7일"].max()), "drift_threshold": 0.2,
        "warn": WARN, "alarm": ALARM, "top_n": TOP_N,
        "band": {s: {"p05": float(q.loc[0.05, s]), "p50": float(q.loc[0.5, s]), "p95": float(q.loc[0.95, s])} for s in SENSORS},
        "normal_since_d": {c: round(float(normal[f"{c}_since_h"]) / 24, 1) for c in ("comp1", "comp2", "comp3", "comp4")}}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

assert d.groupby("date").recommend.sum().min() >= TOP_N
print(f"모델 v{mv.version} ({meta['model_desc']}), {d.date.nunique()}일 × {d.machineID.nunique()}대")
print("하루 점검 추천 대수:", d.groupby("date").recommend.sum().describe()[["mean", "min", "max"]].round(1).to_dict())
print("추천 정책 포착:", int(d[d.recommend].y.sum()), "/", int(d.y.sum()))
print({p.name: f"{p.stat().st_size / 1e6:.2f}MB" for p in OUT.iterdir()})
