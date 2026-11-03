"""5단계: 운영 준비 — MLflow 등록, 골든셋 게이트, drift 감지.

결정 (2026-11-04, 사용자 추천안 채택)
- 후보를 모두 MLflow에 등록하고 게이트를 통과한 것 중 더 나은 모델이 운영 모델(champion)
- 골든셋 = 11~12월 고정 시험지. 합격: 하루 5대 기준 고장 95% 이상, PR-AUC 0.90 이상, 현직 대비 1%p 넘게 나빠지지 않음
- drift: 특징별 PSI(학습 분포 vs 최근 7일) > 0.2면 재학습 알림. 전압 +10% 오차를 넣어 시연
- drift 감시 대상: 측정값(센서·에러)만. 부품 교체 경과 시간은 정비 일정에 따라 원래 바뀌는 값이라 제외 (평소 주간에 오경보 발생 확인 후 결정)
"""
import os
import shutil
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score
from step3_models import Net, train_cnn, predict, daily_topn, FEATS, tr, te, MODELS, ROOT

os.chdir(ROOT)
# shortcut: 시연용이라 실행마다 기록을 새로 시작함. 실제 운영에서는 지우지 않고 계속 쌓아야 함
for p in ("mlflow.db", "mlruns"):
    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pdm-failure-24h")
NAME = "pdm-failure-24h"
client = MlflowClient()
OUT = ROOT / "결과"
FLOOR = {"golden_recall5": 0.95, "golden_prauc": 0.90}


def golden_metrics(net):
    s = predict(net, te)
    t = daily_topn(te, s, 5)[1]
    return {"golden_prauc": average_precision_score(te.y, s),
            "golden_recall5": daily_topn(te, s, 5)[0] / t, "golden_recall3": daily_topn(te, s, 3)[0] / t}


def gate(m, champ):
    checks = {f"{k} >= {v}": m[k] >= v for k, v in FLOOR.items()}
    if champ:
        checks |= {f"{k} >= 현직 {champ[k]:.3f} - 0.01": m[k] >= champ[k] - 0.01 for k in FLOOR}
    return all(checks.values()), checks


# ---------- 후보 3개 ----------
cnn = Net(len(FEATS))
cnn.load_state_dict(torch.load(MODELS / "cnn.pt", weights_only=False)["state"])
mlp, _ = train_cnn(use_seq=False, seed=0, verbose=False)
no_onset, _ = train_cnn(use_seq=False, seed=0, verbose=False, y_train=(tr.datetime == tr.fail_time).to_numpy("int8"))
candidates = [("cnn", cnn, "1D-CNN: 24h 파형 + 30개 특징"),
              ("mlp", mlp, "MLP: 30개 특징만 (파형 뺌)"),
              ("mlp_no_onset", no_onset, "MLP, 라벨 보정 없음 (고장 시각 1시간만 정답)")]

champ, rows = None, []
for name, net, desc in candidates:
    m = golden_metrics(net)
    ok, checks = gate(m, champ and champ[1])
    promote = ok and (champ is None or m["golden_prauc"] > champ[1]["golden_prauc"])
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": name, "desc": desc, "seq_input": net.use_seq})
        mlflow.log_metrics(m | {"gate_passed": int(ok)})
        info = mlflow.pytorch.log_model(net.eval(), name="model", registered_model_name=NAME, pip_requirements=["torch"],
                                         serialization_format="pickle")  # pt2는 입력 2개 모델 미지원
    v = info.registered_model_version
    client.set_model_version_tag(NAME, v, "gate", "passed" if ok else "blocked")
    if promote:
        client.set_registered_model_alias(NAME, "champion", v)
        champ = (name, m)
    rows.append({"후보": name, "설명": desc, **{k: round(x, 3) for k, x in m.items()},
                 "게이트": "통과" if ok else "차단", "미달 항목": ", ".join(k for k, c in checks.items() if not c),
                 "결과": "운영 모델로 승격" if promote else ("통과했지만 현직보다 낫지 않음" if ok else "배포 차단")})
gate_df = pd.DataFrame(rows)
gate_df.to_csv(OUT / "step5_gate.csv", index=False, encoding="utf-8-sig")
assert client.get_model_version_by_alias(NAME, "champion").tags["gate"] == "passed"


# ---------- drift ----------
def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r, c = (np.clip(np.histogram(x, edges)[0] / len(x), 1e-4, None) for x in (ref, cur))
    return float(((c - r) * np.log(c / r)).sum())


assert psi(tr.volt_mean24h.values, tr.volt_mean24h.values) < 1e-9
recent = te[te.datetime >= te.datetime.max() - pd.Timedelta(days=7)]
drifted = recent.copy()
volt_cols = [c for c in FEATS if c.startswith("volt_")]
drifted[volt_cols] *= 1.1  # 센서 값이 1.1배가 되면 평균·표준편차도 정확히 1.1배
DRIFT_FEATS = [f for f in FEATS if not f.endswith("_since_h")]
drift = pd.DataFrame({sc: {f: psi(tr[f].values, d[f].values) for f in DRIFT_FEATS}
                      for sc, d in (("평소_최근7일", recent), ("전압+10%_오차", drifted))}).round(3)
drift.to_csv(OUT / "step5_drift.csv", encoding="utf-8-sig")
with mlflow.start_run(run_name="drift_check"):
    for sc in drift:
        alarms = drift.index[drift[sc] > 0.2].tolist()
        mlflow.log_metric(f"psi_max_{'normal' if sc.startswith('평소') else 'volt_plus10'}", drift[sc].max())
        print(f"[drift] {sc}: PSI 최대 {drift[sc].max():.3f} → {'재학습 알림: ' + str(alarms) if alarms else '이상 없음'}")
    mlflow.log_artifact(str(OUT / "step5_drift.csv"))

print(gate_df.drop(columns="설명").to_string())
print("운영 모델:", champ[0])
print(drift.sort_values("전압+10%_오차", ascending=False).head(8).to_string())
