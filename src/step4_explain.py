"""4단계: 경보마다 판단 근거 보여주기 (SHAP GradientExplainer on 1D-CNN).

결정 (2026-11-04, 사용자 추천안 채택)
- 근거 계산: SHAP → 특징별 기여 + 24시간 파형의 시간별 기여(시계열 attribution)
- 현장 형태: 경보마다 위험을 올린 이유 3줄 + 예시 센서 그래프
- 놓친 고장(하루 5대 기준) 분석
"""
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step3_models import Net, tensors, predict, FEATS, SENSORS, SEQ, tr, te, MODELS, ROOT

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
OUT = ROOT / "결과"
OUT.mkdir(exist_ok=True)
N = 5

ck = torch.load(MODELS / "cnn.pt", weights_only=False)
net = Net(len(FEATS))
net.load_state_dict(ck["state"])
net.eval()

# 하루 5대 경보와 놓친 고장
m = te.assign(score=predict(net, te))
m7 = m[m.datetime.dt.hour == 7].copy()
m7["rank"] = m7.groupby(m7.datetime.dt.date).score.rank(ascending=False, method="first").astype(int)
alerts, missed = m7[m7["rank"] <= N], m7[(m7.y == 1) & (m7["rank"] > N)]
rows = pd.concat([alerts, missed])


class Logit(nn.Module):  # SHAP은 (배치, 출력) 모양을 원함
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, seq, tab):
        return self.net(seq, tab).unsqueeze(1)


bg_pos = np.random.default_rng(0).choice(np.flatnonzero(tr.y.values == 0), 200, replace=False)
explainer = shap.GradientExplainer(Logit(net), list(tensors(tr, bg_pos)))
seq, tab = tensors(te, te.index.get_indexer(rows.index))
sv_seq, sv_tab = (np.asarray(v).reshape(x.shape) for v, x in zip(explainer.shap_values([seq, tab], nsamples=200, rseed=0), (seq, tab)))

# 자체 점검: SHAP 합이 모델 점수(logit)를 잘 따라가야 근거로 믿을 수 있음
with torch.no_grad():
    logit = net(seq, tab).numpy()
corr = np.corrcoef(sv_seq.sum((1, 2)) + sv_tab.sum(1), logit)[0, 1]
assert corr > 0.8, corr

# 현장용으로 묶기: 센서(파형+이동평균), 에러 종류, 부품별 교체 경과, 연식, 모델
fi = {f: i for i, f in enumerate(FEATS)}
groups = {s: (k, [fi[f] for f in FEATS if f.startswith(s + "_")]) for k, s in enumerate(SENSORS)}
groups |= {f.split("_")[0]: (None, [fi[f]]) for f in FEATS if f.startswith(("error", "comp"))}
groups |= {"age": (None, [fi["age"]]), "model": (None, [fi[f] for f in FEATS if f.startswith("model")])}
G = pd.DataFrame({g: (sv_seq[:, k].sum(1) if k is not None else 0) + sv_tab[:, idx].sum(1)
                  for g, (k, idx) in groups.items()}, index=rows.index)

normal = tr[tr.y == 0][FEATS].median()
KO = {"volt": "전압", "rotate": "회전속도", "pressure": "압력", "vibration": "진동"}


def reason(g, r, i):
    if g in SENSORS:
        peak = SEQ - 1 - int(np.argmax(sv_seq[i, SENSORS.index(g)]))
        return f"{KO[g]} 24시간 평균 {r[g + '_mean24h']:.1f} (정상 중앙값 {normal[g + '_mean24h']:.1f}), {peak}시간 전 파형 영향 최대"
    if g.startswith("error"):
        return f"최근 24시간 {g} {int(r[g + '_24h'])}회"
    if g.startswith("comp"):
        h = r[g + "_since_h"]
        return f"{g} 마지막 교체 후 {h / 24:.0f}일 경과 (정상 중앙값 {normal[g + '_since_h'] / 24:.0f}일)"
    if g == "age":
        return f"연식 {int(r.age)}년"
    return "기계 " + next(f for f in FEATS if f.startswith("model") and r[f] == 1)


out = []
for i, (ix, r) in enumerate(rows.iterrows()):
    top = G.loc[ix][G.loc[ix] > 0].nlargest(3)
    out.append({"machineID": r.machineID, "datetime": r.datetime, "rank": r["rank"], "score": round(r.score, 3),
                "실제_24h내_고장": r.y, "고장부품": r.fail_comps, "경보": r["rank"] <= N,
                **{f"이유{j + 1}": reason(g, r, i) for j, g in enumerate(top.index)}})
out = pd.DataFrame(out)
out.to_csv(OUT / "step4_alert_reasons.csv", index=False, encoding="utf-8-sig")

# 전체 경보에서 어떤 근거가 많이 쓰였나
imp = G.loc[alerts.index].abs().mean().sort_values()
ax = imp.rename(lambda g: KO.get(g, g)).plot.barh(figsize=(6, 5), color="#4C78A8")
ax.set(title=f"경보 {len(alerts)}건의 평균 근거 크기 (|SHAP|, 1D-CNN)", xlabel="평균 |SHAP| (logit)")
plt.tight_layout(); plt.savefig(OUT / "step4_global.png", dpi=150); plt.close()

# 예시 1건: 센서가 근거 3줄 안에 든 적중 경보 중 점수 최고 (없으면 적중 경보 최고)
hit = out[out.경보 & (out.실제_24h내_고장 == 1)]
has_sensor = hit[hit.filter(like="이유").apply(lambda c: c.str.contains("파형", na=False)).any(axis=1)]
ex = (has_sensor if len(has_sensor) else hit).nlargest(1, "score").index[0]
raw = seq[ex].numpy().T * ck["sd"] + ck["mu"]
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
hours = np.arange(-SEQ + 1, 1)
for k, s in enumerate(SENSORS):
    a = axs[k]
    a.bar(hours, sv_seq[ex, k], color=np.where(sv_seq[ex, k] > 0, "#E45756", "#9DB6CF"), alpha=0.6)
    a.set_ylabel(f"{KO[s]}\nSHAP")
    b = a.twinx(); b.plot(hours, raw[:, k], color="black", lw=1.2); b.set_ylabel("센서값")
r = out.loc[ex]
axs[0].set_title(f"기계 {r.machineID} · {r.datetime:%m-%d %H시} 경보 (점수 {r.score}, 다음 24h {r.고장부품} 고장)\n"
                 "막대: 시간별 기여(빨강=위험↑), 선: 실제 센서값")
axs[-1].set_xlabel("경보 시점 기준 시간 (h)")
plt.tight_layout(); plt.savefig(OUT / "step4_example.png", dpi=150); plt.close()

print(f"SHAP 합과 모델 점수 상관: {corr:.3f}")
print("근거 크기 순위:", imp.sort_values(ascending=False).round(3).to_dict())
print("\n[예시 경보]\n", out.loc[ex].to_string())
print("\n[놓친 고장]\n", out[~out.경보].to_string())
