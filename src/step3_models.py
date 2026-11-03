"""3단계-②: IsolationForest vs PyTorch 1D-CNN, 같은 경보 수로 비교.

비교 방식: 매일 07시, 100대 중 위험 점수 상위 N대를 점검 목록으로 뽑는다.
적중 = 뽑은 기계가 24시간 안에 고장 (y=1). 두 모델 모두 하루 N대라 경보 수가 같다.
- IsolationForest: 학습 기간의 정상 시간(y=0)만 보고 '평소와 다름'을 점수로 (정답 안 씀)
- 1D-CNN: 최근 24시간 센서 파형 + 30개 특징, 2단계 정답으로 학습
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "데이터/processed", ROOT / "models"
MODELS.mkdir(exist_ok=True)
SENSORS = ["volt", "rotate", "pressure", "vibration"]
SEQ = 24

df = pd.read_parquet(PROC / "features.parquet")
FEATS = [c for c in df if c not in ["datetime", "machineID", "y", "fail_time", "fail_comps", "fail_no_maint", "split", *SENSORS]]
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))


def daily_topn(d, score, n):
    """매일 07시 상위 n대를 뽑았을 때 (잡은 고장 수, 전체 고장 수, 정밀도)."""
    m = d.assign(score=score)[d.datetime.dt.hour == 7]
    pick = m.groupby(m.datetime.dt.date, group_keys=False).apply(lambda x: x.nlargest(n, "score"))
    return int(pick.y.sum()), int(m.y.sum()), pick.y.mean()


# 자체 점검: 정답을 그대로 점수로 쓰면 하루 고장 수 이상 뽑을 때 다 잡아야 함
_c, _t, _ = daily_topn(te, te.y.values, 10)
assert _c == _t > 0

# ---------- 1D-CNN ----------
# 센서 원본을 (기계, 시간, 센서) 배열로. 정규화는 학습 기간 값만 사용
lab = pd.read_parquet(PROC / "labels.parquet", columns=["machineID", "datetime", *SENSORS])
t0 = lab.datetime.min()
arr = lab[SENSORS].to_numpy("float32").reshape(lab.machineID.nunique(), -1, len(SENSORS))
train_hours = int((pd.Timestamp("2015-09-01 06:00") - t0) / pd.Timedelta(hours=1))
mu, sd = arr[:, :train_hours].mean((0, 1)), arr[:, :train_hours].std((0, 1))
arr = (arr - mu) / sd
tab_mu, tab_sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)


def tensors(d, idx):
    rows = d.iloc[idx]
    h = ((rows.datetime - t0) / pd.Timedelta(hours=1)).to_numpy(int)
    seq = arr[rows.machineID.to_numpy() - 1][np.arange(len(rows))[:, None], h[:, None] + np.arange(-SEQ + 1, 1)]
    tab = ((rows[FEATS] - tab_mu) / tab_sd).to_numpy("float32")
    return torch.from_numpy(seq).transpose(1, 2), torch.from_numpy(tab)


class Net(nn.Module):
    """use_seq=False면 파형(합성곱) 없이 30개 특징만 보는 비교용 모델."""
    def __init__(self, n_tab, use_seq=True):
        super().__init__()
        self.use_seq = use_seq
        self.conv = nn.Sequential(nn.Conv1d(len(SENSORS), 32, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
                                  nn.AdaptiveMaxPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32 * use_seq + n_tab, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, seq, tab):
        x = torch.cat([self.conv(seq), tab], 1) if self.use_seq else tab
        return self.head(x).squeeze(1)


@torch.no_grad()
def predict(net, d, bs=8192):
    net.eval()
    return np.concatenate([torch.sigmoid(net(*tensors(d, np.arange(i, min(i + bs, len(d)))))).numpy()
                           for i in range(0, len(d), bs)])


def train_cnn(use_seq=True, seed=0, epochs=10, verbose=True, y_train=None):
    """y_train: 학습 정답을 바꿔 넣을 때 (5단계 게이트 시연용). 조율은 항상 2단계 정답으로."""
    y = tr.y.values if y_train is None else y_train
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(len(FEATS), use_seq)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    best = (-1, None, 0)
    for epoch in range(1, epochs + 1):
        # 매 epoch 고장 구간 전부 + 정상 5배를 무작위로 (드문 사건 불균형 완화)
        idx = rng.permutation(np.concatenate([pos, rng.choice(neg, 5 * len(pos), replace=False)]))
        net.train()
        for i in range(0, len(idx), 512):
            seq, tab = tensors(tr, idx[i:i + 512])
            loss = loss_fn(net(seq, tab), torch.from_numpy(y[idx[i:i + 512]]).float())
            opt.zero_grad(); loss.backward(); opt.step()
        ap = average_precision_score(va.y, predict(net, va))
        if verbose:
            print(f"epoch {epoch}: loss {loss.item():.4f}, val PR-AUC {ap:.3f}")
        if ap > best[0]:
            best = (ap, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
    net.load_state_dict(best[1])
    return net, best


if __name__ == "__main__":
    net, best = train_cnn()
    torch.save({"state": best[1], "feats": FEATS, "mu": mu, "sd": sd, "tab_mu": tab_mu, "tab_sd": tab_sd}, MODELS / "cnn.pt")
    print(f"best epoch {best[2]} (val PR-AUC {best[0]:.3f})")
    cnn_score = {"val": predict(net, va), "test": predict(net, te)}

    # ---------- IsolationForest ----------
    iso = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(tr.loc[tr.y == 0, FEATS])
    joblib.dump(iso, MODELS / "iforest.joblib")
    iso_score = {k: -iso.score_samples(d[FEATS]) for k, d in (("val", va), ("test", te))}

    # ---------- 비교 (시험 기간 11~12월) ----------
    rows = []
    for name, sc in (("IsolationForest", iso_score), ("1D-CNN", cnn_score)):
        r = {"모델": name, "PR-AUC(시간별)": round(average_precision_score(te.y, sc["test"]), 3)}
        for n in (3, 5, 10):
            c, t, p = daily_topn(te, sc["test"], n)
            r[f"하루{n}대: 잡은 고장"] = f"{c}/{t} ({c / t:.0%})"
            r[f"하루{n}대: 적중률"] = f"{p:.0%}"
        rows.append(r)
    res = pd.DataFrame(rows).set_index("모델").T
    print(res.to_string())
    res.to_csv(PROC / "step3_compare.csv")
    pd.DataFrame({"machineID": te.machineID, "datetime": te.datetime, "y": te.y,
                  "iforest": iso_score["test"], "cnn": cnn_score["test"]}).to_parquet(PROC / "scores_test.parquet", index=False)
