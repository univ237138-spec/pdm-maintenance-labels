"""3단계-①: 모델 입력 특징 만들기.

결정 (2026-11-04, 사용자 추천안 채택)
- 특징: 센서 4종의 최근 3h·24h 평균/표준편차, 최근 24h 에러 횟수, 부품별 마지막 교체 후 경과 시간, 기계 모델·연식
- 기간: 1~8월 학습 / 9~10월 조율 / 11~12월 시험 (과거로 배우고 미래로 시험)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "데이터/raw", ROOT / "데이터/processed"
SENSORS = ["volt", "rotate", "pressure", "vibration"]
H = pd.Timedelta(hours=1)

lab = pd.read_parquet(OUT / "labels.parquet")  # 시간순·기계순 정렬, 기계마다 8,761시간 연속
errors = pd.read_csv(RAW / "PdM_errors.csv", parse_dates=["datetime"])
maint = pd.read_csv(RAW / "PdM_maint.csv", parse_dates=["datetime"])
machines = pd.read_csv(RAW / "PdM_machines.csv")

g = lab.groupby("machineID")
feats = {}
for w in (3, 24):
    for s in SENSORS:
        r = g[s].rolling(w, min_periods=w)
        feats[f"{s}_mean{w}h"] = r.mean().values
        feats[f"{s}_std{w}h"] = r.std().values
df = pd.concat([lab, pd.DataFrame(feats, index=lab.index)], axis=1)

# 최근 24h 에러 횟수 (해당 시각 포함)
ec = errors.pivot_table(index=["machineID", "datetime"], columns="errorID", aggfunc="size", fill_value=0)
df = df.merge(ec, on=["machineID", "datetime"], how="left")
err_cols = list(ec.columns)
df[err_cols] = df[err_cols].fillna(0)
df[[f"{c}_24h" for c in err_cols]] = df.groupby("machineID")[err_cols].rolling(24, min_periods=1).sum().values
df = df.drop(columns=err_cols)

# 부품별 마지막 교체 후 경과 시간. 같은 시각 교체는 제외(고장 시각의 교체 기록이 정답을 미리 알려주지 않게)
for comp, m in maint.groupby("comp"):
    j = pd.merge_asof(df[["datetime", "machineID"]].reset_index().sort_values("datetime"), m[["datetime", "machineID"]].rename(columns={"datetime": "last"}).sort_values("last"),
                      left_on="datetime", right_on="last", by="machineID", allow_exact_matches=False).set_index("index")
    # shortcut: 교체 기록이 없으면 정비 기록 시작(2014-06-01)부터 계산, 실제 설비 이력이 있으면 그걸로 대체
    df[f"{comp}_since_h"] = (j.datetime - j["last"].fillna(pd.Timestamp("2014-06-01 06:00"))) / H

df = df.merge(machines, on="machineID", how="left")
df = pd.concat([df.drop(columns="model"), pd.get_dummies(df.model, dtype="int8")], axis=1)

# 기간 나누기. 정답이 24h 뒤 고장을 보므로 경계 직전 24h는 버려서 미래 정보가 섞이지 않게 함
t = df.datetime
df["split"] = "test"
df.loc[t < "2015-11-01 06:00", "split"] = "val"
df.loc[t < "2015-09-01 06:00", "split"] = "train"
gap = ((t >= pd.Timestamp("2015-09-01 06:00") - 24 * H) & (t < "2015-09-01 06:00")) | \
      ((t >= pd.Timestamp("2015-11-01 06:00") - 24 * H) & (t < "2015-11-01 06:00"))
df = df[~gap & df.volt_mean24h.notna()].reset_index(drop=True)

assert df.drop(columns=["fail_time", "fail_comps", "fail_no_maint"]).notna().all().all()
assert (df[[c for c in df if c.endswith("_since_h")]] > 0).all().all()
df.to_parquet(OUT / "features.parquet", index=False)

feat_cols = [c for c in df if c not in ["datetime", "machineID", "y", "fail_time", "fail_comps", "fail_no_maint", "split", *SENSORS]]
print(f"특징 {len(feat_cols)}개: {feat_cols}")
print(df.groupby("split").agg(행=("y", "size"), 고장구간=("y", "sum"), 비율=("y", "mean")).round(4))
