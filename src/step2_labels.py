"""2단계: 정비 기록으로 라벨 바로잡기.

결정 (2026-11-04, 사용자)
1. 정비 기록 없는 고장 18건은 고장으로 두고 fail_no_maint=True로 표시
2. 고장 시각 F 기준 (F-24h, F] 구간을 고장 진행 구간(y=1)으로 표시
3. 같은 시각 여러 부품 고장은 y 하나로 합치고, 부품은 fail_comps에 남김
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "데이터/raw", ROOT / "데이터/processed"
WINDOW = pd.Timedelta(hours=24)

tele = pd.read_csv(RAW / "PdM_telemetry.csv", parse_dates=["datetime"])
maint = pd.read_csv(RAW / "PdM_maint.csv", parse_dates=["datetime"])
fail = pd.read_csv(RAW / "PdM_failures.csv", parse_dates=["datetime"]).rename(columns={"failure": "comp"})

# 정비 분류: 같은 시각·기계·부품 고장이 있으면 고장 교체, 없으면 예방정비
key = ["datetime", "machineID", "comp"]
fail_keys = set(map(tuple, fail[key].values))
maint["type"] = ["corrective" if k in fail_keys else "preventive" for k in map(tuple, maint[key].values)]
maint_keys = set(map(tuple, maint[key].values))
fail["no_maint"] = [k not in maint_keys for k in map(tuple, fail[key].values)]

# 고장 이벤트: (기계, 시각) 단위로 합침
events = (fail.sort_values("comp").groupby(["machineID", "datetime"], as_index=False)
          .agg(fail_comps=("comp", "+".join), fail_no_maint=("no_maint", "any"))
          .rename(columns={"datetime": "fail_time"}))

# 각 시각마다 '다음 고장'을 붙이고, 24시간 안이면 y=1
lab = pd.merge_asof(tele.sort_values("datetime"), events.sort_values("fail_time"),
                    left_on="datetime", right_on="fail_time", by="machineID", direction="forward")
lab["y"] = ((lab.fail_time - lab.datetime) < WINDOW).astype("int8")
lab.loc[lab.y == 0, ["fail_time", "fail_comps", "fail_no_maint"]] = None
lab = lab.sort_values(["machineID", "datetime"]).reset_index(drop=True)

# 자체 점검
assert len(lab) == len(tele)
assert (maint.type == "corrective").sum() == (~fail.no_maint).sum()
hit = lab.merge(events, left_on=["machineID", "datetime"], right_on=["machineID", "fail_time"])
assert len(hit) == len(events) and hit.y.eq(1).all(), "모든 고장 시각은 y=1이어야 함"

OUT.mkdir(parents=True, exist_ok=True)
lab.to_parquet(OUT / "labels.parquet", index=False)
maint.to_csv(OUT / "maint_typed.csv", index=False)

print(f"정비 분류: {maint.type.value_counts().to_dict()}")
print(f"고장 이벤트: {len(events)}건 (부품 기준 {len(fail)}건), 기록 누락 표시 {events.fail_no_maint.sum()}건")
print(f"라벨: 전체 {len(lab):,}시간 중 y=1 {lab.y.sum():,}시간 ({lab.y.mean():.2%})")
