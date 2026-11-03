"""정비팀장 아침 점검 브리핑 (Streamlit).

디자인 기준: ISA-101 고성능 HMI(회색 바탕, 색은 이상 상태에만), IBM Maximo Health/Predict의 작업 큐·기여 요인,
Senseye·Monitron의 위험 순 목록과 3단계 상태. 데이터는 src/step6_dashboard_data.py가 미리 계산한 파일만 읽는다.
"""
import json
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="아침 점검 브리핑", page_icon="🛠️", layout="wide")
DATA = Path(__file__).parent / "data"
SENSORS = {"volt": "전압", "rotate": "회전속도", "pressure": "압력", "vibration": "진동"}
COMPS = ["comp1", "comp2", "comp3", "comp4"]
C = {"ink": "#1F2328", "muted": "#5B616B", "line": "#C9CBCF", "panel": "#F4F5F6", "band": "#D9DBDE",
     "alarm": "#C62828", "warn": "#D99A00", "select": "#1F6FB2", "normal": "#8A9099"}


@st.cache_data
def load():
    return (pd.read_parquet(DATA / "daily.parquet"), pd.read_parquet(DATA / "sensors.parquet"),
            pd.read_parquet(DATA / "events.parquet"), json.loads((DATA / "meta.json").read_text(encoding="utf-8")))


daily, sensors, events, meta = load()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Roboto+Mono:wght@500&display=swap');
:root {{ --ink:{C['ink']}; --muted:{C['muted']}; --line:{C['line']}; --panel:{C['panel']};
        --alarm:{C['alarm']}; --warn:{C['warn']}; --select:{C['select']}; --normal:{C['normal']}; }}
html, body, [class*="css"], .stMarkdown, .stSelectbox, .stDateInput {{ font-family:'Noto Sans KR', sans-serif; }}
.block-container {{ padding-top:1.2rem; max-width:1500px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.num {{ font-family:'Roboto Mono', monospace; font-variant-numeric:tabular-nums; }}
.topbar {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
          background:#2B2F35; color:#E8E9EB; padding:12px 18px; border-radius:6px; margin-bottom:12px; }}
.topbar h1 {{ font-size:1.25rem; margin:0; color:#fff; font-weight:700; }}
.topbar .sub {{ font-size:.82rem; color:#B7BBC2; }}
.chip {{ display:inline-block; border:1px solid #5B616B; border-radius:4px; padding:3px 9px; font-size:.78rem; margin-left:6px; color:#E8E9EB; }}
.chip.ok::before {{ content:'●'; color:#8FBF8F; margin-right:5px; }}
.chip.bad::before {{ content:'●'; color:var(--alarm); margin-right:5px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; margin-bottom:12px; }}
.panel h3 {{ font-size:.95rem; margin:0 0 10px 0; color:var(--ink); font-weight:700; letter-spacing:-.2px; }}
.panel .hint {{ font-size:.75rem; color:var(--muted); font-weight:400; margin-left:6px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin-bottom:12px; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-left:5px solid var(--normal); border-radius:6px; padding:10px 14px; }}
.kpi .label {{ font-size:.78rem; color:var(--muted); }}
.kpi .value {{ font-size:1.7rem; font-weight:700; color:var(--ink); line-height:1.2; }}
.kpi .value small {{ font-size:.8rem; color:var(--muted); font-weight:400; margin-left:3px; }}
.kpi.alarm {{ border-left-color:var(--alarm); }} .kpi.alarm .value {{ color:var(--alarm); }}
.kpi.warn {{ border-left-color:var(--warn); }}
.kpi.select {{ border-left-color:var(--select); }}
table.wq {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
table.wq th {{ text-align:left; color:var(--muted); font-weight:500; border-bottom:1px solid var(--line); padding:6px 6px; white-space:nowrap; }}
table.wq td {{ border-bottom:1px solid #E1E3E6; padding:7px 6px; vertical-align:middle; color:var(--ink); }}
table.wq tr.sel td {{ background:#E3EDF7; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:.74rem; font-weight:700; white-space:nowrap; }}
.badge.위험 {{ background:var(--alarm); color:#fff; }}
.badge.주의 {{ background:var(--warn); color:#1F2328; }}
.badge.정상 {{ background:#E1E3E6; color:var(--muted); }}
.bar {{ position:relative; width:56px; height:8px; background:#E1E3E6; border-radius:2px; display:inline-block; vertical-align:middle; margin-right:6px; }}
.bar > span {{ position:absolute; left:0; top:0; bottom:0; border-radius:2px; }}
.out-hit {{ color:var(--alarm); font-weight:700; }} .out-no {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(10, 1fr); gap:4px; }}
.cell {{ text-align:center; font-size:.72rem; padding:6px 0; border-radius:3px; background:#E1E3E6; color:var(--muted);
         border:2px solid transparent; font-family:'Roboto Mono', monospace; }}
.cell.주의 {{ background:var(--warn); color:#1F2328; font-weight:700; }}
.cell.위험 {{ background:var(--alarm); color:#fff; font-weight:700; }}
.cell.rec {{ border-color:#2B2F35; }}
.cell.sel {{ outline:3px solid var(--select); outline-offset:1px; }}
.legend {{ font-size:.74rem; color:var(--muted); margin-top:8px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 10px; vertical-align:-1px; }}
.head {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; }}
.head .title {{ font-size:1.15rem; font-weight:700; color:var(--ink); }}
.head .meta {{ font-size:.82rem; color:var(--muted); }}
.score {{ font-size:2.1rem; font-weight:700; line-height:1; }}
.reason {{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid #E1E3E6; font-size:.86rem; }}
.reason:last-child {{ border-bottom:none; }}
.reason .n {{ width:20px; height:20px; border-radius:50%; background:#2B2F35; color:#fff; font-size:.72rem; display:flex; align-items:center; justify-content:center; flex:none; }}
.reason .t {{ flex:1; }}
.comp {{ display:grid; grid-template-columns:56px 1fr 120px; align-items:center; gap:8px; font-size:.82rem; padding:4px 0; }}
.comp .track {{ position:relative; height:10px; background:#E1E3E6; border-radius:2px; }}
.comp .fill {{ position:absolute; left:0; top:0; bottom:0; background:#7D838C; border-radius:2px; }}
.comp .fill.over {{ background:var(--warn); }}
.comp .tick {{ position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--ink); }}
table.ev {{ width:100%; border-collapse:collapse; font-size:.8rem; }}
table.ev td {{ padding:4px 6px; border-bottom:1px solid #E1E3E6; color:var(--ink); }}
.sect {{ font-size:.95rem; font-weight:700; color:var(--ink); margin:4px 0 0 2px; }}
.sect .hint {{ font-size:.75rem; color:var(--muted); font-weight:400; margin-left:6px; }}
.foot {{ font-size:.74rem; color:var(--muted); margin-top:6px; }}
@media (max-width: 640px) {{ .grid {{ grid-template-columns:repeat(5, 1fr); }} .bar {{ width:50px; }} }}
</style>""", unsafe_allow_html=True)

# ---------- 상단 바 ----------
drift_ok = meta["drift_psi_max"] < meta["drift_threshold"]
st.markdown(f"""
<div class="topbar">
  <div><h1>아침 점검 브리핑</h1>
       <div class="sub">설비 100대 · 매일 07:00 기준 · 다음 24시간 고장 위험</div></div>
  <div>
    <span class="chip ok">운영 모델 v{meta['model_version']} · 골든셋 통과 (포착 {meta['golden_recall5']:.1%})</span>
    <span class="chip {'ok' if drift_ok else 'bad'}">센서 drift {'정상' if drift_ok else '재학습 필요'} (PSI {meta['drift_psi_max']:.3f} / 기준 {meta['drift_threshold']})</span>
  </div>
</div>""", unsafe_allow_html=True)

dates = sorted(daily.date.dt.date.unique())
c1, c2, c3 = st.columns([2, 2, 6])
day = c1.date_input("기준일", value=pd.Timestamp("2015-12-08").date(), min_value=dates[0], max_value=dates[-1])
verify = c2.toggle("검증 모드 (실제 결과 표시)", value=False,
                   help="과거 데이터를 재생하는 데모라서, 다음 24시간에 실제로 고장 났는지 확인할 수 있습니다.")

today = daily[daily.date.dt.date == day].sort_values("rank")
rec = today[today.recommend]
n = today.status.value_counts()

# ---------- 요약 타일 ----------
kpis = [("select", "오늘 점검 추천", len(rec), "대"), ("alarm", "위험 · 0.9 이상", n.get("위험", 0), "대"),
        ("warn", "주의 · 0.5~0.9", n.get("주의", 0), "대"), ("", "정상", n.get("정상", 0), "대")]
if verify:
    past = daily[(daily.date.dt.date < day) & (daily.date.dt.date >= day - pd.Timedelta(days=30))]
    hit, tot = int(past[past.recommend].actual_fail_24h.sum()), int(past.actual_fail_24h.sum())
    kpis.append(("", "지난 30일 고장 포착", f"{hit}/{tot}" if tot else "-", f"({hit / tot:.0%})" if tot else ""))
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi {cls}"><div class="label">{lab}</div><div class="value num">{val}<small>{unit}</small></div></div>'
    for cls, lab, val, unit in kpis) + "</div>", unsafe_allow_html=True)

left, right = st.columns([5, 6], gap="medium")

# ---------- 설비 선택 (작업 큐 1순위가 기본) ----------
ids = today.machineID.tolist()
with right:
    sel = st.selectbox("설비 상세", ids, index=0, format_func=lambda m: f"설비 #{m:03d}  ·  {today.set_index('machineID').loc[m, 'status']}  ·  위험도 {today.set_index('machineID').loc[m, 'score']:.2f}")
row = today.set_index("machineID").loc[sel]


def score_bar(s):
    color = C["alarm"] if s >= meta["alarm"] else C["warn"] if s >= meta["warn"] else C["normal"]
    return f'<span class="bar"><span style="width:{s * 100:.0f}%;background:{color}"></span></span><span class="num">{s:.2f}</span>'


# ---------- 왼쪽: 점검 작업 큐 + 현황 맵 ----------
with left:
    trs = []
    for _, r in rec.iterrows():
        action = "즉시" if r.status == "위험" else "금일"
        out = ""
        if verify:
            out = f'<td class="out-hit" style="white-space:nowrap">고장 {r.fail_comps}</td>' if r.actual_fail_24h else '<td class="out-no">없음</td>'
        trs.append(f'<tr class="{"sel" if r.machineID == sel else ""}"><td class="num">{r["rank"]}</td>'
                   f'<td style="white-space:nowrap"><b>#{r.machineID:03d}</b> <span class="badge {r.status}">{r.status}</span>'
                   f'<div style="color:var(--muted);font-size:.72rem">{r.model} · {r.age}년</div></td>'
                   f'<td style="white-space:nowrap">{score_bar(r.score)}</td>'
                   f'<td style="font-size:.78rem">{r.reason1 or "-"}</td><td style="white-space:nowrap">{action}</td>{out}</tr>')
    th = "<th>#</th><th>설비</th><th>위험도</th><th>주요 근거</th><th>조치</th>" + ("<th>실제 24h</th>" if verify else "")
    st.markdown(f'<div class="panel"><h3>점검 작업 큐<span class="hint">상위 {meta["top_n"]}대 + 위험도 {meta["alarm"]} 이상 전부</span></h3>'
                f'<table class="wq"><tr>{th}</tr>{"".join(trs)}</table></div>', unsafe_allow_html=True)

    if verify:
        missed = today[~today.recommend & (today.actual_fail_24h == 1)]
        if len(missed):
            items = ", ".join(f"#{r.machineID:03d} (순위 {r['rank']}, 위험도 {r.score:.2f}, {r.fail_comps})" for _, r in missed.iterrows())
            st.markdown(f'<div class="panel" style="border-left:5px solid var(--alarm)"><h3>놓친 고장</h3>'
                        f'<div style="font-size:.84rem">{items}</div></div>', unsafe_allow_html=True)

    cells = "".join(
        f'<div class="cell {r.status} {"rec" if r.recommend else ""} {"sel" if r.machineID == sel else ""}" '
        f'title="#{r.machineID:03d} · {r.status} · 위험도 {r.score:.2f} · 순위 {r["rank"]}">{r.machineID:03d}</div>'
        for _, r in today.sort_values("machineID").iterrows())
    st.markdown(f'<div class="panel"><h3>설비 현황<span class="hint">회색 = 정상, 색은 이상 상태에만 사용</span></h3>'
                f'<div class="grid">{cells}</div><div class="legend"><i style="background:var(--alarm)"></i>위험'
                f'<i style="background:var(--warn)"></i>주의<i style="background:#E1E3E6"></i>정상'
                f'<i style="border:2px solid #2B2F35;background:transparent"></i>점검 추천'
                f'<i style="outline:2px solid var(--select);background:transparent"></i>선택</div></div>', unsafe_allow_html=True)

# ---------- 오른쪽: 설비 상세 ----------
alt.themes.enable("none")


def style(ch, h):
    return (ch.properties(height=h).configure_view(stroke=None)
            .configure_axis(labelColor=C["muted"], titleColor=C["muted"], gridColor="#E1E3E6", domainColor=C["line"],
                            tickColor=C["line"], labelFont="Noto Sans KR", titleFont="Noto Sans KR", labelFontSize=11, titleFontSize=11)
            .configure(background="transparent"))


with right:
    color = C["alarm"] if row.status == "위험" else C["warn"] if row.status == "주의" else C["ink"]
    contribs = [c for c in (row.contrib1, row.contrib2, row.contrib3) if pd.notna(c)]
    reasons = "".join(
        f'<div class="reason"><span class="n">{i + 1}</span><span class="t">{t}</span>'
        f'<span class="bar" style="width:70px"><span style="width:{c / max(contribs) * 100:.0f}%;background:#5B616B"></span></span></div>'
        for i, (t, c) in enumerate(zip([row.reason1, row.reason2, row.reason3], contribs)))
    outcome = ""
    if verify:
        outcome = (f'<div style="margin-top:8px;font-size:.84rem" class="out-hit">실제: 다음 24시간 내 {row.fail_comps} 고장</div>'
                   if row.actual_fail_24h else '<div style="margin-top:8px;font-size:.84rem" class="out-no">실제: 다음 24시간 고장 없음</div>')
    st.markdown(f"""
<div class="panel">
  <div class="head">
    <div><div class="title">설비 #{sel:03d} <span class="badge {row.status}" style="margin-left:6px">{row.status}</span></div>
         <div class="meta">{row.model} · 연식 {row.age}년 · {day:%Y-%m-%d} 07:00</div></div>
    <div style="text-align:right"><div class="meta">위험도 (24h 내 고장 확률)</div>
         <div class="score num" style="color:{color}">{row.score:.2f}</div>
         <div class="meta">전체 100대 중 {row['rank']}위</div></div>
  </div>
  <div style="margin-top:12px;font-size:.8rem;color:var(--muted)">판단 근거 · 위험도를 올린 요인 (SHAP 기여 순)</div>
  {reasons or '<div class="reason">위험도를 올린 뚜렷한 요인 없음</div>'}
  {outcome}
</div>""", unsafe_allow_html=True)

    # 14일 위험도 추이
    hist = daily[(daily.machineID == sel) & (daily.date.dt.date <= day) & (daily.date.dt.date > day - pd.Timedelta(days=14))]
    base = alt.Chart(hist).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", tickCount=7)))
    rules = alt.Chart(pd.DataFrame({"y": [meta["warn"], meta["alarm"]], "c": [C["warn"], C["alarm"]]})).mark_rule(strokeDash=[4, 3]).encode(
        y="y:Q", color=alt.Color("c:N", scale=None))
    trend = (rules + base.mark_line(color=C["ink"], strokeWidth=1.5)
             .encode(y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])))
             + base.mark_circle(size=40).encode(y="score:Q", color=alt.Color("status:N", scale=alt.Scale(
                 domain=["정상", "주의", "위험"], range=[C["normal"], C["warn"], C["alarm"]]), legend=None),
                 tooltip=[alt.Tooltip("date:T", title="날짜", format="%m-%d"), alt.Tooltip("score:Q", title="위험도", format=".2f")]))
    st.markdown('<div class="sect">위험도 추이 · 최근 14일<span class="hint">점선 = 주의 0.5 / 위험 0.9</span></div>', unsafe_allow_html=True)
    st.altair_chart(style(trend, 150), width="stretch")

# ---------- 아래: 센서 · 부품 · 이력 ----------
t07 = pd.Timestamp(day) + pd.Timedelta(hours=7)
with right:  # 부품·이력은 설비 상세 아래로
    mx = max(max(row[f"{c}_since_d"] for c in COMPS), max(meta["normal_since_d"].values())) * 1.1
    comps = "".join(
        f'<div class="comp"><b>{c}</b><div class="track"><div class="fill {"over" if row[f"{c}_since_d"] > 2 * meta["normal_since_d"][c] else ""}" '
        f'style="width:{row[f"{c}_since_d"] / mx * 100:.0f}%"></div><div class="tick" style="left:{meta["normal_since_d"][c] / mx * 100:.0f}%"></div></div>'
        f'<span class="num" style="color:var(--muted)">{row[f"{c}_since_d"]:.0f}일 / 정상 {meta["normal_since_d"][c]:.0f}일</span></div>'
        for c in COMPS)
    errs = " · ".join(f"error{i} <b class='num'>{int(row[f'error{i}_24h'])}</b>" for i in range(1, 6))
    st.markdown(f'<div class="panel"><h3>부품 교체 후 경과<span class="hint">검은 눈금 = 정상 운전 중앙값, 주황 = 2배 초과</span></h3>{comps}'
                f'<div style="margin-top:10px;font-size:.8rem;color:var(--muted)">최근 24h 에러 · {errs}</div></div>', unsafe_allow_html=True)
with right:
    ev = events[(events.machineID == sel) & (events.datetime < t07) & (events.datetime >= t07 - pd.Timedelta(days=30))].sort_values("datetime", ascending=False)
    tag = {"고장": "badge 위험", "고장 교체": "badge 주의", "예방정비": "badge 정상", "에러": "badge 정상"}
    trs = "".join(f'<tr><td class="num">{r.datetime:%m-%d %H:%M}</td><td><span class="{tag[r.event]}">{r.event}</span></td><td>{r.detail}</td></tr>'
                  for _, r in ev.head(12).iterrows())
    st.markdown(f'<div class="panel"><h3>최근 30일 정비 · 에러 이력<span class="hint">기준 시각 이전 기록만</span></h3>'
                f'<table class="ev">{trs or "<tr><td>기록 없음</td></tr>"}</table></div>', unsafe_allow_html=True)

win = sensors[(sensors.machineID == sel) & (sensors.datetime > t07 - pd.Timedelta(hours=24)) & (sensors.datetime <= t07)]
st.markdown('<div class="sect">센서 · 최근 24시간<span class="hint">회색 띠 = 학습 기간 정상 운전의 5~95% 범위, 점선 = 정상 중앙값, 주황 점 = 범위 밖</span></div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, (s, ko) in zip(cols, SENSORS.items()):
    b = meta["band"][s]
    w = win.assign(out=(win[s] < b["p05"]) | (win[s] > b["p95"]))
    lo, hi = min(b["p05"], w[s].min()), max(b["p95"], w[s].max())
    pad = (hi - lo) * 0.1
    sc = alt.Scale(domain=[lo - pad, hi + pad], zero=False)  # 모든 층이 같은 축을 써야 0부터 늘어나지 않음
    y = alt.Y(f"{s}:Q", title=None, scale=sc)
    x = alt.X("datetime:T", title=None, axis=alt.Axis(format="%H시", tickCount=4))
    band = alt.Chart(pd.DataFrame([b])).mark_rect(color=C["band"], opacity=0.8).encode(y=alt.Y("p05:Q", scale=sc), y2="p95:Q")
    med = alt.Chart(pd.DataFrame([b])).mark_rule(color=C["muted"], strokeDash=[3, 3]).encode(y=alt.Y("p50:Q", scale=sc))
    line = alt.Chart(w).mark_line(color=C["ink"], strokeWidth=1.3).encode(x=x, y=y)
    pts = alt.Chart(w[w.out]).mark_circle(color=C["warn"], size=36, opacity=1).encode(
        x=x, y=y, tooltip=[alt.Tooltip("datetime:T", format="%m-%d %H시", title="시각"), alt.Tooltip(f"{s}:Q", format=".1f", title=ko)])
    with col:
        st.markdown(f'<div style="font-size:.82rem;font-weight:700;color:var(--ink);margin:6px 0 0 4px">{ko}</div>', unsafe_allow_html=True)
        st.altair_chart(style(band + med + line + pts, 140), width="stretch")

st.markdown(f'<div class="foot">Microsoft Azure Predictive Maintenance 가상 데이터, 2015년 11~12월 시험 기간 재생 · '
            f'모델 {meta["model_desc"]} · 골든셋 PR-AUC {meta["golden_prauc"]} · 근거는 SHAP(GradientExplainer) 기여 상위 3개</div>',
            unsafe_allow_html=True)
