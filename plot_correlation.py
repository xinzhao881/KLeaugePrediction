"""Visualize how stat-goal correlations evolve match day by match day.

For every match day (season, round) the Spearman correlation between each
selected stat (feature_importance/stat_groups.csv) and goals is recomputed on
  * a trailing window of the last WINDOW_ROUNDS match days, or
  * all match days so far (expanding),
in two modes:
  * predictive : pre-match rolling mean (roll5_*) vs goals in that match
  * same-match : stat in the match vs goals in the same match (reference)

Output: feature_importance/correlation_timeline.html (interactive heatmap +
per-stat line chart, light/dark mode, table view).
"""

import json

import numpy as np
import pandas as pd

import pruning as pr

OUT_PATH = pr.BASE_DIR / "feature_importance" / "correlation_timeline.html"
WINDOW_ROUNDS = 38     # ~ one season of match days
MIN_ROWS = 60          # minimum team-matches for a correlation
TRAIN_RATIO = 0.7

FM_LABELS = {
    "expected_goals": "xG", "expected_goals_open_play": "오픈플레이 xG",
    "expected_goals_set_play": "세트피스 xG", "expected_goals_non_penalty": "논PK xG",
    "expected_goals_on_target": "xGOT", "big_chance": "빅찬스",
    "big_chance_missed_title": "빅찬스 미스", "shotsontarget": "유효슈팅",
    "shots_inside_box": "박스 안 슈팅", "shots_woodwork": "골대 강타",
    "total_shots": "슈팅", "corners": "코너킥", "yellow_cards": "경고",
    "red_cards": "퇴장", "duel_won": "듀얼 승리", "team_rating": "팀 평점",
    "ballpossesion": "점유율", "clearances": "클리어링", "keeper_saves": "키퍼 세이브",
    "long_balls_accurate": "롱볼 성공", "long_balls_accurate_pct": "롱볼 성공률",
    "accurate_passes_pct": "패스 성공률", "own_half_passes": "자기 진영 패스",
    "opposition_half_passes": "상대 진영 패스", "accurate_crosses": "크로스 성공",
    "dribbles_succeeded_pct": "드리블 성공률",
}
GROUP_TAG = {"attack": "att_", "defense": "def_", "strength": "str_"}
GROUP_ORDER = ["attack", "defense", "strength"]
GROUP_TARGET = {"attack": "gf", "defense": "ga", "strength": "gd"}


def label(stat):
    conceded = stat.startswith("opp_")
    s = stat[4:] if conceded else stat
    src, name = s[:2], s[3:]
    text = FM_LABELS.get(name, name) if src == "fm" else name
    return f"{'허용 ' if conceded else ''}{text} ({'FotMob' if src == 'fm' else 'K리그'})"


def load_data():
    groups = pd.read_csv(pr.STAT_GROUPS_PATH, encoding="utf-8-sig")
    groups = groups[groups["group"].isin(GROUP_ORDER)].copy()
    groups["order"] = groups["group"].map(GROUP_ORDER.index)
    groups = groups.assign(a=groups["pred_r"].abs()).sort_values(["order", "a"],
                                                                  ascending=[True, False])
    seasons = sorted(int(p.name) for p in pr.OUTPUT_DIR.iterdir() if p.name.isdigit())
    tm = pd.concat([
        pd.read_csv(pr.OUTPUT_DIR / str(s) / "team_match_stats.csv", encoding="utf-8-sig",
                    low_memory=False)
        for s in seasons if (pr.OUTPUT_DIR / str(s) / "team_match_stats.csv").exists()
    ], ignore_index=True)
    tm = tm[tm["finished"]].copy()
    tm["gf"] = tm["goals_for"].astype(float)
    tm["ga"] = tm["goals_against"].astype(float)
    tm["gd"] = tm["gf"] - tm["ga"]
    return groups, tm


def match_days(tm):
    days = (tm.groupby(["season", "round"])["kst_date"].max().reset_index()
            .sort_values(["season", "round"]).reset_index(drop=True))
    days["idx"] = np.arange(len(days))
    return days


def train_cutoff_index(tm, days):
    times = tm.drop_duplicates("match_id").sort_values(["kst_date", "kst_time"])
    last = times.iloc[int(len(times) * TRAIN_RATIO) - 1]
    hit = days[(days["season"] == last["season"]) & (days["round"] == last["round"])]
    return int(hit["idx"].iloc[0]), str(last["kst_date"])


def correlations(tm, days, stats, feature_cols):
    tm = tm.merge(days[["season", "round", "idx"]], on=["season", "round"])
    out = {}
    for mode, cols in (("pred", feature_cols), ("same", stats)):
        for window in ("roll", "exp"):
            res = {t: [] for t in ("gf", "ga", "gd")}
            ns = []
            for i in days["idx"]:
                lo = i - WINDOW_ROUNDS + 1 if window == "roll" else 0
                w = tm[(tm["idx"] >= lo) & (tm["idx"] <= i)]
                x = w[cols].rank()
                n = x.notna().sum()
                for t in res:
                    r = x.corrwith(w[t].rank())
                    r[n < MIN_ROWS] = np.nan
                    res[t].append([None if pd.isna(v) else round(float(v), 3) for v in r])
                ns.append([int(v) for v in n])
            out[f"{mode}_{window}"] = {**res, "n": ns}
    return out


def main():
    groups, tm = load_data()
    stats = groups["stat"].tolist()
    feature_cols = [f"roll{pr.ROLL_WINDOW}_{GROUP_TAG[g]}{s}"
                    for s, g in zip(groups["stat"], groups["group"])]
    missing = [c for c in stats + feature_cols if c not in tm.columns]
    if missing:
        raise ValueError(f"Missing columns, re-run pruning.py: {missing[:5]}")

    days = match_days(tm)
    cut_idx, cut_date = train_cutoff_index(tm, days)
    data = {
        "stats": [{"key": s, "label": label(s), "group": g,
                   "train_r": round(float(r), 3)}
                  for s, g, r in zip(groups["stat"], groups["group"], groups["pred_r"])],
        "days": [{"season": int(d.season), "round": int(d.round), "date": str(d.kst_date)}
                 for d in days.itertuples()],
        "cutoff": {"idx": cut_idx, "date": cut_date},
        "window_rounds": WINDOW_ROUNDS,
        "group_target": GROUP_TARGET,
        "series": correlations(tm, days, stats, feature_cols),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_PATH} ({len(stats)} stats x {len(days)} match days)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>K리그 지표-득실점 상관 타임라인</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948; --nan: #e1e0d9;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
    --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --nan: #2c2c2a;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
  --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --nan: #2c2c2a;
}
body { margin: 0; background: var(--page); }
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary); background: var(--page); padding: 24px; min-height: 100vh;
  box-sizing: border-box; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); font-size: 13px; margin: 0 0 16px; line-height: 1.5; max-width: 980px; }
.controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 16px; font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: var(--surface-1); color: var(--text-secondary); font: inherit;
  padding: 6px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.ctl-label { color: var(--text-muted); margin-right: 6px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; margin-bottom: 16px; overflow-x: auto; }
.card h2 { font-size: 14px; font-weight: 600; margin: 0 0 4px; }
.card .hint { font-size: 12px; color: var(--text-muted); margin: 0 0 8px; }
svg text { font-family: inherit; }
.tick { font-size: 11px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.rowlabel { font-size: 11px; fill: var(--text-secondary); cursor: pointer; }
.rowlabel.active { fill: var(--text-primary); font-weight: 600; }
.grouplabel { font-size: 11px; font-weight: 600; fill: var(--text-primary); }
.tooltip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; z-index: 10; line-height: 1.5;
  font-variant-numeric: tabular-nums; }
.tooltip .t { font-weight: 600; }
.tooltip .m { color: var(--text-secondary); }
.legend { display: flex; gap: 16px; align-items: center; font-size: 12px; color: var(--text-secondary); margin: 4px 0 8px; }
.sw { display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
table { border-collapse: collapse; font-size: 12px; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--text-muted); font-weight: 600; }
td.num, th.num { text-align: right; }
</style>
</head>
<body>
<div class="viz-root">
  <h1>지표별 득점·실점 상관 타임라인</h1>
  <p class="sub" id="sub"></p>

  <div class="controls">
    <span><span class="ctl-label">상관 종류</span><span class="seg" data-key="mode">
      <button data-v="pred" aria-pressed="true">예측 (직전 5경기 평균 → 이번 경기)</button>
      <button data-v="same">같은 경기 (참고)</button></span></span>
    <span><span class="ctl-label">계산 구간</span><span class="seg" data-key="window">
      <button data-v="roll" aria-pressed="true">최근 1시즌</button>
      <button data-v="exp">누적</button></span></span>
    <span><span class="ctl-label">히트맵 대상</span><span class="seg" data-key="target">
      <button data-v="group" aria-pressed="true">그룹 기준</button>
      <button data-v="gf">득점</button>
      <button data-v="ga">실점</button></span></span>
    <span><span class="ctl-label">보기</span><span class="seg" data-key="view">
      <button data-v="chart" aria-pressed="true">차트</button>
      <button data-v="table">표</button></span></span>
  </div>

  <div id="chartView">
    <div class="card">
      <h2 id="hmTitle"></h2>
      <p class="hint">셀에 마우스를 올리면 값, 행 이름을 클릭하면 아래 선 그래프에서 득점·실점 상관을 함께 봅니다. 회색 셀은 표본 부족.</p>
      <div id="hmLegend" class="legend"></div>
      <svg id="heatmap"></svg>
    </div>
    <div class="card">
      <h2 id="lineTitle"></h2>
      <div class="legend">
        <span><span class="sw" style="background:var(--series-1)"></span>득점과의 상관</span>
        <span><span class="sw" style="background:var(--series-2)"></span>실점과의 상관</span>
      </div>
      <svg id="line"></svg>
    </div>
  </div>
  <div id="tableView" class="card" style="display:none"><table id="tbl"></table></div>
  <div class="tooltip" id="tip"></div>
</div>

<script>
const D = __DATA__;
const state = { mode: "pred", window: "roll", target: "group", view: "chart", stat: 0 };
const GROUP_NAME = { attack: "득점 지표 (attack)", defense: "실점 지표 (defense)", strength: "전력 지표 (strength)" };
const TARGET_NAME = { gf: "득점", ga: "실점", gd: "득실차" };
const css = n => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(n).trim();
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

function hex2rgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function mix(a, b, t) { const A = hex2rgb(a), B = hex2rgb(b);
  return "rgb(" + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",") + ")"; }
function domain() { return state.mode === "pred" ? 0.25 : 0.6; }
function color(r) {
  if (r === null || r === undefined) return css("--nan");
  const t = Math.max(-1, Math.min(1, r / domain()));
  return t < 0 ? mix(css("--div-mid"), css("--div-neg"), -t) : mix(css("--div-mid"), css("--div-pos"), t);
}
function targetOf(stat) { return state.target === "group" ? D.group_target[stat.group] : state.target; }
function series() { return D.series[state.mode + "_" + state.window]; }
function fmt(r) { return r === null || r === undefined ? "–" : (r > 0 ? "+" : "") + r.toFixed(3); }
function dayLabel(d) { return d.season + "시즌 " + d.round + "R (" + d.date + ")"; }

function showTip(evt, html) { tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.style.display = "none"; }

// Season boundary positions on the x axis
function seasonStarts() { const s = []; D.days.forEach((d, i) => { if (i === 0 || d.season !== D.days[i - 1].season) s.push([i, d.season]); }); return s; }

function drawHeatmap() {
  const svg = document.getElementById("heatmap"); svg.innerHTML = "";
  const S = D.stats, N = D.days.length, cw = 5, ch = 14, gapG = 10;
  const left = 240, top = 22, groupGap = {};
  let y = top, rows = [];
  ["attack", "defense", "strength"].forEach(g => {
    const idx = S.map((s, i) => [s, i]).filter(([s]) => s.group === g);
    if (!idx.length) return;
    rows.push({ type: "group", g, y }); y += 16;
    idx.forEach(([s, i]) => { rows.push({ type: "stat", s, i, y }); y += ch; });
    y += gapG;
  });
  const W = left + N * cw + 20, H = y + 24;
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const ser = series();

  // Season separators + labels
  seasonStarts().forEach(([i, season]) => {
    el("line", { x1: left + i * cw, x2: left + i * cw, y1: top - 4, y2: H - 22, stroke: css("--axis"), "stroke-width": 1 }, svg);
    const t = el("text", { x: left + i * cw + 4, y: top - 8, class: "tick" }, svg); t.textContent = season;
  });

  rows.forEach(r => {
    if (r.type === "group") {
      const t = el("text", { x: 0, y: r.y + 11, class: "grouplabel" }, svg);
      t.textContent = GROUP_NAME[r.g] + " · " + (state.target === "group" ? TARGET_NAME[D.group_target[r.g]] : TARGET_NAME[state.target]) + "과의 상관";
      return;
    }
    const tgt = targetOf(r.s), vals = ser[tgt];
    const lab = el("text", { x: left - 8, y: r.y + ch - 4, "text-anchor": "end",
      class: "rowlabel" + (r.i === state.stat ? " active" : "") }, svg);
    lab.textContent = r.s.label;
    lab.addEventListener("click", () => { state.stat = r.i; render(); });
    for (let j = 0; j < N; j++) {
      const v = vals[j][r.i];
      const c = el("rect", { x: left + j * cw, y: r.y + 1, width: cw - 0.5, height: ch - 2, fill: color(v) }, svg);
      c.addEventListener("mousemove", e => showTip(e,
        `<div class="t">${r.s.label}</div><div class="m">${dayLabel(D.days[j])}</div>` +
        `<div>${TARGET_NAME[tgt]}과의 상관 <b>${fmt(v)}</b></div>` +
        `<div class="m">표본 ${ser.n[j][r.i]} 팀-경기 · ${j <= D.cutoff.idx ? "학습 구간" : "테스트 구간"}</div>`));
      c.addEventListener("mouseleave", hideTip);
      c.addEventListener("click", () => { state.stat = r.i; render(); });
    }
  });

  // Train/test cutoff marker
  const cx = left + (D.cutoff.idx + 1) * cw;
  el("line", { x1: cx, x2: cx, y1: top - 4, y2: H - 22, stroke: css("--text-primary"), "stroke-width": 2, "stroke-dasharray": "4 3" }, svg);
  const ct = el("text", { x: cx + 4, y: H - 8, class: "tick" }, svg); ct.textContent = "← 학습 | 테스트 → (" + D.cutoff.date + ")";

  // Color legend
  const d = domain(), steps = 9;
  let html = "<span>음(−) 상관</span>";
  for (let k = 0; k < steps; k++) { const v = -d + (2 * d) * k / (steps - 1);
    html += `<span title="${v.toFixed(2)}" style="display:inline-block;width:20px;height:10px;background:${color(v)};margin:0 -14px 0 0"></span>`; }
  html += `<span style="margin-left:14px">양(+) 상관</span><span class="tick" style="color:var(--text-muted)">범위 ±${d}</span>`;
  document.getElementById("hmLegend").innerHTML = html;
}

function drawLine() {
  const svg = document.getElementById("line"); svg.innerHTML = "";
  const s = D.stats[state.stat], ser = series(), N = D.days.length;
  document.getElementById("lineTitle").textContent = s.label + " — 득점·실점과의 상관 추이 (그룹: " + s.group + ", 학습 구간 예측 r " + fmt(s.train_r) + ")";
  const left = 48, right = 90, top = 12, h = 220, cw = 5;
  const W = left + N * cw + right, H = top + h + 28;
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const g = ser.gf.map(row => row[state.stat]), a = ser.ga.map(row => row[state.stat]);
  const all = g.concat(a).filter(v => v !== null);
  const m = Math.max(domain(), ...all.map(Math.abs)) * 1.05;
  const X = i => left + i * cw + cw / 2, Y = v => top + h / 2 - (v / m) * (h / 2);

  [-m, -m / 2, 0, m / 2, m].forEach(v => {
    el("line", { x1: left, x2: left + N * cw, y1: Y(v), y2: Y(v), stroke: v === 0 ? css("--axis") : css("--grid"), "stroke-width": 1 }, svg);
    const t = el("text", { x: left - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg); t.textContent = v.toFixed(2);
  });
  seasonStarts().forEach(([i, season]) => {
    el("line", { x1: left + i * cw, x2: left + i * cw, y1: top, y2: top + h, stroke: css("--grid"), "stroke-width": 1 }, svg);
    const t = el("text", { x: left + i * cw + 4, y: top + h + 16, class: "tick" }, svg); t.textContent = season;
  });
  const cx = left + (D.cutoff.idx + 1) * cw;
  el("line", { x1: cx, x2: cx, y1: top, y2: top + h, stroke: css("--text-primary"), "stroke-width": 2, "stroke-dasharray": "4 3" }, svg);

  function path(vals, colorVar) {
    let d = "", pen = false;
    vals.forEach((v, i) => { if (v === null) { pen = false; return; } d += (pen ? "L" : "M") + X(i) + "," + Y(v); pen = true; });
    el("path", { d, fill: "none", stroke: css(colorVar), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    // Direct label at the last value
    let li = vals.length - 1; while (li >= 0 && vals[li] === null) li--;
    if (li >= 0) return { i: li, v: vals[li] };
  }
  const lg = path(g, "--series-1"), la = path(a, "--series-2");
  const labels = [[lg, "득점 " + (lg ? fmt(lg.v) : "")], [la, "실점 " + (la ? fmt(la.v) : "")]].filter(x => x[0]);
  if (labels.length === 2 && Math.abs(Y(labels[0][0].v) - Y(labels[1][0].v)) < 14) {
    const up = Y(labels[0][0].v) <= Y(labels[1][0].v) ? 0 : 1; labels[up].yOff = -7; labels[1 - up].yOff = 7; }
  labels.forEach(([p, text], k) => { const t = el("text", { x: X(p.i) + 6, y: Y(p.v) + 4 + (labels[k].yOff || 0), class: "tick" }, svg);
    t.style.fill = css("--text-secondary"); t.textContent = text; });

  // Crosshair + tooltip
  const cross = el("line", { y1: top, y2: top + h, stroke: css("--text-muted"), "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = [el("circle", { r: 4, fill: css("--series-1"), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg),
                el("circle", { r: 4, fill: css("--series-2"), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg)];
  const hit = el("rect", { x: left, y: top, width: N * cw, height: h, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", e => {
    const box = svg.getBoundingClientRect(); const i = Math.max(0, Math.min(N - 1, Math.floor((e.clientX - box.left - left) / cw)));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    [[g[i], dots[0]], [a[i], dots[1]]].forEach(([v, dot]) => { if (v === null) { dot.setAttribute("visibility", "hidden"); return; }
      dot.setAttribute("cx", X(i)); dot.setAttribute("cy", Y(v)); dot.setAttribute("visibility", "visible"); });
    showTip(e, `<div class="t">${dayLabel(D.days[i])}</div>` +
      `<div><span class="sw" style="background:var(--series-1)"></span>득점 ${fmt(g[i])}</div>` +
      `<div><span class="sw" style="background:var(--series-2)"></span>실점 ${fmt(a[i])}</div>` +
      `<div class="m">표본 ${ser.n[i][state.stat]} 팀-경기 · ${i <= D.cutoff.idx ? "학습 구간" : "테스트 구간"}</div>`);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
}

function drawTable() {
  const ser = series(), last = D.days.length - 1, cut = D.cutoff.idx;
  let html = `<thead><tr><th>지표</th><th>그룹</th><th class="num">학습 구간 예측 r</th>` +
    `<th class="num">득점 r (학습 끝)</th><th class="num">실점 r (학습 끝)</th>` +
    `<th class="num">득점 r (최신)</th><th class="num">실점 r (최신)</th></tr></thead><tbody>`;
  D.stats.forEach((s, i) => { html += `<tr><td>${s.label}</td><td>${s.group}</td><td class="num">${fmt(s.train_r)}</td>` +
    `<td class="num">${fmt(ser.gf[cut][i])}</td><td class="num">${fmt(ser.ga[cut][i])}</td>` +
    `<td class="num">${fmt(ser.gf[last][i])}</td><td class="num">${fmt(ser.ga[last][i])}</td></tr>`; });
  document.getElementById("tbl").innerHTML = html + "</tbody>";
}

function render() {
  const modeText = state.mode === "pred" ? "예측 상관: 직전 5경기 평균 → 이번 경기 득실점" : "같은 경기 상관: 이번 경기 지표 ↔ 이번 경기 득실점 (경기 상황 효과로 왜곡됨, 참고용)";
  const winText = state.window === "roll" ? `라운드마다 최근 ${D.window_rounds}라운드(약 1시즌) 창으로 재계산` : "라운드마다 처음부터 누적해서 재계산";
  document.getElementById("sub").textContent = `${modeText}. ${winText}. Spearman 상관, 팀-경기 단위. 점선은 학습(70%)/테스트 경계.`;
  document.getElementById("hmTitle").textContent = "라운드별 상관 히트맵 (" + D.stats.length + "개 선택 지표 × " + D.days.length + "라운드)";
  document.querySelectorAll(".seg").forEach(seg => seg.querySelectorAll("button").forEach(b =>
    b.setAttribute("aria-pressed", String(state[seg.dataset.key] === b.dataset.v))));
  document.getElementById("chartView").style.display = state.view === "chart" ? "" : "none";
  document.getElementById("tableView").style.display = state.view === "table" ? "" : "none";
  if (state.view === "chart") { drawHeatmap(); drawLine(); } else drawTable();
}
document.querySelectorAll(".seg").forEach(seg => seg.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return; state[seg.dataset.key] = b.dataset.v; render(); }));
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
