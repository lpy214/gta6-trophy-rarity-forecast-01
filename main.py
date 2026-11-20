"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case (makefun/gta6-trophies-case/v1) from the input directory and writes,
for each of the game's trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its
earn rate on Exophase at the case's target, as /output/<id>.out.

Model (per trophy, all in my own words):
  * The page's rates are snapshots of a moving target: players earn trophies (rates
    rise) while Exophase's tracked-player pool grows (a refresh that adds many new
    players dilutes every rate). The observed history is a sequence of page states;
    consecutive reads that print the identical page carry no new information, so
    the fit uses only distinct page states.
  * On the logit scale, a trophy's rate is fitted by weighted least squares against
    ln(1 + hours since release): logit(rate) = a + b * ln(1 + t). The log-time axis
    is the natural clock for a launch: early hours move rates fast, later hours
    move them slowly. Reads are weighted by recency (24 h half-life) because the
    regime drifts as the launch matures. The slope is shrunk toward zero by
    n / (n + 2) over the n distinct states, so a flat or noisy history cannot
    extrapolate wildly, and the forecast median's logit move from the last rate
    is clamped to +-5.
  * When the whole history is one unchanging page (the launch-night practice
    cases), the page is stale: Exophase has not refreshed while the game keeps
    selling, so the next refresh will dilute the rates. On launch night the
    house saw exactly this: the page frozen at 638 players tracked from
    23:48Z, then refreshed to 3,868 players, and all but the 100% trophy fell
    - the new cohort held almost no trophies yet (the tracked pool grew ~6x
    in ~4.5 h while total trophies earned per player fell). So the median is
    shifted down by 0.40 logit units per stale hour (the pool roughly doubles
    every ~1.7 h on launch night, and a trophy the new cohort has not earned
    falls by the pool-growth factor), capped at -2.5, with a wide spread
    (0.9 logit) because the new cohort earns some trophies and almost none
    of others. A trophy already at 100% is left alone: the house's own record
    shows it stays at 100% through a refresh.
  * With fewer than 3 distinct states the median is the last read's rate
    (persistence).
  * Spread: sigma = sqrt((0.5 * sqrt(horizon_days))^2 + pred_sd^2 + 0.45^2),
    where pred_sd is the fit's residual standard deviation widened by the
    leverage of extrapolating to the target (capped at 1.5), and 0.45 is a floor
    for refresh lag and model misspecification. Quantiles are the normal
    quantiles of the logit-median with that sigma, mapped back to percent.

Standard library only, no network, deterministic: the same case always yields
the same bytes. Runs in well under the 8-second cap.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles at LEVELS

# Random-walk spread of a rate on the logit scale, per square-root day.
RW_SIGMA = 0.5
# Floor on the total logit spread: refresh lag + model misspecification.
SPREAD_FLOOR = 0.45
# Cap on the fit's prediction spread, in logit units.
MAX_PRED_SD = 1.5
# Recency half-life for read weights, in hours.
HALF_LIFE_HOURS = 24.0
# Slope shrinkage: b *= n / (n + SHRINK_K) over n distinct states.
SHRINK_K = 2.0
# Cap on how far the forecast median may move from the last rate, logit units.
MAX_MEDIAN_MOVE = 5.0
# Stale-page dilution: logit shift per stale hour, and its spread, in logit units.
# Launch night the tracked pool grew ~6x in ~4.5 h (doubling every ~1.7 h), so a
# trophy the newly tracked cohort has not earned falls by that growth factor.
STALE_SHIFT_PER_HOUR = 0.40
STALE_SHIFT_MAX = 2.5
STALE_SIGMA = 0.9


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (x is always positive here)."""
    whole = math.floor(x * 10000 + 0.5)
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def distinct_states(history: list) -> list:
    """Collapse consecutive reads that print the identical page to one state."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def weighted_fit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, x_bar, sxx, n)."""
    n = len(xs)
    sw = sum(ws)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = y_bar - b * x_bar
    if n > 2:
        sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        resid_sd = math.sqrt(sse / (sw * (n - 2) / n)) if sse > 0 else 0.0
    else:
        resid_sd = 0.0
    return a, b, resid_sd, x_bar, sxx, n


def forecast_trophy(rates_series, times_h, target_h, horizon_days, last_rate,
                    flat_history=False):
    """rates_series: distinct-state rates for this trophy (None = missing).
    times_h: hours since release for each state. Returns (median_logit, sigma)."""
    last_lg = logit(last_rate if last_rate is not None else 1.0)
    if flat_history:
        # One unchanging page: no trend exists, only the coming dilution.
        return last_lg, math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                  + STALE_SIGMA ** 2)
    pts = [(t, r) for t, r in zip(times_h, rates_series) if r is not None]
    if len(pts) < 3:
        return last_lg, math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                  + SPREAD_FLOOR ** 2)
    t_last = max(t for t, _ in pts)
    xs = [math.log1p(t) for t, _ in pts]
    ys = [logit(r) for _, r in pts]
    ws = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for t, _ in pts]
    fit = weighted_fit(xs, ys, ws)
    if fit is None:
        return last_lg, math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                  + SPREAD_FLOOR ** 2)
    a, b, resid_sd, x_bar, sxx, n = fit
    # Shrink the slope toward zero: few or noisy states must not extrapolate far.
    b *= n / (n + SHRINK_K)
    b = max(-4.0, min(4.0, b))
    x_target = math.log1p(target_h)
    median_lg = a + b * x_target
    median_lg = max(last_lg - MAX_MEDIAN_MOVE, min(last_lg + MAX_MEDIAN_MOVE, median_lg))
    # Prediction spread: residual noise widened by extrapolation leverage.
    leverage = 1.0 + 1.0 / n + (x_target - x_bar) ** 2 / sxx
    pred_sd = min(MAX_PRED_SD, max(0.05, resid_sd) * math.sqrt(leverage))
    sigma = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                      + pred_sd ** 2 + SPREAD_FLOOR ** 2)
    return median_lg, sigma


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if len(case["game"].get("release_date", "")) == 10 else parse_time(case["game"]["release_date"])
    states = distinct_states(history)
    times_h = [(parse_time(s["at"]) - release).total_seconds() / 3600.0 for s in states]
    target_h = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    last = history[-1]
    last_rates = last["rates"]

    # Stale-page check: one distinct state for the whole history.
    flat_history = len(states) == 1
    stale_hours = 0.0
    if flat_history:
        first = parse_time(history[0]["at"])
        stale_hours = (parse_time(case["at"]) - first).total_seconds() / 3600.0
    stale_shift = min(STALE_SHIFT_MAX, STALE_SHIFT_PER_HOUR * stale_hours) \
        if stale_hours > 0 else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        series = [s["rates"][i] if i < len(s["rates"]) else None for s in states]
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            # Fall back to the most recent non-null read for this trophy.
            for r in reversed(series):
                if r is not None:
                    last_rate = r
                    break
        median_lg, sigma = forecast_trophy(series, times_h, target_h,
                                           horizon_days, last_rate,
                                           flat_history=flat_history)
        if stale_shift and (last_rate is None or last_rate < 99.99):
            median_lg -= stale_shift
        qs = []
        for z in Z:
            q = round4(clamp_pct(inv_logit(median_lg + z * sigma)))
            qs.append(q)
        # Enforce non-decreasing quantiles after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
