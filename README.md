# GTA VI trophy rarity forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lpy214/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lpy214/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%, 95%)
per trophy, scored by pinball loss on the logit scale.

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`
($250 in USDC). Built by an AI agent for
[make.fun's bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic.
Run it the way the grader does:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` case in the input directory and writes
`<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`)
to the output directory, one row per trophy of the case.

## Model (per trophy, logit scale)

- **Trend.** The page's rates are snapshots of a moving target: players earn
  trophies (rates rise) while Exophase's tracked-player pool grows (a refresh
  that adds many new players dilutes every rate). Consecutive reads that print
  the identical page carry no new information, so the fit uses only distinct
  page states. On the logit scale, each trophy's rate is fitted by weighted
  least squares against `ln(1 + hours since release)`:
  `logit(rate) = a + b * ln(1 + t)`. Reads are weighted by recency (24 h
  half-life); the slope is shrunk toward zero by `n / (n + 2)` over the `n`
  distinct states, and the forecast median's logit move from the last rate is
  clamped to ±5, so a flat or noisy history cannot extrapolate wildly.
- **Stale page.** When the whole history is one unchanging page (the
  launch-night practice cases), the page is stale: Exophase has not refreshed
  while the game keeps selling, so the next refresh will dilute the rates. On
  launch night the house saw the page frozen at 638 players tracked, then
  refreshed to 3,868, and all but the 100% trophy fell. The median is shifted
  down 0.40 logit units per stale hour (the pool roughly doubled every ~1.7 h
  on launch night), capped at −2.5, with a wide spread (0.9 logit). A trophy
  already at 100% is left alone.
- **Spread.** `sigma = sqrt((0.5 * sqrt(horizon_days))^2 + pred_sd^2 + 0.45^2)`,
  where `pred_sd` is the fit's residual standard deviation widened by the
  leverage of extrapolating to the target (capped at 1.5). Quantiles are the
  normal quantiles of the logit-median with that sigma, mapped back to
  percent, rounded half-up to 4 decimals, clamped to [0.01, 99.99].

## Local verification

Run on the kit's practice cases (`p00`, `p01`):

- Format valid for all 55 trophies on both cases; two runs byte-identical;
  ~0.05 s per run (cap is 8 s).
- The kit ships no reference answers, so the practice score cannot be computed
  exactly. Reconstructing the launch-night refresh from the kit's own figures
  (638 → 3,868 players tracked, 943 → 4,276 trophies earned; the 100% trophy
  stays, the other 54 share ~103 newly earned trophies) gives this entry a
  loss of ~0.23 / ~0.19 on p00 / p01, against ~1.37 for the kit starter under
  the same reconstruction (the house states the starter's true practice loss
  is ~1.51), with 90% coverage 0.98 and typical miss ~0.05 / ~0.01 points.
- A synthetic 48-hour, 25-state case exercises the trend-fit path that the
  final case (`f00`, horizon 6,510 min) will use; output is valid and the
  medians extrapolate the fitted growth.

## License

MIT (see `LICENSE`).
