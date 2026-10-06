# Premier League Match Predictor 

I'm a medical student with zero prior coding or CS background. I wanted 
to see if I could actually build and understand a working machine 
learning project — not just copy code, but genuinely evaluate whether 
it was any good. This is that project, built step by step with AI 
assistance, and documented honestly including what didn't work.

## What it does
- Loads 3+ seasons of Premier League match data (goals, shots, corners)
- Builds rolling 5-match form features for every team
- Handles newly promoted teams by pulling their prior Championship season 
  stats, discounted for the step up in competition
- Trains an XGBoost classifier to predict Home Win / Draw / Away Win
- Runs a 5,000-iteration Monte Carlo simulation of the remaining season 
  to estimate each team's title, top-4, and relegation odds
- Includes an interactive head-to-head predictor

## How I validated it (this is the part most projects skip)
Instead of just trusting an accuracy number, I tested the model 
retrospectively: trained it on data up to January 2026, then had it 
simulate the rest of the already-completed 2025/26 season — a season 
whose real outcome I could check against.

Results:
- Correctly predicted Arsenal as champion (92.5% confidence) — they won
- Correctly flagged all 3 relegated teams with high confidence 
  (Wolves 100%, Burnley 98.3%, West Ham 83.5%) — all three actually went down
- Badly missed the mid-table: predicted Man United ~8th (they finished 3rd), 
  predicted Brighton near relegation risk (they finished a comfortable 8th)

This told me something specific and useful: the model is good at telling 
the very best and very worst teams apart, but can't distinguish quality 
among the 10-15 teams in between — which matches how football actually 
behaves. The middle of the table is genuinely closer and more volatile; 
that's not a bug, it's reality the model has to grapple with.

## A bug I found myself: Aston Villa
After pointing the model at the live, in-progress 2026/27 season, I 
noticed Aston Villa — a team with a strong recent track record — was 
being predicted to finish 16th. That looked wrong to me, based on 
watching the league for years, not just the numbers.

I dug into why: the model's "form" feature was a rolling average of only 
the last 5 matches. Villa had a rough opening month, and since it was 
early in the season, those 5 rough games were literally their entire 
recent history as far as the model could see — no memory of who they 
normally are.

The fix: I added last season's points-per-game as a separate, stable 
feature — giving the model an anchor for "this team's established level," 
independent of a small, noisy current-season sample. Villa's predicted 
average finishing position improved from 14.9 to 11.5, and their 
relegation risk dropped close to zero — a real, measurable improvement, 
even though their rank label stayed around 16th because of points 
already lost in real matches earlier in the season.

I also tried a more sophisticated version — blending last-season and 
current-season form with a decay weight, to model "regression to the 
mean" (the idea that strong teams with rough starts tend to recover). 
It pushed Villa's number even further, but it measurably hurt the 
model's overall accuracy on the test set (43.0% → 41.3%). I reverted it. 
Not every well-reasoned idea makes the model better — testing beats 
intuition, even your own.

## Fixing the model's blind spot for draws
The model was almost completely unable to predict draws — only 5-7% 
recall, meaning it correctly called a draw about once in every 15-20 
actual draws. I tackled this two ways at once: added a "quality gap" 
feature (the difference in both teams' last-season PPG — a small gap 
signals an even matchup, which tends to produce draws), and told the 
model during training to weight getting draws wrong more heavily.

Draw recall jumped from ~7% to 23% — a real, multi-times improvement — 
and overall accuracy improved alongside it, up to 43-45% depending on 
model type.

![Confusion Matrix and Feature Importance](confusion_matrix.png)

## Checking for the obvious failure mode: memorization
Before trusting any of this, I wanted to rule out the model just 
memorizing the training data rather than learning real patterns — the 
same risk that causes some medical imaging AI to fail (e.g. a skin 
cancer model that "learned" to detect rulers in photos instead of 
cancer). I compared training accuracy against test accuracy on data 
the model had never seen — a small gap meant it was generalizing, not 
memorizing.

## Honest limitations
- Final accuracy: ~43-45% on a 3-way prediction (vs ~33% random baseline) 
  — meaningfully better than chance, well short of "reliable"
- Strong at the extremes (title contenders, relegation zone), weak in 
  the mid-table — a limitation inherent to football's unpredictability, 
  not just this model
- Built on only ~2.5-3 seasons of data — not enough for the model to 
  build a strong independent sense of "team identity" without leaning on 
  recent, sometimes noisy, form
- No Expected Goals (xG), no injury data, no transfer/squad-spending data 
  — all known to meaningfully improve football prediction, out of scope 
  for this project's data sources

## Live, ongoing prediction
The model is currently projecting outcomes for the 2026/27 season, which 
is in progress as of this writing. Unlike the retrospective test, this 
is a genuine forecast with an unknown outcome — worth checking back on 
once the season ends.

## What I'd do with more time
- Expected Goals instead of raw goals (more stable signal, less 
  luck-driven)
- Head-to-head history between specific team pairs
- More historical seasons
- A "days since manager change" feature — a plausible explanation for 
  some of the mid-table misses
