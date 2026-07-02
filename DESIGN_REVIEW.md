# Foresee - Critical Design Review

> Foresee aims to be a safety tool - predict where road users are
> going and flag dangerous situations for an autonomous vehicle. A safety tool that is *wrong in
> ways its users can't see* is worse than no tool. This review catalogues every loophole that
> currently prevents Foresee from being trustworthy enough to "save lives," grades each by
> severity, and lays out a sequenced roadmap to fix them.
>
> Status: the system is a strong *engineering demo* (real AV2 data, end-to-end pipeline, a
> usable dashboard) but is **not yet a sound safety system.** The headline metrics flatter it.

---

## Update (Phase 2 substantially addressed)

Since the first version of this review, the keystone fixes have been implemented and measured:

- **Mode collapse (A2) - fixed** via **goal-anchored modes** ([`anchors.py`](src/foresee/anchors.py),
  [`models/anchored_transformer.py`](src/foresee/models/anchored_transformer.py)): each mode is
  bound to a maneuver-stratified k-means anchor (stop / left / right / straightx3) and ground
  truth is assigned to the nearest *anchor*, so modes specialise instead of all going straight.
- **Miscalibration (A3) - addressed** via post-hoc **temperature scaling** (fitted T saved in the
  checkpoint, applied at inference).
- **Predicted intent for surrounding vehicles (B1, partial)** - the forecaster is now re-run
  per-agent ([`agent_intent.py`](src/foresee/agent_intent.py)) to predict the intent of nearby
  vehicles, not only the focal.

| Metric | WTA (orig) | endpoint-diversity | **anchored** |
|---|---|---|---|
| Distinct maneuvers / 6 modes | 1.44 | 1.50 | **2.50+** |
| Top-1 mode selection accuracy | 28% | 36% | **66%** |
| Deployed top-mode ADE | 3.79 m | 3.43 m | **3.23 m** |
| Calibration temperature | - | 1.01 (no-op) | **1.95 (real)** |
| Oracle minADE (best-of-6) | 1.39 | 1.31 | 1.88 (the trade) |

The trade: anchored modes give up some **oracle** minADE for modes that are *meaningful and
selectable* - the right trade for a safety tool, since oracle minADE was the misleading metric.

**C1 (measuring missed dangers) is now also addressed** by the safety scoreboard
([`analysis/safety_scoreboard.py`](analysis/safety_scoreboard.py)), run on the full 1,500-scenario
val split: 75 real focal-vs-ego near-miss events; the hand-tuned risk gate caught **16%** of them,
while a ground-truth-fitted threshold on the same predictions (predicted closest approach < 4 m)
reaches **93% recall @ 41% false-alarm rate**. It also *measured* B1's blind spot: **85% of all
real near-misses involve non-focal agents** the detector cannot see. B1 (multi-agent prediction)
is therefore the top remaining item; B4 (fitted thresholds) now has its data.

---

## 0. Original findings (historical - the Update above records what was since fixed)

At the time of the first review, the headline minADE 1.31 m was an oracle metric hiding a 3.79 m
deployed error; the 6 modes were near-duplicates of one maneuver; the confidences were close to
noise; and nothing measured missed dangers. Sections 1-2 preserve the evidence and the full
loophole catalogue, since they define what the later fixes were measured against.

---

## 1. Evidence (measured on 416 real AV2 validation scenarios)

| Probe | Result | Interpretation |
|---|---|---|
| Distinct maneuvers among the 6 modes | **1.44 avg**; 271/416 (65%) had **all 6 identical** | Multimodality is largely illusory - modes don't represent turn/straight/stop alternatives |
| Top-1 selection accuracy (argmax-prob == closest-to-truth) | **28%** | Random is ~17% (1/6). The confidence ranking carries little signal |
| ADE of the **most-likely** mode | **3.79 m** | The realistic, deployable accuracy |
| ADE of the **oracle** mode (min over 6) | **1.39 m** | The number we have been advertising |
| Avg probability mass on the actually-best mode | **0.20** | Barely above uniform (0.167) |
| Calibration (predicted mode prob -> empirical "is best") | 0.0-0.1->0.08, 0.1-0.2->0.14, 0.2-0.3->0.25, 0.5+->0.62 (n=13) | Weakly monotonic but the model almost never commits; "25%" != 25% chance |
| Conflict-flag precision vs ground truth (HIGH alerts) | ~**20-25%** came within danger range in reality | Many flags are "approaching a stopped AV" that resolved; some are model error |

Reproduce: `tools/audit_modes.py`-style probe (the script used is in the session log; it loads the
trained checkpoint, runs the val split, and compares per-mode ADE, maneuver labels, and
probability calibration).

**Root cause of #1-4:** the model trains with a **winner-takes-all (WTA)** objective
([`losses.py`](src/foresee/losses.py)). On each example, only the single closest mode receives a
regression gradient, and the confidence head is trained only to point at "which mode was
geometrically closest." Nothing in the objective **forces the modes to be diverse** or the
**probabilities to be calibrated**. The modes therefore collapse toward the dominant maneuver
(going straight), and the confidences are near-noise.

---

## 2. Loophole catalogue

Severity: **Critical** (unsafe to ship / invalidates core claim) | **High** (materially misleads) |
**Medium** (limits usefulness). Effort: **S** (hours) | **M** (days) | **L** (weeks).

### A. Soundness of the prediction

| # | Loophole | Severity | Effort | Why it matters | Fix |
|---|---|---|---|---|---|
| A1 | **Oracle vs deployed gap** - we report best-of-6 (1.31 m) but deployable accuracy is ~3.8 m | Critical | S | Every stakeholder is misled about how good the model is | Always report top-1 (deployed) ADE/FDE next to oracle; headline the honest number |
| A2 | ~~**Mode collapse**~~ **ADDRESSED** - goal-anchored modes; distinct maneuvers 1.44 -> 2.50, top-1 acc 28% -> 66% | ~~Critical~~ Done | M | The system can now represent turn / straight / stop and select the right one | Anchored modes ([`anchored_transformer.py`](src/foresee/models/anchored_transformer.py)) |
| A3 | ~~**Miscalibrated confidence**~~ **ADDRESSED** - temperature scaling (T=1.95) fitted on val, saved in checkpoint | ~~Critical~~ Done | M | Confidences are now calibrated and decisive | `calibrate_temperature` in [`train.py`](src/foresee/train.py) |
| A4 | **Constant-velocity extrapolation** beyond 6 s | Medium | S-M | Wrong exactly through turns/intersections where conflicts occur | Extrapolate with constant turn-rate (CTRV); or train a longer head on a dataset that has longer GT |

### B. Soundness of the risk logic

| # | Loophole | Severity | Effort | Why it matters | Fix |
|---|---|---|---|---|---|
| B1 | **Single-agent prediction** - only the focal agent is predicted | Critical | L | The real threat may be any of 20+ other agents; we are blind to them | Run the forecaster per-agent (batched) or adopt a joint multi-agent model; threat = worst over all agents |
| B2 | **Ego treated as known ground truth** - predicted-focal vs *logged* ego; model never conditions on the AV plan | High | M | Modes "ghost" through the AV (false conflicts); ignores that agents react to the AV | Condition prediction on the AV's planned path; or at least mark conflicts where the focal would have to pass *through* the AV as lower-confidence |
| B3 | **Summing redundant mode probabilities** as "risk" | High | S | With degenerate, miscalibrated modes the sum is not a probability | Define risk on calibrated quantities; e.g. expected danger under the (calibrated) mixture, or P(min-distance < d) with proper uncertainty |
| B4 | **Hand-tuned thresholds** (danger radii, sustained steps, HIGH/MED cutoffs) | High | M | Not validated against real outcomes; chosen to "look right" | Fit thresholds to maximise recall at a fixed false-alarm budget on labelled near-misses (see C1) |

### C. Soundness of evaluation

| # | Loophole | Severity | Effort | Why it matters | Fix |
|---|---|---|---|---|---|
| C1 | **We never measure missed dangers (false negatives)** | Critical | M | A safety screen is defined by what it *misses*. We measure none of it | Mine real near-miss events from GT (all agents vs ego), then report **recall** and false-alarm rate; ROC over thresholds |
| C2 | **minADE/minFDE are not safety metrics** | High | S | Low ADE != "caught the dangerous events" | Add safety metrics (miss-rate of true conflicts, lead-time of warnings) as the primary scoreboard |
| C3 | **In-distribution only** - tested on val of the same cities/distribution | Medium | M | No evidence of behaviour on rare/long-tail/adversarial cases - where crashes live | Curate a hard/rare scenario suite; report separately |

### D. Deployment realism

| # | Loophole | Severity | Effort | Why it matters | Fix |
|---|---|---|---|---|---|
| D1 | **Perfect perception assumed** - uses logged tracks, types, and headings | High | L | Real sensors have noise, latency, occlusion, missed/false detections | Inject perception noise / dropout in training and eval; test degradation |
| D2 | **Map fusion is shallow** - PointNet over lane *centerlines*; no lane connectivity, right-of-way, signals | Medium | M | The model can't reason about who legally yields, red lights, one-ways | Encode the lane graph (successors/neighbors), stop lines, crossings; richer map features |
| D3 | **Object type from logged labels** | Medium | S | In deployment, type is itself a noisy prediction | Train with type noise / "unknown"; don't over-rely on the type embedding |
| D4 | **Trained on 8k/199k scenarios** | Medium | S | Under-trained; more data will move all numbers | Scale to the full split once the objective is fixed (don't scale a broken objective) |

### E. Product / human-factors

| # | Loophole | Severity | Effort | Why it matters | Fix |
|---|---|---|---|---|---|
| E1 | **Over-warning erodes trust** - false alarms train reviewers to ignore alerts | High | - | Alarm fatigue is itself a safety failure | Tune to a defensible false-alarm budget (C1); show calibrated confidence, not raw sums |
| E2 | **Intent-ambiguity uses the same degenerate modes** | Medium | S | The ambiguity meter inherits the mode-collapse problem | Re-derive after A2/A3; validate it correlates with real intent uncertainty |
| E3 | **"Reason" (lane change/swerve) is a geometric heuristic, not learned** | Low | S | Can mislabel the cause | Validate against labelled maneuvers; treat as advisory |

---

## 3. What we should stop claiming today (honesty patch - zero retrain)

Independent of any retraining, the following should change immediately because they are accuracy
*of reporting*:

1. **Report deployed accuracy** (top-1 mode: ~3.8 m ADE) as the headline, with oracle minADE
   clearly labelled "oracle (best-of-6)."
2. **Caveat the risk %** in the dashboard: it is a relative score from an **uncalibrated** model,
   not a probability of collision.
3. **State the scope**: single-agent (focal-vs-AV) detector, offline audit, perfect-perception
   assumption.

These are documentation/label changes and belong in the README and the dashboard.

---

## 4. Sequenced remediation roadmap

The ordering is deliberate: **you cannot improve a safety system you cannot measure**, and you
should not scale or tune a broken objective.

**Phase 0 - Honesty (S, do now).** Implement §3. Stop the oracle-metric framing.

**Phase 1 - Build the safety scoreboard (C1, C2 | M).** Label real near-miss events from ground
truth (all agents vs the ego, over the 6 s horizon). Define a conflict event objectively. Measure
the *current* system's **recall** (fraction of real near-misses flagged) and false-alarm rate, with
an ROC across thresholds. This becomes the metric every later change is judged against.

**Phase 2 - Fix the predictions (A2, A3 | M).** Retrain with a diversity-preserving objective
(mixture-NLL, or WTA + a mode-diversity penalty / anchored modes) so the modes specialise, then
**calibrate** the confidences (temperature scaling, verified by a reliability diagram). Success =
top-1 accuracy and calibration both materially improve, measured against Phase 1.

**Phase 3 - Fix the risk logic (B3, B4 | M).** Redefine risk on the now-calibrated mixture
(e.g. probability that minimum predicted separation < threshold), and **fit** the thresholds to the
Phase 1 ROC at a chosen false-alarm budget.

**Phase 4 - Widen the scene (B1, B2 | L).** Predict all agents and condition on the AV plan, so
the threat can be any road user and modes stop ghosting through the AV.

**Phase 5 - Realism & scale (D1-D4, C3 | L).** Perception noise, richer map/lane-graph,
hard-scenario suite, full-dataset training.

---

## 5. Summary

Foresee today demonstrates the **machinery** of an AV safety system end-to-end on real data, and
the dashboard communicates it well. But as a *safety* claim it has three disqualifying gaps: the
**predictions are weaker than advertised** (oracle metric), the **confidences are not real
probabilities** (mode collapse + miscalibration), and we **do not measure missed dangers** at all.
None of these are fatal to the approach - they are a focused, ordered engineering program (Phases
0-3 are the high-leverage core). Until Phase 1 exists, every other number - including any future
improvement - is unverifiable.
