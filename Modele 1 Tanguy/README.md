# Model 1: dead ball first, possession second

**Earlier exploratory pipeline. This is not the submitted inference model.** Use [the retained model](../modele-bnz/README.md) and [submission guide](../SUBMISSION_BNZ.md) for the deliverable.

Commands below describe the experiment and may train a separate cached LightGBM model when none exists. They are not required for submission reproduction.

```bash
python "Modele 1 Tanguy/pipeline.py"                        # default hybrid preset
python "Modele 1 Tanguy/pipeline.py" --dead all --no-write  # compare presets without writing
python "Modele 1 Tanguy/pipeline.py" --segments             # list predicted D phases
python "Modele 1 Tanguy/eval_dead.py"                       # dead-ball diagnostics alone
python "Modele 1 Tanguy/pipeline.py" --dead rules           # rules without the learned model
python "Modele 1 Tanguy/dead_model.py" --cv                 # external DFL cross-validation
```

## Components

| File | Role |
|---|---|
| `data_io.py` | Tracking cache, Smart Tagging and public ground-truth loading |
| `audio.py` | Per-video-frame 3–5 kHz tactical-audio spectrum |
| `context.py` | Lazily cached `MatchContext` features: speed, in-pitch ball and audio tones |
| `dead_ball.py` | Stop/restart detectors and ALIVE↔DEAD state machine |
| `dead_model.py` | Player-only LightGBM P(dead) trained on external DFL matches |
| `external/` | Dataset downloader and DFL reader; datasets remain outside Git |
| `possession.py` | Interchangeable H/A models; nearest-player baseline with alive-phase voting |
| `pipeline.py` | Assembly, presets and CLI; D overrides H/A |
| `evaluate.py` | Official public-window scores and unlabelled full-match D diagnostics |
| `eval_dead.py` | Frame/phase D diagnostics and full-match descriptive checks |
| `visualisation/video_labels.py` | Local video overlays of truth, predictions, tracking and whistles |
| `visualisation/ball_presence.py` | Local ball-observation timeline plot |

Never redistribute tactical footage or derived tracking visualizations. Outputs are intended for local inspection outside the repository.

## Dead-ball state machine

**Stops.** `OutOfBoundsStop` detects a ball outside the lines. `WhistleStop`, a `RefereeStop`, detects an in-pitch referee whistle. Add a `StopDetector` subclass with `detect(ctx)` to introduce another source.

**Restarts.** Choose the earliest of a tagged set piece or a whistle followed by a kick. `SmartTaggingRestart` accepts throw-in, corner, goal-kick, free-kick and kickoff tags appearing after a stop. Tags are fixed 35-second windows: `FirstKick` uses the first in-pitch ball movement above 4 m/s for three frames; `TagStart` uses the tag start literally. `WhistleRestart` requires a whistle at least three seconds after the stop and a kick within five seconds, ignoring whistles without subsequent kicks.

**Fallback.** If neither source finds a restart within 45 seconds, `KickRestart` searches for an untagged kick at least three seconds after the stop. During at least 80% of the preceding 0.6 seconds, the ball must have been slow (<2 m/s) or unseen, distinguishing a placed/held ball from an ordinary pass. Historical public-window diagnostics were Spell F1 0.400 with this condition, 0.380 without it and 0.271 without fallback, when D occupied 63% of the match. These exploratory observations are not model-selection evidence for the retained submission.

**Half opening.** Use the `Kick Off (Start)` tag, rather than whistles. The second graded half starts during warmup, before actual kickoff, with whistles and a tracked warmup ball.

## External DFL LightGBM (`dead_model.py`, hybrid preset)

The public window contains only four D phases and must not be used to train or tune. The external DFL dataset (Bassek et al., 2025, *Scientific Data*, figshare 28196177, CC BY 4.0) supplies seven corrected 25 Hz Bundesliga matches with ball status, roughly 700 D phases.

The model uses no images and no ball features. About 170 player-derived columns describe speeds, stationary/sprinting fractions, dispersion, inter-team distances, sideline/box/midfield counts and contextual means over 2/10/30-second windows and ±10-second offsets. Columns become per-half ranks. Output is P(dead) per frame.

The corrected DFL ball is frozen in stoppages: using it creates a shortcut absent from automatic challenge tracking (historical target-window D F1 0.40). Ball exits and kicks remain rule inputs. Rank normalization compensates for different player-speed scales (2.1 versus 2.4 m/s during play); without it the exploratory model predicts D on 64% of the match. Training removes 7% of positions in roughly five-second gaps, adds 20 cm noise and recomputes speed. Repeated deterministic fits produced identical predictions.

Leave-one-match-out DFL validation reported alive/dead F1 0.928 and predicted D shares of 43–47%, versus 41–52% in the references. These are external-data diagnostics, separate from the retained network's evaluation.

`ModelStop` requires at least three seconds of predicted D; in `hybrid`, the ball must also be unseen for half of the surrounding ±2 seconds. `ModelRestart` requires confirmation of the stop within 10 seconds and snaps model phase end to a kick in a −2/+1-second interval.

Historical public-window inspection found a rules stop ending 15 seconds late at frames 15889–16400 versus 0.3 seconds late with hybrid. A full-match interval at video time 105:12–108:43 had unseen ball and mean player speed 0.66 m/s, followed by a post-goal kickoff; rules labelled only 34% D. These unlabelled full-match observations are not private-ground-truth measurements. Public Spell F1 nevertheless decreased from 0.400 to 0.390 as H/A fragmentation grew (23 rather than 21 predictions, 17 true possessions).

### Confirm weak evidence

`ModelConfirmedStop` and `ModelConfirmedRestart` require player-model agreement for uncertain cues:

- A whistle shorter than 0.3 seconds or weaker than 20 dB needs model D within five seconds to retain the whistle time; confirmation at five to 15 seconds uses model-transition time. Otherwise ignore it.
- A visible line crossing, or a throw-in/corner/goal-kick tag within 30 seconds, makes an exit trusted. Otherwise require the same model confirmation as a weak whistle.
- A tag/whistle/kick restart needs model alive within three seconds; otherwise try the next proposal.

Historical model confirmation was 90–95% for clear whistles and 60–67% for weak ones. Public-window alive/dead F1 moved from 0.977 to 0.988 and the third D phase began 1.8 rather than 7.3 seconds early. Strong rules/model disagreement fell from 653 to 274 seconds over the match (`rules`: 1389 seconds); this is not independent of the model. Checks with two- to five-second confirmation windows were stable. An oracle public D diagnostic still gave Spell F1 0.390, suggesting H/A segmentation was limiting that experiment. None of these target-window diagnostics may be used to select the submitted model.

Fetch external assets separately with `bash "Modele 1 Tanguy/external/download_external.sh"`, then use `dead_model.py` if reproducing this experiment. Additional dependencies include `lightgbm` and `lxml`. Do not execute this path merely to reproduce the submission.

## Whistle and missing-ball observations

Tactical audio is synchronized with video frames. Whistle detection searches simultaneous tones around 3.7–3.9 and 4.2–4.3 kHz, separated by 400–520 Hz, with harmonics near 8 kHz. Thresholds are 15 and 12 dB above 3–5 kHz background for at least five frames. The experiment detected 88 whistles. Some stoppage whistles were inaudible, including the five tagged free kicks. Whistles can also authorize restarts, so a subsequent kick is required. Unlabelled player-speed changes after whistle-plus-kick restarts were +1.15 m/s median, versus +1.28 after tagged restarts and −0.30 after whistles without kicks.

Ball tracking is absent on 41% of graded frames, including roughly 29% of alive frames in the public diagnostic. Typical gaps are long (10.7 seconds median, none between one and three seconds); the far touchline is difficult. First-half missing-ball share grew from 19% to 64%. These describe observations, not private target labels.

## Historical exploratory presets

The following public-window diagnostics are retained as historical observations only. They must not be interpreted as an authorised tuning protocol or the submitted model's quality.

| Preset | Behavior | Public Spell / macro / alive-dead F1 | Full-match D | D phases >90 s | Tags covered |
|---|---|---|---|---|---|
| `hybrid` (legacy default) | Rules plus DFL model; weak-cue confirmation | 0.381 / 0.886 / 0.988 | 47.1%, 89 phases | 4 | 91% |
| `rules` | Exit/whistle stops; tagged/whistle-plus-kick restarts; fallback | 0.400 / 0.822 / 0.944 | 43.1%, 82 phases | 4 | 79% |
| `whistle_stop` | No restart-whistle source | 0.400 / 0.822 / 0.944 | 43.9%, 81 phases | 5 | 79% |
| `no_whistle` | Tagged kicks plus fallback | 0.343 / 0.643 / 0.914 | 23.0%, 42 phases | 2 | 51% |
| `tag_kick` | D until a kick in the next tag | 0.343 / 0.643 / 0.914 | 35.8% | 6 | 61% |
| `literal` | D until next tag start | 0.343 / 0.629 / 0.911 | 44.7% | 10 | 72% |

No audible restart whistle occurs in the public five minutes. The tracking-only `KickRestart` fallback remains weak: its 30 proposed restarts were followed by only +0.17 m/s player acceleration.

## D-only diagnostics and rules

`eval_dead.py` separates frame metrics, D-phase IoU matching, stop/restart matching within ±1 second, and unlabelled full-match summaries (D share, number of phases, phases >90 seconds and set-piece coverage within ±5 seconds). D-phase F1 is a separate diagnostic, not official H/A Spell F1.

**The public window is only a pipeline check. Never train, calibrate or select models on it, and never add human annotations to the challenge game.** The legacy `--annotations` loader must not be used to add target-game labels. External labelled games require separate inputs and disclosure under the official rules. The final submission comes exclusively from the independent retained package, with external training sources described in its technical report.
