# Possession model

## Inputs and output

The model processes 25 Hz tracking: metric player/ball positions, player speeds, team sides and observation masks. Missing values remain NaN until the encoders mask them. Ball absence does not force D.

Player identities, shirt numbers and roles are not features. XML identifiers assemble tracking channels; the networks receive geometry and team membership.

The model outputs H/A/D inside processed periods. X is written outside the challenge intervals. The CSV uses video frame indices `0..199318`, one row per video frame.

## Two complementary encoders

**Set-temporal.** An MLP encodes each player's position, speed, motion, relation to the ball and opponent proximity. Mean, maximum and attention pooling produce a team representation. This member computes motion from successive tracking slots.

**Relational-temporal.** Two spatial-attention blocks add interactions between players and a ball token. Motion uses soft geometric matching between same-team sets, with an unmatched option. Attention biases encode relative position, distance and team membership. Time since previous and until next ball observations is computed over a bounded 1,500-frame support.

Each member uses an eight-block residual TCN with 128 channels and dilations 1 through 128. It is noncausal, using past and future observations for offline analysis.

Home/away exchange is also evaluated at inference and remapped to symmetrize states and transitions. Float32 stored weights are converted to float64 for reference inference.

## Windows and decoding

Only the central 1,024 outputs of each window are retained. Margins are 512 frames for the set member and 544 for the relational member. The latter additionally receives 1,500 frames of ball-observation context on each side. Windows are padded with missing values at edges and never cross a period interruption.

Member distributions are averaged and calibrated with temperature 6. Transition probabilities are also averaged. Final evidence is rounded to nine decimals before Viterbi. The state-change cost is 2 and boundary-score weight 0.5. These settings are recorded in `weights/manifest.json` and checked when loading.

## Training and data

The supplied weights were trained on other matches:

| Member | IDSSE | SkillCorner | SoccerTrack |
|---|---|---|---|
| Set-temporal | 6 matches | 14 matches | — |
| Relational-temporal | 6 matches | 14 matches | 6 matches, 12 halves |

Relational training uses a 50/40/10 window mixture across those sources. Partial SkillCorner H/A references use a conditional likelihood; they do not impose a negative D target. Dense references supervise boundaries and segment losses. The set member uses the first two sources.

Augmentations include observation gaps, spatial noise, player masking, geometric reflections and team exchange. These are inference checkpoints: optimizer states and training trajectories are not included. The supplied installation and scripts regenerate predictions from those weights.

## Evaluation and limitations

Historical validation uses three match-separated folds and six IDSSE matches, each under clean, moderate-noise and severe-noise observations. Official Spell F1 averages possession-matching F1 over IoU 0.3/0.4/0.5/0.6/0.7.

| Historical development metric | Value |
|---|---:|
| Mean Spell F1 | 0.69312 |
| Clean Spell F1 | 0.77164 |
| Moderate Spell F1 | 0.69447 |
| Severe Spell F1 | 0.61326 |
| H/A/D macro F1 | 0.92502 |
| D F1 | 0.97858 |

The same development matches were reused for selection; the three conditions are not independent matches. Evaluation uses historical fold-excluding checkpoints, whereas final shipped weights use all six allowed matches. The model remains fragile on very short possessions and severely degraded observations. Auxiliary references are partial, and these scores do not measure challenge-match quality.

Two historical independent CPU reproductions generated the exact supplied 199,319-row file. See the [technical report](TECHNICAL_REPORT.md) for the later CPU comparison, finite-context and infinite-coordinate limitations, and [DATA.md](DATA.md) for sources and label provenance.
