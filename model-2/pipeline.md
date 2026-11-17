# Model 2: historical pipeline proposal

**Design note, not the submitted configuration.** The retained model is documented in [the model README](../modele-bnz/README.md). This proposal predates the completed H/A experiments and should not be read as their implemented outcome.

External open-access matches were identified for learning because the provided target-game labels are not an authorised training set. The proposed design separates rule-based dead ball from independently trained H/A possession, with D taking priority.

Inputs: tactical player/ball tracking, tactical-video audio, Smart Tagging and the published half boundaries. Output: H/A/D per processed frame, with X outside periods.

## 1. Compute signals once

- **Ball:** speed and in/out-of-pitch status when observed.
- **Whistles:** simultaneous tones near 3.8 and 4.25 kHz above crowd noise for at least 0.2 seconds.
- **Kicks:** in-pitch ball accelerating above 4 m/s. A stationary-ball kick follows a slow or unseen ball, consistent with a placed or held ball.
- **Set pieces:** throw-in, corner, goal-kick, free-kick and kickoff tags.

## 2. Stop and restart events

A stop is a line crossing or an in-pitch referee whistle. If the ball disappears at the line and reappears outside, backdate to the last observation under the detector's gap limit.

Given a stop at `since`, a restart candidate is:

- the first kick in a new set-piece tag, or its start if no kick is found;
- a kick within five seconds after a whistle at least three seconds after the stop;
- as fallback when neither source arrives within 45 seconds, a stationary-ball kick at least three seconds after the stop.

Whistles without subsequent kicks do not count as restarts.

## 3. ALIVE/DEAD state

```text
FOR each half:
    state = DEAD
    FOR each frame f:
        IF state == DEAD:
            IF kickoff (Kick Off (Start) tag + first kick) OR restart:
                state = ALIVE
                last_restart = f
        ELSE IF stop AND f >= last_restart + 1 second:
            state = DEAD
            since = f
        dead[f] = (state == DEAD)
```

The one-second refractory interval avoids immediate false stops after throw-ins. Without a restart proposal, D lasts to half end. The implementation can jump between stop/restart events instead of checking every frame.

## 4. Proposed H/A learner

The proposal was to train on external IDSSE, SkillCorner and SoccerTrack data and predict only H/A on alive frames. D would come from rules. The retained `modele-bnz` reference instead predicts joint H/A/D; the completed specialized experiments are described in [EXPERIMENTS.md](../modele-bnz/docs/EXPERIMENTS.md).

The proposed input is a tracking window: player positions, velocities and teams, observed ball positions and time since observation, without player identity. For each alive frame, output P(home rather than away).

```text
probability[f] = POSSESSION_MODEL(tracking around f)
FOR each ALIVE interval between DEAD phases:
    choose the H/A sequence fitting probabilities
    while penalizing team changes (Viterbi)
```

No possession crosses a stoppage. Team-change cost suppresses flicker.

## 5. Output and limitations

```text
label[f] = D if dead[f] else possession[f]
```

This is offline processing: future tags, post-whistle kicks, a 45-second fallback horizon and model context are available. A missing ball at restart can hide the kick and leave D active too long. Algorithmic tracking completion could be investigated; manual target-game corrections are prohibited. These are design hypotheses, not validated improvements or the final submission recipe.
