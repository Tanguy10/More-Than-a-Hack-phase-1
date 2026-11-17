# Initial dead-ball-first design

This historical proposal is separate from the retained submission in `modele-bnz`.

1. Detect whether the ball is alive; D overrides every other label.
2. During alive play, detect possession changes.
3. Without a possession change, retain the current team's H/A label.

Out-of-bounds ball positions offer a direct stop cue. Detecting throw-in timing, fouls and referee stoppages is harder and may require restart and whistle evidence. These ideas motivated the exploratory pipeline described in [README.md](README.md).
