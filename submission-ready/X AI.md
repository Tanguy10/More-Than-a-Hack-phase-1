# X AI — Football possession from tracking

We predict who has possession in each video frame: **H** for the home team, **A** for the away team, **D** for a dead ball and **X** outside the processed periods.

## Model

Our final model, **C**, combines two neural networks using player and ball tracking. One looks at each team as a group; the other looks at interactions between players and the ball. Both follow movement over time, using past and future frames.

The model also estimates missing ball positions, keeping track of uncertainty, and learns possession segments to help identify team changes. A missing ball does not automatically mean a stoppage.

We combine the predictions and use **Viterbi decoding** to choose a consistent sequence of labels. This reduces false team changes caused by noisy frames. A brief touch does not automatically start a new possession. The system uses tracking only, without video, audio or tactical tags.

## Training data and sources

We trained on other matches, using three datasets:

- **[IDSSE](https://doi.org/10.6084/m9.figshare.28196177)**: three German professional matches from the DFL and Sportec Solutions, with the dataset's possession and ball-status labels. Licence: CC BY 4.0.
- **[SkillCorner Open Data](https://github.com/SkillCorner/opendata)**: fourteen broadcast-tracking matches. We generated partial possession labels automatically, keeping labels where two estimates agreed. Licence: MIT.
- **[SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2)**, by Scott and colleagues: six training matches. We derived partial labels from the dataset's ball-contact and restart annotations. Licence: CC BY 4.0.

Unknown frames remained unlabelled. For the submitted pair, three other IDSSE matches were reserved for validation and evaluation. No labels from the challenge match were used for training or calibration.

## Choices and limitations

We compared approaches on six IDSSE matches using the organisers' **Spell F1** possession metric, keeping evaluated matches out of training. C and our earlier reference both scored about **0.69**. C was stronger with severely degraded tracking; the reference was stronger with clean tracking. We chose C as a robustness trade-off, without claiming an overall gain.

We also tried a player-based **LightGBM dead-ball detector** and separate Home/Away training. Neither improved the complete pipeline, so we kept C alone.

Short possessions, contested balls and missing or incorrect tracking remain difficult. The final choice followed comparisons including the public excerpt; that excerpt is not independent validation, and the private full-match score remains unknown.

## Reproduction

Trained weights and [installation instructions](https://github.com/Tanguy10/More-Than-a-Hack-phase-1/blob/main/model-c/README.md) are included. After installation, run from `model-c/` with the participant bundle:

```bash
python scripts/infer.py --data /path/to/bundle --weights weights --output possession.csv
```

We reproduced the submitted CSV exactly on CPU from the original XML/TXT tracking files. See the [verification record](https://github.com/Tanguy10/More-Than-a-Hack-phase-1/blob/main/model-c/validation/VERIFICATION.md).
