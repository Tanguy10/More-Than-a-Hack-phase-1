# Data and conventions

## Challenge input

The reader expects the More than a Hack 2027 participant bundle: tactical EPTS tracking and its XML metadata. The match is Barcelona–Athletic Club: H means home, A away. The pitch is 105.3 × 68 m and tracking is sampled at 25 Hz. Normalized coordinates are converted to metres without interpolation.

The reader checks one record per frame and rejects duplicate or out-of-contract indices. The publicly specified processed periods are Python half-open `[11185, 90652)` and `[110808, 190198)`. All other frames receive X.

Participant assets stay under Git-ignored `data/`. The reader uses only `_atd.xml` and `_atd.txt`. The supplied CSV contains predictions, with no coordinates or video data.

The [official repository](https://github.com/metrica-sports/more-than-a-hack-2027) defines the contract and evaluation. Included official code is identified by commit and hashes in `src/possession/_official/provenance.json`.

## IDSSE

M. Bassek et al., *An integrated dataset of synchronized spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025). [Paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177). Data owner: Deutsche Fußball Liga; collection by Sportec Solutions, TRACAB system. Licence: CC BY 4.0.

Six matches with source-provided H/A/D references are used. Speeds are converted from km/h to m/s and centred metric coordinates to a corner origin. Missing observations remain unknown, periods are separated, and labels are stored apart from features. The labels originate from the dataset's provided ball-status and possession information, rather than annotations of the challenge game by this team.

## SkillCorner

[SkillCorner Open Data](https://github.com/SkillCorner/opendata), revision `4340d274572876239c154c90bc507a9b3250a656`, MIT licence. The licence and provenance are preserved under `third_party/skillcorner/`.

Fourteen matches contribute to training. Only observations marked detected are used. Data is resampled from 10 to 25 Hz without interpolating across gaps or periods. Auxiliary H/A references are the consensus of two automatic proxies; possession and phase fields are not input features. Unlabelled frames remain unknown. These are algorithmic proxy labels, not dense human annotations by this team.

## SoccerTrack

Scott, Uchida, Kuroda, Kim and Fujii (2025), [SoccerTrack paper](https://arxiv.org/abs/2508.01802), [official dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2), revision `325bd4c57ee8e135ab652f3921d274526ada496b`, data licence CC BY 4.0.

The six matches in the official training split are used. Event-reconstructed ball positions are excluded from input. Player positions, temporary teams and velocities derived over 400 ms are retained. Source contact events yield narrow H/A references; reliable out/restart pairs yield partial D interiors. Other frames remain unknown. These derived references are not dense possession labels and were not manually added to the target game.

Training datasets are not redistributed with the weights. The target-match labels were used neither for learning nor for selecting the model. Match identifiers and checkpoint hashes are recorded in `weights/manifest.json`.
