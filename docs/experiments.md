# Experiment protocols

All paired comparisons use the same underlying demonstrations, action chunks, model initialization, and training budget. The consistency loss is averaged across the two flow samples, action horizon, and commanded dimensions. At inference all paired models use the ordinary single-view policy.

## Pair coverage

```bash
python scripts/data/select_consistency_episodes.py \
  --dataset data/libero_pairs --output data/episode_membership.json --seed 42
python scripts/train.py coverage_05 --exp-name seed42 --seed 42
python scripts/train.py coverage_25 --exp-name seed42 --seed 42
```

Selection is by whole episode, stratified by task. The 5% set is contained in the 25% set. All episodes continue to contribute paired flow matching; membership affects only the cross-view loss. The loss is normalized by the number of active pairs in a batch, with a zero-safe denominator. `paired_fm` and `paired_cv` supply the 0% and 100% endpoints. With 2,000 episodes, the two selected sets contain 100 and 500 episodes.

## Restricted camera support

```bash
python scripts/data/render_libero_multiview_states.py \
  --libero-root data/libero_original --output-dir data/rendered_restricted \
  --support restricted --seed 42
python scripts/data/export_to_lerobot.py \
  --manifest data/rendered_restricted/pairs.jsonl --output-dir data/libero_restricted
python scripts/train.py restricted_fm --exp-name seed42 --seed 42
python scripts/train.py restricted_cv --exp-name seed42 --seed 42
```

The second training seed is 43. Evaluate each trained policy on the full camera track using `scripts/evaluate.py`. `src/afcv/support.py` defines the support sampler and camera-band classifier. The nominal camera is included as the first view of every pair.

| Axis | Training support | Held-out interpolation | Extrapolation / transfer |
| --- | --- | --- | --- |
| Absolute azimuth | 0–20°, 30–62° | 21–29° | 63–75° |
| Distance scale | 100–128%, 142–175% | 129–141% | 176–200% |
| Signed endpoint angles | ±3°, ±5°, ±7° | ±4°, ±6° | ±8°, ±10° |
| Elevation | 0°, 8° | — | 15° |

Endpoint values apply to both endpoint coordinates. Compound azimuth/elevation conditions are kept separate when aggregating results. Camera specifications are retained in the simulator task names; interpret degrees with the signed modulo-360 convention used by `afcv.support.signed_angle`.

## Generated-view comparisons

The paper evaluates LVSM through [AnyCamVLA](https://github.com/heo0224/AnyCamVLA) and diffusion through [VISTA](https://github.com/s-tian/VISTA). Use the official generator implementations and their model assets to produce the requested target scene views. These generators are external dependencies and are not included in the policy package.

The paired exporter accepts generated views through the same renderer manifest: replace `img_b_path` with the generated target PNG, retaining `img_a_path`, source state identifiers, language, and action source. Keep images upright at 256×256. Use a separate manifest and output directory for each route. Export with `scripts/data/export_to_lerobot.py`, set `AFCV_DATASET` to that directory, and train `paired_fm` and `paired_cv` with seed 42. Report generation failures and fallback rates separately from policy success; do not change action labels to accommodate a generated image. Generation itself is not covered by the package's execution checks.

## Aggregation

For each seed, compute success as successful trials divided by all scheduled trials, including unsuccessful timeouts. Keep the 2,000-trial nominal and 4,797-trial camera denominators separate. Aggregate per-seed rates for mean and population standard deviation. Shard summaries contain their own numerator and denominator; sum those counts before computing a seed's overall rate. A limited `--max-tasks` evaluation is an execution check and is not a benchmark result.
