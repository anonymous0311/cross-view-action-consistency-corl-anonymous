# M7 Boundary Audit Report
_Generated from 3174 matched LeRobot episodes against 1599 reference configs._

## Validation

- Total episodes: 3174
- Matched: 3174
- Match distance: mean=10.55, median=10.30, p10=5.35, p90=15.25
- Confidence margin: mean=0.701, median=0.127, p10=0.019, p90=2.570
- Horizon coverage: 145 / 150 unique values used
- Scale coverage: 81 / 84 unique values used

### Per-suite

- libero_10: 550 episodes
- libero_goal: 856 episodes
- libero_object: 904 episodes
- libero_spatial: 864 episodes

## Boundary candidates

| Name | h_abs | scale_min | er_abs | ev_abs | n_ep | frac | n_cells | min/med/max ep/cell | GATE |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| B-axis-strict | 75 | 195 | 10 | 10 | 339 | 10.7% | 46 | 2/6.0/16 | **PASS** |
| B-axis-lenient | 70 | 180 | 8 | 8 | 746 | 23.5% | 108 | 1/6.0/16 | **PASS** |
| B-axis-permissive | 60 | 150 | 6 | 6 | 1298 | 40.9% | 196 | 1/6.0/20 | FAIL |

### B-axis-strict — detail

- Episodes: 339 / 3174 = 10.7%
- Cells: 46
- ep/cell: min=2, median=6.0, max=16
- Axis breakdown: end_vert=119, end_rot=102, horizon=49, scale=46, end_rot+end_vert=23
- Top 10 cells (by episode count):
  - h=0, v=0, s=100, er=-8, ev=10: 16 ep
  - h=75, v=15, s=100, er=0, ev=0: 14 ep
  - h=0, v=0, s=200, er=0, ev=0: 13 ep
  - h=75, v=0, s=100, er=0, ev=0: 13 ep
  - h=0, v=0, s=100, er=-4, ev=10: 12 ep
  - h=-75, v=15, s=100, er=0, ev=0: 12 ep
  - h=0, v=0, s=100, er=6, ev=10: 12 ep
  - h=0, v=0, s=100, er=-10, ev=4: 10 ep
  - h=0, v=0, s=100, er=4, ev=10: 10 ep
  - h=0, v=0, s=100, er=8, ev=10: 10 ep

### B-axis-lenient — detail

- Episodes: 746 / 3174 = 23.5%
- Cells: 108
- ep/cell: min=1, median=6.0, max=16
- Axis breakdown: end_vert=178, horizon=173, end_rot=143, scale=138, end_rot+end_vert=114
- Top 10 cells (by episode count):
  - h=0, v=0, s=100, er=-8, ev=10: 16 ep
  - h=0, v=0, s=100, er=-8, ev=-8: 16 ep
  - h=0, v=0, s=180, er=0, ev=0: 16 ep
  - h=0, v=0, s=190, er=0, ev=0: 16 ep
  - h=75, v=15, s=100, er=0, ev=0: 14 ep
  - h=0, v=0, s=200, er=0, ev=0: 13 ep
  - h=70, v=15, s=100, er=0, ev=0: 13 ep
  - h=75, v=0, s=100, er=0, ev=0: 13 ep
  - h=0, v=0, s=100, er=-4, ev=10: 12 ep
  - h=0, v=0, s=100, er=4, ev=-8: 12 ep

### B-axis-permissive — detail

- Episodes: 1298 / 3174 = 40.9%
- Cells: 196
- ep/cell: min=1, median=6.0, max=20
- **GATE FAILS**: boundary fraction 40.9% > 30%
- Axis breakdown: horizon=385, scale=331, end_rot+end_vert=242, end_vert=195, end_rot=145
- Top 10 cells (by episode count):
  - h=0, v=0, s=100, er=4, ev=6: 20 ep
  - h=0, v=0, s=100, er=-8, ev=-8: 16 ep
  - h=0, v=0, s=160, er=0, ev=0: 16 ep
  - h=0, v=0, s=100, er=-8, ev=10: 16 ep
  - h=0, v=0, s=180, er=0, ev=0: 16 ep
  - h=0, v=0, s=190, er=0, ev=0: 16 ep
  - h=0, v=0, s=170, er=0, ev=0: 14 ep
  - h=75, v=15, s=100, er=0, ev=0: 14 ep
  - h=0, v=0, s=100, er=2, ev=-6: 14 ep
  - h=75, v=0, s=100, er=0, ev=0: 13 ep

## Chosen BOUNDARY_DEFINITION

**B-axis-strict** — thresholds: `|horizon| >= 75°` OR `scale >= 195` OR `vertical == 15°` OR `|end_rot| >= 10°` OR `|end_vert| >= 10°`

Selected: 339 episodes (10.7% of full), 46 unique cells.
