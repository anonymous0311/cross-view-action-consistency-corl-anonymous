# Real-robot data and inference

The paper uses an RM-75 arm, a two-finger gripper, and three asynchronously recorded RealSense D435i scene cameras, C0–C2. The task is battery placement into a box. There are 98 demonstrations. C1 is the nominal front view; cross-view images are matched by nearest timestamp with a maximum offset of 67 ms. Both images inherit the same base-frame action label.

## Demonstration interchange format

Store one NPZ file per demonstration, with `allow_pickle=False` compatible arrays:

| Key | Shape / contents |
| --- | --- |
| `timestamps_C0`, `timestamps_C1`, `timestamps_C2` | Strictly increasing float64 seconds on a common clock |
| `images_C0`, `images_C1`, `images_C2` | Corresponding upright uint8 RGB arrays `[frames,height,width,3]` |
| `state` | `[N,8]` proprioception at C1 reference timestamps |
| `actions` | `[N,7]` commanded base-frame end-effector action and gripper label |
| `instruction` | Scalar Unicode task instruction |

This interchange adapter uses end-effector position (3), axis-angle orientation (3), and two gripper coordinates (2) for state. Convert robot logs into these channels consistently before export. Preserve the command units, gripper encoding, and base-frame convention when collecting and executing actions; compute normalization from the robot dataset. The exporter does not infer a coordinate transform from camera images.

For every C1 reference observation, the exporter chooses a valid nearest frame from C0 or C2 with a seeded draw. It drops a reference frame if neither partner meets the 67 ms threshold. It takes action chunks from the original C1-indexed action sequence before filtering, so dropped image pairs never shift action labels. Both baseline and consistency training use the same exported rows.

```bash
python scripts/data/export_robot_pairs.py \
  --input-dir data/robot_demonstrations --output-dir data/robot_pairs --seed 42
python scripts/compute_norm_stats.py --dataset data/robot_pairs --output-dir assets/robot
python scripts/train.py robot_nominal --exp-name seed42 --seed 42
python scripts/train.py robot_fm --exp-name seed42 --seed 42
python scripts/train.py robot_cv --exp-name seed42 --seed 42
python scripts/serve.py --config robot_eval \
  --checkpoint-dir checkpoints/robot_cv/seed42/10000
```

All three hardware recipes run 10,000 updates. Paired consistency uses `K=2`, `λ=0.10`, and `Beta(2,3)` flow times. The deployed policy receives only the chosen scene image, instruction, and proprioception. Connect the returned action chunk to the robot's existing controller using the same command representation as the logged labels. Robot-specific controller and camera drivers are not part of this package.

Evaluation uses the same ten object layouts at seven camera placements: three recorded, two between the recorded views, and two beyond the outer views. A ChArUco board is used for measurement of the layout only. Neither training nor inference takes calibration as input. No physical robot trial is part of the software execution check.
