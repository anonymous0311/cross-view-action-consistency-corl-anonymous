# LIBERO pair-data Original LIBERO Rerender Scripts

LIBERO pair-data now uses the original LIBERO HDF5 archives, not the LIBERO-plus
LeRobot/RLDS export. The original HDF5 files contain per-frame
flattened MuJoCo states under `data/demo_*/states`, actions, robot states, and
task metadata. These are sufficient for exact same-state camera rerendering.

Audit the available original HDF5 suites:

```bash
.venv/bin/python scripts/libero_pair_data/audit_libero_hdf5_original.py \
  --libero-root data/libero_hdf5_original \
  --suite-dirs libero_spatial libero_object libero_goal libero_10 \
  --output-dir results/libero_pair_audit
```

If a suite is missing, download the original archives:

```bash
.venv/bin/python scripts/libero_pair_data/download_libero_hdf5_original.py \
  --download-dir data/libero_hdf5_original \
  --datasets all
```

The next LIBERO pair-data script should be
`scripts/libero_pair_data/render_libero_multiview_states.py`: render same-state
canonical/C1/C2/C3 image pairs from original LIBERO states while matching the
LIBERO-plus camera perturbation categories and proportions.
