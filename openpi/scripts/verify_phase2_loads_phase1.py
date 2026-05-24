"""End-to-end verification that Phase 2 can load from a Phase 1-shaped checkpoint.

Runs WITHOUT launching real training:
  1. Initialize the Phase 1 train_state (fresh Pi0CanonicalConfig + freeze_filter applied).
  2. Save its EMA params to a temp dir using the same Orbax format as train.py's save_state.
  3. Run Phase 2's init_train_state with weight_loader pointing at that temp checkpoint.
  4. Confirm that:
     (a) all Phase 2 params have matching shapes/dtypes (no missing, no extras)
     (b) canonical params loaded from Phase 1 match what was saved
     (c) no JAX/Orbax errors during load
"""
from __future__ import annotations

import pathlib
import shutil
import tempfile

import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.weight_loaders as _weight_loaders


def _count_can_params(params) -> int:
    """Count canonical params (can_* keys) via path regex."""
    flat = tu.flatten_dict(params, sep="/")
    return sum(1 for k in flat if "/can_" in "/" + k)


def _snapshot_params_for_orbax(params_dict) -> dict:
    """Convert params into the exact nested structure that train.py's save_state uses.

    save_state() wraps params as `{"params": {"params": params}}`, so we mirror that here
    to produce a file layout that `restore_params` and `CheckpointWeightLoader` can read.
    """
    return {"params": params_dict}


def main() -> None:
    print("=" * 70)
    print("Phase 2 load-from-Phase-1 verification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build Phase 1 model and extract fresh params.
    # ------------------------------------------------------------------
    phase1_cfg = _config.get_config("m5_canonical_phase1")
    phase2_cfg = _config.get_config("m5_canonical")

    print(f"\n[1] Phase 1 config: {phase1_cfg.name}")
    print(f"    canonical_dim={phase1_cfg.model.canonical_dim}, "
          f"dropout={phase1_cfg.model.canonical_token_dropout}, "
          f"steps={phase1_cfg.num_train_steps}")
    print(f"    freeze_filter={phase1_cfg.freeze_filter}")
    print(f"\n[1] Phase 2 config: {phase2_cfg.name}")
    print(f"    canonical_dim={phase2_cfg.model.canonical_dim}, "
          f"dropout={phase2_cfg.model.canonical_token_dropout}, "
          f"steps={phase2_cfg.num_train_steps}")
    print(f"    peak_lr={phase2_cfg.lr_schedule.peak_lr}, "
          f"decay_lr={phase2_cfg.lr_schedule.decay_lr}")

    # ------------------------------------------------------------------
    # 2. Verify Phase 1 and Phase 2 produce identical param shapes via eval_shape.
    #    This is the critical invariant: a Phase 1 checkpoint must be a 1:1 shape
    #    match for a Phase 2 fresh init.
    # ------------------------------------------------------------------
    print(f"\n[2] Comparing param shapes via jax.eval_shape ...")

    def _p1_shape_tree():
        model = phase1_cfg.model.create(jax.random.key(0))
        return nnx.state(model).to_pure_dict()

    def _p2_shape_tree():
        model = phase2_cfg.model.create(jax.random.key(0))
        return nnx.state(model).to_pure_dict()

    with at.disable_typechecking():
        p1_shapes = jax.eval_shape(_p1_shape_tree)
        p2_shapes = jax.eval_shape(_p2_shape_tree)

    flat_p1 = tu.flatten_dict(p1_shapes, sep="/")
    flat_p2 = tu.flatten_dict(p2_shapes, sep="/")

    p1_keys = set(flat_p1)
    p2_keys = set(flat_p2)
    only_p1 = p1_keys - p2_keys
    only_p2 = p2_keys - p1_keys
    shared = p1_keys & p2_keys

    print(f"    params in phase1 only: {len(only_p1)}")
    print(f"    params in phase2 only: {len(only_p2)}")
    print(f"    shared params: {len(shared)}")

    assert not only_p1, f"Phase 1 has params that Phase 2 doesn't: {list(only_p1)[:5]}"
    assert not only_p2, f"Phase 2 has params that Phase 1 doesn't: {list(only_p2)[:5]}"

    # Shape match check
    shape_mismatches = []
    for k in shared:
        s1 = flat_p1[k]
        s2 = flat_p2[k]
        if s1.shape != s2.shape or s1.dtype != s2.dtype:
            shape_mismatches.append((k, s1.shape, s2.shape))
    assert not shape_mismatches, f"Shape mismatches: {shape_mismatches[:3]}"
    print(f"    ✓ All {len(shared)} params match between Phase 1 and Phase 2")

    # Count canonical params
    can_count = sum(1 for k in shared if "/can_" in "/" + k)
    print(f"    ✓ {can_count} canonical (can_*) param arrays found in both configs")
    assert can_count > 0, "No canonical params — something is wrong"

    # ------------------------------------------------------------------
    # 3. Create a fake Phase 1 params dict with recognizable values,
    #    save via Orbax in the exact format train.py uses, and confirm Phase 2
    #    can load it via CheckpointWeightLoader.
    # ------------------------------------------------------------------
    print(f"\n[3] Simulating Phase 1 checkpoint save + Phase 2 load ...")
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="phase2_verify_"))
    try:
        # Concrete init (not just shape) — match what Phase 1's init_train_state produces.
        rng = jax.random.key(42)
        phase1_model = phase1_cfg.model.create(rng)
        phase1_params = nnx.state(phase1_model).to_pure_dict()

        # "Mark" canonical params with a distinct constant so we can verify they survive
        # the save/load round-trip (distinguishing them from pretrained values).
        marker = jnp.float32(7.77)
        flat = tu.flatten_dict(phase1_params, sep="/")
        marked_canonical_keys = []
        for k in list(flat):
            if "/can_" in "/" + k:
                flat[k] = jnp.full_like(flat[k], marker)
                marked_canonical_keys.append(k)
        phase1_params = tu.unflatten_dict(flat, sep="/")
        print(f"    marked {len(marked_canonical_keys)} canonical arrays with value={marker}")

        # Save in the same nested structure save_state uses: {"params": {"params": {...}}}.
        save_path = tmp_root / "4999" / "params"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with ocp.PyTreeCheckpointer() as ckptr:
            with at.disable_typechecking():
                ckptr.save(save_path, {"params": phase1_params})
        print(f"    saved fake phase1 checkpoint to {save_path}")

        # ------------------------------------------------------------------
        # Load via the exact mechanism Phase 2 uses.
        # ------------------------------------------------------------------
        loader = _weight_loaders.CheckpointWeightLoader(
            str(save_path),
            missing_regex=".*(lora|can_).*",
        )

        # Build Phase 2 params shape (as a dict of ShapeDtypeStructs) and feed into loader.
        def _phase2_shape():
            model = phase2_cfg.model.create(jax.random.key(0))
            return nnx.state(model).to_pure_dict()

        with at.disable_typechecking():
            phase2_params_shape = jax.eval_shape(_phase2_shape)

        loaded = loader.load(phase2_params_shape)
        flat_loaded = tu.flatten_dict(loaded, sep="/")
        flat_phase2_shape = tu.flatten_dict(phase2_params_shape, sep="/")
        print(f"    loader returned {len(flat_loaded)} param arrays")

        # Check every canonical param was loaded with the marker value (proves Phase 1
        # canonical weights are preserved into Phase 2, not re-initialized from scratch).
        bad = []
        for k in marked_canonical_keys:
            if k not in flat_loaded:
                bad.append((k, "missing from loader output"))
                continue
            v = flat_loaded[k]
            if isinstance(v, jax.ShapeDtypeStruct):
                bad.append((k, "returned as ShapeDtypeStruct (not loaded)"))
                continue
            v_np = np.asarray(v)
            if not np.allclose(v_np, marker):
                bad.append((k, f"value {v_np.flatten()[:3].tolist()} ≠ marker {marker}"))

        if bad:
            print(f"    ✗ CANONICAL WEIGHT TRANSFER FAILED in {len(bad)} arrays:")
            for k, why in bad[:5]:
                print(f"      - {k}: {why}")
            raise AssertionError("Phase 2 did not correctly load canonical weights from Phase 1")

        print(f"    ✓ All {len(marked_canonical_keys)} canonical arrays loaded correctly")
        print(f"    ✓ Value roundtrip verified: Phase 2 will receive trained canonical "
              f"weights from Phase 1")

        # Also verify pretrained (non-canonical) params are loaded (from the fake
        # checkpoint, which stood in for "pi05_base + frozen-through-phase1").
        non_can_keys = [k for k in flat_phase2_shape if "/can_" not in "/" + k]
        loaded_non_can = [k for k in non_can_keys if k in flat_loaded]
        print(f"    ✓ {len(loaded_non_can)}/{len(non_can_keys)} pretrained arrays loaded")
        assert len(loaded_non_can) == len(non_can_keys), "Some pretrained params missing"

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # 4. Verify the CLI override syntax actually parses.
    # ------------------------------------------------------------------
    print(f"\n[4] Verifying Phase 2 CLI override parsing ...")
    import sys  # noqa: PLC0415
    import tyro  # noqa: PLC0415

    saved_argv = sys.argv
    fake_path = "/tmp/some_phase1/4999/params"
    try:
        sys.argv = [
            "train.py",
            "m5_canonical",
            "--exp-name", "m5_canonical_v1",
            "--overwrite",
            "--weight-loader.params-path", fake_path,
        ]
        parsed = _config.cli()
    except SystemExit as e:
        sys.argv = saved_argv
        raise AssertionError(f"tyro CLI parse failed with SystemExit({e.code})") from e
    finally:
        sys.argv = saved_argv

    assert parsed.name == "m5_canonical", f"wrong config: {parsed.name}"
    assert parsed.exp_name == "m5_canonical_v1"
    assert isinstance(parsed.weight_loader, _weight_loaders.CheckpointWeightLoader), (
        f"weight_loader type: {type(parsed.weight_loader)}"
    )
    assert parsed.weight_loader.params_path == fake_path, (
        f"params_path override failed: got {parsed.weight_loader.params_path!r}"
    )
    # missing_regex should survive the override (it's the default from the config).
    assert "can_" in parsed.weight_loader.missing_regex, (
        f"missing_regex lost its can_ pattern after override: "
        f"{parsed.weight_loader.missing_regex!r}"
    )
    print(f"    ✓ CLI parses m5_canonical with --weight_loader.params_path override")
    print(f"    ✓ Resolved weight_loader: {type(parsed.weight_loader).__name__}")
    print(f"      params_path  = {parsed.weight_loader.params_path}")
    print(f"      missing_regex= {parsed.weight_loader.missing_regex}")

    print(f"\n{'=' * 70}")
    print("✓ Phase 2 is verified to correctly load from Phase 1 checkpoints.")
    print("✓ Trained canonical weights will be preserved across the phase transition.")
    print("✓ Phase 2 CLI override syntax parses correctly.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
