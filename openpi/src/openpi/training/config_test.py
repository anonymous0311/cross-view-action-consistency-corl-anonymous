import dataclasses

from openpi.training import config as _config


def test_phase0b_cv010_no_spatial_aug_matches_cv010_except_pair_aug_and_steps():
    baseline = _config.get_config("pi05_v4_pair_cv010")
    no_spatial = _config.get_config("pi05_v4_pair_cv010_no_spatial_aug")

    excluded_config_fields = {"name", "model", "num_train_steps", "policy_metadata"}
    for field in dataclasses.fields(baseline):
        if field.name not in excluded_config_fields:
            assert getattr(no_spatial, field.name) == getattr(baseline, field.name)

    excluded_model_fields = {"pair_spatial_aug_mode", "pair_photometric_aug_mode"}
    for field in dataclasses.fields(baseline.model):
        if field.name not in excluded_model_fields:
            assert getattr(no_spatial.model, field.name) == getattr(baseline.model, field.name)

    assert no_spatial.model.lambda_cv == 0.10
    assert no_spatial.model.cv_pair_mode == "matched"
    assert baseline.model.pair_spatial_aug_mode == "current"
    assert baseline.model.pair_photometric_aug_mode == "current"
    assert no_spatial.model.pair_spatial_aug_mode == "none"
    assert no_spatial.model.pair_photometric_aug_mode == "independent"

    assert no_spatial.num_train_steps == 10_000
    assert no_spatial.model.total_train_steps == 10_000


def test_phaseb_multi_sample_asymmetric_cross_view010_config_is_primary_cv010():
    matched = _config.get_config("pi05_v4_pair_multi_sample_asymmetric_cross_view010")
    clean_wrong = _config.get_config("pi05_v4_clean_wrong_multi_sample_asymmetric_cross_view010")

    for cfg in (matched, clean_wrong):
        assert cfg.model.lambda_cv == 0.10
        assert cfg.model.cv_loss_mode == "multi_sample_asymmetric"
        assert cfg.model.cv_num_samples == 2
        assert cfg.model.cv_stopgrad_anchor is True
        assert cfg.model.cv_time_distribution == "legacy"
        assert cfg.model.cv_eps_shared_across_views is True
        assert cfg.model.cv_average_over_samples is True
        assert cfg.model.pair_spatial_aug_mode == "none"
        assert cfg.model.pair_photometric_aug_mode == "independent"
        assert cfg.num_train_steps == 10_000
        assert cfg.policy_metadata["primary_lambda"] is True

    assert matched.model.cv_pair_mode == "matched"
    assert clean_wrong.model.cv_pair_mode == "clean_wrong_batch_derangement"
    assert clean_wrong.policy_metadata["matched_reference"] == "pi05_v4_pair_multi_sample_asymmetric_cross_view010"


def test_phaseb_multi_sample_bilateral_cross_view010_only_changes_stopgrad():
    asymmetric = _config.get_config("pi05_v4_pair_multi_sample_asymmetric_cross_view010")
    bilateral = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010")

    assert bilateral.model.cv_loss_mode == "multi_sample_asymmetric"
    assert bilateral.model.cv_num_samples == 2
    assert bilateral.model.cv_stopgrad_anchor is False
    assert bilateral.policy_metadata["cv_gradient_mode"] == "bilateral"
    assert bilateral.policy_metadata["ablation_reference"] == asymmetric.name
    assert bilateral.num_train_steps == 10_000

    excluded_model_fields = {"cv_stopgrad_anchor"}
    for field in dataclasses.fields(asymmetric.model):
        if field.name not in excluded_model_fields:
            assert getattr(bilateral.model, field.name) == getattr(asymmetric.model, field.name)


def test_phaseb_b6b_action_biased_time_config_only_changes_time_from_bilateral():
    bilateral = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010")
    action_biased = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time")

    assert action_biased.model.cv_loss_mode == "multi_sample_asymmetric"
    assert action_biased.model.cv_num_samples == 2
    assert action_biased.model.cv_stopgrad_anchor is False
    assert action_biased.model.cv_time_distribution == "beta_2p0_3p0"
    assert action_biased.policy_metadata["phase_step"] == "B6b"
    assert action_biased.policy_metadata["cv_gradient_mode"] == "bilateral"
    assert action_biased.policy_metadata["cv_time_bias"] == "action_biased"
    assert (
        action_biased.policy_metadata["bilateral_legacy_time_reference"]
        == "pi05_v4_pair_multi_sample_bilateral_cross_view010"
    )
    assert action_biased.num_train_steps == 10_000

    excluded_model_fields = {"cv_time_distribution"}
    for field in dataclasses.fields(bilateral.model):
        if field.name not in excluded_model_fields:
            assert getattr(action_biased.model, field.name) == getattr(bilateral.model, field.name)


def test_phaseb_b6b_clean_wrong_only_changes_cv_pairing_from_matched_b6b():
    matched = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time")
    clean_wrong = _config.get_config("pi05_v4_clean_wrong_multi_sample_bilateral_cross_view010_action_biased_time")

    assert matched.model.cv_pair_mode == "matched"
    assert clean_wrong.model.cv_pair_mode == "clean_wrong_batch_derangement"
    assert clean_wrong.model.cv_loss_mode == "multi_sample_asymmetric"
    assert clean_wrong.model.cv_num_samples == 2
    assert clean_wrong.model.cv_stopgrad_anchor is False
    assert clean_wrong.model.cv_time_distribution == "beta_2p0_3p0"
    assert clean_wrong.model.cv_eps_shared_across_views is True
    assert clean_wrong.model.cv_average_over_samples is True
    assert clean_wrong.model.pair_spatial_aug_mode == "none"
    assert clean_wrong.model.pair_photometric_aug_mode == "independent"
    assert clean_wrong.policy_metadata["matched_reference"] == matched.name
    assert clean_wrong.policy_metadata["cv_gradient_mode"] == "bilateral"
    assert clean_wrong.policy_metadata["cv_time_bias"] == "action_biased"

    excluded_config_fields = {"name", "model", "policy_metadata"}
    for field in dataclasses.fields(matched):
        if field.name not in excluded_config_fields:
            assert getattr(clean_wrong, field.name) == getattr(matched, field.name)

    excluded_model_fields = {"cv_pair_mode"}
    for field in dataclasses.fields(matched.model):
        if field.name not in excluded_model_fields:
            assert getattr(clean_wrong.model, field.name) == getattr(matched.model, field.name)

    assert type(clean_wrong.data).__name__ == "LeRobotV4PairDataConfig"
    assert clean_wrong.data.repo_id == matched.data.repo_id


def test_phaseb_b6b_independent_ablation_train_configs_isolate_one_axis():
    baseline = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time")
    specs = {
        "pi05_v4_pair_multi_sample_stopgrad_cross_view010_action_biased_time": {
            "excluded_model_fields": {"cv_stopgrad_anchor"},
            "ablation_axis": "gradient_direction",
            "cv_stopgrad_anchor": True,
            "cv_num_samples": 2,
            "cv_time_distribution": "beta_2p0_3p0",
        },
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_k1_action_biased_time": {
            "excluded_model_fields": {"cv_num_samples"},
            "ablation_axis": "num_flow_samples",
            "cv_stopgrad_anchor": False,
            "cv_num_samples": 1,
            "cv_time_distribution": "beta_2p0_3p0",
        },
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_k4_action_biased_time": {
            "excluded_model_fields": {"cv_num_samples"},
            "ablation_axis": "num_flow_samples",
            "cv_stopgrad_anchor": False,
            "cv_num_samples": 4,
            "cv_time_distribution": "beta_2p0_3p0",
        },
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_uniform_time": {
            "excluded_model_fields": {"cv_time_distribution"},
            "ablation_axis": "time_distribution",
            "cv_stopgrad_anchor": False,
            "cv_num_samples": 2,
            "cv_time_distribution": "uniform",
        },
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_beta_1p0_1p5_time": {
            "excluded_model_fields": {"cv_time_distribution"},
            "ablation_axis": "time_distribution",
            "cv_stopgrad_anchor": False,
            "cv_num_samples": 2,
            "cv_time_distribution": "beta_1p0_1p5",
        },
    }

    for name, spec in specs.items():
        cfg = _config.get_config(name)

        assert type(cfg.data).__name__ == "LeRobotV4PairDataConfig"
        assert cfg.data.repo_id == baseline.data.repo_id
        assert cfg.model.lambda_cv == 0.10
        assert cfg.model.cv_loss_mode == "multi_sample_asymmetric"
        assert cfg.model.cv_stopgrad_anchor is spec["cv_stopgrad_anchor"]
        assert cfg.model.cv_num_samples == spec["cv_num_samples"]
        assert cfg.model.cv_time_distribution == spec["cv_time_distribution"]
        assert cfg.model.cv_eps_shared_across_views is True
        assert cfg.model.cv_average_over_samples is True
        assert cfg.model.pair_spatial_aug_mode == "none"
        assert cfg.model.pair_photometric_aug_mode == "independent"
        assert cfg.num_train_steps == 10_000
        assert cfg.policy_metadata["phase"] == "B-independent-ablation"
        assert cfg.policy_metadata["ablation_axis"] == spec["ablation_axis"]
        assert cfg.policy_metadata["ablation_baseline"] == baseline.name
        assert cfg.policy_metadata["cv_num_samples"] == spec["cv_num_samples"]
        assert cfg.policy_metadata["cv_time_distribution"] == spec["cv_time_distribution"]

        for field in dataclasses.fields(baseline.model):
            if field.name not in spec["excluded_model_fields"]:
                assert getattr(cfg.model, field.name) == getattr(baseline.model, field.name)


def test_phaseb_b6b_independent_ablation_eval_configs_use_single_view_eval_data():
    train_names = [
        "pi05_v4_pair_multi_sample_stopgrad_cross_view010_action_biased_time",
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_k1_action_biased_time",
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_k4_action_biased_time",
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_uniform_time",
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_beta_1p0_1p5_time",
    ]

    for train_name in train_names:
        train = _config.get_config(train_name)
        eval_cfg = _config.get_config(f"{train_name}_eval")

        assert type(eval_cfg.data).__name__ == "LiberoPhase0BEvalDataConfig"
        assert type(train.data).__name__ == "LeRobotV4PairDataConfig"
        assert eval_cfg.policy_metadata["eval_only"] is True
        assert eval_cfg.policy_metadata["train_config"] == train.name
        assert eval_cfg.policy_metadata["inference_inputs"] == "single_scene_rgb_language_state"
        assert eval_cfg.policy_metadata["ablation_axis"] == train.policy_metadata["ablation_axis"]

        for field in dataclasses.fields(train.model):
            assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_phaseb_multi_sample_asymmetric_eval_config_uses_single_view_eval_data():
    train = _config.get_config("pi05_v4_pair_multi_sample_asymmetric_cross_view010")
    eval_cfg = _config.get_config("pi05_v4_pair_multi_sample_asymmetric_cross_view010_eval")

    assert type(eval_cfg.data).__name__ == "LiberoPhase0BEvalDataConfig"
    assert type(train.data).__name__ == "LeRobotV4PairDataConfig"
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["inference_inputs"] == "single_scene_rgb_language_state"

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)

    clean_eval = _config.get_config("pi05_v4_clean_wrong_multi_sample_asymmetric_cross_view010_eval")
    assert type(clean_eval.data).__name__ == "LiberoPhase0BEvalDataConfig"
    assert clean_eval.model.cv_pair_mode == "clean_wrong_batch_derangement"
    assert clean_eval.policy_metadata["matched_reference"] == "pi05_v4_pair_multi_sample_asymmetric_cross_view010"


def test_phaseb_multi_sample_bilateral_eval_config_uses_single_view_eval_data():
    train = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010")
    eval_cfg = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_eval")

    assert type(eval_cfg.data).__name__ == "LiberoPhase0BEvalDataConfig"
    assert type(train.data).__name__ == "LeRobotV4PairDataConfig"
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["cv_gradient_mode"] == "bilateral"
    assert eval_cfg.policy_metadata["inference_inputs"] == "single_scene_rgb_language_state"

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_phaseb_b6b_action_biased_time_eval_config_uses_single_view_eval_data():
    train = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time")
    eval_cfg = _config.get_config("pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time_eval")

    assert type(eval_cfg.data).__name__ == "LiberoPhase0BEvalDataConfig"
    assert type(train.data).__name__ == "LeRobotV4PairDataConfig"
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["phase_step"] == "B6b"
    assert eval_cfg.policy_metadata["cv_time_distribution"] == "beta_2p0_3p0"
    assert eval_cfg.policy_metadata["cv_time_bias"] == "action_biased"
    assert eval_cfg.policy_metadata["inference_inputs"] == "single_scene_rgb_language_state"

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_phaseb_b6b_clean_wrong_eval_config_uses_single_view_eval_data():
    train = _config.get_config("pi05_v4_clean_wrong_multi_sample_bilateral_cross_view010_action_biased_time")
    eval_cfg = _config.get_config("pi05_v4_clean_wrong_multi_sample_bilateral_cross_view010_action_biased_time_eval")

    assert type(eval_cfg.data).__name__ == "LiberoPhase0BEvalDataConfig"
    assert type(train.data).__name__ == "LeRobotV4PairDataConfig"
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["phase_step"] == "B6b-clean-wrong"
    assert eval_cfg.policy_metadata["matched_reference"] == (
        "pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time"
    )
    assert eval_cfg.policy_metadata["inference_inputs"] == "single_scene_rgb_language_state"

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_realman_task123_b6b_config_uses_combined_pair_data_and_realman_actions():
    cfg = _config.get_config("pi05_realman_task123_b6b")

    assert type(cfg.data).__name__ == "LeRobotV4PairDataConfig"
    assert cfg.data.repo_id == "data/real_robot/task123_pair"
    assert cfg.data.assets.assets_dir == "assets/pi05_realman_task123_b6b"
    assert cfg.data.assets.asset_id == "anonymous/realman_task123_corl"
    assert cfg.data.output_action_dim == 8
    assert cfg.data.use_wrist_image is False
    assert cfg.data.dataset_episodes[:2] == (0, 1)
    assert cfg.data.dataset_episodes[-2:] == (302, 303)
    assert len(cfg.data.dataset_episodes) == 280

    assert cfg.model.lambda_cv == 0.10
    assert cfg.model.cv_action_dim == 8
    assert cfg.model.cv_loss_mode == "multi_sample_asymmetric"
    assert cfg.model.cv_num_samples == 2
    assert cfg.model.cv_stopgrad_anchor is False
    assert cfg.model.cv_time_distribution == "beta_2p0_3p0"
    assert cfg.model.cv_warmup_start_fraction == 0.0
    assert cfg.model.cv_warmup_end_fraction == 0.05
    assert cfg.model.pair_spatial_aug_mode == "none"
    assert cfg.model.pair_photometric_aug_mode == "independent"
    assert cfg.num_train_steps == 10_000
    assert cfg.policy_metadata["phase_step"] == "Task123_CoRL-B6b"
    assert cfg.policy_metadata["cv_warmup_start_fraction"] == 0.0
    assert cfg.policy_metadata["cv_warmup_end_fraction"] == 0.05
    assert cfg.policy_metadata["output_action_dim"] == 8


def test_realman_task123_b6b_eval_uses_single_view_val_split():
    train = _config.get_config("pi05_realman_task123_b6b")
    eval_cfg = _config.get_config("pi05_realman_task123_b6b_eval")

    assert type(eval_cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert eval_cfg.data.repo_id == train.data.repo_id
    assert eval_cfg.data.assets == train.data.assets
    assert eval_cfg.data.output_action_dim == 8
    assert eval_cfg.data.use_wrist_image is False
    assert eval_cfg.data.dataset_episodes == (
        *range(128, 144),
        *range(210, 218),
        *range(304, 314),
    )
    assert len(eval_cfg.data.dataset_episodes) == 34
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_realman_task123_b6b_action_expert_only_small_config_is_conservative_short_run():
    base = _config.get_config("pi05_realman_task123_b6b")
    cfg = _config.get_config("pi05_realman_task123_b6b_action_expert_only_5k_lr2e5")

    assert type(cfg.data).__name__ == "LeRobotV4PairDataConfig"
    assert cfg.data == base.data
    assert cfg.batch_size == base.batch_size
    assert cfg.num_train_steps == 5_000
    assert cfg.save_interval == 1_000
    assert cfg.keep_period == 1_000
    assert cfg.lr_schedule.warmup_steps == 300
    assert cfg.lr_schedule.peak_lr == 2e-5
    assert cfg.lr_schedule.decay_steps == 5_000
    assert cfg.lr_schedule.decay_lr == 2e-6
    assert cfg.model.total_train_steps == 5_000
    assert cfg.model.lambda_cv == base.model.lambda_cv
    assert cfg.model.cv_action_dim == 8
    assert cfg.model.cv_loss_mode == "multi_sample_asymmetric"
    assert cfg.model.cv_num_samples == 2
    assert cfg.model.cv_stopgrad_anchor is False
    assert cfg.model.cv_time_distribution == "beta_2p0_3p0"
    assert cfg.model.cv_warmup_start_fraction == 0.0
    assert cfg.model.cv_warmup_end_fraction == 0.05
    assert cfg.policy_metadata["method"] == "b6b_cv_action_expert_only"
    assert cfg.policy_metadata["base_train_config"] == base.name
    assert cfg.policy_metadata["keep_period"] == 1_000
    assert cfg.policy_metadata["freeze_scope"] == "vision_language_backbone_frozen"
    assert cfg.policy_metadata["trainable_scope"] == "action_expert_and_action_heads"
    assert "llm.*_1" in cfg.policy_metadata["trainable_regex"]
    assert "action_out_proj" in cfg.policy_metadata["trainable_regex"]
    assert cfg.freeze_filter is not base.freeze_filter


def test_realman_task123_b6b_action_expert_only_small_eval_matches_train_model():
    train = _config.get_config("pi05_realman_task123_b6b_action_expert_only_5k_lr2e5")
    eval_cfg = _config.get_config("pi05_realman_task123_b6b_action_expert_only_5k_lr2e5_eval")

    assert type(eval_cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert eval_cfg.data.repo_id == train.data.repo_id
    assert eval_cfg.data.assets == train.data.assets
    assert eval_cfg.data.output_action_dim == 8
    assert eval_cfg.batch_size == 1
    assert eval_cfg.wandb_enabled is False
    assert eval_cfg.num_train_steps == train.num_train_steps
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["trainable_scope"] == train.policy_metadata["trainable_scope"]

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_realman_task123_b6b_eval_h20_only_changes_inference_horizon():
    base_eval = _config.get_config("pi05_realman_task123_b6b_eval")
    h20 = _config.get_config("pi05_realman_task123_b6b_eval_h20")

    assert type(h20.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert h20.data == base_eval.data
    assert h20.model.action_horizon == 20
    assert h20.policy_metadata["train_config"] == "pi05_realman_task123_b6b"
    assert h20.policy_metadata["inference_action_horizon"] == 20
    assert h20.policy_metadata["base_train_action_horizon"] == 10
    assert h20.policy_metadata["output_action_dim"] == 8

    for field in dataclasses.fields(base_eval.model):
        if field.name != "action_horizon":
            assert getattr(h20.model, field.name) == getattr(base_eval.model, field.name)


def test_realman_task123_all_fm_only_is_pure_pi05_on_all_nominal_data():
    cfg = _config.get_config("pi05_realman_task123_all_fm_only")

    assert type(cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert cfg.data.repo_id == "data/real_robot/task123_pair"
    assert cfg.data.assets.assets_dir == "assets/pi05_realman_task123_pi05_all"
    assert cfg.data.assets.asset_id == "anonymous/realman_task123_corl"
    assert cfg.data.output_action_dim == 8
    assert cfg.data.use_wrist_image is False
    assert cfg.data.scene_only_image_inputs is True
    assert cfg.data.dataset_episodes == tuple(range(314))

    assert type(cfg.model).__name__ == "Pi0Config"
    assert cfg.model.pi05 is True
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is False
    assert not hasattr(cfg.model, "lambda_cv")
    assert cfg.batch_size == 384
    assert cfg.num_train_steps == 10_000
    assert cfg.policy_metadata["method"] == "pi05_fm_only"
    assert cfg.policy_metadata["cv_loss"] is False
    assert cfg.policy_metadata["uses_perturbed_view"] is False
    assert cfg.policy_metadata["train_pair_episodes"] == "0:314"
    assert cfg.policy_metadata["output_action_dim"] == 8


def test_realman_task123_all_fm_only_eval_matches_train_transforms():
    train = _config.get_config("pi05_realman_task123_all_fm_only")
    eval_cfg = _config.get_config("pi05_realman_task123_all_fm_only_eval")

    assert type(eval_cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert eval_cfg.data == train.data
    assert eval_cfg.batch_size == 1
    assert eval_cfg.wandb_enabled is False
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)


def test_realman_task123_action_expert_only_fm_config_freezes_backbone():
    base = _config.get_config("pi05_realman_task123_all_fm_only")
    cfg = _config.get_config("pi05_realman_task123_all_fm_only_action_expert_only")

    assert type(cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert cfg.data == base.data
    assert cfg.model == base.model
    assert cfg.batch_size == base.batch_size
    assert cfg.num_train_steps == base.num_train_steps
    assert cfg.policy_metadata["method"] == "pi05_fm_only"
    assert cfg.policy_metadata["cv_loss"] is False
    assert cfg.policy_metadata["freeze_scope"] == "vision_language_backbone_frozen"
    assert cfg.policy_metadata["trainable_scope"] == "action_expert_and_action_heads"
    assert "llm.*_1" in cfg.policy_metadata["trainable_regex"]
    assert "action_out_proj" in cfg.policy_metadata["trainable_regex"]
    assert cfg.freeze_filter is not base.freeze_filter


def test_realman_task123_action_expert_only_eval_matches_train_transforms():
    train = _config.get_config("pi05_realman_task123_all_fm_only_action_expert_only")
    eval_cfg = _config.get_config("pi05_realman_task123_all_fm_only_action_expert_only_eval")

    assert type(eval_cfg.data).__name__ == "LeRobotLiberoPlusDataConfig"
    assert eval_cfg.data == train.data
    assert eval_cfg.batch_size == 1
    assert eval_cfg.wandb_enabled is False
    assert eval_cfg.policy_metadata["eval_only"] is True
    assert eval_cfg.policy_metadata["train_config"] == train.name
    assert eval_cfg.policy_metadata["trainable_scope"] == train.policy_metadata["trainable_scope"]

    for field in dataclasses.fields(train.model):
        assert getattr(eval_cfg.model, field.name) == getattr(train.model, field.name)
