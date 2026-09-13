"""Paper-facing training recipes and data contracts."""

import abc
import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import tyro
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import cross_view_config, pi0_config
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import libero_policy
from openpi.shared import download as _download
from openpi.shared import normalize as _normalize
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional subset of LeRobot episodes to expose from a local dataset root.
    dataset_episodes: Sequence[int] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional video decoder backend for LeRobot datasets. When unset, LeRobot
    # will use its own default backend selection.
    video_backend: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional LeRobot episode subset to expose from repo_id.
    dataset_episodes: tyro.conf.Suppress[Sequence[int] | None] = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        dataset_episodes = self.dataset_episodes
        if dataset_episodes is None and self.base_config is not None:
            dataset_episodes = self.base_config.dataset_episodes
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            dataset_episodes=dataset_episodes,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class PairedDataConfig(DataConfigFactory):
    """paired same-state multiview pair data config.

    The LeRobot dataset stores one row per simulator state with both the nominal
    and perturbed scene-camera images. This config preserves that pair axis for
    the cross-view loss while keeping the policy inference contract unchanged.
    """

    extra_delta_transform: bool = False
    use_wrist_image: bool = False
    video_backend: str | None = "pyav"
    output_action_dim: int = 7
    # Exported datasets store one complete [H,7] action chunk per row.
    # Disable LeRobot temporal delta queries so sparse/repeated manifest rows
    # can never be mistaken for consecutive source timesteps.
    prechunked_actions: bool = True
    pair_membership_path: str | None = None
    pair_coverage_percent: int | None = None
    # Spec passed to ``make_bool_mask`` when ``extra_delta_transform`` is True.
    # None preserves the legacy 7-dim LIBERO mask of (6, -1). Realman 8-dim
    # (7 joints delta + 1 absolute gripper) should pass (7, -1).
    delta_action_mask_spec: tuple[int, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_structure = {
            "observation/image": "observation.images.front",
            "observation/image_perturbed": "observation.images.perturbed",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
            "task_index": "task_index",
            "episode_index": "episode_index",
            "sample_index": "index",
        }
        if self.use_wrist_image:
            # The scene-only full-support export deliberately omits this key;
            # wrist-on controls use a separate export with the exact HDF5
            # eye-in-hand stream under this canonical LeRobot feature name.
            repack_structure["observation/wrist_image"] = "observation.images.wrist"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])

        if (self.pair_membership_path is None) != (self.pair_coverage_percent is None):
            raise ValueError("pair_membership_path and pair_coverage_percent must be set together")
        pair_mask_transforms = (
            [
                _transforms.PairMaskFromEpisodeIndex.from_membership_file(
                    self.pair_membership_path,
                    self.pair_coverage_percent,
                )
            ]
            if self.pair_membership_path is not None and self.pair_coverage_percent is not None
            else []
        )

        data_transforms = _transforms.Group(
            inputs=[
                *pair_mask_transforms,
                libero_policy.LiberoPairInputs(
                    model_type=model_config.model_type,
                    use_wrist_image=self.use_wrist_image,
                ),
            ],
            outputs=[libero_policy.LiberoOutputs(action_dim=self.output_action_dim)],
        )

        if self.extra_delta_transform:
            mask_spec = self.delta_action_mask_spec if self.delta_action_mask_spec is not None else (6, -1)
            delta_action_mask = _transforms.make_bool_mask(*mask_spec)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config).push(inputs=[libero_policy.LiberoPairTokenInputs()])

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.prechunked_actions else ("action",),
            video_backend=self.video_backend,
        )


@dataclasses.dataclass(frozen=True)
class SingleViewDataConfig(DataConfigFactory):
    """Eval-only single-view inference config for paired pair-trained models.

    paired training batches carry a nominal/perturbed pair axis, but LIBERO-plus
    rollout inference supplies one scene-camera observation at a time. This config
    keeps the paired model and norm-stat asset contract while using the regular
    single-view LIBERO policy inputs at eval time.
    """

    extra_delta_transform: bool = False
    use_wrist_image: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    model_type=model_config.model_type,
                    use_wrist_image=self.use_wrist_image,
                )
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "action-flow-consistency"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _recipe(
    name: str,
    *,
    weight: float = 0.10,
    shuffled: bool = False,
    action_weighting: bool = False,
    photo: str = "independent",
    evaluation: bool = False,
    coverage: int | None = None,
    restricted: bool = False,
    observation_mode: str = "paired",
    robot: bool = False,
) -> TrainConfig:
    root = pathlib.Path(__file__).resolve().parents[3]
    dataset = os.environ.get("AFCV_DATASET", str(root / "data" / ("robot_pairs" if robot else "libero_pairs")))
    assets = AssetsConfig(
        assets_dir=os.environ.get("AFCV_NORM_DIR", str(root / "assets")), asset_id="robot" if robot else "libero"
    )
    base = DataConfig(prompt_from_task=True)
    if restricted:
        dataset = os.environ.get("AFCV_DATASET", str(root / "data/libero_restricted"))
    data = (
        SingleViewDataConfig(repo_id=dataset, assets=assets, base_config=base)
        if evaluation
        else PairedDataConfig(
            repo_id=dataset,
            assets=assets,
            base_config=base,
            prechunked_actions=True,
            pair_membership_path=os.environ.get("AFCV_MEMBERSHIP", str(root / "data/episode_membership.json"))
            if coverage is not None
            else None,
            pair_coverage_percent=coverage,
        )
    )
    model = cross_view_config.CrossViewConfig(
        lambda_cv=weight,
        observation_mode=observation_mode,
        shuffle_pairs=shuffled,
        action_space_weighting=action_weighting,
        photometric_mode=photo,
        cv_warmup_start=0,
        cv_warmup_steps=500,
    )
    return TrainConfig(
        name=name,
        model=model,
        data=data,
        batch_size=192,
        num_workers=8,
        num_train_steps=10_000,
        ema_decay=0.999,
        save_interval=2000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get("AFCV_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params")
        ),
        policy_metadata={
            "inference_inputs": "scene_rgb_language_proprioception",
            "eval_image_flip_mode": "flipud",
            "protocol": "paper",
        },
    )


_CONFIGS = [
    _recipe("nominal_only", weight=0, observation_mode="nominal"),
    _recipe("mixed_camera", weight=0, observation_mode="mixed"),
    _recipe("paired_cv"),
    _recipe("paired_fm", weight=0),
    _recipe("shuffled", shuffled=True),
    _recipe("action_consistency", weight=0.50, action_weighting=True),
    _recipe("shared_jitter", photo="shared"),
    _recipe("coverage_05", coverage=5),
    _recipe("coverage_25", coverage=25),
    _recipe("restricted_cv", restricted=True),
    _recipe("restricted_fm", weight=0, restricted=True),
    _recipe("paired_cv_eval", evaluation=True),
    _recipe("robot_nominal", robot=True, weight=0, observation_mode="nominal"),
    _recipe("robot_fm", robot=True, weight=0),
    _recipe("robot_cv", robot=True),
    _recipe("robot_eval", robot=True, evaluation=True),
]
_CONFIGS_DICT = {c.name: c for c in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({name: (name, cfg) for name, cfg in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        raise ValueError(f"Unknown config {config_name!r}. Choices: {', '.join(_CONFIGS_DICT)}")
    return _CONFIGS_DICT[config_name]
