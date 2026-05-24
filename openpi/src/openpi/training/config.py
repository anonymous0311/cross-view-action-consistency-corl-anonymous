"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.cross_view_action_consistency_config as cross_view_action_consistency_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.droid_sharded_dataset as droid_sharded_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
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

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Only used for the sharded modern DROID data loader.
    sharded_droid_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


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
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
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
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    use_wrist_image: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_structure = {
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        if self.use_wrist_image:
            repack_structure["observation/wrist_image"] = "wrist_image"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type, use_wrist_image=self.use_wrist_image)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoPlusDataConfig(DataConfigFactory):
    """Scene-camera-only LIBERO-plus data config.

    This config is intentionally separate from the generic LIBERO config:
    - scene camera is stored as `observation.images.front`
    - wrist camera exists in the dataset but must be dropped for this project
    - actions are stored under `action`
    - local training currently relies on `pyav` for robust video decoding
    """

    extra_delta_transform: bool = False
    use_wrist_image: bool = False
    video_backend: str | None = "pyav"
    output_action_dim: int = 7
    scene_only_image_inputs: bool = False
    # Spec passed to ``make_bool_mask`` when ``extra_delta_transform`` is True.
    # None preserves the standard 7-dim LIBERO mask of (6, -1). Realman 8-dim
    # (7 joints delta + 1 absolute gripper) should pass (7, -1).
    delta_action_mask_spec: tuple[int, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                        "task_index": "task_index",
                        "episode_index": "episode_index",
                        "sample_index": "index",
                    }
                )
            ]
        )

        image_input_transform = (
            libero_policy.LiberoSceneInputs()
            if self.scene_only_image_inputs
            else libero_policy.LiberoInputs(model_type=model_config.model_type, use_wrist_image=self.use_wrist_image)
        )
        data_transforms = _transforms.Group(
            inputs=[
                image_input_transform,
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

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            video_backend=self.video_backend,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCrossViewPairDataConfig(DataConfigFactory):
    """cross-view same-state multiview pair data config.

    The LeRobot dataset stores one row per simulator state with both the nominal
    and perturbed scene-camera images. This config preserves that pair axis for
    the cross-view loss while keeping the policy inference contract unchanged.
    """

    extra_delta_transform: bool = False
    use_wrist_image: bool = False
    video_backend: str | None = "pyav"
    output_action_dim: int = 7
    # Spec passed to ``make_bool_mask`` when ``extra_delta_transform`` is True.
    # None preserves the standard 7-dim LIBERO mask of (6, -1). Realman 8-dim
    # (7 joints delta + 1 absolute gripper) should pass (7, -1).
    delta_action_mask_spec: tuple[int, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/image_perturbed": "observation.images.perturbed",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                        "task_index": "task_index",
                        "episode_index": "episode_index",
                        "sample_index": "index",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoPairInputs(
                    model_type=model_config.model_type,
                    use_wrist_image=self.use_wrist_image,
                )
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
            action_sequence_keys=("action",),
            video_backend=self.video_backend,
        )


@dataclasses.dataclass(frozen=True)
class LiberoCrossViewEvalDataConfig(DataConfigFactory):
    """Eval-only single-view inference config for cross-view pair-trained models.

    cross-view training batches carry a nominal/perturbed pair axis, but LIBERO-plus
    rollout inference supplies one scene-camera observation at a time. This config
    keeps the cross-view model and norm-stat asset contract while using the regular
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
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class ShardedDROIDDataConfig(DataConfigFactory):
    """
    Config for training on DROID from the local sharded substrate used by the modern JAX stack.
    """

    sharded_droid_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.sharded_droid_dir is not None, "Need to set sharded DROID dir for the modern DROID data loader."
        manifest = droid_sharded_dataset.load_manifest(self.sharded_droid_dir)
        if self.action_space is not None and manifest.action_space != self.action_space.name.lower():
            raise ValueError(
                f"Shard manifest action space {manifest.action_space} does not match config {self.action_space.name.lower()}."
            )
        if manifest.action_horizon != model_config.action_horizon:
            raise ValueError(
                f"Shard manifest action horizon {manifest.action_horizon} does not match model action horizon "
                f"{model_config.action_horizon}."
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            sharded_droid_dir=self.sharded_droid_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

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
    wandb_enabled: bool = True

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


_CROSS_VIEW_MATCHED_REPO_ID = "data/libero_multiview_lerobot"
_CROSS_VIEW_WRONG_REPO_ID = "data/libero_multiview_lerobot_wrong"
_CROSS_VIEW_NORM_ASSET_ID = "anonymous/libero_multiview_lerobot"
_CROSS_VIEW_NUM_TRAIN_STEPS = 10_000
# Pair batch 192 means 384 effective image/action forwards because each sample
# contains nominal and perturbed scene-camera views.
_CROSS_VIEW_PAIR_BATCH_SIZE = 192
_CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE = 2 * _CROSS_VIEW_PAIR_BATCH_SIZE
_CROSS_VIEW_NUM_WORKERS = 24
_CROSS_VIEW_SAVE_INTERVAL = 2_000
_CROSS_VIEW_COMMON_ASSETS = AssetsConfig(
    assets_dir="assets/pi05_cross_view_fm_only",
    asset_id=_CROSS_VIEW_NORM_ASSET_ID,
)

_REALMAN_TASK1_PAIR_REPO_ID = "data/real_robot/task1_pair"
_REALMAN_TASK1_NORM_ASSET_ID = "anonymous/realman_task1_corl"
_REALMAN_TASK123_PAIR_REPO_ID = "data/real_robot/task123_pair"
_REALMAN_TASK123_NORM_ASSET_ID = "anonymous/realman_task123_corl"
_REALMAN_TASK1_NUM_TRAIN_STEPS = 10_000
_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS = 5_000
_REALMAN_TASK1_PAIR_BATCH_SIZE = _CROSS_VIEW_PAIR_BATCH_SIZE
_REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE = 2 * _REALMAN_TASK1_PAIR_BATCH_SIZE
_REALMAN_TASK1_NUM_WORKERS = _CROSS_VIEW_NUM_WORKERS
_REALMAN_TASK1_SAVE_INTERVAL = _CROSS_VIEW_SAVE_INTERVAL
_REALMAN_TASK123_SMALL_SAVE_INTERVAL = 1_000
_REALMAN_TASK1_TRAIN_EPISODES = tuple(range(128))
_REALMAN_TASK1_VAL_EPISODES = tuple(range(128, 144))
_REALMAN_TASK123_TRAIN_EPISODES = (
    *range(0, 128),  # Task1 source episodes 0-63, two pair episodes each.
    *range(144, 210),  # Task2 source episodes 0-32.
    *range(218, 304),  # Task3 source episodes 0-42.
)
_REALMAN_TASK123_VAL_EPISODES = (
    *range(128, 144),  # Task1 source episodes 64-71.
    *range(210, 218),  # Task2 source episodes 33-36.
    *range(304, 314),  # Task3 source episodes 43-47.
)
_REALMAN_TASK123_ALL_EPISODES = tuple(range(314))
_REALMAN_TASK1_COMMON_ASSETS = AssetsConfig(
    assets_dir="assets/pi05_realman_task1_cross_view_action_consistency",
    asset_id=_REALMAN_TASK1_NORM_ASSET_ID,
)
_REALMAN_TASK123_COMMON_ASSETS = AssetsConfig(
    assets_dir="assets/pi05_realman_task123_cross_view_action_consistency",
    asset_id=_REALMAN_TASK123_NORM_ASSET_ID,
)
_REALMAN_TASK123_PI05_ALL_ASSETS = AssetsConfig(
    assets_dir="assets/pi05_realman_task123_pi05_all",
    asset_id=_REALMAN_TASK123_NORM_ASSET_ID,
)
# Delta-action assets (norm stats computed in delta space — first 7 joints are
# action[i] - state[t], gripper remains absolute). See
# scripts/real_robot/compute_realman_pair_norm_stats_fast_delta.py.
_REALMAN_TASK123_DELTA_NORM_ASSET_ID = "anonymous/realman_task123_corl_delta"
_REALMAN_TASK123_DELTA_ASSETS = AssetsConfig(
    assets_dir="assets/pi05_realman_task123_cross_view_action_consistency_delta",
    asset_id=_REALMAN_TASK123_DELTA_NORM_ASSET_ID,
)
# 8-dim mask: first 7 joints delta, gripper absolute.
_REALMAN_TASK123_DELTA_MASK_SPEC: tuple[int, ...] = (7, -1)
_REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX = (
    ".*(llm.*_1|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
)


def _realman_pi05_action_expert_only_freeze_filter() -> nnx.filterlib.Filter:
    """Freeze vision/language backbone; train action expert and action heads."""
    return nnx.All(
        nnx.Param,
        nnx.Not(nnx_utils.PathRegex(_REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX)),
    )


def _cross_view_optimizer() -> _optimizer.AdamW:
    return _optimizer.AdamW(clip_gradient_norm=1.0)


def _cross_view_schedule() -> _optimizer.CosineDecaySchedule:
    return _optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=30_000,
        decay_lr=5e-6,
    )


def _realman_small_action_expert_schedule() -> _optimizer.CosineDecaySchedule:
    return _optimizer.CosineDecaySchedule(
        warmup_steps=300,
        peak_lr=2e-5,
        decay_steps=_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
        decay_lr=2e-6,
    )


def _cross_view_pair_data(repo_id: str) -> LeRobotCrossViewPairDataConfig:
    return LeRobotCrossViewPairDataConfig(
        repo_id=repo_id,
        assets=_CROSS_VIEW_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
    )


def _cross_view_eval_data() -> LiberoCrossViewEvalDataConfig:
    return LiberoCrossViewEvalDataConfig(
        repo_id=_CROSS_VIEW_MATCHED_REPO_ID,
        assets=_CROSS_VIEW_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
    )


def _realman_task1_pair_data(
    *,
    episodes: Sequence[int] = _REALMAN_TASK1_TRAIN_EPISODES,
) -> LeRobotCrossViewPairDataConfig:
    return LeRobotCrossViewPairDataConfig(
        repo_id=_REALMAN_TASK1_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK1_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
    )


def _realman_task123_pair_data(
    *,
    episodes: Sequence[int] = _REALMAN_TASK123_TRAIN_EPISODES,
) -> LeRobotCrossViewPairDataConfig:
    return LeRobotCrossViewPairDataConfig(
        repo_id=_REALMAN_TASK123_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK123_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
    )


def _realman_task123_pair_data_delta(
    *,
    episodes: Sequence[int] = _REALMAN_TASK123_TRAIN_EPISODES,
) -> LeRobotCrossViewPairDataConfig:
    """Pair training data with delta-joint actions (gripper absolute).

    Subtracts current state from joints 0..6 of every action in the chunk; this
    breaks the ``action ≈ state`` shortcut that absolute-joint training induced
    on the Realman dataset.
    """
    return LeRobotCrossViewPairDataConfig(
        repo_id=_REALMAN_TASK123_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK123_DELTA_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
        delta_action_mask_spec=_REALMAN_TASK123_DELTA_MASK_SPEC,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
    )


def _realman_task123_single_view_data_delta(
    *,
    episodes: Sequence[int] = _REALMAN_TASK123_VAL_EPISODES,
) -> LeRobotLiberoPlusDataConfig:
    """Single-view eval data with delta-joint actions (gripper absolute)."""
    return LeRobotLiberoPlusDataConfig(
        repo_id=_REALMAN_TASK123_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK123_DELTA_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
        delta_action_mask_spec=_REALMAN_TASK123_DELTA_MASK_SPEC,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
        scene_only_image_inputs=True,
    )


def _realman_task1_single_view_data(
    *,
    episodes: Sequence[int] = _REALMAN_TASK1_VAL_EPISODES,
) -> LeRobotLiberoPlusDataConfig:
    return LeRobotLiberoPlusDataConfig(
        repo_id=_REALMAN_TASK1_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK1_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
        scene_only_image_inputs=True,
    )


def _realman_task123_single_view_data(
    *,
    episodes: Sequence[int] = _REALMAN_TASK123_VAL_EPISODES,
) -> LeRobotLiberoPlusDataConfig:
    return LeRobotLiberoPlusDataConfig(
        repo_id=_REALMAN_TASK123_PAIR_REPO_ID,
        dataset_episodes=episodes,
        assets=_REALMAN_TASK123_COMMON_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
        scene_only_image_inputs=True,
    )


def _realman_task123_pi05_all_single_view_data() -> LeRobotLiberoPlusDataConfig:
    return LeRobotLiberoPlusDataConfig(
        repo_id=_REALMAN_TASK123_PAIR_REPO_ID,
        dataset_episodes=_REALMAN_TASK123_ALL_EPISODES,
        assets=_REALMAN_TASK123_PI05_ALL_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        use_wrist_image=False,
        video_backend="pyav",
        output_action_dim=8,
        scene_only_image_inputs=True,
    )


def _cross_view_model_config(
    lambda_cv: float,
    *,
    total_train_steps: int = _CROSS_VIEW_NUM_TRAIN_STEPS,
    action_horizon: int = 10,
    cv_action_dim: int = 7,
    cv_pair_mode: str = "matched",
    cv_loss_mode: str = "symmetric",
    cv_num_samples: int = 1,
    cv_stopgrad_anchor: bool = True,
    cv_time_distribution: str = "beta_1p5_1",
    cv_eps_shared_across_views: bool = True,
    cv_average_over_samples: bool = True,
    cv_warmup_start_fraction: float = 0.10,
    cv_warmup_end_fraction: float = 0.30,
    pair_spatial_aug_mode: str = "current",
    pair_photometric_aug_mode: str = "current",
) -> cross_view_action_consistency_config.Pi0CrossViewActionConsistencyConfig:
    return cross_view_action_consistency_config.Pi0CrossViewActionConsistencyConfig(
        pi05=True,
        action_horizon=action_horizon,
        discrete_state_input=False,
        lambda_cv=lambda_cv,
        total_train_steps=total_train_steps,
        cv_warmup_start_fraction=cv_warmup_start_fraction,
        cv_warmup_end_fraction=cv_warmup_end_fraction,
        cv_action_dim=cv_action_dim,
        cv_pair_mode=cv_pair_mode,
        cv_loss_mode=cv_loss_mode,
        cv_num_samples=cv_num_samples,
        cv_stopgrad_anchor=cv_stopgrad_anchor,
        cv_time_distribution=cv_time_distribution,
        cv_eps_shared_across_views=cv_eps_shared_across_views,
        cv_average_over_samples=cv_average_over_samples,
        pair_spatial_aug_mode=pair_spatial_aug_mode,
        pair_photometric_aug_mode=pair_photometric_aug_mode,
    )


def _method_ablation_model(
    *,
    cv_num_samples: int,
    cv_stopgrad_anchor: bool,
    cv_time_distribution: str,
) -> cross_view_action_consistency_config.Pi0CrossViewActionConsistencyConfig:
    return _cross_view_model_config(
        lambda_cv=0.10,
        cv_loss_mode="multi_sample_asymmetric",
        cv_num_samples=cv_num_samples,
        cv_stopgrad_anchor=cv_stopgrad_anchor,
        cv_time_distribution=cv_time_distribution,
        cv_eps_shared_across_views=True,
        cv_average_over_samples=True,
        pair_spatial_aug_mode="none",
        pair_photometric_aug_mode="independent",
    )


def _method_ablation_metadata(
    *,
    method_variant: str,
    ablation_axis: str,
    cv_num_samples: int,
    cv_stopgrad_anchor: bool,
    cv_time_distribution: str,
    comparison_role: str,
    baseline_reference: str = "pi05_cross_view_action_consistency",
) -> dict[str, Any]:
    cv_gradient_mode = "stop_gradient_nominal_anchor" if cv_stopgrad_anchor else "bilateral"
    return {
        "experiment_group": "cross_view_action_consistency_ablation",
        "method_variant": method_variant,
        "pair_type": "matched",
        "ablation_axis": ablation_axis,
        "comparison_role": comparison_role,
        "lambda_cv_target": 0.10,
        "cv_loss_mode": "multi_sample_asymmetric",
        "cv_gradient_mode": cv_gradient_mode,
        "cv_num_samples": cv_num_samples,
        "cv_anchor_view": "nominal",
        "cv_student_view": "perturbed",
        "cv_stopgrad_anchor": cv_stopgrad_anchor,
        "cv_time_distribution": cv_time_distribution,
        "cv_eps_shared_across_views": True,
        "pair_spatial_aug_mode": "none",
        "pair_photometric_aug_mode": "independent",
        "matched_reference": "pi05_cross_view_single_sample",
        "ablation_baseline": baseline_reference,
        "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
        "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        "primary_lambda": True,
    }


def _method_ablation_train_config(
    name: str,
    *,
    method_variant: str,
    ablation_axis: str,
    cv_num_samples: int,
    cv_stopgrad_anchor: bool,
    cv_time_distribution: str,
    comparison_role: str,
    baseline_reference: str = "pi05_cross_view_action_consistency",
) -> TrainConfig:
    return TrainConfig(
        name=name,
        model=_method_ablation_model(
            cv_num_samples=cv_num_samples,
            cv_stopgrad_anchor=cv_stopgrad_anchor,
            cv_time_distribution=cv_time_distribution,
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata=_method_ablation_metadata(
            method_variant=method_variant,
            ablation_axis=ablation_axis,
            cv_num_samples=cv_num_samples,
            cv_stopgrad_anchor=cv_stopgrad_anchor,
            cv_time_distribution=cv_time_distribution,
            comparison_role=comparison_role,
            baseline_reference=baseline_reference,
        ),
    )


def _method_ablation_eval_config(
    train_config_name: str,
    *,
    method_variant: str,
    ablation_axis: str,
    cv_num_samples: int,
    cv_stopgrad_anchor: bool,
    cv_time_distribution: str,
    comparison_role: str,
    baseline_reference: str = "pi05_cross_view_action_consistency",
) -> TrainConfig:
    return TrainConfig(
        name=f"{train_config_name}_eval",
        model=_method_ablation_model(
            cv_num_samples=cv_num_samples,
            cv_stopgrad_anchor=cv_stopgrad_anchor,
            cv_time_distribution=cv_time_distribution,
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            **_method_ablation_metadata(
                method_variant=method_variant,
                ablation_axis=ablation_axis,
                cv_num_samples=cv_num_samples,
                cv_stopgrad_anchor=cv_stopgrad_anchor,
                cv_time_distribution=cv_time_distribution,
                comparison_role=comparison_role,
                baseline_reference=baseline_reference,
            ),
            "eval_only": True,
            "train_config": train_config_name,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_scene",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            use_wrist_image=False,
        ),
        batch_size=384,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=4e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_plus_scene",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoPlusDataConfig(
            repo_id="data/libero_plus_camera_perturbation",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            use_wrist_image=False,
            video_backend="pyav",
        ),
        batch_size=384,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=4e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # ---------------------------------------------------------------------------
    # cross-view action consistency sweep: same-state cross-view action-flow consistency.
    #
    # Data rows contain two scene-camera views of the same simulator state. The
    # dataloader batch size is pair count; effective view forwards are 2x larger.
    # These sweep configs are included for ablations and controls.
    # ---------------------------------------------------------------------------
    TrainConfig(
        name="pi05_cross_view_fm_only",
        model=_cross_view_model_config(lambda_cv=0.0),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "fm_loss": "0.5 * (FM_nominal + FM_perturbed)",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda005",
        model=_cross_view_model_config(lambda_cv=0.05),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "matched",
            "lambda_cv_target": 0.05,
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda010",
        model=_cross_view_model_config(lambda_cv=0.10),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_single_sample",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "matched_reference": "pi05_cross_view_lambda010",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_stopgrad_lambda005",
        model=_cross_view_model_config(
            lambda_cv=0.05,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "pair_type": "matched",
            "lambda_cv_target": 0.05,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": True,
            "cv_time_distribution": "beta_1p5_1",
            "cv_eps_shared_across_views": True,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": False,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_stopgrad",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": True,
            "cv_time_distribution": "beta_1p5_1",
            "cv_eps_shared_across_views": True,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "matched_reference": "pi05_cross_view_single_sample",
            "selection_note": "lambda_cv=0.05 full was only 0.2pp above lambda_cv=0.10; use lambda_cv=0.10 as cross-view primary configuration",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": True,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_bilateral_beta_1p5_1_time",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_1p5_1",
            "cv_eps_shared_across_views": True,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "matched_reference": "pi05_cross_view_single_sample",
            "ablation_reference": "pi05_cross_view_multisample_stopgrad",
            "decision_note": (
                "K=2 bilateral-gradient CV: same stop-gradient objective except CV gradients update both nominal and "
                "perturbed branches."
            ),
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": True,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_action_consistency",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "method_variant": "cross_view_action_consistency",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_eps_shared_across_views": True,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "matched_reference": "pi05_cross_view_single_sample",
            "negative_reference": "pi05_cross_view_multisample_stopgrad",
            "bilateral_beta_1p5_1_time_reference": "pi05_cross_view_multisample_bilateral_beta_1p5_1_time",
            "decision_note": (
                "cross_view_action_consistency unified fix: bilateral CV gradients plus Beta(2,3) action-biased time sampling."
            ),
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": True,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_action_consistency_clean_wrong",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "method_variant": "clean-wrong-control",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_eps_shared_across_views": True,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "matched_reference": "pi05_cross_view_action_consistency",
            "control_note": (
                "Clean wrong for cross_view_action_consistency: training uses the matched same-state pair repo and row-local FM labels; "
                "only the CV nominal anchor indices are deranged inside the multi-sample loss."
            ),
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": True,
        },
    ),
    _method_ablation_train_config(
        name="pi05_cross_view_action_consistency_stopgrad",
        method_variant="cross_view_action_consistency-ablate-gradient-stopgrad-beta2p0-3p0",
        ablation_axis="gradient_direction",
        cv_num_samples=2,
        cv_stopgrad_anchor=True,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency to isolate bilateral vs stop-gradient with K=2 and Beta(2,3).",
    ),
    _method_ablation_train_config(
        name="pi05_cross_view_action_consistency_k1",
        method_variant="cross_view_action_consistency-ablate-K1-beta2p0-3p0",
        ablation_axis="num_flow_samples",
        cv_num_samples=1,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency K=2 and K=4 to isolate sample count.",
    ),
    _method_ablation_train_config(
        name="pi05_cross_view_action_consistency_k4",
        method_variant="cross_view_action_consistency-ablate-K4-beta2p0-3p0",
        ablation_axis="num_flow_samples",
        cv_num_samples=4,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency K=2 and K=1 to isolate sample count.",
    ),
    _method_ablation_train_config(
        name="pi05_cross_view_action_consistency_uniform_time",
        method_variant="cross_view_action_consistency-ablate-uniform-time",
        ablation_axis="time_distribution",
        cv_num_samples=2,
        cv_stopgrad_anchor=False,
        cv_time_distribution="uniform",
        comparison_role="Compare against cross_view_action_consistency Beta(2,3) and Beta(1,1.5) to isolate time distribution.",
    ),
    _method_ablation_train_config(
        name="pi05_cross_view_action_consistency_beta_1p0_1p5_time",
        method_variant="cross_view_action_consistency-ablate-beta1p0-1p5-time",
        ablation_axis="time_distribution",
        cv_num_samples=2,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_1p0_1p5",
        comparison_role="Compare against cross_view_action_consistency Beta(2,3) and uniform to isolate time distribution.",
    ),
    TrainConfig(
        name="pi05_realman_task1_cross_view_action_consistency",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task1_pair_data(),
        batch_size=_REALMAN_TASK1_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task1_CoRL-cross_view_action_consistency",
            "robot": "realman",
            "task": "pick up the blue batery and place it into cardboard box.",
            "pair_type": "matched",
            "source_dataset": "data/real_robot/task1",
            "pair_dataset": _REALMAN_TASK1_PAIR_REPO_ID,
            "train_episodes": [0, 63],
            "val_episodes": [64, 71],
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_eps_shared_across_views": True,
            "cv_action_dim": 8,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _REALMAN_TASK1_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task1_cross_view_action_consistency_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task1_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task1_CoRL-cross_view_action_consistency-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task1_cross_view_action_consistency",
            "robot": "realman",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_action_dim": 8,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_pair_data(),
        batch_size=_REALMAN_TASK1_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "pair_type": "matched",
            "pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "source_datasets": [
                "data/real_robot/task1",
                "data/real_robot/task2",
                "data/real_robot/task3",
            ],
            "train_pair_episodes": ["0:128", "144:210", "218:304"],
            "val_pair_episodes": ["128:144", "210:218", "304:314"],
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_eps_shared_across_views": True,
            "cv_action_dim": 8,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _REALMAN_TASK1_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_delta",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_pair_data_delta(),
        batch_size=_REALMAN_TASK1_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-delta",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "method": "cross_view_cv_delta_joint",
            "action_representation": "joint_delta_from_state_gripper_absolute",
            "delta_action_mask_spec": list(_REALMAN_TASK123_DELTA_MASK_SPEC),
            "pair_type": "matched",
            "pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "source_datasets": [
                "data/real_robot/task1",
                "data/real_robot/task2",
                "data/real_robot/task3",
            ],
            "train_pair_episodes": ["0:128", "144:210", "218:304"],
            "val_pair_episodes": ["128:144", "210:218", "304:314"],
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_eps_shared_across_views": True,
            "cv_action_dim": 8,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _REALMAN_TASK1_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_delta_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_single_view_data_delta(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-delta-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_cross_view_action_consistency_delta",
            "robot": "realman",
            "method": "cross_view_cv_delta_joint",
            "action_representation": "joint_delta_from_state_gripper_absolute",
            "delta_action_mask_spec": list(_REALMAN_TASK123_DELTA_MASK_SPEC),
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_action_dim": 8,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        # FM-only ablation of pi05_realman_task123_cross_view_action_consistency_delta. Every hyperparam
        # matches the cross_view_delta train config (full model, delta-joint targets,
        # pair data with both nominal+perturbed views, lr 5e-5, 10k steps,
        # batch 192 pair, multi_sample_asymmetric with K=2 / beta_2p0_3p0).
        # The only change is lambda_cv = 0.0, which zeroes out the cross-view
        # term in cross_view_action_consistency._pair_loss_components (cross_view_action_consistency.py:651:
        #     total = loss_fm + lambda_cv * loss_cv
        # and loss_fm already averages over BOTH views, so both still receive
        # full flow-matching supervision — this is the correct paired-FM-only
        # ablation, not a single-view baseline.
        name="pi05_realman_task123_cross_view_action_consistency_delta_fm_only",
        model=_cross_view_model_config(
            lambda_cv=0.0,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_pair_data_delta(),
        batch_size=_REALMAN_TASK1_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-delta-fm-only",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "method": "fm_only_delta_joint_paired",
            "ablation_of": "pi05_realman_task123_cross_view_action_consistency_delta",
            "action_representation": "joint_delta_from_state_gripper_absolute",
            "delta_action_mask_spec": list(_REALMAN_TASK123_DELTA_MASK_SPEC),
            "pair_type": "matched",
            "pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "source_datasets": [
                "data/real_robot/task1",
                "data/real_robot/task2",
                "data/real_robot/task3",
            ],
            "train_pair_episodes": ["0:128", "144:210", "218:304"],
            "val_pair_episodes": ["128:144", "210:218", "304:314"],
            "lambda_cv_target": 0.0,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral_disabled",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_eps_shared_across_views": True,
            "cv_action_dim": 8,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _REALMAN_TASK1_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_delta_fm_only_eval",
        model=_cross_view_model_config(
            lambda_cv=0.0,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_single_view_data_delta(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-delta-fm-only-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_cross_view_action_consistency_delta_fm_only",
            "ablation_of": "pi05_realman_task123_cross_view_action_consistency_delta",
            "robot": "realman",
            "method": "fm_only_delta_joint_paired",
            "action_representation": "joint_delta_from_state_gripper_absolute",
            "delta_action_mask_spec": list(_REALMAN_TASK123_DELTA_MASK_SPEC),
            "lambda_cv_target": 0.0,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral_disabled",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_action_dim": 8,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_action_expert_only_5k_lr2e5",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            total_train_steps=_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_pair_data(),
        batch_size=_REALMAN_TASK1_PAIR_BATCH_SIZE,
        lr_schedule=_realman_small_action_expert_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_realman_pi05_action_expert_only_freeze_filter(),
        num_train_steps=_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK123_SMALL_SAVE_INTERVAL,
        keep_period=_REALMAN_TASK123_SMALL_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-action-expert-only-small",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "method": "cross_view_cv_action_expert_only",
            "base_train_config": "pi05_realman_task123_cross_view_action_consistency",
            "pair_type": "matched",
            "pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "source_datasets": [
                "data/real_robot/task1",
                "data/real_robot/task2",
                "data/real_robot/task3",
            ],
            "train_pair_episodes": ["0:128", "144:210", "218:304"],
            "val_pair_episodes": ["128:144", "210:218", "304:314"],
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_anchor_view": "nominal",
            "cv_student_view": "perturbed",
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_eps_shared_across_views": True,
            "cv_action_dim": 8,
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "freeze_scope": "vision_language_backbone_frozen",
            "trainable_scope": "action_expert_and_action_heads",
            "trainable_regex": _REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX,
            "pair_batch_size": _REALMAN_TASK1_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "num_train_steps": _REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
            "save_interval": _REALMAN_TASK123_SMALL_SAVE_INTERVAL,
            "keep_period": _REALMAN_TASK123_SMALL_SAVE_INTERVAL,
            "lr_schedule": "cosine_warmup300_peak2e-5_decay5k_end2e-6",
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_cross_view_action_consistency",
            "robot": "realman",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_action_dim": 8,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_action_expert_only_5k_lr2e5_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            total_train_steps=_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-action-expert-only-small-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_cross_view_action_consistency_action_expert_only_5k_lr2e5",
            "robot": "realman",
            "method": "cross_view_cv_action_expert_only",
            "base_train_config": "pi05_realman_task123_cross_view_action_consistency",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_warmup_start_fraction": 0.0,
            "cv_warmup_end_fraction": 0.05,
            "cv_action_dim": 8,
            "freeze_scope": "vision_language_backbone_frozen",
            "trainable_scope": "action_expert_and_action_heads",
            "trainable_regex": _REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX,
            "num_train_steps": _REALMAN_TASK123_SMALL_NUM_TRAIN_STEPS,
            "lr_schedule": "cosine_warmup300_peak2e-5_decay5k_end2e-6",
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_cross_view_action_consistency_eval_h20",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            action_horizon=20,
            cv_action_dim=8,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            cv_warmup_start_fraction=0.0,
            cv_warmup_end_fraction=0.05,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_realman_task123_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123_CoRL-cross_view_action_consistency-eval-h20",
            "eval_only": True,
            "train_config": "pi05_realman_task123_cross_view_action_consistency",
            "robot": "realman",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_action_dim": 8,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
            "inference_action_horizon": 20,
            "base_train_action_horizon": 10,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_all_fm_only",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_realman_task123_pi05_all_single_view_data(),
        batch_size=_REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123-CoRL-pi05-all-fm-only",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "method": "pi05_fm_only",
            "train_config_role": "all_data_nominal_scene_camera",
            "source_pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "train_pair_episodes": "0:314",
            "num_pair_episodes": 314,
            "uses_perturbed_view": False,
            "cv_loss": False,
            "lambda_cv_target": 0.0,
            "action_horizon": 10,
            "batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_all_fm_only_eval",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_realman_task123_pi05_all_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123-CoRL-pi05-all-fm-only-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_all_fm_only",
            "robot": "realman",
            "method": "pi05_fm_only",
            "train_config_role": "all_data_nominal_scene_camera",
            "source_pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "train_pair_episodes": "0:314",
            "num_pair_episodes": 314,
            "uses_perturbed_view": False,
            "cv_loss": False,
            "lambda_cv_target": 0.0,
            "action_horizon": 10,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_all_fm_only_action_expert_only",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_realman_task123_pi05_all_single_view_data(),
        batch_size=_REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_REALMAN_TASK1_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_realman_pi05_action_expert_only_freeze_filter(),
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        save_interval=_REALMAN_TASK1_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123-CoRL-pi05-fm-only-action-expert-only",
            "robot": "realman",
            "tasks": [
                "pick up the blue batery and place it into cardboard box.",
                "close the laptop lid.",
                "takeoff the headphone from the stand.",
            ],
            "method": "pi05_fm_only",
            "train_config_role": "all_data_nominal_scene_camera_action_expert_only",
            "source_pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "train_pair_episodes": "0:314",
            "num_pair_episodes": 314,
            "uses_perturbed_view": False,
            "cv_loss": False,
            "lambda_cv_target": 0.0,
            "freeze_scope": "vision_language_backbone_frozen",
            "trainable_scope": "action_expert_and_action_heads",
            "trainable_regex": _REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX,
            "action_horizon": 10,
            "batch_size": _REALMAN_TASK1_EFFECTIVE_VIEW_BATCH_SIZE,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_realman_task123_all_fm_only_action_expert_only_eval",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_realman_task123_pi05_all_single_view_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_REALMAN_TASK1_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "real_robot",
            "method_variant": "Task123-CoRL-pi05-fm-only-action-expert-only-eval",
            "eval_only": True,
            "train_config": "pi05_realman_task123_all_fm_only_action_expert_only",
            "robot": "realman",
            "method": "pi05_fm_only",
            "train_config_role": "all_data_nominal_scene_camera_action_expert_only",
            "source_pair_dataset": _REALMAN_TASK123_PAIR_REPO_ID,
            "train_pair_episodes": "0:314",
            "num_pair_episodes": 314,
            "uses_perturbed_view": False,
            "cv_loss": False,
            "lambda_cv_target": 0.0,
            "freeze_scope": "vision_language_backbone_frozen",
            "trainable_scope": "action_expert_and_action_heads",
            "trainable_regex": _REALMAN_PI05_ACTION_EXPERT_TRAINABLE_REGEX,
            "action_horizon": 10,
            "inference_inputs": "single_scene_rgb_language_state",
            "output_action_dim": 8,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_multisample_stopgrad_lambda005",
        model=_cross_view_model_config(
            lambda_cv=0.05,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.05,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "control_note": "FM stays row-local; only the CV nominal anchor index is deranged inside the loss.",
            "matched_reference": "pi05_cross_view_multisample_stopgrad_lambda005",
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": False,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_multisample_stopgrad",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "control_note": "FM stays row-local; only the CV nominal anchor index is deranged inside the loss.",
            "matched_reference": "pi05_cross_view_multisample_stopgrad",
            "selection_note": "lambda_cv=0.05 full was only 0.2pp above lambda_cv=0.10; use lambda_cv=0.10 as cross-view primary configuration",
            "pair_spatial_aug_mode": "none",
            "pair_photometric_aug_mode": "independent",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
            "primary_lambda": True,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda020",
        model=_cross_view_model_config(lambda_cv=0.20),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "matched",
            "lambda_cv_target": 0.20,
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda050",
        model=_cross_view_model_config(lambda_cv=0.50),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "matched",
            "lambda_cv_target": 0.50,
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_wrong_state_lambda005",
        model=_cross_view_model_config(lambda_cv=0.05),
        data=_cross_view_pair_data(_CROSS_VIEW_WRONG_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "wrong_state",
            "lambda_cv_target": 0.05,
            "control_note": "wrong-state pair confidence is intentionally ignored; CV weight is 1.0",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_wrong_state_lambda010",
        model=_cross_view_model_config(lambda_cv=0.10),
        data=_cross_view_pair_data(_CROSS_VIEW_WRONG_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "wrong_state",
            "lambda_cv_target": 0.10,
            "control_note": "wrong-state pair confidence is intentionally ignored; CV weight is 1.0",
            "matched_reference": "pi05_cross_view_lambda010",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_single_sample",
        model=_cross_view_model_config(lambda_cv=0.10, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "control_note": (
                "FM uses correct per-view labels from the matched repo; only the CV branch deranges the batch index."
            ),
            "matched_reference": "pi05_cross_view_lambda010",
            "poisoned_control_to_deprecate": "pi05_cross_view_wrong_state_lambda010",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_lambda020",
        model=_cross_view_model_config(lambda_cv=0.20, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.20,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "control_note": (
                "FM uses correct per-view labels from the matched repo; only the CV branch deranges the batch index."
            ),
            "matched_reference": "pi05_cross_view_lambda020",
            "poisoned_control_to_deprecate": "pi05_cross_view_wrong_state_lambda010",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_lambda050",
        model=_cross_view_model_config(lambda_cv=0.50, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_pair_data(_CROSS_VIEW_MATCHED_REPO_ID),
        batch_size=_CROSS_VIEW_PAIR_BATCH_SIZE,
        lr_schedule=_cross_view_schedule(),
        optimizer=_cross_view_optimizer(),
        ema_decay=0.999,
        num_workers=_CROSS_VIEW_NUM_WORKERS,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        save_interval=_CROSS_VIEW_SAVE_INTERVAL,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.50,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "control_note": (
                "FM uses correct per-view labels from the matched repo; only the CV branch deranges the batch index."
            ),
            "matched_reference": "pi05_cross_view_lambda050",
            "poisoned_control_to_deprecate": "pi05_cross_view_wrong_state_lambda010",
            "pair_batch_size": _CROSS_VIEW_PAIR_BATCH_SIZE,
            "effective_view_batch_size": _CROSS_VIEW_EFFECTIVE_VIEW_BATCH_SIZE,
        },
    ),
    # Eval-only variants for LIBERO-plus rollouts. These load the same cross-view
    # checkpoint parameters but use single-view inference inputs instead of the
    # training-time nominal/perturbed pair transform.
    TrainConfig(
        name="pi05_cross_view_fm_only_eval",
        model=_cross_view_model_config(lambda_cv=0.0),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_fm_only",
            "lambda_cv_target": 0.0,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda005_eval",
        model=_cross_view_model_config(lambda_cv=0.05),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_lambda005",
            "pair_type": "matched",
            "lambda_cv_target": 0.05,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda010_eval",
        model=_cross_view_model_config(lambda_cv=0.10),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_lambda010",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda020_eval",
        model=_cross_view_model_config(lambda_cv=0.20),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_lambda020",
            "pair_type": "matched",
            "lambda_cv_target": 0.20,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_lambda050_eval",
        model=_cross_view_model_config(lambda_cv=0.50),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_lambda050",
            "pair_type": "matched",
            "lambda_cv_target": 0.50,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_stopgrad_lambda005_eval",
        model=_cross_view_model_config(
            lambda_cv=0.05,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "eval_only": True,
            "train_config": "pi05_cross_view_multisample_stopgrad_lambda005",
            "pair_type": "matched",
            "lambda_cv_target": 0.05,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_stopgrad_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "eval_only": True,
            "train_config": "pi05_cross_view_multisample_stopgrad",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "primary_lambda": True,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_multisample_bilateral_beta_1p5_1_time_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "eval_only": True,
            "train_config": "pi05_cross_view_multisample_bilateral_beta_1p5_1_time",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "primary_lambda": True,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_action_consistency_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "method_variant": "cross_view_action_consistency",
            "eval_only": True,
            "train_config": "pi05_cross_view_action_consistency",
            "pair_type": "matched",
            "lambda_cv_target": 0.10,
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "primary_lambda": True,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_action_consistency_clean_wrong_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=False,
            cv_time_distribution="beta_2p0_3p0",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "method_variant": "clean-wrong-control",
            "eval_only": True,
            "train_config": "pi05_cross_view_action_consistency_clean_wrong",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_gradient_mode": "bilateral",
            "cv_num_samples": 2,
            "cv_stopgrad_anchor": False,
            "cv_time_distribution": "beta_2p0_3p0",
            "cv_time_bias": "action_biased",
            "matched_reference": "pi05_cross_view_action_consistency",
            "primary_lambda": True,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    _method_ablation_eval_config(
        "pi05_cross_view_action_consistency_stopgrad",
        method_variant="cross_view_action_consistency-ablate-gradient-stopgrad-beta2p0-3p0",
        ablation_axis="gradient_direction",
        cv_num_samples=2,
        cv_stopgrad_anchor=True,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency to isolate bilateral vs stop-gradient with K=2 and Beta(2,3).",
    ),
    _method_ablation_eval_config(
        "pi05_cross_view_action_consistency_k1",
        method_variant="cross_view_action_consistency-ablate-K1-beta2p0-3p0",
        ablation_axis="num_flow_samples",
        cv_num_samples=1,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency K=2 and K=4 to isolate sample count.",
    ),
    _method_ablation_eval_config(
        "pi05_cross_view_action_consistency_k4",
        method_variant="cross_view_action_consistency-ablate-K4-beta2p0-3p0",
        ablation_axis="num_flow_samples",
        cv_num_samples=4,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_2p0_3p0",
        comparison_role="Compare against cross_view_action_consistency K=2 and K=1 to isolate sample count.",
    ),
    _method_ablation_eval_config(
        "pi05_cross_view_action_consistency_uniform_time",
        method_variant="cross_view_action_consistency-ablate-uniform-time",
        ablation_axis="time_distribution",
        cv_num_samples=2,
        cv_stopgrad_anchor=False,
        cv_time_distribution="uniform",
        comparison_role="Compare against cross_view_action_consistency Beta(2,3) and Beta(1,1.5) to isolate time distribution.",
    ),
    _method_ablation_eval_config(
        "pi05_cross_view_action_consistency_beta_1p0_1p5_time",
        method_variant="cross_view_action_consistency-ablate-beta1p0-1p5-time",
        ablation_axis="time_distribution",
        cv_num_samples=2,
        cv_stopgrad_anchor=False,
        cv_time_distribution="beta_1p0_1p5",
        comparison_role="Compare against cross_view_action_consistency Beta(2,3) and uniform to isolate time distribution.",
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_multisample_stopgrad_lambda005_eval",
        model=_cross_view_model_config(
            lambda_cv=0.05,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "eval_only": True,
            "train_config": "pi05_cross_view_clean_wrong_multisample_stopgrad_lambda005",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.05,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "matched_reference": "pi05_cross_view_multisample_stopgrad_lambda005",
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_multisample_stopgrad_eval",
        model=_cross_view_model_config(
            lambda_cv=0.10,
            cv_pair_mode="clean_wrong_batch_derangement",
            cv_loss_mode="multi_sample_asymmetric",
            cv_num_samples=2,
            cv_stopgrad_anchor=True,
            cv_time_distribution="beta_1p5_1",
            cv_eps_shared_across_views=True,
            cv_average_over_samples=True,
            pair_spatial_aug_mode="none",
            pair_photometric_aug_mode="independent",
        ),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_multisample",
            "eval_only": True,
            "train_config": "pi05_cross_view_clean_wrong_multisample_stopgrad",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "cv_pair_mode": "clean_wrong_batch_derangement",
            "cv_loss_mode": "multi_sample_asymmetric",
            "cv_num_samples": 2,
            "matched_reference": "pi05_cross_view_multisample_stopgrad",
            "primary_lambda": True,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_wrong_state_lambda005_eval",
        model=_cross_view_model_config(lambda_cv=0.05),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_wrong_state_lambda005",
            "pair_type": "wrong_state",
            "lambda_cv_target": 0.05,
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_wrong_state_lambda010_eval",
        model=_cross_view_model_config(lambda_cv=0.10),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_wrong_state_lambda010",
            "pair_type": "wrong_state",
            "lambda_cv_target": 0.10,
            "matched_reference": "pi05_cross_view_lambda010",
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_single_sample_eval",
        model=_cross_view_model_config(lambda_cv=0.10, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_clean_wrong_single_sample",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.10,
            "matched_reference": "pi05_cross_view_lambda010",
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_lambda020_eval",
        model=_cross_view_model_config(lambda_cv=0.20, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_clean_wrong_lambda020",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.20,
            "matched_reference": "pi05_cross_view_lambda020",
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="pi05_cross_view_clean_wrong_lambda050_eval",
        model=_cross_view_model_config(lambda_cv=0.50, cv_pair_mode="clean_wrong_batch_derangement"),
        data=_cross_view_eval_data(),
        batch_size=1,
        num_workers=0,
        wandb_enabled=False,
        num_train_steps=_CROSS_VIEW_NUM_TRAIN_STEPS,
        policy_metadata={
            "experiment_group": "cross_view_lambda_sweep",
            "eval_only": True,
            "train_config": "pi05_cross_view_clean_wrong_lambda050",
            "pair_type": "clean_wrong_cv",
            "lambda_cv_target": 0.50,
            "matched_reference": "pi05_cross_view_lambda050",
            "inference_inputs": "single_scene_rgb_language_state",
        },
    ),
    TrainConfig(
        name="m1_naive_mixing",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoPlusDataConfig(
            repo_id="data/libero_plus_camera_perturbation",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            use_wrist_image=False,
            video_backend="pyav",
        ),
        batch_size=384,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=4e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=24,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # Modern full-DROID training path backed by local HDF5 shards.
        name="pi0_fast_full_droid_modern_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=ShardedDROIDDataConfig(
            repo_id="droid",
            sharded_droid_dir="<path_to_droid_shards>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=8,
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # Modern full-DROID training path backed by local HDF5 shards.
        name="pi05_full_droid_modern_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=ShardedDROIDDataConfig(
            repo_id="droid",
            sharded_droid_dir="data/processed_droid_parallel",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=384,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=8,
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
