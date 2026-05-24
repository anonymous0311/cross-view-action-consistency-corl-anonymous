import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_required_alias(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required key. Expected one of: {keys}. Got keys: {sorted(data)}")


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    # If false, or if the wrist key is absent, the wrist stream is zero-padded.
    use_wrist_image: bool = True

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(
            _get_required_alias(
                data,
                (
                    "observation/image",
                    "observation.scene_image",
                    "nominal_image",
                    "observation.images.front",
                    "observation.images.nominal_image",
                ),
            )
        )
        wrist_image_value = None
        for wrist_key in ("observation/wrist_image", "observation.wrist_image"):
            if wrist_key in data:
                wrist_image_value = data[wrist_key]
                break
        has_wrist_image = self.use_wrist_image and wrist_image_value is not None
        wrist_image = _parse_image(wrist_image_value) if has_wrist_image else np.zeros_like(base_image)
        state = _get_required_alias(data, ("observation/state", "observation.state"))

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist_image or self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "canonical_tokens" in data:
            inputs["canonical_tokens"] = data["canonical_tokens"]
        if "canonical_tokens_neg" in data:
            inputs["canonical_tokens_neg"] = data["canonical_tokens_neg"]
        if "canonical_tokens_mean" in data:
            inputs["canonical_tokens_mean"] = data["canonical_tokens_mean"]
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "sample_index" in data:
            inputs["sample_index"] = data["sample_index"]
        if "camera_bin_id" in data:
            inputs["camera_bin_id"] = data["camera_bin_id"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoSceneInputs(transforms.DataTransformFn):
    """LIBERO inputs that emit only the third-person scene image."""

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(
            _get_required_alias(
                data,
                (
                    "observation/image",
                    "observation.scene_image",
                    "nominal_image",
                    "observation.images.front",
                    "observation.images.nominal_image",
                ),
            )
        )
        state = _get_required_alias(data, ("observation/state", "observation.state"))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "canonical_tokens" in data:
            inputs["canonical_tokens"] = data["canonical_tokens"]
        if "canonical_tokens_neg" in data:
            inputs["canonical_tokens_neg"] = data["canonical_tokens_neg"]
        if "canonical_tokens_mean" in data:
            inputs["canonical_tokens_mean"] = data["canonical_tokens_mean"]
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "sample_index" in data:
            inputs["sample_index"] = data["sample_index"]
        if "camera_bin_id" in data:
            inputs["camera_bin_id"] = data["camera_bin_id"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoPairInputs(transforms.DataTransformFn):
    """LIBERO same-state pair inputs for Phase 0B cross-view consistency.

    The output keeps a pair axis in image/state/action fields:

    - images: ``[2, H, W, C]`` for nominal and perturbed scene views
    - state: ``[2, state_dim]``
    - actions: ``[2, action_horizon, action_dim]``

    The model loss flattens this pair axis for the forward pass while sharing
    the same flow time/noise across both views.
    """

    model_type: _model.ModelType
    use_wrist_image: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        perturbed_image = _parse_image(data["observation/image_perturbed"])
        pair_images = np.stack([base_image, perturbed_image], axis=0)

        state = np.asarray(data["observation/state"])
        pair_state = np.stack([state, state], axis=0)

        has_wrist_image = self.use_wrist_image and "observation/wrist_image" in data
        images = {
            "base_0_rgb": pair_images,
        }
        image_masks = {
            "base_0_rgb": np.asarray([np.True_, np.True_]),
        }
        if has_wrist_image:
            wrist_image = _parse_image(data["observation/wrist_image"])
            images["left_wrist_0_rgb"] = np.stack([wrist_image, wrist_image], axis=0)
            image_masks["left_wrist_0_rgb"] = np.asarray([np.True_, np.True_])

        inputs = {
            "state": pair_state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = np.stack([actions, actions], axis=0)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "task_index" in data:
            value = np.asarray(data["task_index"])
            inputs["task_index"] = np.stack([value, value], axis=0)
        if "episode_index" in data:
            value = np.asarray(data["episode_index"])
            inputs["episode_index"] = np.stack([value, value], axis=0)
        if "sample_index" in data:
            value = np.asarray(data["sample_index"])
            inputs["sample_index"] = np.stack([value, value], axis=0)

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoPairTokenInputs(transforms.DataTransformFn):
    """Replicate tokenized shared fields onto the pair axis after tokenization."""

    def __call__(self, data: dict) -> dict:
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            if key in data:
                value = np.asarray(data[key])
                if value.ndim == 0 or value.shape[0] != 2:
                    data[key] = np.stack([value, value], axis=0)
        return data


@dataclasses.dataclass(frozen=True)
class LiberoAnchorInputs(transforms.DataTransformFn):
    """LIBERO inputs for M6-A, where RGB is unused and canonical tokens are precomputed."""

    def __call__(self, data: dict) -> dict:
        dummy_image = np.zeros((1, 1, 3), dtype=np.float32)
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": dummy_image,
                "left_wrist_0_rgb": dummy_image,
                "right_wrist_0_rgb": dummy_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "canonical_tokens" in data:
            inputs["canonical_tokens"] = data["canonical_tokens"]
        if "canonical_tokens_neg" in data:
            inputs["canonical_tokens_neg"] = data["canonical_tokens_neg"]
        if "canonical_tokens_mean" in data:
            inputs["canonical_tokens_mean"] = data["canonical_tokens_mean"]
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "sample_index" in data:
            inputs["sample_index"] = data["sample_index"]
        if "camera_bin_id" in data:
            inputs["camera_bin_id"] = data["camera_bin_id"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For realman Task1, configs set this to 8 to keep the gripper dimension.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
