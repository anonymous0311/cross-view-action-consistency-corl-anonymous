import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def test_libero_inputs_accept_realman_robot_client_aliases():
    base_image = np.full((256, 256, 3), 17, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)

    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05, use_wrist_image=False)
    out = transform(
        {
            "observation.scene_image": base_image,
            "observation.state": state,
            "prompt": "pick up the blue batery and place it into cardboard box.",
        }
    )

    np.testing.assert_array_equal(out["image"]["base_0_rgb"], base_image)
    np.testing.assert_array_equal(out["state"], state)
    assert out["image_mask"]["base_0_rgb"]
    assert not out["image_mask"]["left_wrist_0_rgb"]
    assert out["prompt"] == "pick up the blue batery and place it into cardboard box."


def test_libero_inputs_accept_nominal_image_alias_from_robot_client():
    base_image = np.full((256, 256, 3), 23, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)

    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05, use_wrist_image=False)
    out = transform(
        {
            "nominal_image": base_image,
            "observation/state": state,
            "prompt": "close the laptop lid.",
        }
    )

    np.testing.assert_array_equal(out["image"]["base_0_rgb"], base_image)
    np.testing.assert_array_equal(out["state"], state)
    assert out["prompt"] == "close the laptop lid."


def test_libero_scene_inputs_omit_masked_wrist_slots():
    base_image = np.full((256, 256, 3), 31, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)

    transform = libero_policy.LiberoSceneInputs()
    out = transform(
        {
            "nominal_image": base_image,
            "observation/state": state,
            "prompt": "takeoff the headphone from the stand.",
        }
    )

    np.testing.assert_array_equal(out["image"]["base_0_rgb"], base_image)
    np.testing.assert_array_equal(out["state"], state)
    assert out["image_mask"]["base_0_rgb"]
    assert "left_wrist_0_rgb" not in out["image"]
    assert "right_wrist_0_rgb" not in out["image"]
    assert out["prompt"] == "takeoff the headphone from the stand."


def test_libero_pair_inputs_shapes_and_wrist_masks():
    base_image = np.full((256, 256, 3), 17, dtype=np.uint8)
    perturbed_image = np.full((256, 256, 3), 29, dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    actions = np.ones((10, 7), dtype=np.float32)

    transform = libero_policy.LiberoPairInputs(model_type=_model.ModelType.PI05, use_wrist_image=False)
    out = transform(
        {
            "observation/image": base_image,
            "observation/image_perturbed": perturbed_image,
            "observation/state": state,
            "actions": actions,
            "prompt": "pick up the bowl",
            "task_index": np.asarray(3),
            "episode_index": np.asarray(12),
            "sample_index": np.asarray(34),
        }
    )

    assert out["image"]["base_0_rgb"].shape == (2, 256, 256, 3)
    np.testing.assert_array_equal(out["image"]["base_0_rgb"][0], base_image)
    np.testing.assert_array_equal(out["image"]["base_0_rgb"][1], perturbed_image)
    assert out["state"].shape == (2, 8)
    np.testing.assert_array_equal(out["state"][0], state)
    np.testing.assert_array_equal(out["state"][1], state)
    assert out["actions"].shape == (2, 10, 7)
    np.testing.assert_array_equal(out["actions"][0], actions)
    np.testing.assert_array_equal(out["actions"][1], actions)
    np.testing.assert_array_equal(out["image_mask"]["base_0_rgb"], np.asarray([True, True]))
    assert "left_wrist_0_rgb" not in out["image"]
    assert "right_wrist_0_rgb" not in out["image"]
    assert "left_wrist_0_rgb" not in out["image_mask"]
    assert "right_wrist_0_rgb" not in out["image_mask"]
    assert out["prompt"] == "pick up the bowl"
    np.testing.assert_array_equal(out["task_index"], np.asarray([3, 3]))
    np.testing.assert_array_equal(out["episode_index"], np.asarray([12, 12]))
    np.testing.assert_array_equal(out["sample_index"], np.asarray([34, 34]))
