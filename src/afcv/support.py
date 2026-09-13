"""Camera-pose support used by the paper's restricted-support experiment."""


def signed_angle(value):
    return (int(value) + 180) % 360 - 180


def sample_restricted_camera(rng):
    axis = rng.choice(["azimuth", "distance", "endpoint", "elevation"])
    params = {"horizon": 0, "vertical": 0, "scale": 100, "end_rot": 0, "end_vert": 0, "category": "C2"}
    if axis == "azimuth":
        magnitude = int(rng.choice(list(range(1, 21)) + list(range(30, 63))))
        params["horizon"] = (magnitude * int(rng.choice([-1, 1]))) % 360
    elif axis == "distance":
        params.update(scale=int(rng.choice(list(range(100, 129)) + list(range(142, 176)))), category="C1")
    elif axis == "endpoint":
        params.update(
            end_rot=int(rng.choice([-7, -5, -3, 3, 5, 7])) % 360,
            end_vert=int(rng.choice([-7, -5, -3, 3, 5, 7])) % 360,
            category="C3",
        )
    else:
        params["vertical"] = 8
    return params


def classify_camera(params):
    """Return an axis and evaluation band; compound changes remain separate."""
    az = abs(signed_angle(params["horizon"]))
    el = int(params["vertical"])
    distance = int(params["scale"])
    endpoint = [abs(signed_angle(params[k])) for k in ["end_rot", "end_vert"]]
    if az and el:
        return "compound", "held_out"
    if az:
        return "azimuth", "extrapolation" if az >= 63 else "interpolation" if 21 <= az <= 29 else "support"
    if el:
        return "elevation", "transfer" if el == 15 else "support"
    if distance != 100:
        return (
            "distance",
            "extrapolation" if distance >= 176 else "interpolation" if 129 <= distance <= 141 else "support",
        )
    if any(endpoint):
        return "endpoint", "extrapolation" if any(x in [8, 10] for x in endpoint) else "interpolation" if any(
            x in [4, 6] for x in endpoint
        ) else "support"
    return "nominal", "support"
