"""Serve a trained single-scene-view policy over the OpenPI websocket protocol."""

import argparse

from afcv.evaluation.libero_plus_eval import load_pi05_checkpoint
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--config", default="paired_cv_eval")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    policy = load_pi05_checkpoint(args.checkpoint_dir, config_name=args.config)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
