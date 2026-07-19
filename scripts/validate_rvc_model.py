"""Validate that a generated .pth is an inference-ready official RVC model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sample-rate", required=True, choices=("32k", "40k", "48k"))
    parser.add_argument("--version", required=True, choices=("v1", "v2"))
    parser.add_argument("--use-f0", required=True, choices=("0", "1"))
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model is missing: {model_path}")
    try:
        try:
            payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(model_path), map_location="cpu")
    except Exception as exc:
        raise SystemExit(f"Model cannot be loaded by PyTorch: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("Model payload is not a mapping")
    required = {"weight", "config", "version", "f0", "sr"}
    missing = sorted(required.difference(payload))
    if missing:
        raise SystemExit("Inference model is missing keys: " + ", ".join(missing))
    weights = payload["weight"]
    if not isinstance(weights, Mapping) or not weights:
        raise SystemExit("Inference model has no weight tensors")
    if any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise SystemExit("Inference model weight mapping contains non-tensor values")
    config = payload["config"]
    if not isinstance(config, Sequence) or isinstance(config, (str, bytes)) or len(config) < 18:
        raise SystemExit("Inference model config is missing or too short")
    if str(payload["version"]) != args.version:
        raise SystemExit(
            f"Model version {payload['version']!r} does not match {args.version!r}"
        )
    if int(payload["f0"]) != int(args.use_f0):
        raise SystemExit(f"Model f0={payload['f0']!r} does not match {args.use_f0}")
    if str(payload["sr"]) != args.sample_rate:
        raise SystemExit(
            f"Model sample-rate label {payload['sr']!r} does not match {args.sample_rate!r}"
        )
    expected_hz = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sample_rate]
    if int(config[-1]) != expected_hz:
        raise SystemExit(
            f"Model config sample rate {config[-1]!r} does not match {expected_hz}"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "model": str(model_path),
                "version": str(payload["version"]),
                "f0": int(payload["f0"]),
                "sample_rate": str(payload["sr"]),
                "weight_count": len(weights),
                "speaker_capacity": int(config[-3]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
