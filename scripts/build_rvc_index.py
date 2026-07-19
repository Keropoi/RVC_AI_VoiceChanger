"""Build a deterministic FAISS index from official RVC feature arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("auto", "flat", "ivf"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum-vectors", type=int, default=200_000)
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "FAISS is missing from the RVC virtual environment; install faiss-cpu."
        ) from exc

    feature_dir = args.feature_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not feature_dir.is_dir():
        raise SystemExit(f"Feature directory is missing: {feature_dir}")
    if output.suffix.lower() != ".index":
        raise SystemExit("--output must end in .index")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing index: {output}")
    if args.maximum_vectors < 1:
        raise SystemExit("--maximum-vectors must be positive")

    arrays = []
    feature_files = sorted(feature_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    if not feature_files:
        raise SystemExit(f"No .npy features found in {feature_dir}")
    dimension = None
    for path in feature_files:
        array = np.load(str(path), allow_pickle=False)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise SystemExit(f"Feature array must be non-empty and 2-D: {path}")
        if dimension is None:
            dimension = int(array.shape[1])
        if int(array.shape[1]) != dimension:
            raise SystemExit(f"Feature dimensions do not match: {path}")
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise SystemExit(f"Feature array contains NaN or Infinity: {path}")
        arrays.append(array)
    vectors = np.concatenate(arrays, axis=0)
    generator = np.random.default_rng(args.seed)
    order = generator.permutation(vectors.shape[0])
    if order.size > args.maximum_vectors:
        order = order[: args.maximum_vectors]
    vectors = np.ascontiguousarray(vectors[order], dtype=np.float32)
    count = int(vectors.shape[0])
    assert dimension is not None

    n_ivf = min(int(16 * math.sqrt(count)), count // 39)
    algorithm = args.algorithm
    if algorithm == "auto":
        algorithm = "ivf" if n_ivf >= 2 else "flat"
    if algorithm == "ivf" and n_ivf < 2:
        raise SystemExit(
            f"Only {count} feature vectors are available; use flat/auto for a tiny dataset."
        )
    if algorithm == "flat":
        index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.index_factory(dimension, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(vectors)
    index.add(vectors)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        faiss.write_index(index, str(temporary))
        if not temporary.is_file() or temporary.stat().st_size < 128:
            raise SystemExit("FAISS wrote an empty or implausibly small index")
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        "schema_version": 1,
        "feature_dir": str(feature_dir),
        "feature_files": [path.name for path in feature_files],
        "feature_file_sha256": {path.name: _sha256(path) for path in feature_files},
        "dimension": dimension,
        "vector_count": count,
        "algorithm": algorithm,
        "n_ivf": n_ivf if algorithm == "ivf" else None,
        "seed": args.seed,
        "index_sha256": _sha256(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built {algorithm} index with {count} vectors x {dimension} dimensions: {output}",
        flush=True,
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
