"""Convert the public dataset into VERL inputs without changing labels or splits."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_aligned_external_eval import format_aligned_messages


def to_verl(row):
    labels = [s["label"] for s in row["steps"]]
    return {
        "data_source": "lean_nl_verifier",
        "prompt": format_aligned_messages(
            row["problem"], [s["text"] for s in row["steps"]]
        ),
        "ability": "proof_verification",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"steps": labels, "overall": row["overall"]}, separators=(",", ":")
            ),
        },
        "extra_info": {
            "id": row["id"],
            "proof_index": row["proof_index"],
            "category": row["category"],
            "source": row["training_source"],
            "theorem_group": row["theorem_group"],
            "overall": row["overall"],
            "step_count": len(labels),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", required=True, help="Local HF dataset directory, or Hub dataset ID"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--historical-validation",
        action="store_true",
        help="Use original 200 validation examples for paper-run reproduction only",
    )
    args = p.parse_args()
    local = Path(args.dataset)
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output must be empty")
    if local.is_dir():
        data = {
            s: pq.read_table(
                local / ("benchmark" if s == "test" else "data") / f"{s}.parquet"
            ).to_pylist()
            for s in ["train", "validation", "test"]
        }
        get_file = lambda name: local / name
    else:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        data = {
            s: list(rows) for s, rows in load_dataset(args.dataset).items()
        }
        get_file = lambda name: Path(
            hf_hub_download(args.dataset, name, repo_type="dataset")
        )
    if args.historical_validation:
        data["validation"] = pq.read_table(
            get_file("provenance/validation_during_training.parquet")
        ).to_pylist()
    rows = {s: [to_verl(r) for r in rs] for s, rs in data.items()}
    order = json.loads(get_file("provenance/train_order.json").read_text())
    lookup = {
        (r["extra_info"]["id"], r["extra_info"]["proof_index"]): r
        for r in rows["train"]
    }
    weighted = [lookup[tuple(k)] for k in order]
    rows["train_weighted"] = weighted
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        pq.write_table(
            pa.Table.from_pylist(rs),
            args.output / f"{split}.parquet",
            compression="zstd",
        )
    (args.output / "preparation.json").write_text(
        json.dumps(
            {
                "historical_validation": args.historical_validation,
                "counts": {s: len(r) for s, r in rows.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print({s: len(rs) for s, rs in rows.items()})


if __name__ == "__main__":
    main()
