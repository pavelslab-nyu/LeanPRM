# LeanPRM

Lean-grounded process supervision for natural-language proof verification.

[Training data](https://huggingface.co/datasets/pavelslab-nyu/LeanPRM/tree/main/data) · [LeanPRM-Bench](https://huggingface.co/datasets/pavelslab-nyu/LeanPRM/tree/main/benchmark)

## Dataset and benchmark

Both are in the same Hugging Face repository.

| Resource | HF split | Code |
| --- | --- | --- |
| LeanPRM training data | `train` | `prepare_data.py` and `train.py` |
| LeanPRM validation data | `validation` | `prepare_data.py` |
| LeanPRM-Bench | `test` | `evaluate_in_domain.py` |

```python
from datasets import load_dataset

train = load_dataset("pavelslab-nyu/LeanPRM", split="train")
benchmark = load_dataset("pavelslab-nyu/LeanPRM", split="test")
```

## Setup

```bash
pip install -e '.[pipeline]'
```

Training and local model evaluation use Python 3.12, PyTorch 2.8, CUDA 12.8, and the dependencies in `requirements-training.txt`. The pinned VERL fork requires access to the lab repository. Install a matching flash-attention 2.8.3 build for the training environment.

```bash
pip install -r requirements-training.txt
```

## Generate data

Install Lean through elan and export `ANTHROPIC_API_KEY`. The pipeline pins Lean 4.25.0-rc2 and Mathlib. It generates aligned natural-language and Lean proofs, checks compilation, and reviews the alignment.

```bash
python -m pipeline.generate --input examples/statements.json --output-dir runs/generation --dry-run
```

Remove `--dry-run` to generate. Use `--model` and `--review-model` to select available models. Current generation preserves the initial natural-language steps during review. Some historical runs allowed natural-language cleanup, so the published corpus is the reference for reproducing the paper.

`pipeline/select_numina.py` and `pipeline/select_balanced_hf_statements.py` select source statements. `pipeline/classify_difficulty.py` provides the difficulty classifier. `pipeline/filter_corpus.py` filters generated proofs and supports excluding held-out theorem groups.

## Train

Authenticate with Hugging Face to access the private dataset.

```bash
python prepare_data.py --dataset pavelslab-nyu/LeanPRM --historical-validation --output data/paper
python train.py --data-dir data/paper --model Qwen/Qwen3-4B --output runs/leanprm-4b --dry-run
```

Remove `--dry-run` to train. The launcher uses the paper's 300-update GRPO configuration. `verifier_task.py` contains the prompt and reward. Omit `--historical-validation` to use the final validation split. LeanPRM-Bench is the dataset’s test split. The dataset card explains historical validation overlap.

`solver_task.py` prepares the solver baseline and supplies its reward. `convert_prm800k.py` prepares the PRM800K baseline. Pass separately prepared solver data to `train.py --task solver`.

## Evaluate

`evaluate_in_domain.py` evaluates LeanPRM-Bench. `format_aligned_external_eval.py` evaluates Hard2Verify, ProcessBench, and PRMBench. `imo_gradingbench_eval.py` evaluates IMO-GradingBench with `--protocol ours`. Their shared adapters are in `external_eval.py`.

Use temperature 1 and decoding seeds 42, 43, and 44 for the paper's local-model repeats. `frontier_api_eval.py` and `codex_frontier_eval.py` provide the hosted evaluation paths. Hosted repetitions are independent runs, not controlled API seeds. Each entry point exposes its arguments through `--help`.
