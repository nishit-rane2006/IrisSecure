"""Train the embedding model only when explicitly requested."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import register_existing_checkpoint


def _checkpoint_is_valid() -> bool:
    from config import CFG
    from irisguard.artifact_manager import ArtifactError, ArtifactManager

    try:
        manager = ArtifactManager()
        entries = manager.read_manifest()["artifacts"]
        if "model" in entries:
            path = manager.validate("model")
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if int(checkpoint.get("embedding_dim", -1)) != CFG.embedding_dim:
                return False
        else:
            register_existing_checkpoint("adopted-legacy-checkpoint")
        return True
    except (ArtifactError, FileNotFoundError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--approve-training", action="store_true")
    parser.add_argument("--force", action="store_true", help="Back up and retrain even when a checkpoint is valid.")
    args = parser.parse_args()

    from config import CFG
    from config.hardware_config import configure_torch, resource_plan

    hardware = configure_torch()
    plan = resource_plan(hardware)
    print(f"Backend: {hardware.selected_backend}; CPU threads: {hardware.cpu_threads}")
    print(f"CPU: {hardware.cpu_model}; logical processors: {hardware.logical_processors}")
    print(
        f"PyTorch: {hardware.pytorch_version}; GPU detected: {hardware.gpu_devices or 'none'}; "
        f"resource plan: {plan.to_dict()}"
    )

    checkpoint_valid = _checkpoint_is_valid()
    if checkpoint_valid and not args.force:
        path = register_existing_checkpoint("reused-existing-checkpoint")
        print(f"Reusing validated checkpoint: {path}")
        return
    if not args.force and not args.approve_training:
        parser.error(
            "No compatible checkpoint is available. Review the dataset and run again with "
            "--approve-training to start training with the detected resource plan."
        )

    from stages.s0_clean import run_clean
    from stages.s1_preprocess import run_preprocess
    from stages.s2_train import run_train

    CFG.ensure_dirs()
    run_clean(force=False)
    run_preprocess(force=False)
    run_train(
        force=True,
        epochs=args.epochs if args.epochs is not None else CFG.epochs,
        batch_size=args.batch_size if args.batch_size is not None else plan.training_batch_size,
        grad_accum=args.grad_accum if args.grad_accum is not None else plan.gradient_accumulation_steps,
        max_classes=args.max_classes if args.max_classes is not None else CFG.max_classes,
        num_workers=args.num_workers if args.num_workers is not None else plan.dataloader_workers,
        early_stopping_patience=args.patience if args.patience is not None else CFG.early_stopping_patience,
    )
    path = register_existing_checkpoint("trained-by-train-once-script")
    print(f"Training complete. Registered checkpoint: {path}")


if __name__ == "__main__":
    main()
