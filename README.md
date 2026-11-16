# IrisSecure + IrisGuard

This repository preserves the original IrisSecure baseline for deep iris biometric recognition and adds an IrisGuard research extension for adaptive template protection against gradual poisoning.

The baseline remains useful for recognition, registration, verification, and evaluation on CASIA-Iris-Thousand. The IrisGuard layer introduces immutable enrollment references, adaptive template management, AFSDD drift detection, and synthetic attack experiments.

The dataset and generated biometric artifacts are not included in this repository. Obtain the dataset from its authorized source and follow its terms before use.

## Baseline setup

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Place the CASIA-Iris-Thousand dataset under `dataset/CASIA-Iris-Thousand/...` (or `data/CASIA-Iris-Thousand/...` if you prefer to mirror the default config) and then run:

```bash
python pipeline.py
```

This runs five cached stages in order: clean → preprocess → train → embed → evaluate.

## IrisGuard research experiments

The research extension is implemented in `irisguard/` and can be exercised directly:

```bash
python main.py
```

This runs a synthetic embedding-level evaluation of:
- legitimate template adaptation,
- unprotected adaptive updates,
- quality-aware updates,
- AFSDD-protected updates,
- gradual poisoning and slow drift trajectories.

The simulator is intentionally synthetic and is designed to study poisoning dynamics in a controlled research environment. It does not claim to use real iris datasets or real forensic attack traces unless explicitly provided by the user.

## Useful commands

```bash
python pipeline.py --status
python pipeline.py --only train
python pipeline.py --from train
python pipeline.py --force
python pipeline.py --max_classes 800 --epochs 10
python register.py --user_id alice_L --image data/CASIA-Iris-Thousand/000/L/S5000L00.jpg --eye L
python verify.py --image data/CASIA-Iris-Thousand/000/L/S5000L01.jpg --against alice_L
python verify.py --image data/CASIA-Iris-Thousand/000/L/S5000L02.jpg --identify --top_k 5
python app.py
python main.py
pytest -q
```

## IrisGuard design summary

The project implements:
- immutable enrollment templates stored separately from adaptive templates,
- template versioning with rollback,
- multiple update policies: static, unprotected, quality-aware, and AFSDD-protected,
- AFSDD anomaly scoring combining reference deviation, temporal drift, cumulative drift, and quality penalties,
- attack scenarios for legitimate adaptation, single-step poisoning, gradual poisoning, and slow drift,
- reproducible synthetic experiment summaries.

This architecture is intended to support research on gradual template poisoning without silently overwriting the original biometric enrollment reference.
