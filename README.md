# IrisSecure + IrisGuard

This repository preserves the original IrisSecure baseline for deep iris biometric recognition and adds an IrisGuard research extension for adaptive template protection against gradual poisoning.

The baseline remains useful for recognition, registration, verification, and evaluation on CASIA-Iris-Thousand. The IrisGuard layer introduces immutable enrollment references, adaptive template management, AFSDD drift detection, and synthetic attack experiments.

The dataset and generated biometric artifacts are not included in this repository. Obtain CASIA-Iris-Thousand from its authorized source and follow its access and usage terms. Keep the dataset local; the repository ignores the `IrisSecure-Dataset/` folder.

## Baseline setup

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Configure the local dataset

Put the authorized dataset inside the project directory using either of these layouts:

```text
IrisSecure-Dataset/
└── CASIA-Iris-Thousand/
    └── <subject_id>/
        ├── L/
        └── R/
```

or, if the archive contains a duplicate nested directory:

```text
IrisSecure-Dataset/
└── CASIA-Iris-Thousand/
    └── CASIA-Iris-Thousand/
        └── <subject_id>/
            ├── L/
            └── R/
```

The loader searches for `IrisSecure-Dataset/` or `CASIA-Iris-Thousand/` and handles the nested layout. Left and right eyes are treated as separate identities. Do not rename or move the original image files. If the data is stored elsewhere, set `Config.data_root` in `config.py` to its actual subject-folder root.

Generate the dataset manifest and observed statistics without starting training:

```bash
python -c "from irisguard.dataset_inspector import write_manifest_and_statistics; print(write_manifest_and_statistics()[:2])"
```

This writes `data/processed/dataset_manifest.csv` and `data/processed/dataset_statistics.json`. These generated files contain paths and image metadata, not copies of the images. If discovery fails, verify the folder layout above or configure `data_root`.

### Run the baseline pipeline

After setting up the dataset, start the training and evaluation pipeline:

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
