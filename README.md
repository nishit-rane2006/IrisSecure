# IrisSecure

Iris-based biometric authentication using ResNet50 + ArcFace on the CASIA-Iris-Thousand dataset. Produces 512-D L2-normalized embeddings for 1:1 verification, 1:N identification, pairwise comparison, and webcam capture.

The dataset and generated biometric artifacts are not included in this repository. Obtain the dataset from its authorized source and follow its terms before use.

## Setup (once)

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
```

## Optional VS Code launch settings

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Pipeline (cached)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/pipeline.py",
      "console": "integratedTerminal"
    },
    {
      "name": "Pipeline (force)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/pipeline.py",
      "args": ["--force"],
      "console": "integratedTerminal"
    },
    {
      "name": "Gradio UI",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/app.py",
      "console": "integratedTerminal"
    }
  ]
}
```
