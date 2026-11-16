# Artifact integrity and lifecycle

Generated artifacts live under ignored `artifacts/` and `work/` directories. `artifacts/manifest.json` tracks the path, SHA-256, version, timestamp, model hash, dataset fingerprint, preprocessing fingerprint, compatibility state, and metadata for each registered artifact.

`ArtifactManager` atomically replaces JSON and data files. When a registered script replaces an existing artifact, the prior file is copied into a timestamped `artifacts/backups/` directory. Backups are retained; cleanup is deliberately manual.

Validate current artifacts:

```powershell
python scripts/validate_artifacts.py
python scripts/validate_artifacts.py --require-all
```

Back up named manifest entries:

```powershell
python scripts/backup_artifacts.py model embeddings calibration
```

Restore one exact backup file and update its manifest hash:

```powershell
python scripts/restore_artifacts.py --name model --source "C:\path\to\backup\best_model.pth" --destination "C:\path\to\repo\work\artifacts\train\best_model.pth"
```

Biometric embeddings and model checkpoints are sensitive local data. They are not encrypted by the artifact manifest; a checksum detects accidental changes but does not prevent deliberate tampering. Keep local artifact directories protected and out of source control.
