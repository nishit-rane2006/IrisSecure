"""Template protection: AES-GCM encryption + cancelable biometrics."""
import os
from typing import Dict

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def encrypt_embedding(emb: np.ndarray, key: bytes) -> Dict[str, str]:
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, emb.astype(np.float32).tobytes(), None)
    return {"nonce": nonce.hex(), "ct": ct.hex()}


def decrypt_embedding(blob: Dict[str, str], key: bytes) -> np.ndarray:
    aes = AESGCM(key)
    pt = aes.decrypt(bytes.fromhex(blob["nonce"]), bytes.fromhex(blob["ct"]), None)
    return np.frombuffer(pt, dtype=np.float32).copy()


def random_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def apply_projection(emb: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return (Q @ emb.astype(np.float32)).astype(np.float32)
