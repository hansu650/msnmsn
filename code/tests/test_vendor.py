from pathlib import Path

import pytest
import torch

from paano_k0.vendor import (
    VendorMismatchError,
    build_encoder,
    load_vendor_symbols,
    verify_vendor_repo,
)


VENDOR_ROOT = Path(r"C:\Users\qintian\Desktop\msn\vendor\PaAno")
VENDOR_SHA = "d4c67116190efa4592dc6a8a157ced0def68b6af"


def test_vendor_sha_guard() -> None:
    fingerprint = verify_vendor_repo(VENDOR_ROOT, VENDOR_SHA)
    assert fingerprint.git_sha == VENDOR_SHA
    assert fingerprint.root == VENDOR_ROOT.resolve()
    with pytest.raises(VendorMismatchError, match="SHA mismatch"):
        verify_vendor_repo(VENDOR_ROOT, "0" * 40)


def test_encoder_forward_contract() -> None:
    symbols = load_vendor_symbols(VENDOR_ROOT, VENDOR_SHA)
    model = build_encoder(symbols, channels=1, use_revin=True, device=torch.device("cpu"))
    model.eval()
    batch = torch.randn(2, 1, 96)
    with torch.no_grad():
        embedding = model.embedding(batch)
        projection = model.projection(embedding)
    assert embedding.shape == (2, 64)
    assert projection.shape == (2, 256)

