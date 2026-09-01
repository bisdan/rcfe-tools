import sys

import numpy as np
import pytest

from fast_pcc_rmsd.cuda_backend import CudaBackendError, cuda_minimize_vectors, load_cupy


def test_cuda_minimum_image_rejects_malformed_dimensions():
    with pytest.raises(CudaBackendError, match="expected 6 unit-cell values"):
        cuda_minimize_vectors(np.zeros((1, 3)), [10.0, 10.0, 10.0], np)


def test_cuda_minimum_image_matches_orthogonal_reference_without_gpu():
    vectors = np.array([[6.0, -6.0, 1.0], [-6.0, 6.0, -1.0]])
    actual = cuda_minimize_vectors(
        vectors,
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        np,
    )
    np.testing.assert_allclose(actual, [[-4.0, 4.0, 1.0], [4.0, -4.0, -1.0]])


def test_missing_cupy_error_names_both_installation_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(CudaBackendError, match=r"\[cuda12\].*\[cuda13\]"):
        load_cupy()
