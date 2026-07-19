"""ctypes binding for the Dendritron RBF forward-pass HIP kernel.

Loads libdendritron_rbf.so (built via hipcc from dendritron_rbf.hip.cpp) and
exposes a clean numpy-in / numpy-out interface: rbf_forward_gpu().
"""

import ctypes
import os

import numpy as np

_SO_NAME = "libdendritron_rbf.so"
_here = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_here, _SO_NAME)

_lib = ctypes.CDLL(_lib_path)

_lib.rbf_forward.restype = ctypes.c_int
_lib.rbf_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.POINTER(ctypes.c_float),  # centers
    ctypes.POINTER(ctypes.c_float),  # sigmas
    ctypes.POINTER(ctypes.c_float),  # outputs
    ctypes.POINTER(ctypes.c_int),    # active_mask
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_int,  # N
    ctypes.c_int,  # B
    ctypes.c_int,  # D
    ctypes.c_int,  # O
]


def _f32_ptr(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def rbf_forward_gpu(
    X: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    outputs: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Compute the Dendritron RBF forward pass (integration="sum",
    Euclidean single chart) on the GPU.

    Parameters
    ----------
    X:        [N, D] float array of input samples
    centers:  [B, D] float array of branch centers
    sigmas:   [B]    float array of branch RBF widths (sigma)
    outputs:  [B, O] float array of per-branch output vectors
    active:   [B]    bool (or 0/1) array marking active branches

    Returns
    -------
    out: [N, O] float32 numpy array, out[n, o] = sum over active branches b
         of exp(-0.5*(||X[n]-centers[b]||/max(sigma[b],1e-8))^2) * outputs[b,o]

    Note: computation happens in float32 on-device for throughput on gfx803
    (weak fp64). Expect ~1e-5 to 1e-4 absolute deviation from a float64 numpy
    reference; validated at atol=1e-4, rtol=1e-3 in correctness_test.py.
    """
    X = np.atleast_2d(X)
    centers = np.atleast_2d(centers)
    outputs = np.atleast_2d(outputs)
    sigmas = np.asarray(sigmas)
    active = np.asarray(active)

    N, D = X.shape
    B, D_c = centers.shape
    B_o, O = outputs.shape
    assert D == D_c, f"X dim {D} != centers dim {D_c}"
    assert B == B_o, f"centers branches {B} != outputs branches {B_o}"
    assert sigmas.shape == (B,), f"sigmas shape {sigmas.shape} != ({B},)"
    assert active.shape == (B,), f"active shape {active.shape} != ({B},)"

    X_f, X_p = _f32_ptr(X)
    centers_f, centers_p = _f32_ptr(centers)
    sigmas_f, sigmas_p = _f32_ptr(sigmas)
    outputs_f, outputs_p = _f32_ptr(outputs)

    active_i = np.ascontiguousarray(active.astype(np.int32))
    active_p = active_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    out = np.empty((N, O), dtype=np.float32)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = _lib.rbf_forward(
        X_p, centers_p, sigmas_p, outputs_p, active_p, out_p, N, B, D, O
    )
    if rc != 0:
        raise RuntimeError(f"rbf_forward HIP kernel failed with code {rc}")

    return out
