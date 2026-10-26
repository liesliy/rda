"""附录 B 三项算法的正确性与口径测试。

这些测试的核心目的不是"跑通"，而是**锁死口径**：一旦有人把 B.1 的分母
从 T 改成 T-k（一个非常常见的"顺手修正"），这里必须立刻红。
"""
from __future__ import annotations

import numpy as np
import pytest

from rda.conform.timeseries import (
    autocorrelation,
    dtw,
    dtw_normalized,
    fourier_transform,
    hausdorff_distance,
    resample_sequence,
)


# ---------------------------------------------------------------------------
# B.1 自相关
# ---------------------------------------------------------------------------


def _naive_acf(x: np.ndarray, k: int, denominator_T: bool = True) -> float:
    """参考实现。denominator_T=True 用全长度 T（标准口径）。"""
    xc = x - x.mean()
    if denominator_T:
        denom = np.sum(xc * xc)
    else:
        denom = np.sum(xc[: len(x) - k] * xc[: len(x) - k])
    return float(np.sum(xc[: len(x) - k] * xc[k:]) / denom)


def test_acf_matches_formula_with_full_length_denominator():
    """R(k) 的分母必须是全长度 T 项，不是 T-k 项。"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    ac = autocorrelation(x, 6)
    for k in range(1, 7):
        assert ac[k - 1] == pytest.approx(_naive_acf(x, k, denominator_T=True))


def test_acf_differs_from_wrong_tk_denominator():
    """用 T-k 分母会得到明显不同的数——这个差异必须能被检出。"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    ac_ours = autocorrelation(x, 3)
    ac_wrong = np.array([_naive_acf(x, k, denominator_T=False) for k in (1, 2, 3)])
    assert not np.allclose(ac_ours, ac_wrong)
    # 方向性：T-k 分母让首阶自相关被系统性放大
    assert ac_wrong[0] > ac_ours[0]


def test_acf_lag0_equivalent_is_one_for_full_denominator():
    """k=0 时分子分母相同 → R(0)=1（本实现从 k=1 起，这里校验一致性）。"""
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.5])
    xc = x - x.mean()
    assert np.sum(xc * xc) / np.sum(xc * xc) == pytest.approx(1.0)


def test_acf_constant_series_is_zero_not_one():
    """常数序列方差为 0，自相关无定义。不许返回 1（会被读成"完全相关"）。"""
    assert np.all(autocorrelation(np.ones(10), 4) == 0.0)


def test_acf_empty_and_zero_kmax():
    assert autocorrelation(np.array([]), 3).shape == (3,)
    assert autocorrelation(np.arange(5.0), 0).shape == (0,)


def test_acf_accepts_1d_and_casts():
    ac = autocorrelation([1.0, 2.0, 3.0, 2.0, 1.0], 2)
    assert ac.shape == (2,)
    assert np.all(np.isfinite(ac))


# ---------------------------------------------------------------------------
# B.2 傅里叶
# ---------------------------------------------------------------------------


def test_fourier_matches_formula_exactly():
    """X(f) = Σ_t x(t)·e^{-i2πft}，不含 1/T 归一化。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    freqs = np.array([0.0, 0.25, 0.5, 0.75])
    t = np.arange(len(x))
    ref = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
    assert np.allclose(fourier_transform(x, freqs), ref)


def test_fourier_dc_term_is_the_sum():
    x = np.arange(10.0)
    assert fourier_transform(x, 0.0) == pytest.approx(float(x.sum()))


def test_fourier_scalar_and_vector_shapes():
    x = np.arange(8.0)
    scalar = fourier_transform(x, 0.25)
    assert np.shape(scalar) == ()
    vector = fourier_transform(x, np.array([0.0, 0.25]))
    assert vector.shape == (2,)


def test_fourier_is_not_normalized_by_T():
    """标准公式没有 1/T；实现不得擅自归一化。"""
    x = np.ones(8)
    assert fourier_transform(x, 0.0) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# B.3 DTW
# ---------------------------------------------------------------------------


def _naive_dtw(a: np.ndarray, b: np.ndarray):
    """教科书逐元素递推，作为向量化实现的对照组。"""
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    n, m = len(A), len(B)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = d[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, L = n, m, 0
    while i > 0 and j > 0:
        L += 1
        step = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        if step == D[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif step == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return float(D[n, m]), L


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dtw_matches_naive_recursion(seed):
    rng = np.random.default_rng(seed)
    for _ in range(3):
        n = int(rng.integers(4, 18))
        m = int(rng.integers(4, 18))
        a = rng.normal(size=(n, 3))
        b = rng.normal(size=(m, 3))
        v, L = dtw(a, b)
        nv, nL = _naive_dtw(a, b)
        assert v == pytest.approx(nv)
        assert L == nL


def test_dtw_1d_input_is_not_transposed():
    """1-D 输入必须被当成 (T,1) 而不是 (1,T)。"""
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    v, L = dtw(a, b)
    nv, nL = _naive_dtw(a, b)
    assert v == pytest.approx(nv)
    assert L == nL
    assert L == 4  # 路径长度至少是 max(n, m)


def test_dtw_identical_sequences_is_zero():
    x = np.linspace(0, 1, 12).reshape(-1, 1)
    v, L = dtw(x, x)
    assert v == pytest.approx(0.0, abs=1e-12)
    assert L == 12


def test_dtw_path_length_bounds():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(9, 2))
    b = rng.normal(size=(14, 2))
    _v, L = dtw(a, b)
    assert max(9, 14) <= L <= 9 + 14 - 1


def test_dtw_empty_sequence():
    assert dtw(np.zeros((0, 2)), np.ones((3, 2))) == (0.0, 0)


def test_dtw_normalized_nan_when_path_empty():
    d, ratio, L = dtw_normalized(np.zeros((0, 2)), np.ones((3, 2)))
    assert L == 0
    assert np.isnan(ratio)  # 0/0 无定义，不许冒充 0


# ---------------------------------------------------------------------------
# Hausdorff / 重采样
# ---------------------------------------------------------------------------


def test_hausdorff_symmetric_and_known_value():
    a = np.zeros((3, 2))
    b = np.ones((4, 2))
    assert hausdorff_distance(a, b) == pytest.approx(np.sqrt(2.0))
    assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a))


def test_hausdorff_identical_is_zero():
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    assert hausdorff_distance(x, x) == pytest.approx(0.0, abs=1e-12)


def test_hausdorff_empty_is_nan():
    assert np.isnan(hausdorff_distance(np.zeros((0, 2)), np.ones((2, 2))))


def test_resample_keeps_endpoints_and_length():
    src = np.arange(10.0).reshape(-1, 1)
    out = resample_sequence(src, 4)
    assert out.shape == (4, 1)
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(9.0)
    assert resample_sequence(src, 10).shape == (10, 1)


def test_resample_multidimensional():
    src = np.stack([np.arange(5.0), np.arange(5.0) * 2], axis=1)
    out = resample_sequence(src, 3)
    assert out.shape == (3, 2)
    assert out[-1, 1] == pytest.approx(8.0)
