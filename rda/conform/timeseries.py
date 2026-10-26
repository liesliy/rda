"""附录 B 三项纯算法（GB/Z 218.1-2026）。

只依赖 numpy，不需要视频像素层。三个函数的 docstring 都写明对应公式编号，
实现必须与公式逐字一致——尤其是 B.1 的分母口径。

B.1 分母口径警告
----------------
标准 B.1 的分母是**全长度 T 项求和**：

    R(k) = Σ_{t=1}^{T-k} (x_t - x̄)(x_{t+k} - x̄)  /  Σ_{t=1}^{T} (x_t - x̄)²

而不是常见的 T-k 项。这不是笔误——用 T-k 会让 R(k) 在 k 接近 T 时被人为
放大，是本标准明确采用的特殊口径。改成分母 T-k 即构成"方法不符"，
会使整份报告失去可比性。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_sequence(x) -> np.ndarray:
    """Coerce input to a ``(T, D)`` float64 array.

    1-D input becomes ``(T, 1)`` — ``np.atleast_2d`` would instead produce
    ``(1, T)``, which silently transposes every 1-D series.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def autocorrelation(x, k_max: int) -> np.ndarray:
    """附录 B.1 自相关函数。

    公式（B.1）::

        R(k) = Σ_{t=1}^{T-k} (x_t - x̄)(x_{t+k} - x̄) / Σ_{t=1}^{T} (x_t - x̄)²

    分母为**全长度 T 项**求和，不是 T-k 项（本标准特殊口径，见模块 docstring）。

    Args:
        x: 一维实数序列，长度 T。
        k_max: 需要计算的滞后阶数上界，返回长度 k_max 的数组，
            第 k-1 个元素即 R(k)。

    Returns:
        ``np.ndarray``，shape ``(k_max,)``，dtype float64。两种"无定义"的情况：
        序列方差为 0（常数序列）时返回 0；``k >= T`` 时分子是空和，返回
        ``nan``。两者都不能被读成"相关"或"不相关"——调用方必须显式处理。
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    T = arr.size
    if T == 0 or k_max <= 0:
        return np.zeros(max(k_max, 0), dtype=np.float64)
    centred = arr - arr.mean()
    # 分母：全长度 T 项求和（B.1 口径）
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        # 常数序列：自相关无定义。返回 0 而不是 1，避免被误读成"完全相关"。
        return np.zeros(k_max, dtype=np.float64)
    out = np.full(k_max, np.nan, dtype=np.float64)
    for k in range(1, k_max + 1):
        if k >= T:
            # 分子是空和（T-k ≤ 0 项），R(k) 无定义。返回 nan 而不是 0，
            # 免得被当成"该滞后阶不相关"。
            continue
        # 分子：T-k 项求和
        out[k - 1] = float(np.dot(centred[: T - k], centred[k:]) / denom)
    return out


def fourier_transform(x, freqs) -> np.ndarray:
    """附录 B.2 离散傅里叶变换。

    公式（B.2）::

        X(f) = Σ_{t=0}^{T-1} x(t) · e^{-i·2π·f·t}

    注意：标准写法中不含 1/T 归一化因子，也不含 1/√T 对称归一化；
    实现必须与公式逐字一致，归一化留给调用方在需要时自行处理
    （本模块不代为归一化，否则无法与标准原文对账）。

    Args:
        x: 一维实数或复数序列，长度 T。
        freqs: 频率点。可以是标量或一维数组（归一化频率 f，与公式中的
            f 同义；t 取 0..T-1 的整数下标）。

    Returns:
        ``np.ndarray``，dtype complex128，与 ``freqs`` 同形状。
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    T = arr.size
    t = np.arange(T, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    scalar_input = f.ndim == 0
    f_flat = np.atleast_1d(f).ravel()
    # e^{-i·2π·f·t}：外层是 f，内层是 t
    kernel = np.exp(-1j * 2.0 * np.pi * np.outer(f_flat, t))
    out = kernel @ arr
    if scalar_input:
        return out.reshape(())[()]
    return out.reshape(f.shape)


def dtw(a, b) -> Tuple[float, int]:
    """附录 B.3 动态时间规整（DTW）。

    标准递推::

        D(0,0) = 0
        D(i,0) = D(0,j) = +∞   (i,j ≥ 1)
        D(i,j) = d(i,j) + min( D(i-1,j), D(i,j-1), D(i-1,j-1) )

    其中 d(i,j) 为 a_i 与 b_j 之间的距离（本实现用欧氏距离，支持多维
    序列，shape ``(T, D)``）。

    输出 DTW 与按最优路径长度归一化的 ``DTW / L``。归一化用最优路径的
    实际步数 L（而不是 n+m 或 max(n,m)），因为 L 才是参与累加的真实项数。

    Args:
        a: 序列 A，shape ``(n,)`` 或 ``(n, D)``。
        b: 序列 B，shape ``(m,)`` 或 ``(m, D)``。

    Returns:
        ``(dtw_distance, path_length)``。调用方用 ``distance / path_length``
        得到归一化值。两者为 0 长度的退化输入时返回 ``(0.0, 0)``。
    """
    A = _as_sequence(a)
    B = _as_sequence(b)
    n, m = A.shape[0], B.shape[0]
    if n == 0 or m == 0:
        return 0.0, 0

    # 距离矩阵 d(i,j)
    diff = A[:, None, :] - B[None, :, :]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # 沿反对角线推进：D(i,j) 只依赖 s-1 与 s-2 两条对角线，可整条向量化。
    # 数学上与逐元素递推完全等价，只是把 O(n·m) 的 Python 循环压成
    # O(n+m) 次 numpy 运算——大序列上快一到两个数量级。
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for s in range(2, n + m + 1):
        i_lo = max(1, s - m)
        i_hi = min(n, s - 1)
        if i_lo > i_hi:
            continue
        i = np.arange(i_lo, i_hi + 1)
        j = s - i
        D[i, j] = d[i - 1, j - 1] + np.minimum(
            np.minimum(D[i - 1, j], D[i, j - 1]), D[i - 1, j - 1]
        )

    distance = float(D[n, m])

    # 回溯最优路径求 L
    i, j, length = n, m, 0
    while i > 0 and j > 0:
        length += 1
        step = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        if step == D[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif step == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return distance, length


def dtw_normalized(a, b) -> Tuple[float, float, int]:
    """DTW 便捷封装，直接给出归一化值。

    Args:
        a: 序列 A。
        b: 序列 B。

    Returns:
        ``(dtw_distance, dtw_over_L, path_length)``。
        ``path_length`` 为 0 时 ``dtw_over_L`` 返回 ``nan``（0/0 无定义），
        调用方应据此判 UNDETERMINED 而不是当成 0。
    """
    distance, length = dtw(a, b)
    if length == 0:
        return distance, float("nan"), 0
    return distance, distance / length, length


def resample_sequence(arr, T: int) -> np.ndarray:
    """把长度可变的序列线性重采样到固定长度 T。

    用于 Hausdorff 这类要求"两条轨迹点数可比"的算法。逐维线性插值，
    端点对齐（0 与 1 分别映射到首末样本）。

    Args:
        arr: shape ``(n, D)``。
        T: 目标长度。

    Returns:
        shape ``(T, D)`` 的 float64 数组。
    """
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim == 1:
        src = src.reshape(-1, 1)
    n = src.shape[0]
    if n == 0:
        return np.zeros((0, src.shape[1]), dtype=np.float64)
    if n == T:
        return src.astype(np.float64)
    src_grid = np.linspace(0.0, 1.0, n)
    dst_grid = np.linspace(0.0, 1.0, T)
    out = np.empty((T, src.shape[1]), dtype=np.float64)
    for d in range(src.shape[1]):
        out[:, d] = np.interp(dst_grid, src_grid, src[:, d])
    return out


def hausdorff_distance(a, b) -> float:
    """两条轨迹（视为点集）之间的对称 Hausdorff 距离。

    ``H(A,B) = max( sup_{a∈A} inf_{b∈B} ||a-b||, sup_{b∈B} inf_{a∈A} ||b-a|| )``

    Args:
        a: shape ``(n, D)``。
        b: shape ``(m, D)``。

    Returns:
        对称 Hausdorff 距离（float）。任一点集为空时返回 ``nan``，
        调用方必须据此判 UNDETERMINED。
    """
    A = _as_sequence(a)
    B = _as_sequence(b)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    diff = A[:, None, :] - B[None, :, :]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
