# 多 LF 帧下一阶段：固定基础空间上的候选轨道评分

本阶段比较一个**有限、已给定**的候选池。复习 [单候选轨道阶段](lf_mr_orbit_trial_stage.md)：每个候选的分数是在同一个基础空间上，分别加入它的完整平移轨道后得到的最低 NOCI 能量。此阶段不生成候选、不自动接受候选，也不联合再优化位移。

## 1. 数学定义与选择边界

设基础帧张成 $V_B$，第 $j$ 个候选帧的完整平移轨道张成 $V_j$。对每个 $j=0,\ldots,J-1$，分别求

$$
E_j=\min_{\Psi\in V_B+V_j}
\frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle},
\qquad
G_j=E_B-E_j,
\qquad
\Delta r_j=r_j-r_B.
$$

$G_j$ 是相对于**当前**基础空间的能量收益，$\Delta r_j$ 是重叠矩阵保留秩的增加。两者都要报告：只看 `lam` 是否不同会混淆规范等价、平移等价与真正增加的物理方向。有限 `overlap_cut` 可导致极小数值误差，原始 $E_j$ 不应裁剪或排序。

这里的 $E_j$ 每次只加入第 $j$ 个候选，而不是同时加入所有候选。因此 $\min_j E_j$ 不等于全部候选合并后的能量。若以后进行逐步选择，选入一个轨道后基础空间改变，其余候选必须重新评分。本阶段只把分数公开，暂不规定收益阈值或贪心停止准则。

不必局限于 LF-HF 的局部极小值。α=2.4 的只读试算中，以对称帧加四个破缺帧平移副本为基础，$E_B=-3.038160786265$、$r_B=20$。沿两套 LF 位移做线性路径，两个端点的轨道已包含在基础空间；延长路径到参数 1.5 的候选得到 $E_j=-3.040366167030$、$r_j=36$，收益约 0.00220538。这个例子说明候选池可包含非驻点位移，但单次改善仍不等于 exact。该扩充空间的最小重叠本征值约为 0.01869；把 `overlap_cut` 在 `1e-8` 到 `1e-12` 间改变，能量和秩未变。

## 2. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_noci_score_orbits(
    tmat: np.ndarray,
    g: float,
    omega: float,
    base_lam: np.ndarray,
    base_shift: np.ndarray,
    candidate_lam: np.ndarray,
    candidate_shift: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    ...
```

| 数据 | dtype | shape 与顺序 |
| --- | --- | --- |
| `tmat` | `float64` | `(L,L)`，`[p,q]`，实对称 |
| `g`, `omega`, `overlap_cut` | 实标量 | `g` 有限，`omega>0`，`overlap_cut>0` |
| `base_lam`, `base_shift` | `float64` | `(K,L,L)`、`(K,L)`；`K>=1` |
| `candidate_lam`, `candidate_shift` | `float64` | `(J,L,L)`、`(J,L)`；`J>=1`；轴 0 是候选编号 |
| 返回 `E_base`, `rank_base` | 标量 | 基础空间的最低能量与保留秩 |
| 返回 `E_trial`, `rank_trial` | `float64`、`int64` | 两个 `(J,)` 数组，保持输入候选顺序 |

所有数组有限，输入不修改；不排序、不删除重复候选、不对能量收益取绝对值。`E_trial[j]` 是基础空间**只加入候选 j** 的结果。例行输入检查由我在验收时补齐，你重点保证 `j` 轴、返回顺序和固定基础空间的语义。

伪代码：

1. 获取 `J`，创建长度为 `J` 的浮点能量数组和整数秩数组。
2. 对每个 `j` 调用已有 `lf_noci_trial_orbit(...)`，传入同一个 `base_lam/base_shift` 和第 `j` 个候选。
3. 保存该次的扩充能量与秩；基础能量与秩从任意一次调用取出，推荐第一次。
4. 返回 `(E_base, rank_base, E_trial, rank_trial)`。不在循环中把候选加进基础集合。

## 3. 验收

两站点例子：$t_{01}=t_{10}=-1$、$g=0.8$、$\omega=1$、基础帧位移为零。候选 `lam=0` 重复基础空间；`lam=diag(0.4,0.4)` 与 `lam=diag(0.8,0.8)` 分别产生约 $-1.332547205409$ 与 $-1.326306686855$ 的扩充能量。基础能量为 $-1$，基础秩为 2。测试还会核对候选顺序置换、规范等价和输入不修改。

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_noci_score_orbits.py
```

实现前请想清楚：为何最小的 $E_j$ 仍不是全部候选共同加入时的能量？为何选入一个候选之后必须重算剩余候选的 $G_j$？完成函数后回复“请验收”。
