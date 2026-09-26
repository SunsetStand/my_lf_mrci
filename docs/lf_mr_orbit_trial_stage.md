# 多 LF 帧下一阶段：一个候选平移轨道的变分能量收益

本阶段只评价**一套**候选 LF 帧。输入是已经选定的基础帧集合与一个候选种子；把种子的完整周期平移轨道加入基础集合，比较两次 NOCI 最低能量。不要在这一函数里搜索优化初值、选择多个候选、再优化位移或运行参数扫描。

先复习 [NOCI 求解阶段](lf_mr_noci_stage.md) 的广义 Rayleigh 商与重叠秩，以及 [平移轨道阶段](lf_mr_translation_stage.md) 的两个 site/mode 平移轴。

## 1. 理论：加入的是整个变分子空间

记基础空间为 $V_B=\operatorname{span}\{|A,p\rangle\}$，候选种子 $T$ 的完整平移轨道为 $\{T_R\}_{R=0}^{L-1}$。扩充空间是

$$
V_{B+T}=\operatorname{span}\bigl(V_B\cup\{|T_R,p\rangle:\;R,p=0,\ldots,L-1\}\bigr).
$$

两次求解均使用同一未中心化 Hamiltonian。若 $C^\mathsf{T}SC=1$，最低能量是广义 Rayleigh 商的极小值：

$$
E_B=\min_{\Psi\in V_B}\frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle},\qquad
E_{B+T}=\min_{\Psi\in V_{B+T}}\frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle}.
$$

精确算术和完整保留物理空间时，$V_B\subseteq V_{B+T}$，所以 $E_{B+T}\le E_B$。定义收益 $\Delta E=E_B-E_{B+T}$，越大表示该候选在当前空间上增加的变分能力越强。候选单独的最低能量不能替代 $\Delta E$：合并后的交叉 $H_{BT}$ 与 $S_{BT}$ 块会影响结果。

完整矩阵可按基础帧与候选轨道分块：

$$
H_{B+T}=\begin{pmatrix}H_{BB}&H_{BT}\\H_{TB}&H_{TT}\end{pmatrix},\qquad
S_{B+T}=\begin{pmatrix}S_{BB}&S_{BT}\\S_{TB}&S_{TT}\end{pmatrix}.
$$

重复、规范等价或平移等价的帧可能不增加物理空间的秩，故同时记录重叠矩阵的保留秩。`lf_noci_lowest` 的有限 `overlap_cut` 可能改变保留子空间；数值上若出现极小负收益，不要擅自截成零，先检查秩和阈值。

若基础帧集合本身对平移封闭，加入候选的**完整轨道**仍保持空间的平移封闭性。只加入一个局域候选通常会破坏这一性质。

## 2. 本阶段唯一函数

在 `src/lf_mr.py` 增加：

```python
def lf_noci_trial_orbit(
    tmat: np.ndarray,
    g: float,
    omega: float,
    base_lam: np.ndarray,
    base_shift: np.ndarray,
    seed_lam: np.ndarray,
    seed_shift: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
) -> tuple[float, float, int, int]:
    ...
```

| 数据 | dtype | shape / 索引 |
| --- | --- | --- |
| `tmat` | `float64` | `(L,L)`，`[p,q]`，实对称 |
| `g`, `omega`, `overlap_cut` | 实标量 | `g` 有限，`omega>0`，`overlap_cut>0` |
| `base_lam` | `float64` | `(K,L,L)`，`[A,x,p]`，`K>=1` |
| `base_shift` | `float64` | `(K,L)`，`[A,x]` |
| `seed_lam` | `float64` | `(L,L)`，`[x,p]` |
| `seed_shift` | `float64` | `(L,)`，`[x]` |
| 返回 | 标量 | `(E_base, E_aug, rank_base, rank_aug)` |

基础集合通常是前一步选出的完整平移轨道；函数不强制检查其对称性。所有数组输入必须有限且不被修改。仅计算能量和保留秩；`gain = E_base - E_aug` 可由调用者直接计算。例行输入检查我会在验收时补齐，你先集中处理物理流程与正确的帧轴。

伪代码：

1. 用已有的 `lf_frame_overlap`、`lf_frame_hamiltonian` 和 `lf_noci_lowest` 求基础能量与秩。
2. 调用 `lf_translation_orbit(seed_lam, seed_shift)`；它已经同时平移声子模轴和电子 site 轴。
3. 沿**帧轴 0**拼接基础帧和候选轨道的 `lam`、`shift`。
4. 在拼接后的全部帧上重新建立完整的 `S`、`H`，包含基础与候选之间的交叉块；再求最低能量与秩。
5. 返回四个标量。不独立求候选能量、不手工展平、不裁剪能量收益。

## 3. 手算与验收

先说明：为何加轨道不能升高精确变分最低能量；为何重复候选应保持能量和秩；为何不能只比较候选轨道自身与基础空间的两个最低能量。

验收用两站点例子：$t_{01}=t_{10}=-1$、$g=0.8$、$\omega=1$，基础帧的位移为零，候选的 `seed_lam=diag(0.8,0.8)`、`seed_shift=0`。基础能量为 $-1$、秩为 2；合并后能量约为 $-1.326306686855$、秩为 4。还将测试重复候选不增加秩，以及候选同时平移或作 LF 规范平移后结果不变。

运行验收：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_noci_trial_orbit.py
```

完成函数后回复“请验收”。本阶段不修改旧 NOCI 核或求解器。
