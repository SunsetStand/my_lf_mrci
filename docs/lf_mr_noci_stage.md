# 多 LF 帧第三阶段：NOCI 最低能广义本征解

本阶段只完成一个边界清楚的求解器：输入已验收的物理 Hamiltonian 核
`H[A,p,B,q]` 与重叠核 `S[A,p,B,q]`，输出最低变分能量
和每个 LF 帧、电子 site 的 CI 系数。参考帧的产生、平移对称投影、
位移再优化与 MRPT 均留待后续阶段。

## 1. 阅读与物理空间

复习前两阶段讲义中的 `|A,p>=|p>|eta[A,:,p]>`；
参看导师原型 `/mnt/d/code/LF_MRCI/code/core.py` 中
`generalized_lowest` 的思路，但不要复制其输出或将其
“完整参考态混合”当作本项目的 `K*L` 空间。
本地 Cui 论文 Sec. 2.3 描述 LF-HF 单参考；这里的广义本征求解
是额外的多帧变分步骤。

写出变分态与 Rayleigh 商：

$$
|\Psi_C\rangle=\sum_{A=0}^{K-1}\sum_{p=0}^{L-1}
C_{Ap}|A,p\rangle,\qquad
E[C]=\frac{C^\mathsf{T}HC}{C^\mathsf{T}SC}.
$$

约束 `C^T S C=1`，对每个实系数变分，得到

$$
HC=ESC.
$$

`C[A,p]` 是非正交基上的展开系数；单个
`C[A,p]**2` 不是物理概率。最终态可以整体变号，
因此测试不能逐元素固定本征向量的符号。

## 2. 为什么先正交化重叠

固定 `a=A*L+p`，只在求解器边界将四指标核展平为
`(M,M)`，其中 `M=K*L`。
`S` 是物理态的 Gram 矩阵，故理论上半正定。
若两帧完全重复，`S` 有零特征值，不能直接求逆或
在未经处理时交给要求正定 `S` 的求解器。

令 `S=V diag(s) V^T`。设
`tau=overlap_cut*max(1,s_max)`，默认
`overlap_cut=1e-10`；若
`s_min < -tau`，重叠明显不符合 Gram 矩阵，应报错。
保留 `s_i>tau` 的方向，定义

$$
X=V_{\rm keep}\operatorname{diag}(s_{\rm keep}^{-1/2}),
\qquad X^\mathsf{T}SX=I.
$$

剩下的是普通对称本征问题：

$$
(X^\mathsf{T}HX)y=Ey,\qquad C=Xy.
$$

取最低特征值，返回 `C.reshape(K,L)`。
返回 `rank=len(s_keep)`，使调用者知道剔除了多少
线性相关方向。不要给小特征值加常数、使用未经说明的伪逆，
也不要静默改写 `H` 或 `S`。

通过正交化后，`C^TSC=1`。如果没有舍弃方向，
广义残差 `||HC-ESC||` 应很小；若舍弃近相关方向，
应看保留空间的残差 `||X^T(HC-ESC)||`。
此时原始坐标中的完整残差不一定同样小。
完全重复的帧在解析核中通常也给出很小的完整残差。
只在未被 cutoff 破坏的嵌套空间里，加入参考帧才保证最低
变分能量不升高。

## 3. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_noci_lowest(
    hmat: np.ndarray,
    smat: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
) -> tuple[float, np.ndarray, int]:
    ...
```

| 数据 | dtype | shape / 索引 |
| --- | --- | --- |
| `hmat` | `float64` | `(K,L,K,L)` / `[A,p,B,q]` |
| `smat` | `float64` | 与 `hmat` 相同 |
| 能量 | `float` | 最低变分能量 |
| `coeff` | `float64` | `(K,L)` / `[A,p]` |
| `rank` | `int` | 保留的重叠特征方向数 |

输入来自同一批帧、同一个单电子物理 Hamiltonian，均为有限
实对称核。`K,L>0`、形状匹配，`overlap_cut`
为有限正数。此阶段在函数边界检查这些基本契约与明显负的
重叠特征值；不用为任意 Python 类型写复杂兼容层。
函数不修改输入，也不依赖调用者提前把张量展平。

伪代码：

1. 检查两个四指标核的形状和参数，按 `a=A*L+p` 展平。
2. 用 `np.linalg.eigh` 对 `S` 对角化，计算
   `tau` 并拒绝明显负的特征值。
3. 选取 `s>tau` 的特征方向；若没有方向，报错。
4. 构造列正交化矩阵 `X`，再对
   `X.T @ H @ X` 对角化。
5. 取最低能本征向量 `y`，计算 `C=X@y`，
   reshape 回 `(K,L)`，返回能量、系数和秩。

## 4. 手算与验收

请先用纸笔回答：

1. 对两份完全相同的 L-site 帧，写出展平后的 `S`、
   非零和零特征值及其重数；解释能量为何不能因复制帧而改善。
2. 从约束 `C^TSC=1` 推出 `HC=ESC`，
   再验证 `X^TSX=I` 与
   `C^TSC=1`。
3. 说明为什么不能把 `C[A,p]**2` 直接当作概率，
   以及为什么“丢弃近零重叠方向”之后完整坐标残差
   不一定等于保留空间残差。

验收测试将覆盖：`g=0` 的四站点单帧能量 `-2`；
重复帧不改变能量且保留秩 L；非重复帧与独立广义求解器对照；
规范归一化和残差；近相关方向的 cutoff；拒绝明显不定的重叠。
测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_noci_lowest.py
```

你完成函数后回复“请验收”；我会 review 数学、检查和测试，通过后
补充 docstring。只有验收过的能量与系数才能用于后续物理分析。