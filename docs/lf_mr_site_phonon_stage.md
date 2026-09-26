# 多 LF 帧第六阶段：电子–声子联合位移矩

本阶段只计算一个联合可观测量

$$
M_{xp}=\langle\Psi|n_p(b_x+b_x^\dagger)|\Psi\rangle.
$$

电子 site 边缘密度已可能均匀；`M[x,p]` 进一步记录
电子位于 `p` 时，物理声子模 `x` 的相关位移。
它仍只是对波函数的一个诊断，不能单独证明与 exact 态一致。
它也给出未中心化 Holstein 耦合项的期望值：

$$
E_{\mathrm{ep}}=g\sum_p M_{pp}.
$$

总能量还需要动能与声子数项，

$$
E=E_t+E_{\mathrm{ph}}+E_{\mathrm{ep}}.
$$

因此，求最低 NOCI 能量只需要已有的 Hamiltonian 核、重叠核与
广义本征求解；本函数不是能量计算的前置条件。它可用于解释能量
变化，以及检查电子密度均匀时的局域电子–声子关联。

## 1. 从相干态矩阵元推导

复习第二阶段的

$$
\langle\eta_A|b_x|\eta_B\rangle
=\eta_{B,x}\langle\eta_A|\eta_B\rangle,\qquad
\langle\eta_A|b_x^\dagger|\eta_B\rangle
=\eta_{A,x}\langle\eta_A|\eta_B\rangle.
$$

当前实数单电子 NOCI 态为

$$
|\Psi\rangle=\sum_{A,p} C_{Ap}|p\rangle|\eta_{Ap}\rangle,
\qquad
\eta_{Ap,x}=z_{A,x}-\lambda_{A,xp},
\qquad C^\mathsf{T}SC=1.
$$

`n_p` 保留电子 site `p`，而不同 LF 帧之间
的声子态并不正交。因此

$$
\boxed{
M_{xp}=
\sum_{A,B}C_{Ap}C_{Bp}\,S_{Ap,Bp}
\left(\eta_{Ap,x}+\eta_{Bp,x}\right).
}
$$

一帧时 `M[x,p]=2*rho[p]*eta[0,x,p]`；
重复两帧时仍有跨帧交叉项。没有声子零点位移。
如要定义“电子位于 p 时的平均位移”，才在
`rho[p]>0` 时另算 `M[x,p]/rho[p]`；
本阶段的函数不做这个除法。

对完整的周期平移轨道，若 NOCI 基态是唯一的平移本征态，
`M[(x+R)%L,(p+R)%L]=M[x,p]`。
矩阵可随 `x-p` 而变化；每个电子 site 的密度均匀
并不要求所有 `M[x,p]` 相等。

## 2. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_noci_site_phonon_moment(
    coeff: np.ndarray,
    smat: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    ...
```

| 数据 | dtype | shape / 索引 |
| --- | --- | --- |
| `coeff` | `float64` | `(K,L)` / `[A,p]` |
| `smat` | `float64` | `(K,L,K,L)` / `[A,p,B,q]` |
| `lam` | `float64` | `(K,L,L)` / `[A,x,p]` |
| `shift` | `float64` | `(K,L)` / `[A,x]` |
| 输出 `M` | `float64` | `(L,L)` / `[x,p]` |

四个输入来自同一批 LF 帧；`coeff` 已由 NOCI
求解器按 `S` 归一化。只计算原始联合矩，
不重归一化系数，不裁剪结果，不计算条件比值。
形状、dtype、有限值检查由导师在验收时补齐；
你的作业重点是正确的 `A,B,x,p` 索引与两个相干
位移项。

伪代码：

1. 构造 `eta=shift[:,:,None]-lam`，顺序 `[A,x,p]`。
2. 分配 `M[x,p]`。
3. 对每个 `x,p`，遍历帧对 `A,B`，
   累加 `coeff[A,p]*coeff[B,p]*smat[A,p,B,p]`
   乘 `eta[A,x,p]+eta[B,x,p]`。
4. 返回 `M`；不用第一阶段的
   `S[A,p,B,q]` 去替代同 site 的声子重叠以外
   的完整算符矩阵元。

## 3. 手算与验收

请先手算：

1. 单帧 `K=1` 和两份完全相同的
   `L=1` 帧，验证交叉项与因子 2。
2. 从相干态的左右作用推导括号中的
   `eta_A+eta_B`，说明为什么不是
   `2*eta_A` 或 `2*eta_B`。
3. 对四站点平移轨道说明：`rho[p]=1/4`
   时，为什么 `M[x,p]` 仍可表现出
   在 `x=p` 附近更强的局域声子云？

验收测试将核对单帧极限、重复帧干涉、
不同帧的解析矩阵元、规范不变性、
独立有限 Fock 空间的算符期望值，以及
四站点联合平移协变性。测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_noci_site_phonon_moment.py
```

完成手算和函数后回复“请验收”。

`alpha=2.4` 的未中心化 exact ED 复核采用
`Nmax=20,22,24`，能量依次为
`-3.043101750554`、`-3.043101753204`、
`-3.043101753313`。相邻能量变化为
`2.65e-9` 与 `1.09e-10`；三次本征残差均小于 `8e-7`。
因此现有 `data/fig2b_exact.csv` 的该点可用于当前能量比较。