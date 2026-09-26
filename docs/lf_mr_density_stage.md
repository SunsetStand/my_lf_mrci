# 多 LF 帧第五阶段：从 NOCI 系数计算电子 site 密度

本阶段只实现一个物理诊断量 `rho[p]=<Psi|n_p|Psi>`，
用于检查平移轨道混合后的电子密度。前四阶段已经提供
`S`、`H`、NOCI 系数和完整平移轨道。
本阶段不优化新参考帧、不改本征求解器、不扫描参数。

## 1. 阅读与推导

复习 `src/cs_mp.py` 中的 `cs_site_density` 与
第三阶段讲义的非正交归一化。当前实系数波函数是

$$
|\Psi\rangle=\sum_{A,p} C_{Ap}|p\rangle|\eta_{Ap}\rangle,
\qquad C^\mathsf{T}SC=1.
$$

在单电子 site 基中，`n_p=|p><p|`。从电子基正交性出发，

$$
\langle A,r|n_p|B,q\rangle
=\delta_{rp}\delta_{qp}
\langle\eta_{Ar}|\eta_{Bq}\rangle
=\delta_{rp}\delta_{qp}S_{Ar,Bq}.
$$

所以

$$
\boxed{
\rho_p=\sum_{A,B}C_{Ap}\,S_{Ap,Bp}\,C_{Bp}.
}
$$

只有电子 site 固定为同一个 `p`，不同 LF 帧的交叉项仍存在。
`sum_A C[A,p]**2` 会漏掉这些交叉项。
`S[:,p,:,p]` 是 Gram 子矩阵，故 `rho_p>=0`
（允许浮点舍入级误差）；而

$$
\sum_p\rho_p=C^\mathsf{T}SC=1
$$

是本阶段的独立验收恒等式。单帧 `K=1` 时
`S[0,p,0,p]=1`，退化为
`rho_p=|coeff[p]|**2`。

注意：均匀的电子边缘密度是恢复平移对称性的一个诊断，
但它本身不足以证明电子–声子联合波函数与 exact 态一致。
不能从非正交系数平方直接读取每个参考帧的“概率”。

## 2. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_noci_site_density(
    coeff: np.ndarray,
    smat: np.ndarray,
) -> np.ndarray:
    ...
```

| 数据 | dtype | shape / 索引 |
| --- | --- | --- |
| `coeff` | `float64` | `(K,L)` / `[A,p]` |
| `smat` | `float64` | `(K,L,K,L)` / `[A,p,B,q]` |
| `rho` | `float64` | `(L,)` / `[p]` |

`coeff` 来自 `lf_noci_lowest`，与 `smat`
属于同一批帧，并已满足 `C^T S C≈1`。不修改输入，
不在函数里重新归一化系数或裁剪负密度。
本阶段仍只实现实数单电子情形；复系数需要共轭，暂不扩展。
形状、dtype、有限值等边界检查由导师在验收时补齐；
你的编码重点是式中的四个索引。

伪代码：

1. 从 `coeff.shape` 读取 K、L，创建长度 L 的密度数组。
2. 对每个 `p`，取同一电子 site 的
   `smat[:,p,:,p]`，它的形状是 `(K,K)`。
3. 用 `coeff[:,p]` 在这个 Gram 子矩阵两侧收缩，
   写入 `rho[p]`。
4. 返回 `rho`。只在验收测试中检查
   `rho.sum()≈1`，不要通过重新归一化掩盖错误。

## 3. 手算与验收

请先手算：

1. `L=1,K=2` 且两帧完全相同，令
   `C[0,0]=C[1,0]=1/2`。算出
   `C^TSC`、正确的 `rho[0]` 和
   错误公式 `sum_A C[A,0]**2` 的结果。
2. 对 `K=1`，证明本公式退化为
   `cs_site_density`；对一般 K 证明
   `sum_p rho[p]=C^TSC`。
3. 说明为什么 `rho[p]=1/L` 不能单独证明
   电子–声子联合态已经是 exact 态。

验收测试涵盖单帧极限、重复参考帧的交叉项、
带正负 CI 系数的两帧解析例子，以及现有四站点
平移轨道 NOCI 解的均匀密度与总粒子数。
测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_noci_site_density.py
```

完成手算和函数后回复“请验收”；我会 review 并运行测试，
通过后补齐 docstring。