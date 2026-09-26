# 多 LF 帧第一阶段：物理态与重叠

本阶段是学习材料与作业，不是已完成的 MRCI 实现。先完成文末理论题，
再由学生创建 `src/lf_mr.py`、实现唯一函数；验收后补齐 docstring。
本阶段不实现 Hamiltonian 核、广义本征求解器或参考帧选择算法。

## 1. 阅读与约定

阅读本地论文 `/mnt/d/下载/Cui24-Lang-Firsov-MP.pdf` 的 Sec. 2.3、
Eqs. (27)–(36)，以及 `/mnt/d/code/LF_MRCI/THEORY.md` 第 1–2 节。
论文定义单参考方法；以下多帧空间是基于这些定义的扩展。

采用单电子 `nelec=(1,0)`，周期性四站点，默认 `omega=0.5`、
最近邻 hopping 为 `-1`，保留全部局域声子模。Hamiltonian 为

$$
H=\sum_{pq}t_{pq}|p\rangle\langle q|
+\omega\sum_x b_x^\dagger b_x
+g\sum_p |p\rangle\langle p|(b_p+b_p^\dagger).
$$

Hubbard 双占据项为零，声子没有零点常数。耦合未中心化，不再补加
`-alpha/4`。重叠本身不依赖 `tmat,g,omega`，这些参数不进入本阶段接口。

## 2. 从变换得到物理条件位移

定义实位移算符

$$
D(\boldsymbol v)=\exp\!\left[\sum_x v_x(b_x^\dagger-b_x)\right],
\quad |\boldsymbol v\rangle=D(\boldsymbol v)|0\rangle.
$$

第 A 帧的变换为

$$
U_A=\exp\!\left[\sum_{xp}\lambda_{A,xp}n_p(b_x-b_x^\dagger)\right]
D(\boldsymbol z_A).
$$

利用 `n_r|p> = delta[r,p]|p>`，LF 部分在电子位于 p 时变成
`D(-lam[A,:,p])`。实位移沿相同振子坐标，因此两个位移指数对易，没有
额外相位，得到

$$
|A,p\rangle=U_A|p\rangle|0\rangle
=|p\rangle|\boldsymbol\eta_{Ap}\rangle,
\qquad \eta_{Ap,x}=z_{A,x}-\lambda_{A,xp}.
$$

这里 eta 是条件位移，不是耦合参数 `alpha=g**2/omega`。
同时令 `lam[A,x,p] += u[A,x]`、`shift[A,x] += u[A,x]`，物理态不变。
现有 `lf_hf_full_optimize` 返回 `lam`、`shift`、`coeff`；其中 `shift=0`，
`lam=mu`，所以 eta 等于负的 lam。`lf_hf_full_multistart` 已返回所有起点
的结果，不仅是 best；日后研究参考帧时仍须检查各结果的优化状态。

## 3. 从 Fock 展开推导重叠

先考虑一个模。正规排序给出

$$
|\eta\rangle=e^{-\eta^2/2}\sum_{n=0}^{\infty}
\frac{\eta^n}{\sqrt{n!}}|n\rangle.
$$

使用数态正交性与指数级数：

$$
\langle\eta|\xi\rangle
=e^{-(\eta^2+\xi^2)/2}\sum_{n=0}^{\infty}\frac{(\eta\xi)^n}{n!}
=e^{-(\eta-\xi)^2/2}.
$$

独立多模的重叠是各模乘积，再乘电子重叠，故

$$
\boxed{S_{Ap,Bq}=\delta_{pq}\exp\!\left[-\frac12\sum_x
(\eta_{Ap,x}-\eta_{Bq,x})^2\right].}
$$

不同 p、q 的声子云可以重叠，但总态依旧因电子正交而重叠为零。
同帧电子展开给出正交的 L 个基态；跨帧的同 site 基态才是非正交的。

选择的变分空间是

$$
|\Psi\rangle=\sum_{Ap} C_{Ap}|A,p\rangle,\qquad C^\dagger S C=1.
$$

这里有 K*L 个独立线性系数。完整 LF-HF 参考态的混合只允许
`C[A,p] = mix[A] * coeff[A,p]`，是此空间的一个子空间。
固定一个 LF 帧时，在全部 site 上变分已等价于其零声子有效一体矩阵的
对角化；因此单帧电子 CI 不会超过该帧已优化电子轨道的单电子 LF-HF。
增加不同条件声子云才扩大空间。

S 是 Gram 矩阵，因此半正定，不一定正定。重复两帧时展平得到
`[[I,I],[I,I]]`，秩为 L；不能要求它可逆。非正交系数的模平方不是概率。

## 4. 唯一函数作业

理论题确认后，在 `src/lf_mr.py` 实现

```python
def lf_frame_overlap(lam, shift, nelec=(1, 0)):
    ...
```

| 量 | dtype | shape / 索引 |
| --- | --- | --- |
| lam | float64 ndarray | `(K,L,L)` / `[A,x,p]` |
| shift | float64 ndarray | `(K,L)` / `[A,x]` |
| 内部 eta | float64 ndarray | `(K,L,L)` / `[A,x,p]` |
| 输出 S | float64 ndarray | `(K,L,K,L)` / `[A,p,B,q]` |

输入边界：检查 K、L 非零，lam 两个末轴相等，shift 形状相符，
两个数组均为有限 float64 数据。当前公式只推导和验收单电子
`nelec=(1,0)`；`nelec` 暂不参与计算或输入检查，保留给后续扩展。
形状或非有限数值抛出 ValueError；不支持的数组 dtype 抛出 TypeError。
不承诺自动接受列表或转换 dtype，不静默丢弃复数部分。
不修改输入，不引入电子轨道参数，不归一化或正则化 S。

伪代码：

1. 检查边界契约，读取 K、L。
2. 沿电子 site 轴广播 shift，构造 eta。
3. 创建 `(K,L,K,L)` 的 float64 零张量。
4. 遍历 A、B、p：取两帧在同一 p 的位移差，平方后沿声子 x 求和，
   将指数值写入 `S[A,p,B,p]`。
5. 返回 S。

先用显式循环保证索引正确，不必急于向量化。展平只用于测试或后续
求解器边界：`S.reshape(K*L,K*L)`，行编号 `a=A*L+p`。
导师原型采用 `[p,x]` 位移并消去均匀模，不能直接搬用其数组。
现有 `lf_franck_condon` 是同帧 hopping 的声子因子，不能直接当作这里
包含电子正交性的 S。

## 5. 理论提交与代码验收

现在先提交以下三个问题的推导或解释，不需要开始写 Hamiltonian：

1. 从 U_A 推出 eta 的负号，再从 Fock 展开推出重叠；解释为什么
   `S[A,p,B,q]` 在 p!=q 时必须为零。
2. 证明一帧、两份重复帧的 S 分别是什么，并解释单帧 CI 与 LF-HF 的关系。
3. 定义平移 R：`T_R|p>=|p+R>` 且 `T_R b_x† T_R†=b_(x+R)†`。
   写出平移后的 eta，说明为什么电子和声子 site 都必须平移。
   思考：恢复均匀电子密度为何不等于恢复全部电子–声子关联？

理论确认后完成函数，回复“请验收”。测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_frame_overlap.py
```

测试包含解析极限、重复帧、规范不变性、Gram 半正定性及独立 Fock 展开。
Fock 对照每模保留 0..Nmax，使用未经重归一化的投影相干态；截断态的
范数小于 1 是丢失尾部的真实结果。测试的 S 可以随 Nmax 收敛，但不能
据此声称 Hamiltonian 或能量也已收敛。

函数尚不存在时，依赖它的测试明确 skip，辅助 Fock 检查可以运行；
skip 不算学生作业验收通过。此后若函数存在但出错，测试应直接失败。
