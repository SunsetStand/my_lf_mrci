# 多 LF 帧第二阶段：Hamiltonian 核

本阶段只推导并实现物理 Hamiltonian 在第一阶段基态
`|A,p> = |p>|eta[A,:,p]>` 间的矩阵元。先阅读本讲义并手算文末
验收题，再由学生实现一个函数。暂不求解广义本征值问题，也不选择参考帧。

## 1. 同一个物理 Hamiltonian

沿用第一阶段的实位移、完整局域声子模和未中心化耦合：

$$
H=\sum_{pq}t_{pq}|p\rangle\langle q|
+\omega\sum_x b_x^\dagger b_x
+g\sum_p|p\rangle\langle p|(b_p+b_p^\dagger),
\qquad \eta_{Ap,x}=z_{A,x}-\lambda_{A,xp}.
$$

这里 `p,q` 是电子 site，`x` 是物理局域声子模。采用
`tmat[p,q]`，其对角元可以非零；`omega>0`，`g` 可正可负。
在 Fig. 2b 的四站点环上，`tmat` 的最近邻元是 `-1`。
单电子 Hubbard 双占据项为零，声子能量无零点常数；不添加
`-alpha/4` 的中心化修正。

## 2. 从相干态算符关系推导

令两个物理条件位移为 `a=eta[A,:,p]` 和 `b=eta[B,:,q]`，
先只定义声子重叠

$$
s(a,b)=\langle a|b\rangle
=\exp[-\tfrac12\sum_x(a_x-b_x)^2].
$$

从 `b_x|b> = b_x|b>` 及其厄米共轭得到

$$
\langle a|b_x|b\rangle=b_x\,s(a,b),\qquad
\langle a|b_x^\dagger|b\rangle=a_x\,s(a,b),\qquad
\langle a|b_x^\dagger b_x|b\rangle=a_xb_x\,s(a,b).
$$

上式里的 `b_x` 同时被用于算符和位移分量，推导时可把右侧位移写成
`beta_x` 来避免混淆。电子 site 的正交性只作用在对角的声子与耦合项；
hopping 算符 `|p><q|` 可以连接不同 site。逐项合并后：

$$
\boxed{
H_{Ap,Bq}=s(\eta_{Ap},\eta_{Bq})
\left[
t_{pq}+
\delta_{pq}\left(
\omega\sum_x\eta_{Ap,x}\eta_{Bq,x}
+g(\eta_{Ap,p}+\eta_{Bq,p})
\right)
\right].
}
$$

注意第一阶段的 `S[A,p,B,q]` 自带 `delta[p,q]`。
若把 `tmat[p,q]` 逐元素乘 `S`，会误删所有非对角 hopping。
需要计算不带电子 `delta` 的声子重叠。两种张量分别有不同的物理意义。

当 `A=B`、`p=q` 时，`s=1`，得到

$$
H_{Ap,Ap}
=t_{pp}+\omega\sum_x\eta_{Ap,x}^2+2g\eta_{Ap,p}.
$$

如果 `shift=0`，则 `eta=-lam`，线性耦合贡献是负的。
这一符号是本阶段最重要的独立检查之一。

## 3. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_frame_hamiltonian(tmat, g, omega, lam, shift, nelec=(1, 0)):
    ...
```

与 `lf_frame_overlap` 一样，`lam[A,x,p]` 为
`(K,L,L)` 的实 `float64`，`shift[A,x]`
为 `(K,L)`；`tmat[p,q]` 是
`(L,L)` 的实对称 `float64` 矩阵。`g` 和
`omega` 是有限实数，`omega>0`。
输出 `H[A,p,B,q]` 为 `(K,L,K,L)` 的
`float64` 张量，不修改输入。`nelec` 保留作未来扩展；
当前仅推导单电子公式，不在本函数内检查它。输入检查保持在函数边界，
不要放进 `A,B,p,q` 循环。

伪代码：

1. 检查形状、dtype、有限值、`tmat` 的对称性与正频率。
2. 构造 `eta=shift[:,:,None]-lam`，形状 `[A,x,p]`。
3. 创建 `H[A,p,B,q]`；遍历两个帧和两个电子 site。
4. 沿声子模轴求 `eta[A,:,p]-eta[B,:,q]` 的范数，得到
   不带电子 `delta` 的声子重叠。
5. 先加 `tmat[p,q]*s`；只有 `p==q` 时再加
   `omega*sum_x(eta[A,x,p]*eta[B,x,p])*s` 与
   `g*(eta[A,p,p]+eta[B,p,p])*s`。
6. 返回四指标张量。仅在未来求解器边界按
   `a=A*L+p` 展平成 `(K*L,K*L)`。

这一阶段不调用导师原型的核函数，也不调用自己的原始 site-basis
`contract_all` 来代替公式；它们只可用于独立验证。

## 4. 手算与验收

先手算并提交三点：

1. 分别写出 `p!=q` 和 `p=q` 的矩阵元，说明前者为何
   没有声子数项，却仍有 hopping。
2. 令 `K=1`、`shift=0`，验证对角线的
   `-2g*lam[p,p]` 项；再说明一般 `shift` 时，
   本核怎样对应现有 `lf_effective_one_body` 加上
   `omega*shift@shift` 的全局常数。
3. 令 `lam=shift=0`，写出单帧和两份重复帧的 `H`；
   解释为什么重复帧不能给出新的物理态。

代码验收将比对非对角 hopping 的解析值、对角局域能、规范不变性、
单帧现有 LF 有效一体矩阵，以及独立构造的 tiny dense Fock Hamiltonian。
后一个对照使用有限投影的相干态，其结果随截断趋于解析核；
有限截断误差不应通过重新归一化投影态掩盖。

测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_frame_hamiltonian.py
```

完成函数和手算后回复“请验收”。