# CS-HF 与 LF-HF：一般多电子推导阶段

本阶段只做理论推导，不实现 SCF、LF 优化或 MP。论文对应 Sec. II.1--II.3，
重点是 Eqs. (1)--(4)、(11)--(20) 和 (27)--(36)。

## 1. 记号、数据类型和维数

一般电子--玻色 Hamiltonian 写成

\[
\begin{aligned}
\hat H={}&\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\frac12\sum_{pqrs\sigma\tau}V_{pqrs}
a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}\\
&+\sum_x\omega_x b_x^\dagger b_x
+\sum_{xpq\sigma}g^x_{pq}a^\dagger_{p\sigma}a_{q\sigma}
(b_x+b_x^\dagger).
\end{aligned}
\]

约定 `p,q,r,s` 是正交局域 AO/site index，`x` 是 boson-mode index，
`i,j` 和 `a,b` 分别只用于 occupied 和 virtual MO。定义

\[
\gamma^\sigma_{qp}=\langle a^\dagger_{p\sigma}a_{q\sigma}\rangle,
\qquad
\gamma=\gamma^\alpha+\gamma^\beta.
\]

实现阶段将使用以下形状：

| quantity | basis | dtype | shape |
|---|---|---|---|
| \(h\), \(F^\sigma\) | local AO/site | `float64` or `complex128` | `(norb, norb)` |
| \(g\) | local AO/site + mode | same as integrals | `(nmode, norb, norb)` |
| \(\gamma^\alpha,\gamma^\beta\) | local AO/site | same as orbitals | `(norb, norb)` |
| \(z\) | mode | `float64` | `(nmode,)` |
| \(\lambda\) | mode + local orbital | `float64` | `(nmode, norb)` |
| \(C^\sigma\) | AO rows, MO columns | same as Fock | `(norb, norb)` |
| \(\kappa^\sigma\) | virtual rows, occupied columns | same as orbitals | `(nvir, nocc)` |

先自行用 Wick 定理验证

\[
\langle a^\dagger_{p\sigma}a^\dagger_{r\tau}
a_{s\tau}a_{q\sigma}\rangle
=\gamma^\sigma_{qp}\gamma^\tau_{sr}
-\delta_{\sigma\tau}\gamma^\sigma_{sp}\gamma^\sigma_{qr}.
\]

这一步确定后续 Coulomb 与 exchange 的索引顺序。

## 2. CS-HF 推导路线

论文采用

\[
U_{\rm CS}=\prod_x e^{-z_x(b_x-b_x^\dagger)},
\qquad U_{\rm CS}^\dagger b_xU_{\rm CS}=b_x+z_x.
\]

令

\[
G_x[\gamma]=\sum_{pq}g^x_{pq}\gamma_{qp}.
\]

在电子 Slater determinant 与变换后 boson vacuum 上取期望值，应得到

\[
E_{\rm CS}[\gamma,z]
=E_{\rm elec}^{\rm HF}[\gamma]
+\sum_x\left(\omega_xz_x^2+2z_xG_x[\gamma]\right).
\]

接下来亲手完成两步：

1. 对每个 \(z_x\) 求偏导，推出论文 Eq. (16)。
2. 对 \(\gamma^\sigma\) 求变分，验证
   \(F^\sigma=F^\sigma_{\rm elec}+2\sum_xz_xg^x\)。

注意：消去 \(z\) 后的能量含有 \(-G_x^2/\omega_x\)，但 Fock 不能通过
“把能量中的一半漏掉”来构造；必须对完整能量泛函求导。

## 3. LF-HF 推导路线

对局域 AO/site number operator 使用 diagonal LF 变换

\[
U_{\rm LF}=\exp\left[\sum_{xp}\lambda_p^x a_p^\dagger a_p
(b_x-b_x^\dagger)\right].
\]

先用 BCH 展开到嵌套对易子终止的位置，证明

\[
U_{\rm LF}^\dagger b_xU_{\rm LF}
=b_x-\sum_p\lambda_p^x a_p^\dagger a_p,
\]

\[
U_{\rm LF}^\dagger a_qU_{\rm LF}
=a_q\exp\left[\sum_y\lambda_q^y(b_y-b_y^\dagger)\right].
\]

因此 hopping 会带有位移算符。定义 vacuum Franck--Condon 因子

\[
S_{pq}=\exp\left[-\frac12\sum_y(\lambda_q^y-\lambda_p^y)^2\right],
\]

以及二体对应因子

\[
S_{pqrs}=\exp\left[-\frac12\sum_y
(\lambda_q^y-\lambda_p^y+\lambda_s^y-\lambda_r^y)^2\right].
\]

零声子平均后的一个关键一体量是

\[
\bar h_{pq}=h_{pq}+\sum_x
\left(2z_xg^x_{pq}-\lambda_p^xg^x_{pq}-g^x_{pq}\lambda_q^x\right),
\]

\[
h^{\rm eff}_{pq}=\bar h_{pq}S_{pq}
+\delta_{pq}\sum_x\left[\omega_x(\lambda_p^x)^2
-2\omega_xz_x\lambda_p^x\right].
\]

二体部分还包含三类项：原始 \(V_{pqrs}S_{pqrs}\)、
\(2\omega_x\lambda_p^x\lambda_q^x\) 诱导项，以及
\(-4g^x_{pq}\lambda_r^xS_{pq}\) 项。请从论文 Eq. (31) 按
\(a_p^\dagger a_r^\dagger a_sa_q\) 的固定顺序重新标出四个指标，暂时不要编码。

最后用普通 HF 规则由这些有效积分构造 Fock。LF-HF 能量为

\[
E_{\rm LF-HF}=\sum_x\omega_xz_x^2
+\operatorname{Tr}\left[(h^{\rm eff}+\tfrac12v^{\rm HF})\gamma\right].
\]

与 CS-HF 不同，\(\lambda,z\) 和 occupied--virtual orbital rotation
\(\kappa\) 要共同做变分优化。

## 4. AO、MO 与两种“ED”

局域 AO/site 基是定义 \(g^x_{pq}\)、\(\lambda_p^x\) 和局域电子密度的基。
HF 求解产生

\[
a^\dagger_{m\sigma}=\sum_p C^\sigma_{pm}a^\dagger_{p\sigma},
\qquad C^{\sigma\dagger}F^\sigma C^\sigma=\varepsilon^\sigma.
\]

canonical MO 基用于 occupied/virtual 分类、MP 分母和激发矩阵元。例如
\(g^x_{mn}=\sum_{pq}C^*_{pm}g^x_{pq}C_{qn}\)。不要把 `site index`
和 `MO index` 共用同一变量名。

“做 ED”必须区分：

1. 对完整 \(U^\dagger HU\) 做电子--声子 ED；无截断时它与原始 \(H\)
   幺正等价，有限 Fock 截断时需重新检查收敛。
2. 对 \(\langle0|U^\dagger HU|0\rangle\) 做纯电子 ED；这是 CS/LF-ED
   近似，比单 determinant HF 的电子空间更大，但不是原模型 exact ED。

论文的 CS-MP/LF-MP 使用第二类有效 Hamiltonian 的 HF canonical MO
作为零阶参考，而不是用 ED 本征态直接套入普通单参考 MP 公式。

## 5. 理论验收题

提交一页推导，依次回答：

1. 从 CS 能量泛函推出 \(z_x\) 和 CS Fock；说明两个因子 2 的来源。
2. 令 \(\lambda=0\)，逐项指出 LF 表达式如何退化到 CS。
3. 令 \(g=0,z=0,\lambda=0\)，说明为何恢复普通电子 HF。
4. 对 Fig. 2b 的 \(c=(1,1,1,1)^T/2\)，推出
   \(E_{\rm CS-HF}=-2-\alpha/4\)。
5. 用投影算符 \(P_0=|0\rangle\langle0|\) 解释：为什么
   \(U^\dagger HU\) 与 \(H\) 等谱，而
   \(P_0U^\dagger HUP_0\) 通常不等谱。

通过这些题后，再开始单电子 CS-HF SCF；本阶段不提前实现 LF-HF 或 MP。
