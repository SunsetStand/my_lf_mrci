# Fig. 2d：多电子非均匀密度的 LF 固定参考态检查

## 当前结论与本阶段边界

当前 `src/lf_mp.py` 的 LF-HF 能量、优化器和 LF-MP2 reference 都是单电子接口；
特别是 MP2 reference 固定 `nocc=1`，LF-HF 能量缺少 Hubbard 二体项。
它不能直接计算 Fig. 2d 的四电子 LF 曲线。论文 Fig. 2d 使用周期性四站点、
四电子、`U=4`；本文只取其中 `omega=0.5`、`alpha=g²/omega=2.4`，
检查一个**指定的非均匀 Slater 参考态**。这个态未经变分优化，下面的能量
不是 Fig. 2d 的曲线点。

本阶段只请你实现**固定轨道、固定完整 `lam` 和 `shift` 的 LF-HF 能量求值**。
先不做轨道或 LF 参数优化，也不做 MP2。保留现有单电子函数及其接口。

阅读：论文 Sec. II.1--II.3 的 Hubbard--Holstein Hamiltonian 和 LF 零声子平均；
先前的 [多电子 gauge 证明](lf_gauge_multielectron_report.md)。

## 数学定义与数据流

取 `L=nmode=norb=4`，固定 `nelec=(2,2)`，站点 `p,q=0..3`，声子模
`x=0..3`；`tmat[p,q]=-1` 当 `p,q` 为周期相邻站点，其他为零。
Hamiltonian 使用论文的未中心化耦合

$$
H=\sum_{pq\sigma}t_{pq}c^\dagger_{p\sigma}c_{q\sigma}
+U\sum_p n_{p\alpha}n_{p\beta}
+\omega\sum_x b_x^\dagger b_x
+g\sum_x n_x(b_x+b_x^\dagger),\quad n_x=n_{x\alpha}+n_{x\beta}.
$$

令 `occ_a[p,i]`、`occ_b[p,i]` 是各自自旋的正交 occupied 轨道列。
定义 `D_sigma = occ_sigma @ occ_sigma.T`，`rho_sigma[p]=D_sigma[p,p]`，
`rho=rho_a+rho_b`。对实轨道，Slater 态的站点占据二阶矩为

$$
Q_{pq}=\langle n_pn_q\rangle
=\rho_p\rho_q-(D_\alpha)_{pq}(D_\alpha)_{qp}
 -(D_\beta)_{pq}(D_\beta)_{qp}+\delta_{pq}\rho_p.
$$

请先手算检验 `Q[0,0] = rho[0] + 2*rho_a[0]*rho_b[0]`，
并确认 `sum_p Q[p,q] = N_e*rho[q]` 和
`sum_pq Q[p,q] = N_e**2`。再实现下列四项：

$$
S_{pq}=\exp[-\tfrac12\sum_x(\lambda_{xp}-\lambda_{xq})^2],
\qquad E_t=\sum_{pq}t_{pq}S_{pq}(D_\alpha+D_\beta)_{qp},
$$

$$
E_U=U\sum_p\rho_{p\alpha}\rho_{p\beta},
$$

$$
E_{\rm ph}=\omega\sum_x\left[z_x^2-2z_x\sum_p\lambda_{xp}\rho_p
+\sum_{pq}\lambda_{xp}\lambda_{xq}Q_{pq}\right],
$$

$$
E_{\rm ep}=2g\sum_x\left[z_x\rho_x-\sum_p\lambda_{xp}Q_{xp}\right],
\qquad E=E_t+E_U+E_{\rm ph}+E_{\rm ep}.
$$

数据流：`occ_a,occ_b` → `D_a,D_b` → `rho,Q`；`lam` → `S`；
再由 `tmat,U,g,omega,shift` 合成四项与总能量。
这里的 `Q` 很关键：把 `n_p n_q` 直接替换成 `rho_p rho_q` 会丢掉
同自旋交换项和站点上的占据涨落，不能正确检查多电子 gauge。

## 小编码作业：接口

在 `src/lf_mp.py` 增加一个独立函数，建议接口：

```python
def lf_hf_multi_fixed_energy(
    tmat: np.ndarray, U: float, g: float, omega: float,
    nelec: tuple[int, int], occ_a: np.ndarray, occ_b: np.ndarray,
    lam: np.ndarray, shift: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return (energy, spin_density) for a fixed unrestricted LF reference."""
```

本阶段全用实数 `float64`。`tmat.shape=(L,L)`；
`occ_a.shape=(L,nelec[0])`，`occ_b.shape=(L,nelec[1])`；
`lam.shape=(L,L)`，行是 mode `x`，列是 site `p`；
`shift.shape=(L,)`，即上式的 `z_x`；输出 `spin_density.shape=(2,L)`，
第一行 alpha，第二行 beta。验证轨道列正交、输入形状以及 `omega>0`。
允许在此阶段仅支持实数和 `L=4` 的周期模型，但公式按一般 `L` 编写。

## 可执行验收点

你实现函数后，在项目根目录运行下面的片段。轨道产生的密度是
`rho_a=(0.8,0.2,1,0)`、`rho_b=(1,0,1,0)`，所以总密度
`(1.8,0.2,2,0)` 确实非均匀；该态含两个电子行列式分量。

```python
import sys
import numpy as np
sys.path.insert(0, "src")
from lf_mp import lf_hf_multi_fixed_energy

L = 4
t = np.zeros((L, L))
for p in range(L):
    q = (p + 1) % L
    t[p, q] = t[q, p] = -1.0
ca = np.zeros((L, 2))
ca[0, 0], ca[1, 0], ca[2, 1] = np.sqrt(.8), np.sqrt(.2), 1.0
cb = np.zeros((L, 2))
cb[0, 0], cb[2, 1] = 1.0, 1.0
lam = np.array([[.4, .1, .05, -.05],
                [0., .25, -.1, .1],
                [.1, -.15, .45, .05],
                [-.05, .1, .2, .3]])
z = np.zeros(L)
c = np.array([.17, -.11, .09, .05])
args = (t, 4.0, np.sqrt(1.2), .5, (2, 2), ca, cb)
energy, density = lf_hf_multi_fixed_energy(*args, lam, z)
e0, _ = lf_hf_multi_fixed_energy(*args, np.zeros_like(lam), z)
eg, dg = lf_hf_multi_fixed_energy(*args, lam + c[:, None], z + 4*c)
mean = lam.mean(axis=1)
er, dr = lf_hf_multi_fixed_energy(*args, lam - mean[:, None], z - 4*mean)
ewrong, _ = lf_hf_multi_fixed_energy(*args, lam + c[:, None], z + c)
print(energy, density)
assert abs(energy - (-0.5622409685425955)) < 1e-11
assert abs(e0 - 6.4) < 1e-12
assert np.allclose(density, [[.8, .2, 1., 0.], [1., 0., 1., 0.]])
assert abs(eg - energy) < 1e-12 and np.allclose(dg, density)
assert abs(er - energy) < 1e-12 and np.allclose(dr, density)
assert abs(ewrong - energy) > 1.0
```

独立行列式收缩已核对：该态的裸跳跃能是 `-0.8`，Hubbard 能是 `7.2`。
相干位移与电子声子项合计 `-7.051817144175838`，LF 跳跃项为
`-0.7104238243667571`，故总能量为上述数值。

## Gauge 的判据

固定 `N_e=4` 时，任意行列式的条件位移是
`eta[D,x] = shift[x] - sum_p lam[x,p]*n[D,p]`。因此
`lam[x,p] += c[x]` 必须同时配 `shift[x] += 4*c[x]`；
两种规范 `shift=0` 和 `sum_p lam[x,p]=0` 均可选，但不能同时强加。
上面的错误 `shift += c` 刻意提供一个反例。本阶段通过后，
再进入多电子 LF-HF 的轨道与 `lam` 优化；LF-MP2 还需电子双激发项。
