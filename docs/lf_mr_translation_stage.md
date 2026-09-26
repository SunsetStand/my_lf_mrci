# 多 LF 帧第四阶段：破缺参考帧的完整平移轨道

本阶段只从**一套已给定**的 LF 参数生成周期环上的 L 个平移参考帧。
第一至三阶段已能计算每个帧的 `H`、`S` 并求最低 NOCI 能量。
本阶段不运行 LF 优化、不筛选多起点结果、不显式构造动量投影，
也不宣称平移轨道本身就是 exact 态。

## 1. 阅读与物理问题

复习本项目前三阶段的 `eta[x,p]=shift[x]-lam[x,p]`，
以及导师原型 `/mnt/d/code/LF_MRCI/code/core.py` 中的
`translate_frame`。导师原型使用消去均匀模后的
`alpha[p,mode]`；这里保留全部 L 个局域物理模，
索引是 `lam[x,p]`，不可直接搬用其数组操作。

对长度 L 的环，定义平移算符 `T_R`：

$$
T_R|p\rangle=|p+R\bmod L\rangle,\qquad
T_R b_x^\dagger T_R^\dagger=b_{x+R\bmod L}^\dagger.
$$

因此整个条件位移云和电子一起平移。令新指标
`x'=x+R`、`p'=p+R`，则

$$
\eta^{(R)}_{x',p'}=\eta_{x'-R,\,p'-R},
\quad
\lambda^{(R)}_{x',p'}=\lambda_{x'-R,\,p'-R},
\quad
z^{(R)}_{x'}=z_{x'-R}.
$$

所有 site 指标都按 L 取模。对单个完整 LF-HF 态，
电子轨道同时满足 `coeff_R[p']=coeff[p'-R]`；
本项目的 NOCI 每个帧已包含全部电子 site 基态，
所以本阶段的函数**只返回平移后的 lambda 和 shift**，
后续系数由 `lf_noci_lowest` 重新变分。

只移动电子而不移动声子会改变电子–声子相对位置，
所得状态不是 `T_R|A,p>`。四站点环的 hopping、
局域耦合和各模同一频率使 `[H,T_R]=0`，
故任意完整态与其平移态的能量相同。

## 2. 唯一函数作业

在 `src/lf_mr.py` 增加：

```python
def lf_translation_orbit(
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

| 数据 | dtype | shape / 索引 |
| --- | --- | --- |
| 输入 `lam` | `float64` | `(L,L)` / `[x,p]` |
| 输入 `shift` | `float64` | `(L,)` / `[x]` |
| 输出 `lam_orbit` | `float64` | `(L,L,L)` / `[R,x,p]` |
| 输出 `shift_orbit` | `float64` | `(L,L)` / `[R,x]` |

输入必须非空、形状一致、有限且为 float64；不修改输入。
保留 R=0 原帧，并依序生成 R=1 到 L-1 的平移帧。
即使种子本身平移对称，也保留 L 个位置；后续 NOCI 的
重叠正交化负责剔除重复方向。本阶段不要按近似相等的 lambda
删除帧，因为不同参数规范也可能表示同一物理位移。

伪代码：

1. 检查输入边界，读取 L，分配输出数组。
2. 遍历 `R=0,...,L-1`。
3. 在 mode 轴和 electron-site 轴同时循环平移 `lam`；
   只在 mode 轴平移 `shift`。
4. 写入第 R 帧，返回两个数组。可用显式模索引或
   `np.roll`，但先在纸上核对正方向。

把输出直接送给现有
`lf_frame_overlap(lam_orbit,shift_orbit)`、
`lf_frame_hamiltonian(tmat,g,omega,lam_orbit,shift_orbit)`、
`lf_noci_lowest(H,S)` 即可做固定种子的实验；
本阶段不在 `src` 增加编排驱动器。

## 3. 手算与验收

请先手算：

1. `L=4` 时，若唯一非零条件位移是
   `eta[0,0]`，`R=1` 后它位于哪个
   `[x,p]`？分别只移动 x 或只移动 p 会发生什么？
2. 用上面的指标式验证 `T_R T_Q=T_(R+Q)`，
   以及同时做规范变换
   `lam[x,p]+=u[x], shift[x]+=u[x]` 后
   平移轨道的物理 `eta` 不变。
3. 解释为什么 L 个破缺态的平移副本能恢复一个完整的
   **平移轨道**，但未必张成 exact 声子态的全部关联空间。

验收测试覆盖：非对称种子的明确坐标迁移、轨道的循环顺序、
`eta` 的规范不变性、四站点 `H/S` 的联合平移
协变性，以及接入现有 NOCI 后的变分能量不高于种子单帧能量。
测试入口：

```bash
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_lf_translation_orbit.py
```

完成函数和手算后回复“请验收”；我会审查物理平移的两个索引、
运行测试，通过后补齐 docstring。