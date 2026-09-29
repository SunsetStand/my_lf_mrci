# Fig. 2b：用 α=2.4 的破缺帧扩充破缺前 MR-LF 空间

本实验在原有 MR-LF 扫描的 α=0,0.2,…,2.2 共 12 个点上，加入同一组来自 α=2.4 的对称性破缺 LF-HF 帧。模型为四点周期环、单电子、`t=-1`、`omega=0.5`、论文的未中心化耦合，`alpha=g²/omega`。所有新增能量都使用**目标 α 自己的** `g=sqrt(alpha*omega)` 计算，没有加 MP2 修正。

## 帧空间与数据流

每个 LF 帧由 `lam[x,p]` 和 `shift[x]` 表示，均为 `float64`，shape 分别为 `(4,4)` 与 `(4,)`；`x` 是声子 mode，`p` 是电子 site。物理条件位移为 `eta[x,p]=shift[x]-lam[x,p]`。一次四点平移生成 shape `(R=4,x=4,p=4)` 的轨道，并同时平移 mode 与 site。

对目标 `alpha` 重新优化 LF-HF，取其最低成功分支和 CS 起点优化后的分支，各自的完整平移轨道组成原 MR-LF 基础空间 `V_base(alpha)`。α=2.4 的最低成功 LF-HF 分支具有密度不均衡度 0.693945981；把其完整平移轨道 `O_broken(2.4)` 加入：

\[
V_{\rm new}(\alpha)=V_{\rm base}(\alpha)+O_{\rm broken}(2.4),\qquad
E_{\rm new}(\alpha)=\min_{\psi\in V_{\rm new}}
\frac{\langle\psi|H(\alpha)|\psi\rangle}{\langle\psi|\psi\rangle}.
\]

原空间名义上有 8 帧、32 个 `(frame,site)` 基态，但因重复的对称帧只保留秩 4；扩充后是 12 帧、48 个名义基态，全部目标点的重叠矩阵保留秩均为 20。求解器仅在广义本征问题边界把 `(K,4,K,4)` 的核展平。

## 结果

下表误差是 `E_method-E_ED`，单位取 `|t|=1`。所有 α>0 点都满足 `0 < E_new-E_ED < E_original-E_ED`。α=0 的原方法已经达到 −2。

| α | 原 MR-LF 误差 | 新 MR-LF 误差 | LF-MP2 绝对误差 |
| ---: | ---: | ---: | ---: |
| 0.2 | 0.000373329 | 0.000325277 | 0.000009844 |
| 1.0 | 0.011431991 | 0.006153597 | 0.001651048 |
| 1.4 | 0.025414935 | 0.008669695 | 0.005429561 |
| 1.6 | 0.035670028 | 0.008743869 | 0.008977768 |
| 2.0 | 0.065933188 | 0.005926370 | 0.022166566 |
| 2.2 | 0.088104157 | 0.004270999 | 0.033740251 |

这组固定破缺帧缓解了原 MR-LF 在对称区与 LF-HF 重合的问题，但不能在整个区域替代 LF-MP2：在所用网格中，α≤1.4 时 LF-MP2 更接近 ED；α≥1.6 时新 MR-LF 更接近 ED。后一比较只谈能量误差，不意味着两种方法的计算成本相同。

每一点的重叠阈值取 `1e-10`；再用 `1e-8` 和 `1e-12` 求解，能量在 `1e-10` 内一致且秩仍为 20。全部扩充空间中最小保留重叠本征值大于 0.28，电子密度不均衡小于 `1e-10`。这是给定帧集合内的数值稳定性检查，不是帧数收敛证明。

ED 参考来自现有 [fig2b_exact.csv](../data/fig2b_exact.csv)，使用 `Nmax=24`、`D=1,562,500`、同一未中心化耦合；本实验没有重跑 ED。α=2.2 的保存值为 −2.935080429839，有限矩阵残差约 `9.97e-7`。残差不等于声子截断误差。新 MR-LF 在 α=2.2 的能量为 −2.930809431151。α=2.4 帧是指定多起点搜索中的最优成功结果，没有全局极小值证明；本实验也没有重新优化跨 α 的帧参数。

## 复现

```bash
/home/enovo/.venvs/mr_lf/bin/python -m scripts.fig2b_mr_broken_seed_scan
/home/enovo/.venvs/mr_lf/bin/python -m scripts.plot_fig2b_mr_broken_seed
/home/enovo/.venvs/mr_lf/bin/python -m pytest -q tests/test_fig2b_mr_broken_seed.py
```

逐点能量、秩、密度和重叠阈值检查见 [CSV](../data/fig2b_mr_broken_seed.csv)；实际 LF 参数见 [NPZ 帧归档](../data/fig2b_mr_broken_seed_frames.npz)；对照图见 [PNG](../figures/fig2b_mr_broken_seed.png) 和 [PDF](../figures/fig2b_mr_broken_seed.pdf)。原扫描文件与原图均未改写。
