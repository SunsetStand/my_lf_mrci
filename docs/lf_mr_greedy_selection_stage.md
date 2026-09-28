# 下一阶段：固定候选池的迭代 NOCI 选择

本阶段目标：你亲手把收敛脚本中的迭代选择写成一个可复用的核心函数，并能说明每一步为什么接受或拒绝候选。参考 [结果说明](lf_mr_results_and_selection_notes.md) 第 5–8 节，以及现有脚本的 while 循环。

只新增下面一个函数到 src/lf_mr.py。核心代码由你实现；输入检查中的常规细节可以在“请验收”时由我补齐。不在本阶段实现新的候选生成法、MR-PT2 或位移再优化。

## 1. 数学规则与边界

给定当前参考帧集 $\mathcal B$，候选 seed $j$ 通过已有 lf_translation_orbit 生成完整轨道 $\mathcal O_j$。

$$
E_j=E(\mathcal B\cup\mathcal O_j),\qquad
\Delta E_j=E(\mathcal B)-E_j.
$$

本阶段复现现有脚本的基线接受规则：

$$
j\text{ 合格}\iff r_j>r_{\mathcal B}
\quad\mathrm{且}\quad
\Delta E_j>\mathrm{min\_gain}.
$$

在合格者中选 $E_j$ 最小者；精确并列时选原始候选编号最小者。只添加这一条完整轨道，重新优化全部 CI 系数，再对剩余候选重新评分。

保留秩增加是当前算法的选择条件，不是物理定理。重新截断后，即使秩相同，保留空间的方向也可能改变。当前规则以及 cutoff 的影响都必须在记录中可见。

当没有合格候选时只报告 no_eligible_candidate，不能报告 converged。固定 cut 下没有正收益可能与截断导致的非嵌套有关。

## 2. 唯一函数接口

~~~python
lf_noci_greedy_orbits(
    tmat,
    g,
    omega,
    base_lam,
    base_shift,
    candidate_lam,
    candidate_shift,
    *,
    overlap_cut=1e-10,
    min_gain=1e-8,
    max_additions=None,
) -> dict
~~~

输入数组均为有限 float64：

| 输入 | shape / 定义 |
| --- | --- |
| tmat | (L,L)，实对称 |
| base_lam | (K,L,L)，[frame,mode,site]；K,L>0 |
| base_shift | (K,L) |
| candidate_lam | (J,L,L)，[candidate,mode,site]；允许 J=0 |
| candidate_shift | (J,L) |
| g / omega | 有限实数 / 有限正实数 |
| overlap_cut | 有限正数，所有评分与最终解统一使用 |
| min_gain | 有限非负数，严格大于才接受 |
| max_additions | None 或非负整数；限制新增轨道的条数，不是帧数 |

继续使用当前单电子模型约定。函数不接收 alpha、theta、exact_energy 或路径；候选可以来自任意生成器。不会修改输入，不会在函数内做 LF-HF 优化。

返回字典：

| key | 内容 |
| --- | --- |
| lam / shift | 最终帧数组，float64；shape (K_final,L,L)/(K_final,L) |
| coeff | 最终 NOCI 系数，float64，shape (K_final,L)，按 S 归一化 |
| energy / rank | 最终能量 float / 保留秩 int |
| selected_indices | int64，shape (n_selected,)，按接受顺序记录输入池的原始编号 |
| history | 初始状态及每次接受后的状态列表，长度 n_selected+1 |
| trials | 每轮实际评分的完整记录列表，包括未能选出候选的最后一轮 |
| stop_reason | pool_exhausted、max_additions 或 no_eligible_candidate |

history 每项为一个字典，包含：

- candidate_index：初始项为 None，之后为原始候选编号；
- n_frames、energy、rank；
- gain：初始项为 None，之后为上一步能量减本步能量。

trials 每项包含：

- energy_base、rank_base；
- candidate_indices：本轮参与评分的原始编号，int64；
- energy_trial、rank_trial：与编号逐一对应；
- gain、eligible：float64 / bool 数组；
- selected_index：原始编号或 None。

保存数组时复制，避免后续删除候选或更新变量改变历史记录。负 gain 必须原样保存，不能 clip 为零。

## 3. 依赖关系与伪代码

~~~text
已有基础帧 + 候选 seeds
    ↓
lf_noci_score_orbits：在同一基础空间上逐个试加完整轨道
    ↓
合格掩码 + argmin
    ↓
lf_translation_orbit：生成选中 seed 的轨道
    ↓
拼接基础帧，记录状态，移除候选
    ↓
下一轮重新评分
~~~

~~~text
复制基础帧
维护 remaining_ids = [0, ..., J-1]，与输入池原始编号对应
求解初始 NOCI，记录 history[0]

循环：
    若 remaining_ids 为空：
        stop_reason = pool_exhausted；退出
    若已达到 max_additions：
        stop_reason = max_additions；退出

    对 remaining_ids 对应的 seeds 调用 lf_noci_score_orbits
    计算 gain 与 eligible
    准备本轮 trials 记录，保存所有原始编号与分数

    若无合格候选：
        selected_index = None；保存本轮记录
        stop_reason = no_eligible_candidate；退出

    在 eligible 中按最低试加能量选择候选
    把局部数组位置映射回原始 candidate_index
    生成并追加完整平移轨道
    保存本轮 trials 与接受后的 history
    将原始 candidate_index 加入 selected_indices
    从 remaining_ids 删除该编号

在最终帧集上构造 H、S 并求解最终 coeff、energy、rank
检查最终结果与最后一条 history 一致（在合理数值容差内）
返回结果字典
~~~

优先检查空池，再检查 max_additions，因此同时满足这两种终止条件时返回 pool_exhausted。

可复用现有 lf_noci_score_orbits，暂不要求优化它反复计算基础能量的成本。所有新的物理选择逻辑只在这一个函数中定义；脚本之后只负责候选构造、模型参数和输出。

## 4. 验收安排

你完成后说“请验收”，我准备并运行针对性测试：

1. 空候选池和 max_additions=0：初始解保持，返回原因准确，不调用拒绝空池的评分函数。
2. 重复 seed 的完整轨道：不因帧数增加而误认为增加了有效空间。
3. alpha=2.4、基础 8 帧、只有 theta=1.5：应追加 4 帧，得到 rank=36，能量约 -3.040366167。
4. 多候选实际模型：每轮选取结果等于对当前基础空间独立试加比较的结果；不是预先排序一次后照单追加。
5. 原始编号与输入所有权：候选删除后编号不串位；输入不变；history/trials 不因后续更新改变。
6. 固定同一批 LF 参数，复现现有完整七候选脚本的顺序、能量与秩。原始 theta 编号为 [0.25,0.5,0.75,1.25,1.5,1.75,2]，已有数据的接受编号为 [6,0,1,2,3,4]。
7. 最后 theta=1.75 的负收益需保留在 trials，停机原因为 no_eligible_candidate；不把它标记为物理收敛。该末段检查对照同一参数数组求解，不依赖两次优化所得参数逐位相同。

本阶段尚未实现、尚未验收。通过后再由我补齐 docstring，并将数据脚本改为调用该函数；旧数据作为基线保留。

## 5. 需要你先想明白的两个问题

- 为什么加入一条轨道后，剩余所有候选都需要重新评分？
- 为什么固定 overlap_cut 并不保证扩展后的保留子空间包含旧保留子空间？

这两个问题分别决定未来是否需要前瞻选择，以及是否需要保留旧空间的增量正交化。先完成可审计的选择驱动器，再对这些策略作有控制的比较。
