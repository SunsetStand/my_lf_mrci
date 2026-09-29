# Fig. 2d 四电子 ED：扫描与截断验收

本次计算复用 src/my_direct_ep.py 中的电子、Hubbard、声子及中心化电子–声子收缩函数。新增脚本只组织迭代求解、初猜、缓存、截断扫描与绘图。

## 模型和能量约定

四格点周期环，$t=-1$、$U=4$、$(N_\alpha,N_\beta)=(2,2)$，四个局域声子模。扫描 $\alpha=0,0.4,\ldots,4$，分别取 $\omega=0.5,5$，并令 $g=\sqrt{\alpha\omega}$。

中心化 Hamiltonian 的耦合使用 $g(n_i-1)(b_i+b_i^\dagger)$。结果转换为论文的未中心化能量：

$$
E_{\mathrm{paper}}=E_{\mathrm{centered}}-\frac{g^2N_e^2}{L\omega}
=E_{\mathrm{centered}}-4\alpha.
$$

这一关系在完整声子空间严格成立；有限局域截断不保持位移变换。因此表中的有限 cutoff 能量是用上述常数换算的中心化计算结果，只有通过下述截断收敛检验后才作为论文约定下的 ED 估计。
具体地，在 $\omega=0.5,\alpha=4,N_{\max}=16$，先前直接使用未中心化耦合得到 $-19.68509498$，而本次中心化结果换算后为 $-24.65391201$。这种同 cutoff 差异正是截断空间不保持位移变换的表现，不能误判为两个 Hamiltonian 的完整空间能量不同。

声子 Hamiltonian 不含零点能。每个局域模截断在 $0,\ldots,N_{\max}$，完整电子 sector 有 $\binom{4}{2}^2=36$ 个基态：

$$
D=36(N_{\max}+1)^4.
$$

本次保留四个局域声子模，没有通过模式消元或新的 LF ansatz 缩小模型。
在固定四电子 sector，$\sum_i(n_i-1)=0$，所以中心化耦合与均匀声子模解耦；计算仍保留四个局域模的完整张量表示。
在固定 $\alpha$ 下，非零局域中心化电荷对应的相干位移尺度是 $g/\omega=\sqrt{\alpha/\omega}$。例如 $\alpha=4$ 时，$\omega=0.5$ 的该尺度约为 $2.83$，而 $\omega=5$ 约为 $0.89$；前者因此需要检查更高的局域声子截断。

参数核对来源：[Cui 等，Fig. 2 与 III.1](https://arxiv.org/html/2310.13084v2#S3.SS1)。本次图展示两组 ED 曲线；现有单电子 LF-HF/MP2 接口没有用于四电子近似曲线。

## 收敛标准

每一个非零 alpha 点单独增大 cutoff：

1. 有限矩阵最低态求解成功，重新计算的 $\|H\psi-E\psi\|<10^{-8}$。
2. 连续两次增大 cutoff 后，$|\Delta E|<10^{-6}$，对应三次合格的求解。
3. 检查增大空间没有导致超出舍入误差的能量上升。
4. alpha=0 时声子真空已包含电子基态，不需要声子尾部检验。

这是明确容差下的数值收敛判据，不是无限精度保证。计算中没有放宽残差标准来接受 Davidson 停滞的点。

## 求解器与计算成本

- Hamiltonian action 与对角预条件数据均直接来自用户的 src/my_direct_ep.py。
- 在一次求解中缓存原有 phonon_configs 的输出，避免每次矩阵乘都重新枚举。
- 更大 cutoff 使用前一个波函数的零填充嵌入作为初猜。常规扫描另加电子真空初猜，使其他对称性分量可以参与求解；低频高耦合补算只使用已经收敛的嵌入态，以降低大维度求解的内存占用。
- 初始生产批次使用嵌入态与少量电子真空分量混合；这些初猜方式使用同一 Hamiltonian 和验收阈值。
- PySCF Davidson 设置内存预算，较大 Krylov 空间可使用临时磁盘存储。
- 在观察到残差停滞时，当前驱动器改用 SciPy Lanczos/eigsh，仍调用相同 Hamiltonian action，并重新检查 residual。
- BLAS/OpenMP 线程设为 1，避免张量小操作中线程开销过大。

原脚本 kernel 不支持传入初猜或 max_memory，因此数据脚本直接组织通用本征求解器；src 中的模型函数没有改动。

长作业中途因运行环境重启而中断；已完成的 cutoff 行留在原始 CSV。恢复批次将波函数保存在被 Git 忽略的 `.fig2d_checkpoints/`，重算了 $\alpha=3.2$ 并接着计算 $3.6$。原始高耦合 CSV 中已经收敛的 $2.4,2.8$ 两点另存为不含重复行的输入文件，原始记录不作删改。$\alpha=4$ 的最终补算由 [fig2d_complete_anchor.py](../scripts/fig2d_complete_anchor.py) 先重建 $N_{\max}=32$ 波函数，与已有能量核对，再热启动下一 cutoff。重建初猜是各个配对电子行列式与其条件相干态的叠加，局域位移 $\eta_x=-(g/\omega)(n_x-1)$；它只决定迭代起点，最终能量仍由原 Hamiltonian 收缩和残差检验确定。

## 文件与复现

运行完整扫描：

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/enovo/.venvs/mr_lf/bin/python -m scripts.fig2d_exact_scan \
  --output data/fig2d_exact_convergence.csv
~~~

该命令覆盖指定输出文件。首次完整计算可能耗时较长。实际生产过程分为 omega=5、omega=0.5 的低/高 alpha 段，以及 alpha=4 的困难点，分别保存中间 CSV；每个 cutoff 完成后立即刷新文件。

绘制完整扫描：

~~~bash
/home/enovo/.venvs/mr_lf/bin/python -m scripts.plot_fig2d_exact \
  data/fig2d_exact_convergence.csv
~~~

也可以向绘图命令传入多个没有重复点/截断的分段 CSV。绘图前要求 22 个点完整，且最终 cutoff 与残差标志均合格。它会写出：

- data/fig2d_exact.csv：每个 alpha/omega 的最终通过点。
- data/fig2d_exact_convergence.csv：所有 cutoff 记录。
- figures/fig2d_exact.png/pdf：两组 ED 能量曲线。
- figures/fig2d_exact_cutoff_convergence.png/pdf：代表性 alpha 的截断变化图。

脚本：[fig2d_exact_scan.py](../scripts/fig2d_exact_scan.py)、[plot_fig2d_exact.py](../scripts/plot_fig2d_exact.py)。

## 验证

[test_fig2d_exact_scan.py](../tests/test_fig2d_exact_scan.py) 包括：与原 kernel 的零/非零耦合对照、真实小规模 cutoff 序列、能量常数、未收敛标记、重复 cutoff 拒绝、初猜维度与输入所有权，以及停滞后的 Lanczos 备用求解。新增测试共 7 项已通过。旧的 $\omega=0.5,\alpha=4,N_{\max}=16$ 中心化测试能量 $-24.65391200554646$，与本次独立批次的 $-24.653912005546$ 一致。另用此前独立收敛的未中心化序列核对 $\omega=5,\alpha=2$：其 $N_{\max}=16$ 能量为 $-11.096623035063125$；本次中心化扫描换算后为 $-11.096623035077718$，相差约 $1.5\times10^{-11}$。

完整生产扫描状态与最终数值以生成后的 CSV 为准。本说明不把单元测试通过当作完整扫描已经结束。
