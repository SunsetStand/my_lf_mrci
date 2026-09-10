# MR_LF：导师式计算物理与量子化学学习项目

## 教学协作
- 用户是本科生，希望亲手理解并实现核心代码。助手扮演导师，不一次生成完整复现程序。
- 每次只安排一个边界清楚的小阶段：阅读材料 → 公式推导 → 接口、数据类型和数组维数 → 小编码作业 → 可执行验收测试。
- 用户提交实现后，先 review 并解释问题，通过本阶段验收后再推进。可以介绍整体依赖关系，但不要同时布置后续所有阶段。
- 允许伪代码、局部示例、测试与环境辅助代码；模型特有核心函数由用户实现或明确主导。不要因模型能力或工具可用而自行扩大范围。
- 实现前明确数学定义、依赖图、数据流、输入输出、dtype、shape 和索引顺序。
- 用户在会话中的明确授权和最新指令优先于本文件；不要把教学约束解释为额外的确认流程。

## 项目与环境
- 主项目为 `/mnt/d/code/MR_LF`；`/home/enovo/MR_LF` 是另一个目录，不混用。
- 使用 WSL2 `ubuntu-d`，Linux 解释器为 `/home/enovo/.venvs/mr_lf/bin/python`。
- 保留项目原有 Windows `.venv`，不在其中安装或编译 PySCF。
- 修改前检查磁盘现状、适用指令及 Git 状态；保留用户已有代码，不覆盖未提交工作。
- 依赖和解释器记录使用现有 `requirements.txt`、`.vscode/settings.json`、README；避免无关脚手架。
- 论文参考：`/mnt/d/下载/Cui24-Lang-Firsov-MP.pdf`。交接记录与磁盘不一致时，报告差异，以实际文件为准。

## 表示与物理约定
- Fig. 2b 目标：周期性一维 Hubbard–Holstein，L=4、nelec=(1,0)、omega=0.5、|t|=1，alpha=g**2/omega；g=0 基态能量为 -2。
- 单电子下 Hubbard 双占据项为零。声子项采用无零点常数的 omega * sum(b_i† b_i)。
- 明确记录耦合约定：论文未中心化 n_i；PySCF direct_ep 使用 n_i-N_e/L。
- 固定电子数、完整声子空间下，未中心化能量为中心化能量减 g**2*N_e**2/(L*omega)，Fig. 2b 为减 alpha/4。有限截断空间的位移不保证严格等价，须分别验证截断收敛；直接使用论文耦合时不再加此修正。
- d=Nmax+1，P=d**L；单电子结构化表示 psi_site.shape=(L,P)，D=L*P。
- 一般 FCI 张量 shape=(nstra,nstrb)+(d,)*L，nstr_sigma=C(L,N_sigma)。仅在求解器边界 flatten。
- 区分 site index、determinant address、bit string 整数值和 MO index。exact/site-basis 用 psi_site；LF-MP/MO-basis 用 chi_mo。

## PySCF 使用边界
- NumPy 用于张量、局部算符和数值验证；cistring 用于行列式枚举、link table 和费米符号；lib.davidson 用于 matrix-free 求解。
- ao2mo、rdm 留到需要的积分变换和密度矩阵阶段，不强行引入。
- 核心接口显式接收 (neleca,nelecb)，不依赖私有 _unpack_nelec；可仅在环境测试中检查其兼容性。
- direct_ep 仅作为接口和算法参考，不复制或直接调用其模型函数来替代学生实现。
- make_hdiag 必须从所选 Hamiltonian 独立推导，以 tiny dense 对角线验证，不照搬 direct_ep.make_hdiag。

## 推进与验证
- 顺序：basis/局部算符 → tiny dense ED → dense 与 matrix-free 交叉验证 → Davidson → 声子截断收敛 → CS-HF/MP2 → variational LF-HF → LF-MP2/MP4 → 参数扫描与 Fig. 2b。
- 第一阶段限定基础函数与 cistring 练习，不提前实现完整 Hamiltonian 或 kernel。
- 基准：四格点 t=-1 hopping 谱为 (-2,0,0,2)；Nmax=2 时 P=81；g=0 Davidson 能量为 -2，独立 residual <1e-10。
- 只报告实际运行过的检查；环境测试不代表模型测试通过。未完成的学生作业、未运行的验收和未收敛的曲线必须明确标注。
- exact 数据须记录参数、耦合/能量约定、Nmax、维数、能量、residual 与截断收敛证据，再用于论文比较。
- 复用基础设施前核对表象和算符定义；LF 变换后的位移算符、MP 分母和高阶修正不能直接当作原始 site-basis contraction。
