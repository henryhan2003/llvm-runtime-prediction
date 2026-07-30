# 问题1：dataset 代码分类、应用场景与资源密集型判断

## 1. 结论先行

本项目的目标是把 C/C++/CUDA 源码转换为 AST、CFG、LLVM IR 等程序表示，再从表示中提取影响运行时间的属性，形成 `Representation -> Resource Demand -> Runtime` 的建模数据集。结合 `0604研究进展整理.docx` 中的路线，`datasets` 目录应按“基准集来源 + 任务类型 + 资源瓶颈 + 输入规模”四层打标签，而不是只按文件扩展名打标签。

建议采用如下用途划分：

| 数据集目录 | 代码形态 | 主要用途 | 是否适合正式训练 | 资源特征概括 |
|---|---|---|---|---|
| `datasets/test` | 少量 `.c` 测试程序 | 调试 AST/IR/CFG 生成脚本、验证运行时间采集流程 | 否，只做 smoke test | 规模小，通常无法代表真实瓶颈 |
| `datasets/PolyBenchC-4.2.1` | 规整 C 计算内核 | 优先用于脚本调试、特征定义、输入规模实验 | 是，适合作为第一批 CPU 训练集 | 循环密集，矩阵/数组访问多，适合分析计算/内存边界 |
| `datasets/rodinia_3.1` | C/C++/CUDA/OpenMP/OpenCL 应用内核 | 正式训练集主力，覆盖图算法、图像、物理仿真、数据挖掘等 | 是，尤其适合研究异构/并行程序 | 同时包含 CPU 密集、内存密集、分支不规则、同步密集任务 |
| `datasets/cuda-samples-13.3` | CUDA C++/Python 示例、库示例、平台示例 | 增强泛化能力和 GPU 特性覆盖 | 有选择地使用 | 需要筛选；不是所有样例都适合运行时间建模 |

## 2. 分类标签体系

每个程序建议至少标注以下字段：

| 字段 | 取值示例 | 说明 |
|---|---|---|
| `dataset` | `PolyBenchC` / `Rodinia` / `CUDA Samples` / `test` | 来源数据集 |
| `language` | `C` / `C++` / `CUDA` / `OpenMP` / `OpenCL` | 编译和运行方式不同，应进入环境元数据 |
| `task_type` | 矩阵运算、图算法、排序、图像处理、物理仿真、数据库查找等 | 建模时用于分层评估 |
| `parallel_model` | serial / OpenMP / CUDA / OpenCL / library | 影响运行时间的硬件依赖很强 |
| `dominant_resource` | CPU compute / memory bandwidth / cache locality / branch irregular / synchronization / I/O | 理论瓶颈标签 |
| `input_scale_keys` | `N,M,K` / `nodes,edges` / `width,height,frames` | 运行时间不可只由代码结构解释，必须记录输入规模 |
| `include_in_training` | yes / maybe / no | 过滤掉设备查询、GUI、纯 API 演示等不稳定样例 |

## 3. PolyBenchC 子目录应用场景

PolyBenchC 的优点是结构规整、输入规模由宏控制，适合做第一批可控实验。多数程序是数组/矩阵循环内核，能够很好地检验 AST 循环特征、CFG 循环结构和 LLVM IR 中 load/store/arithmetic 的关系。

| 子目录/程序 | 具体应用场景 | 典型输入规模 | 资源密集型判断 |
|---|---|---|---|
| `datamining/correlation` | 相关系数矩阵计算，常见于统计分析和特征相关性分析 | 样本数、特征数 | 内存带宽 + 浮点计算密集 |
| `datamining/covariance` | 协方差矩阵计算，常见于 PCA、统计建模 | 样本数、特征数 | 内存带宽 + 浮点计算密集 |
| `linear-algebra/kernels/2mm` | 两次矩阵乘法链 | 矩阵维度 `NI,NJ,NK,NL` | CPU/FPU 计算密集，缓存局部性重要 |
| `linear-algebra/kernels/3mm` | 三次矩阵乘法链 | 多个矩阵维度 | CPU/FPU 计算密集，内存流量大 |
| `linear-algebra/kernels/atax` | 矩阵转置乘向量组合 | `M,N` | 内存带宽密集，访问模式影响缓存 |
| `linear-algebra/kernels/bicg` | BiCG 子步骤，稀疏迭代法的简化核心 | `M,N` | 内存带宽密集，计算量中等 |
| `linear-algebra/kernels/doitgen` | 三维张量与矩阵乘法 | `NR,NQ,NP` | 计算密集 + cache blocking 敏感 |
| `linear-algebra/kernels/mvt` | 矩阵向量乘和转置矩阵向量乘 | `N` | 内存带宽密集 |
| `linear-algebra/blas/gemm` | 通用矩阵乘 | `NI,NJ,NK` | 典型 CPU/FPU 计算密集 |
| `linear-algebra/blas/gemver` | 矩阵 rank-1 更新和矩阵向量乘 | `N` | 计算 + 内存混合 |
| `linear-algebra/blas/gesummv` | 两个矩阵向量乘的加权求和 | `N` | 内存带宽密集 |
| `linear-algebra/blas/symm` | 对称矩阵乘 | `M,N` | 计算密集，数据复用明显 |
| `linear-algebra/blas/syr2k` | 对称 rank-2k 更新 | `N,M` | 计算密集，缓存复用重要 |
| `linear-algebra/blas/syrk` | 对称 rank-k 更新 | `N,M` | 计算密集 |
| `linear-algebra/blas/trmm` | 三角矩阵乘 | `M,N` | 计算密集，分支/循环边界略复杂 |
| `linear-algebra/solvers/cholesky` | Cholesky 分解 | `N` | 计算密集，数据依赖强 |
| `linear-algebra/solvers/durbin` | Toeplitz 系统求解 | `N` | 数据依赖强，向量操作为主 |
| `linear-algebra/solvers/gramschmidt` | Gram-Schmidt 正交化 | `M,N` | 计算 + 内存混合，归约敏感 |
| `linear-algebra/solvers/lu` | LU 分解 | `N` | 计算密集，数据依赖强 |
| `linear-algebra/solvers/ludcmp` | LU 分解和求解 | `N` | 计算密集，内层循环明显 |
| `linear-algebra/solvers/trisolv` | 三角线性系统求解 | `N` | 数据依赖强，串行性较强 |
| `medley/deriche` | Deriche 图像滤波 | 图像宽高 | 图像处理，内存带宽 + 递推依赖 |
| `medley/floyd-warshall` | 全源最短路 | 节点数 `N` | 计算密集，三重循环，缓存敏感 |
| `medley/nussinov` | RNA 二级结构动态规划 | 序列长度 | 动态规划，内存 + 分支/依赖混合 |
| `stencils/adi` | ADI 数值求解 | 网格规模、时间步 | stencil，内存带宽 + 数据依赖 |
| `stencils/fdtd-2d` | 2D 电磁场有限差分 | 网格规模、时间步 | stencil，内存带宽密集 |
| `stencils/heat-3d` | 3D 热扩散 | 三维网格、时间步 | stencil，内存带宽密集，缓存压力高 |
| `stencils/jacobi-1d` | 1D Jacobi 迭代 | `N,T` | 内存带宽密集 |
| `stencils/jacobi-2d` | 2D Jacobi 迭代 | `N,T` | 内存带宽密集，空间局部性重要 |
| `stencils/seidel-2d` | 2D Seidel 迭代 | `N,T` | 内存带宽 + 数据依赖 |

## 4. Rodinia 子目录应用场景

Rodinia 是正式训练集的主力，因为它覆盖真实应用抽取出的计算 kernel，并且同一任务常有 CUDA、OpenMP、OpenCL 版本。建议优先选择能够稳定命令行运行、输入文件明确、无 GUI 依赖的程序。

| 子目录 | 具体应用场景 | 典型输入规模 | 资源密集型判断 |
|---|---|---|---|
| `b+tree` | 数据库/索引结构查询 | 记录数、查询数、树阶数 | 随机访存密集，cache miss 和分支预测敏感 |
| `backprop` | 神经网络反向传播 | 层规模、样本数、迭代数 | 计算密集 + 内存带宽混合 |
| `bfs` | 图广度优先搜索 | 节点数、边数、图稀疏度 | 不规则内存访问密集，分支不规则 |
| `cfd` | 计算流体力学 | 网格单元数、迭代数 | 浮点计算 + 内存带宽混合 |
| `dwt2d` | 二维离散小波变换 | 图像宽高、分解层数 | 图像/信号处理，内存带宽密集 |
| `gaussian` | 高斯消元 | 矩阵维度 | 计算密集，数据依赖强 |
| `heartwall` | 医学视频心壁跟踪 | 帧数、图像宽高、跟踪点数 | 图像处理，内存带宽 + 控制流混合 |
| `hotspot` | 芯片热仿真 2D stencil | 网格大小、迭代数 | 内存带宽密集，局部性重要 |
| `hotspot3D` | 芯片热仿真 3D stencil | 三维网格、迭代数 | 内存带宽密集，cache 压力更高 |
| `huffman` | Huffman 编码/压缩 | 输入字节数、符号分布 | 分支/位操作 + 内存混合 |
| `hybridsort` | 混合排序 | 元素数量、键分布 | 内存带宽 + 分支/同步敏感 |
| `kmeans` | 聚类 | 点数、维度、簇数、迭代数 | 内存带宽 + 浮点计算混合 |
| `lavaMD` | 分子动力学 | 粒子数、盒子数 | 计算密集，邻域访问影响缓存 |
| `leukocyte` | 医学图像白细胞跟踪 | 图像帧数、窗口大小 | 图像处理，内存密集 |
| `lud` | LU 分解 | 矩阵维度、块大小 | 计算密集，块大小影响缓存 |
| `mummergpu` | 生物序列匹配 | 参考序列长度、reads 数、read 长度 | 字符串/不规则访存密集 |
| `myocyte` | 心肌细胞仿真 | 时间步、模型状态变量数 | CPU/FPU 计算密集 |
| `nn` | 最近邻搜索 | 数据点数、查询数、维度 | 内存带宽密集，距离计算为主 |
| `nw` | Needleman-Wunsch 序列比对 | 两条序列长度 | 动态规划，内存 + 数据依赖 |
| `particlefilter` | 粒子滤波 | 粒子数、帧数、随机种子 | 内存 + 随机数 + 分支混合 |
| `pathfinder` | 网格路径动态规划 | 网格宽高、时间步 | 内存带宽密集，动态规划依赖 |
| `srad` | Speckle Reducing Anisotropic Diffusion 图像降噪 | 图像宽高、迭代数 | stencil/图像处理，内存带宽密集 |
| `streamcluster` | 流式聚类 | 点数、维度、中心数、chunk 大小 | 内存带宽 + 同步/归约敏感 |

Rodinia 的 `cuda`、`openmp`、`opencl` 三个目录可以作为同一算法在不同并行模型下的对照。建模时不要把它们混成同一种程序；应加入 `parallel_model`、`num_threads`、`gpu_name`、`block_size`、`work_group_size` 等环境/运行参数。

## 5. CUDA Samples 子目录应用场景

CUDA Samples 的价值在于补充 GPU 编程特性和库调用模式，但它不是纯 benchmark 数据集。建议先按一级目录筛选，再按 README 判断是否适合自动运行。

| 一级目录 | 应用场景 | 建模建议 |
|---|---|---|
| `cpp/0_Introduction` | 入门 kernel、向量加法、矩阵乘、流、统一内存、原子操作等 | 可选；保留 `vectorAdd`、`matrixMul`、`simpleStreams` 等可重复运行样例，剔除纯打印或设备查询类 |
| `cpp/1_Utilities` | `deviceQuery`、拓扑查询等工具 | 通常不纳入运行时间预测训练，因为主要测环境查询，不测算法复杂度 |
| `cpp/2_Concepts_and_Techniques` | reduction、scan、sort、histogram、texture、convolution、Monte Carlo 等算法技术 | 建议重点使用；覆盖访存、分支、并行归约、排序等 GPU 典型模式 |
| `cpp/3_CUDA_Features` | CUDA Graph、cooperative groups、dynamic parallelism、tensor core、async copy 等特性 | 可作为 GPU 特性补充集，但要单独打 `cuda_feature` 标签 |
| `cpp/4_CUDA_Libraries` | cuBLAS、cuSolver、NPP、nvJPEG、CUB、FFT 等库调用 | 不宜与纯手写 kernel 混合建模；应标记 `library_call_heavy=yes` |
| `cpp/5_Domain_Specific` | N-body、图像/视频、金融、信号处理等领域样例 | 适合泛化测试，但输入文件和外部依赖要固定 |
| `cpp/6_Performance` | 性能优化、带宽、异步拷贝、occupancy 等 | 适合研究资源需求，但部分样例更像 microbenchmark |
| `cpp/7_libNVVM` | NVVM/编译相关示例 | 更适合编译流程研究，不一定适合算法运行时间建模 |
| `cpp/8_Platform_Specific` | Tegra、DirectX/OpenGL/Vulkan/平台互操作 | 多数依赖图形或平台环境，建议默认剔除 |
| `cpp/9_CUDA_Tile` | tile 级矩阵乘、转置、SpMV 等 | 适合 GPU 内存访问和并行粒度研究 |
| `python/*` | CUDA Python 示例 | 与 C/C++/LLVM IR 生成路线不完全一致，建议暂不纳入第一阶段 |

## 6. CPU 密集型、内存密集型等判断规则

建议不要只凭主观经验判断瓶颈，而是使用“静态初判 + 动态校准”的两阶段方法。

| 类型 | 静态初判依据 | 动态校准指标 | 典型程序 |
|---|---|---|---|
| CPU/FPU 计算密集 | `add/mul/fmul/fadd` 等算术指令多，循环中 load/store 相对少，矩阵乘/分解明显 | 高 IPC、低 cache miss、FLOPs 占比高 | `gemm`、`2mm`、`3mm`、`lavaMD`、`myocyte` |
| 内存带宽密集 | `load/store/getelementptr` 多，数组扫描多，算术强度低 | cache miss 多、内存带宽接近上限、IPC 下降 | `jacobi-2d`、`heat-3d`、`mvt`、`pathfinder` |
| Cache 局部性敏感 | 多维数组、stride 访问、转置访问、stencil 邻域访问 | L1/L2 miss 对运行时间解释力强 | `atax`、`doitgen`、`srad`、`hotspot3D` |
| 分支/控制流不规则 | CFG 分支节点多，循环边界依赖数据，`switch/if` 密集 | branch miss 较高，运行时间方差较大 | `bfs`、`b+tree`、`huffman`、`hybridsort` |
| 同步/并行开销敏感 | OpenMP/CUDA/OpenCL 版本，存在 barrier、atomic、reduction | 线程数变化下加速比不线性 | `streamcluster`、`kmeans`、CUDA reduction/scan |
| I/O 或初始化影响明显 | 程序读取大输入文件，计时包含加载/解析 | page fault、磁盘读写、冷/热缓存差异大 | `heartwall`、`mummergpu`、图像/视频样例 |

## 7. 纳入训练集的建议顺序

1. 第一阶段：`datasets/test` + PolyBenchC。目标是跑通递归源码扫描、AST/CFG/IR 生成、指标采集、运行时间采集。
2. 第二阶段：Rodinia OpenMP/CPU 版本。目标是引入真实应用类型和输入文件规模。
3. 第三阶段：Rodinia CUDA/OpenCL 版本。目标是建立 GPU/异构程序的并行特征。
4. 第四阶段：筛选 CUDA Samples。只纳入可命令行稳定运行、输入规模可控、非 GUI/非设备查询/非纯 API 演示的样例。

