# Rodinia 数据集程序用途详解

本文用于补充说明 `datasets/rodinia_3.1` 中各个 benchmark 子目录的应用背景、输入输出规模含义，以及它们适合用于哪些执行时间建模问题。Rodinia 和 PolyBenchC 不一样：PolyBenchC 更像规整的单文件数值 kernel，而 Rodinia 更接近真实应用抽取出的并行程序，通常包含 CUDA、OpenMP、OpenCL 多种实现。

本地 Rodinia 目录下没有发现 `data/` 输入数据目录，但很多 `run` 文件会引用 `../../data/...`。因此下面的命令示例主要来自各程序自带的 `run` 文件，用来说明它期望的输入规模字段；真正运行实验前，需要补齐 Rodinia 的输入数据文件，或者改用可由命令行生成规模的运行方式。

## 1. `b+tree`

`b+tree` 是 B+ 树查找 benchmark，应用背景是数据库索引、键值存储、文件系统索引等场景。B+ 树是一种多路平衡树，适合大量键值数据的范围查询和点查询。

它的典型命令类似：

```text
./b+tree.out file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt
./b+tree.out core 2 file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt
```

其中 `mil.txt` 可以理解为树中已有的数据记录，`command.txt` 是查询、插入或操作命令集合。OpenMP 版本里的 `core 2` 表示使用 2 个 CPU 核/线程。

输入规模应该记录：记录数量、key 数量、命令数量、查询/插入比例、树阶或节点容量、线程数。输出通常是每条命令的查询结果、匹配位置或执行状态。

对执行时间建模来说，它是典型的不规则内存访问程序。B+ 树查找不是连续数组扫描，而是根据 key 沿树节点跳转，容易产生 cache miss。AST/CFG 中的分支、循环和指针访问，以及 IR 中的 load/store、branch、比较指令都比较重要。

## 2. `backprop`

`backprop` 是神经网络反向传播 benchmark，应用背景是机器学习中的多层感知机训练。它会执行前向传播、误差计算和反向权重更新。

典型命令是：

```text
./backprop 65536
```

这里的 `65536` 一般表示输入层或训练样本相关规模。实际程序会按照该规模分配输入、隐藏层、权重和梯度数组。

输入规模应该记录：输入节点数、隐藏层节点数、输出节点数、训练样本数、迭代次数、线程数或 GPU block 配置。输出通常是更新后的权重、误差或校验信息。

它的主要计算是矩阵/向量乘加和激活函数相关操作，通常偏计算密集，但权重矩阵较大时也会受内存带宽影响。建模时应重点关注浮点算术指令、数组访问、load/store、循环嵌套和并行配置。

## 3. `bfs`

`bfs` 是图的广度优先搜索，应用背景包括社交网络分析、路径搜索、图可达性分析和稀疏图遍历。

典型命令是：

```text
./bfs ../../data/bfs/graph1MW_6.txt
./bfs 4 ../../data/bfs/graph1MW_6.txt
```

OpenMP 版本中的 `4` 表示线程数。`graph1MW_6.txt` 是图输入文件，通常包含节点、边、邻接关系等信息。

输入规模应该记录：图节点数、边数、平均度、最大度、起始源点、frontier 规模变化、线程数。输出通常是从源点到各节点的层数、距离或访问状态。

BFS 是 Rodinia 中非常重要的不规则 benchmark。它的运行时间不仅取决于节点和边的数量，还取决于图结构。链式图、星型图、社交网络图的 frontier 变化完全不同。它通常表现为访存不规则、分支不规则、负载不均衡，适合观察 CFG 分支、IR load/store、branch 指令和运行时间方差。

## 4. `cfd`

`cfd` 是计算流体力学 benchmark，用有限体积或类似数值方法求解流场。应用背景包括空气动力学、飞行器外形仿真、气体流动模拟等。

典型命令包括：

```text
./euler3d ../../data/cfd/fvcorr.domn.097K
./euler3d ../../data/cfd/fvcorr.domn.193K
./euler3d ../../data/cfd/missile.domn.0.2M
```

这些输入文件名中的 `097K`、`193K`、`0.2M` 可以理解为网格单元规模。不同实现还包括 `euler3d`、`euler3d_double`、`pre_euler3d` 等版本。

输入规模应该记录：网格单元数量、边/邻接数量、变量数量、迭代步数、单精度/双精度版本、线程数或 GPU 配置。输出通常是更新后的流体变量，例如密度、动量、能量或残差。

CFD 程序通常有大量浮点计算和邻域访存，既有计算密集特征，也有内存带宽压力。对模型来说，`fadd/fmul/fdiv`、load/store、数组访问和循环结构都很关键。

## 5. `dwt2d`

`dwt2d` 是二维离散小波变换，应用背景是 JPEG2000 图像压缩、图像多尺度分析和信号处理。

README 中的用法类似：

```text
./dwt2d [options] src_img.rgb <out_img.dwt>
-d, --dimension  例如 1920x1080
-c, --components 颜色通道数，默认 3
-l, --level      小波分解层数，默认 3
-f, --forward    正向变换
-5, --53         使用 5/3 小波变换
```

输入规模应该记录：图像宽度、高度、通道数、小波分解层数、变换类型、输入字节数。输出通常是小波系数图像或压缩前的变换结果。

它主要是规则图像处理计算，包含行/列方向的滤波、下采样和多层级处理。访存模式比 BFS 规则，但会涉及二维数组访问、边界处理和多层循环。建模时应关注 image size、level、load/store、浮点或整数算术、内存访问局部性。

## 6. `gaussian`

`gaussian` 是高斯消元，用于求解线性方程组。应用背景包括科学计算、工程仿真和数值线性代数。

典型命令是：

```text
./gaussian -f ../../data/gaussian/matrix4.txt
./gaussian -s 16
./gaussian -s 256
```

`-f` 表示从矩阵文件读取输入，`-s` 表示生成指定规模的矩阵。`-s 256` 就是一个 256 阶矩阵。

输入规模应该记录：矩阵阶数 `N`、是否从文件读取、数据类型、是否进行 pivot、线程数或 GPU 配置。输出通常是消元后的矩阵、解向量或校验结果。

它的理论计算量通常接近 `O(N^3)`，对规模非常敏感。它包含嵌套循环、浮点除法、乘减更新和大量矩阵访存，是计算密集和内存访问共同作用的程序。

## 7. `heartwall`

`heartwall` 是医学图像/视频中的心脏壁运动跟踪程序。应用背景是超声心动图分析，通过视频帧追踪心脏边界或特征点。

典型命令是：

```text
./heartwall ../../data/heartwall/test.avi 20
./heartwall ../../data/heartwall/test.avi 20 4
```

其中 `test.avi` 是输入视频，`20` 表示处理帧数，OpenMP 版本中的 `4` 表示线程数。OpenCL 版本的 run 文件中也有只传 `20` 的形式，说明部分路径可能内置了视频文件位置。

输入规模应该记录：视频帧数、帧宽、帧高、跟踪点数量、搜索窗口大小、线程数。输出通常是每帧跟踪到的心壁位置、轨迹或中间计算结果。

它同时具有图像处理、模板匹配、局部搜索和时间序列处理特征。运行时间受帧数、每帧像素数、候选搜索范围影响。AST/CFG 中循环和分支较多，IR 中 load/store 与算术指令都重要。

## 8. `hotspot`

`hotspot` 是二维芯片热传导仿真。应用背景是芯片热设计、电路温度分布预测和热管理。

典型命令是：

```text
./hotspot 512 2 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out
./hotspot 1024 1024 2 4 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.out
```

不同版本参数略有差异，但核心含义包括网格行列数、迭代步数、线程数、初始温度文件、功率输入文件和输出文件。

输入规模应该记录：网格宽度、高度、时间步/迭代次数、输入温度数组大小、功率数组大小、线程数或 GPU block 大小。输出通常是迭代后的温度分布。

它是典型 stencil 程序，每个网格点的新温度由邻近点和功率值共同决定。运行时间通常近似随 `grid_x * grid_y * iterations` 增长。它适合观察规则访存、cache 局部性、循环嵌套和内存带宽压力。

## 9. `hotspot3D`

`hotspot3D` 是三维芯片热仿真，是 `hotspot` 的三维扩展。应用背景仍然是芯片或三维集成电路的热扩散模拟。

典型命令是：

```text
./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out
```

这里可以理解为平面尺寸 512、层数 8、迭代次数 100。输入文件包含三维功率和温度数据。

输入规模应该记录：平面宽度/高度、z 方向层数、体素总数、迭代次数、线程数或 GPU 配置。输出通常是三维温度场。

相比二维 hotspot，它的工作集更大，邻域访问更多，cache 压力和内存带宽压力更明显。建模时不能只记录 `N`，还要记录层数和迭代次数。

## 10. `huffman`

`huffman` 是 Huffman 编码/解码相关 benchmark，应用背景是无损压缩、熵编码和数据压缩管线。该目录在本地 Rodinia 中主要出现在 CUDA 版本。

典型命令是：

```text
./pavle ../../data/huffman/test1024_H2.206587175259.in
```

输入文件名中的 `test1024` 暗示输入规模约为 1024 级别，`H2...` 可能和数据熵或生成参数有关。具体规模应以输入文件大小、符号数量和频率分布为准。

输入规模应该记录：输入字节数、符号种类数、频率分布、熵估计、编码块大小。输出通常是编码后的比特流、码表或压缩结果。

Huffman 的执行时间不只取决于输入大小，还取决于符号分布。频率统计、树构建、编码阶段的分支和位操作都很重要。它适合观察分支不规则、整数操作、内存访问和数据分布对时间的影响。

## 11. `hybridsort`

`hybridsort` 是混合排序 benchmark，应用背景是通用数据排序、GPU/并行排序算法评估。它通常结合不同排序策略，例如 bitonic sort、merge sort 或 radix-like 分阶段处理。

典型命令是：

```text
./hybridsort r
```

其中 `r` 通常表示随机输入模式。具体输入规模可能由程序内部常量或生成逻辑决定。

输入规模应该记录：元素数量、key 类型、输入分布类型、是否随机/有序/逆序、线程数或 GPU 配置。输出通常是排序后的数组或校验信息。

排序程序对输入分布敏感。随机、有序、重复 key 会影响分支、数据交换和访存模式。建模时应把输入分布作为重要字段，否则同样元素数量可能对应不同运行时间。

## 12. `kmeans`

`kmeans` 是 K-means 聚类 benchmark，应用背景是数据挖掘、机器学习、图像分割和向量量化。

典型命令是：

```text
./kmeans -o -i ../../data/kmeans/kdd_cup
./kmeans_openmp/kmeans -n 4 -i ../../data/kmeans/kdd_cup
```

README 中还支持参数：

```text
-m max_nclusters
-n min_nclusters
-t threshold
-l nloops
-b binary input
-r calculate RMSE
-o output cluster centers
```

输入规模应该记录：样本数量、维度数、cluster 数量、最大迭代次数、收敛阈值、输入格式、线程数。输出通常是聚类中心、样本所属类别或 RMSE。

K-means 的核心是反复计算样本到聚类中心的距离，计算量大致和 `points * dimensions * clusters * iterations` 相关。它既有浮点计算，也有大量数组扫描。收敛轮数会显著影响运行时间。

## 13. `lavaMD`

`lavaMD` 是分子动力学/粒子相互作用 benchmark。应用背景是分子模拟、粒子系统和邻域相互作用计算。

典型命令是：

```text
./lavaMD -boxes1d 10
./lavaMD -cores 4 -boxes1d 10
```

`boxes1d 10` 表示每个维度有 10 个 box，总 box 数约为 `10^3`。每个 box 中包含若干粒子，并与周围邻域 box 发生相互作用。

输入规模应该记录：`boxes1d`、总 box 数、每个 box 粒子数、邻域 box 数、粒子总数、线程数或 GPU block 配置。输出通常是每个粒子的力、势能或更新后的属性。

它是计算密集型代表之一。粒子间作用包含大量浮点乘加、距离计算和可能的指数/除法操作。邻域访问也会影响 cache。对模型来说，浮点指令、循环嵌套、数组访问和输入规模三次方增长都很重要。

## 14. `leukocyte`

`leukocyte` 是白细胞跟踪 benchmark，应用背景是医学图像分析和细胞运动追踪。它在视频中检测并跟踪白细胞位置。

典型命令是：

```text
./CUDA/leukocyte ../../data/leukocyte/testfile.avi 5
./OpenMP/leukocyte 5 4 ../../data/leukocyte/testfile.avi
./leukocyte ../../../data/leukocyte/testfile.avi 10
```

参数通常包括输入视频文件、处理帧数和线程数。不同并行版本参数顺序略有差异。

输入规模应该记录：帧数、帧宽、帧高、细胞/目标数量、搜索窗口大小、线程数。输出通常是目标位置、轨迹或检测结果。

它与 `heartwall` 类似，属于视频目标跟踪类程序，但目标对象和图像处理流程不同。运行时间受帧数、图像大小、搜索区域和检测算法复杂度影响。它适合分析图像处理中的规则扫描和局部不规则控制流。

## 15. `lud`

`lud` 是 LU decomposition，即矩阵 LU 分解。应用背景是线性方程组求解、数值线性代数和科学计算。

典型命令是：

```text
cuda/lud_cuda -s 256 -v
./omp/lud_omp -s 8000
./lud -s 1024 -v
```

也可以从矩阵文件读取：

```text
lud_cuda -i ../../data/lud/256.dat
```

输入规模应该记录：矩阵阶数 `N`、是否校验 `-v`、是否从文件读取、线程数或 GPU block 大小。输出通常是分解后的 L/U 矩阵或验证结果。

LU 分解通常是 `O(N^3)` 级别，对矩阵规模非常敏感。它有规则嵌套循环、浮点除法和乘减更新。与 PolyBench 的 `lu/ludcmp` 类似，但 Rodinia 版本包含并行实现和更多工程代码。

## 16. `mummergpu`

`mummergpu` 是 DNA 序列精确匹配/比对 benchmark，应用背景是生物信息学中的基因组序列比对。它来自 MUMmerGPU，用 GPU 加速 reference 和 query 序列之间的匹配。

典型命令是：

```text
bin/mummergpu ../../data/mummergpu/NC_003997.fna ../../data/mummergpu/NC_003997_q100bp.fna > NC_00399.out
bin/mummergpu -C ../../data/mummergpu/NC_003997.fna ../../data/mummergpu/NC_003997_q100bp.fna > NC_00399.out
```

其中 reference 是参考基因组，query 是查询序列集合。OpenMP run 文件说明 `-C` 可运行 CPU 版本。

输入规模应该记录：reference 长度、query 数量、query 平均长度、字符集大小、最小匹配长度、线程数或 GPU 配置。输出通常是匹配位置、匹配长度和序列对应关系。

它是字符串/序列处理程序，具有不规则访存、分支和数据依赖。运行时间不只取决于总字符数，也取决于序列重复度和匹配分布。

## 17. `myocyte`

`myocyte` 是心肌细胞模型仿真 benchmark，应用背景是生物医学中的细胞电生理和 ODE 系统模拟。

典型命令是：

```text
./myocyte.out 100 1 0
./myocyte.out 100 1 0 4
./myocyte.out -time 100
```

README 中说明参数含义包括：模拟时间间隔、仿真实例数量、并行化方法。OpenMP 版本还会增加线程数。

输入规模应该记录：模拟时间、实例数量、并行模式、时间步长、状态变量数量、线程数。输出通常是细胞状态变量随时间变化的结果。

它属于数值模拟程序，包含大量浮点计算、函数调用和状态更新。由于 ODE 模型内部公式复杂，函数调用和浮点操作对运行时间影响较大。

## 18. `nn`

`nn` 是 nearest neighbor 最近邻搜索 benchmark，应用背景包括地理位置检索、空间数据库、KNN 分类和相似数据查找。

典型命令是：

```text
./nn filelist_4 -r 5 -lat 30 -lng 90
./nn filelist_4 5 30 90
```

`filelist_4` 是数据文件列表，`-r 5` 表示返回 5 个最近邻，`lat/lng` 是查询点经纬度。

输入规模应该记录：记录数量、文件数量、返回邻居数 `r`、查询点数量、特征维度、线程数。输出通常是最近的若干记录及其距离。

它的核心是对大量记录计算距离并选择最小值，通常是内存扫描加少量浮点计算。规模主要由记录数和查询数决定。若数据布局连续，则访存较规则；若多文件或结构体布局复杂，则 cache 行为会更明显。

## 19. `nw`

`nw` 是 Needleman-Wunsch 序列比对，应用背景是生物信息学中的全局序列比对。它用动态规划计算两个序列的最佳匹配得分。

典型命令是：

```text
./needle 2048 10
./needle 2048 10 2
./nw 2048 10 ./nw.cl
```

这里 `2048` 通常表示序列长度或动态规划矩阵规模，`10` 是 penalty，OpenMP 版本的额外参数可能表示线程数。

输入规模应该记录：序列 A 长度、序列 B 长度、penalty、block size、线程数或 GPU 配置。输出通常是动态规划得分矩阵、最终比对得分或 traceback 结果。

NW 的计算量通常接近 `O(N^2)`，因为需要填充二维 DP 矩阵。它有明显的数据依赖，常按反对角线并行。建模时应关注二维规模、内存访问、循环结构和同步。

## 20. `particlefilter`

`particlefilter` 是粒子滤波 benchmark，应用背景是目标跟踪、机器人定位、状态估计和视频分析。

典型命令是：

```text
./particlefilter_naive -x 128 -y 128 -z 10 -np 1000
./particle_filter -x 128 -y 128 -z 10 -np 10000
./OCL_particlefilter_single -x 128 -y 128 -z 10 -np 400000
```

其中 `x/y` 是图像宽高，`z` 可理解为帧数或时间维度，`np` 是粒子数量。

输入规模应该记录：图像宽度、高度、帧数、粒子数量、重采样次数、线程数或 GPU 配置。输出通常是估计目标位置、粒子权重或滤波结果。

它的运行时间通常随粒子数和帧数增长。核心操作包括粒子状态更新、权重计算、归一化和重采样。它既有浮点计算，也有随机访问和并行归约/同步。

## 21. `pathfinder`

`pathfinder` 是网格路径动态规划 benchmark。应用背景类似在二维代价矩阵中寻找从上到下的最小代价路径，可类比图像 seam、路径规划或动态规划网格问题。

典型命令是：

```text
./pathfinder 100000 100 > out
./pathfinder 100000 100 20 > result.txt
```

通常可以理解为网格列数 100000、行数 100，CUDA/OpenCL 版本里的额外参数 `20` 可能用于 pyramid height 或并行分块参数。

输入规模应该记录：网格宽度、网格高度、时间步/行数、pyramid height、线程数或 GPU block 配置。输出通常是最终路径代价数组或最小路径结果。

它是动态规划型 stencil 程序，每一行依赖上一行相邻位置。访存较规则，但存在跨行依赖。运行时间大致随 `width * height` 增长，也受分块策略影响。

## 22. `srad`

`srad` 是 Speckle Reducing Anisotropic Diffusion，即斑点噪声抑制的各向异性扩散。应用背景是医学图像、雷达图像或超声图像去噪。

Rodinia 中有 `srad_v1` 和 `srad_v2`。典型命令是：

```text
./srad 100 0.5 502 458
./srad 100 0.5 502 458 4
./srad 2048 2048 0 127 0 127 0.5 2
./srad 2048 2048 0 127 0 127 2 0.5 2
```

`srad_v1` 参数常包含迭代次数、lambda、图像行列数；`srad_v2` 参数包含图像尺寸、感兴趣区域边界、lambda、迭代次数，OpenMP 版本还包含线程数。

输入规模应该记录：图像宽度、高度、ROI 范围、迭代次数、lambda、线程数或 GPU 配置。输出通常是去噪后的图像矩阵。

它是图像 stencil/扩散类程序。每轮迭代会访问像素的上下左右邻域，计算扩散系数并更新像素。运行时间通常随 `width * height * iterations` 增长，内存带宽和 cache 局部性非常重要。

## 23. `streamcluster`

`streamcluster` 是流式聚类 benchmark，应用背景是大规模数据流中的在线聚类、数据挖掘和近似中心选择。

典型命令是：

```text
./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1
./sc_omp 10 20 256 65536 65536 1000 none output.txt 4
./streamcluster 10 20 256 65536 65536 1000 none output.txt 1 -t gpu -d 0
```

参数通常可理解为：最小/最大中心数、点维度、总点数、chunk size、中心候选规模或迭代控制、输入文件、输出文件、线程数/设备参数。不同版本命令略有差异。

输入规模应该记录：点数量、维度数、最小中心数、最大中心数、chunk size、输入来源、线程数或 GPU 设备。输出通常是聚类中心及其权重。

它比普通 K-means 更接近真实数据流聚类，既有距离计算，也有中心选择、局部搜索和数据分块。运行时间受点数、维度、中心数量和 chunk size 共同影响。

## 24. Rodinia 建模时应特别记录的通用字段

Rodinia 不能像 PolyBenchC 那样只靠一个 `N` 或几个矩阵维度解释运行时间。建议至少按任务类型记录以下输入规模字段：

1. 图算法：节点数、边数、平均度、最大度、源点、frontier 统计。
2. 图像/视频：宽度、高度、帧数、通道数、ROI、搜索窗口。
3. 矩阵/线性代数：矩阵阶数、行列数、数据类型、是否校验。
4. 网格/stencil：grid_x、grid_y、grid_z、time_steps、stencil radius。
5. 聚类/机器学习：样本数、维度、cluster 数、迭代次数、收敛阈值。
6. 粒子/仿真：粒子数、box 数、时间步、邻域大小、实例数。
7. 序列/字符串：reference 长度、query 数量、query 长度、字符集、匹配参数。
8. 并行环境：parallel_model、线程数、GPU 型号、block size、work group size、是否使用 OpenMP/CUDA/OpenCL。

如果后续要把 Rodinia 加入当前 pipeline，不建议直接递归扫描所有 `.c/.cpp/.cu` 后混在一起建模。更合理的做法是为每个 benchmark 建一个 manifest，明确主程序、辅助源码、编译命令、运行命令、输入文件和规模字段。Rodinia 的价值正在于它更真实，但也正因为真实，必须把输入和并行配置记录清楚。
