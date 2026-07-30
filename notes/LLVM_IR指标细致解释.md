# LLVM IR 指标细致解释

本文解释当前流水线中 LLVM IR 指标的含义、它们在程序中对应什么、在 `.ll` 文件中通常怎样表现，以及理论上为什么可能影响程序执行时间。

LLVM IR 是 LLVM 的中间表示，比 C 源码更接近机器执行语义，但仍然保持跨平台。它使用 SSA 形式，大多数临时值只赋值一次。源码中的数组访问、循环、函数调用和算术表达式，会被降低为 `load`、`store`、`getelementptr`、`br`、`phi`、`fadd`、`fmul` 等 IR 指令。

例如源码：

```c
c[i] = a[i] + b[i];
```

在 LLVM IR 中可能出现：

```llvm
%idx = getelementptr inbounds float, ptr %a, i64 %i
%av = load float, ptr %idx
%bv = load float, ptr %bidx
%sum = fadd float %av, %bv
store float %sum, ptr %cidx
```

LLVM IR 指标比 AST 指标更适合估计底层资源需求，例如访存压力、算术强度、分支密度和函数调用开销。但当前指标仍是静态统计，不等于动态执行次数。循环内的一条 `load` 如果执行一百万次，静态 `load_count` 仍然只加 1。

## `ir_instruction_count`

`ir_instruction_count` 统计 `.ll` 文件中被视为 IR 指令的行数。当前脚本会排除空行、注释、标签行、`define`、`declare`、`attributes`、`source_filename`、`target` 等元信息。

例如这些会计入：

```llvm
%1 = load i32, ptr %i
%2 = add nsw i32 %1, 1
br label %loop
```

理论影响执行时间的方式：静态 IR 指令越多，说明编译后的中间操作越多，潜在执行成本越高。它比 `ast_node_count` 更接近执行层面。但它仍然没有乘上动态执行次数，因此需要结合循环深度、CFG 回边和输入规模。一个三重循环体里 10 条 IR 指令可能比循环外 1000 条初始化 IR 指令更重要。

## `ir_line_count`

`ir_line_count` 统计 `.ll` 文件总行数，包括元信息、类型声明、函数声明、属性和指令等。

理论影响执行时间的方式：它是非常粗的 IR 文件规模代理。行数越多通常表示程序更大、函数更多或调试/属性信息更多，但它包含很多非执行内容，所以解释性弱于 `ir_instruction_count`。

## `load_count`

`load_count` 统计 `load` 指令数量。

`load` 表示从内存读数据到寄存器/SSA 临时值中：

```llvm
%x = load double, ptr %array_element
```

源码中的变量读取、数组读取、指针解引用都可能产生 `load`。

理论影响执行时间的方式：`load` 是内存读压力的核心指标。读内存如果命中寄存器或 L1 cache 很快；如果产生 L2/L3/DRAM 访问就会显著变慢。大量 `load` 说明程序可能受内存带宽、cache miss 或访存延迟影响。对矩阵、stencil、图算法等程序，这个字段非常重要。

## `store_count`

`store_count` 统计 `store` 指令数量。

`store` 表示把值写回内存：

```llvm
store double %sum, ptr %c_element
```

源码中的赋值到数组、写指针、写局部变量地址都可能产生 `store`。

理论影响执行时间的方式：`store` 表示内存写压力。写操作可能引起 cache line 写分配、写回、写合并和一致性成本。对大数组初始化、矩阵更新、stencil 写回，`store_count` 可以帮助判断写带宽需求。

## `memory_inst_count`

`memory_inst_count` 是当前脚本对内存相关 IR 操作的合计，包含：

```text
load
store
alloca
getelementptr
llvm.memcpy
llvm.memset
```

它不是严格的“真实内存访问次数”，因为 `getelementptr` 只是地址计算，`alloca` 是栈对象分配，未必直接访问数据内存。

理论影响执行时间的方式：它用于粗略判断程序偏内存密集还是偏计算密集。值高说明程序有大量读写、地址计算、栈对象或内存批量操作。它需要和 `memory_arithmetic_ratio` 一起看。

## `getelementptr_count`

`getelementptr_count` 统计 `getelementptr` 指令数量，简称 GEP。

GEP 用来计算数组、结构体或指针偏移地址。例如：

```llvm
%arrayidx = getelementptr inbounds double, ptr %A, i64 %idx
```

源码中的 `A[i]`、`A[i][j]`、`struct.field`、指针偏移都可能产生 GEP。

理论影响执行时间的方式：GEP 多说明地址计算复杂、数组/结构体访问密集。它本身通常是整数地址计算，不一定很慢，但它常常伴随 `load/store`，说明程序有大量访存。多维数组和复杂索引会增加 GEP 数量，也可能影响编译器向量化和内存访问模式分析。

## `alloca_count`

`alloca_count` 统计 `alloca` 指令数量。

`alloca` 表示在栈帧中分配对象：

```llvm
%i = alloca i32
%array = alloca [100 x double]
```

理论影响执行时间的方式：少量标量 `alloca` 在优化后常会被提升为寄存器，不一定有运行时成本。大量或大对象 `alloca` 可能增加栈空间、栈访问和初始化开销。在 `-O0` IR 中，`alloca` 往往很多；在 `-O2` 后会减少。因此这个指标强烈依赖优化等级。

## `memcpy_count`

`memcpy_count` 统计 `llvm.memcpy` intrinsic。

IR 示例：

```llvm
call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
```

理论影响执行时间的方式：`memcpy` 表示批量内存复制。复制量大时，运行时间可能直接受内存带宽限制。静态 `memcpy_count` 只表示出现了几处复制，真正成本还要看复制字节数参数。

## `memset_count`

`memset_count` 统计 `llvm.memset` intrinsic。

IR 示例：

```llvm
call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 %n, i1 false)
```

理论影响执行时间的方式：`memset` 常用于数组清零或初始化。对大数组，初始化成本可能很高，甚至主导短 kernel 的运行时间。和 `memcpy` 一样，静态次数需要结合字节数。

## `call_count_ir`

`call_count_ir` 统计 `call` 和 `invoke` 指令数量。

IR 示例：

```llvm
call void @foo(i32 %x)
%p = call ptr @malloc(i64 %n)
```

理论影响执行时间的方式：调用可能带来函数调用开销，也可能隐藏大量外部工作。普通小函数如果被内联，优化后 `call` 会消失；库函数、系统调用、内存分配、I/O、GPU runtime 调用则可能非常昂贵。该指标应结合 `external_call_count` 使用。

## `external_call_count`

`external_call_count` 是当前脚本对外部函数调用的近似统计。它统计调用中出现 `@function` 的行，并排除 `@llvm.` intrinsic。

理论影响执行时间的方式：外部调用更难从当前 IR 内部看出成本。例如 `printf`、`malloc`、`fopen`、BLAS、CUDA runtime 都可能使运行时间主要由库实现决定。外部调用多的程序，仅靠当前文件的静态 IR 指令预测时间会比较困难。

## `branch_count`

`branch_count` 统计 `br` 指令数量。

无条件跳转：

```llvm
br label %loop
```

条件跳转：

```llvm
br i1 %cmp, label %then, label %else
```

理论影响执行时间的方式：`br` 是控制流跳转。循环、if、goto 都会产生分支。分支多说明控制流频繁变化，可能增加 branch prediction 成本。循环中的 `br` 会被大量动态执行，是运行时间的重要结构信号。

## `switch_count_ir`

`switch_count_ir` 统计 LLVM IR 的 `switch` 指令。

IR 示例：

```llvm
switch i32 %op, label %default [
  i32 0, label %case0
  i32 1, label %case1
]
```

理论影响执行时间的方式：`switch` 表示多路跳转。它可能被后端降低成跳转表或比较链。执行成本取决于 case 数量、输入分布、跳转表局部性和分支预测行为。

## `phi_count`

`phi_count` 统计 `phi` 指令数量。

SSA 中不同路径汇合时，需要用 `phi` 选择变量值：

```llvm
%i = phi i32 [ 0, %entry ], [ %inc, %loop ]
```

循环归纳变量和 if/else 合流变量经常产生 `phi`。

理论影响执行时间的方式：`phi` 本身通常不是最终机器上的普通指令，但它表示数据流合并、循环变量和控制依赖。`phi_count` 高通常说明循环和分支结构复杂。它对解释“为什么 CFG 分支多、循环多”很有帮助。

## `select_count`

`select_count` 统计 `select` 指令。

IR 示例：

```llvm
%x = select i1 %cond, i32 %a, i32 %b
```

`select` 可以理解为无显式跳转的条件选择。

理论影响执行时间的方式：`select` 可能减少分支跳转，从而降低 branch miss，但它会让两个候选值相关的数据依赖进入同一路径。对简单条件赋值，它通常比分支更稳定；对复杂表达式，是否更快取决于后端优化和硬件。

## `icmp_count`

`icmp_count` 统计整数比较指令。

IR 示例：

```llvm
%cmp = icmp slt i32 %i, %n
```

循环条件、数组边界、if 判断都会产生 `icmp`。

理论影响执行时间的方式：整数比较通常较便宜，但它是循环控制和分支判断的基础。`icmp_count` 高说明程序条件判断多或循环控制多。它经常和 `branch_count` 配合解释控制流成本。

## `fcmp_count`

`fcmp_count` 统计浮点比较指令。

IR 示例：

```llvm
%cmp = fcmp olt double %x, %eps
```

理论影响执行时间的方式：浮点比较常用于数值算法的阈值判断、收敛条件或分类逻辑。它可能受 NaN、有序/无序比较语义影响，通常比简单整数比较更复杂一些。对迭代收敛类程序，`fcmp` 可能提示数据依赖的退出条件。

## `add_count` 和 `sub_count`

`add_count`、`sub_count` 统计整数加法和减法。

IR 示例：

```llvm
%inc = add nsw i32 %i, 1
%diff = sub nsw i32 %a, %b
```

理论影响执行时间的方式：整数加减通常很快，但在循环索引、地址计算和整数算法中出现频繁。它们单次成本低，动态次数大时仍然重要。需要注意，当前 `add_count` 不包含浮点 `fadd`。

## `mul_count`

`mul_count` 统计整数乘法。

IR 示例：

```llvm
%p = mul nsw i32 %a, %b
```

理论影响执行时间的方式：整数乘法通常比加减更重，但现代 CPU 上也相对高效。它常出现在索引计算、哈希、整数数值算法中。对于数组线性化下标，多维索引可能带来乘法或等价优化。

## `div_count` 和 `rem_count`

`div_count` 统计 `sdiv` 和 `udiv`，`rem_count` 统计 `srem` 和 `urem`。

IR 示例：

```llvm
%q = sdiv i32 %a, %b
%r = srem i32 %a, %b
```

理论影响执行时间的方式：整数除法和取余通常是高延迟指令，明显慢于加减乘。如果它们位于热循环中，会显著影响执行时间。编译器有时能把除以常数优化为乘法和移位，但除数是变量时通常成本较高。

## `fadd_count` 和 `fsub_count`

`fadd_count`、`fsub_count` 统计浮点加法和减法。

IR 示例：

```llvm
%s = fadd double %a, %b
%d = fsub double %a, %b
```

理论影响执行时间的方式：浮点加减是数值程序的核心操作之一。对矩阵、stencil、滤波、仿真程序，浮点加减动态次数常和运行时间高度相关。但如果程序受内存带宽限制，浮点操作多不一定代表更慢。

## `fmul_count`

`fmul_count` 统计浮点乘法。

IR 示例：

```llvm
%p = fmul double %a, %b
```

理论影响执行时间的方式：矩阵乘、卷积、滤波、物理模拟中浮点乘法非常关键。浮点乘法多通常表示计算密集度高。现代 CPU/GPU 对乘加可高度流水化，若编译器生成 FMA，IR 中的 `fmul` 和 `fadd` 可能在后端合并，因此静态计数只是近似。

## `fdiv_count`

`fdiv_count` 统计浮点除法。

IR 示例：

```llvm
%q = fdiv double %a, %b
```

理论影响执行时间的方式：浮点除法通常比 `fadd`、`fmul` 高延迟得多。数值归一化、求解器、滤波参数计算中如果在热循环内大量出现 `fdiv`，可能明显拖慢程序。

## `vector_inst_count`

`vector_inst_count` 当前通过匹配 IR 中 `<N x type>` 形式来估计向量操作数量。

IR 示例：

```llvm
%v = load <4 x float>, ptr %p
%r = fadd <4 x float> %a, %b
```

理论影响执行时间的方式：向量操作表示 SIMD 程度。更多向量指令可能说明编译器成功向量化，同样数据量下运行时间可能降低。但静态向量指令多也可能意味着向量化后的宽操作变多，不应简单理解为更慢。它应和输入规模、IR 总指令数和运行时间一起分析。

## `atomic_count`

`atomic_count` 统计 `atomicrmw` 和 `cmpxchg`。

IR 示例：

```llvm
atomicrmw add ptr %p, i32 1 monotonic
%old = cmpxchg ptr %p, i32 %expected, i32 %new monotonic monotonic
```

理论影响执行时间的方式：原子操作常见于并行程序中的共享计数、锁、队列和归约。它们可能导致缓存一致性流量、序列化和线程竞争。对多线程或 GPU 程序，少量原子热点就可能严重影响可扩展性。

## `barrier_or_sync_count`

`barrier_or_sync_count` 当前通过文本匹配同步相关名称来估计，例如 `__syncthreads`、`barrier`、`omp_`、`cudaDeviceSynchronize`、`clFinish`。

理论影响执行时间的方式：同步会让快的线程等待慢的线程，也可能强制内存可见性或设备/主机同步。同步越多，并行效率越可能下降。这个字段对 CUDA/OpenMP/OpenCL 程序比对普通 C 程序更重要。

## `constant_count_ir`

`constant_count_ir` 统计 IR 文本中的整数常量数量，过滤掉绝对值特别大的数。

IR 示例：

```llvm
%inc = add i32 %i, 1
%cmp = icmp slt i32 %i, 1000
```

理论影响执行时间的方式：常量可能表示循环边界、数组偏移、步长、对齐、状态码等。它本身不是耗时操作，但可以辅助推断固定规模和控制逻辑。和 AST 常量一样，它不能替代正式输入规模解析。

## `max_constant_ir`

`max_constant_ir` 是 IR 中最大整数常量。

理论影响执行时间的方式：如果该常量对应循环上界、数组大小或内存操作字节数，它可能强烈影响运行时间。但如果它来自类型大小、对齐、调试信息或宏展开，就可能没有实际意义。因此应把它作为辅助特征。

## `avg_constant_ir`

`avg_constant_ir` 是 IR 中整数常量平均值。

理论影响执行时间的方式：它可以提供程序常量规模的粗略信号，但解释性较弱。大量 `0`、`1`、`2` 会拉低均值，少数大常量会抬高均值。建模时它不应被过度解释。

## `load_store_ratio`

`load_store_ratio` 是：

```text
load_count / store_count
```

如果 `store_count` 为 0，当前脚本会避免除零并返回近似值。

理论影响执行时间的方式：该指标用于区分读主导和写主导程序。读多写少可能是扫描、查找、矩阵读取型 kernel；写多可能是初始化、填充、转置或生成型 kernel。不同读写比例会影响 cache、内存带宽和写回行为。

## `memory_arithmetic_ratio`

`memory_arithmetic_ratio` 是：

```text
memory_inst_count / arithmetic_instruction_count
```

其中 arithmetic 包括整数和浮点加减乘除、取余等。

理论影响执行时间的方式：这是判断内存密集型和计算密集型的重要静态代理。比值高说明访存/地址计算相对算术更多，程序可能受内存带宽或 cache 影响；比值低说明算术操作相对更多，程序可能更计算密集。但最终还要看动态执行次数、访存局部性、向量化和硬件。

## `branch_instruction_ratio`

`branch_instruction_ratio` 是：

```text
control_instruction_count / ir_instruction_count
```

当前 control 包括 `br`、`switch`、`indirectbr`、`invoke`、`ret`。

理论影响执行时间的方式：该指标表示控制流指令在总 IR 指令中的占比。占比高说明程序分支、跳转或函数返回相对密集，可能受分支预测、路径不确定性和控制依赖影响。规则数值 kernel 通常该比例较低；搜索、解析、图遍历、状态机程序可能较高。

## 使用 LLVM IR 指标时的注意点

LLVM IR 指标的优势是更接近底层执行：`load/store` 对应访存，`fadd/fmul/fdiv` 对应浮点计算，`br/phi` 对应控制流和循环结构。

但它仍然有三个重要限制。

第一，它是静态计数，不知道循环执行多少次。必须结合输入规模和循环结构。

第二，它受优化等级影响很大。`-O0` 会保留大量 `alloca/load/store`，`-O2` 会做 mem2reg、内联、循环优化和向量化，指标值会明显变化。因此实验中必须记录 `opt_level`。

第三，IR 还不是最终机器码。后端可能把 IR 指令合并、展开、向量化或替换成目标平台指令。尤其是 FMA、SIMD、内存寻址模式和库调用，IR 静态指标只能作为运行时间预测的中间代理。
