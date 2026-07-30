# AST 指标细致解释

本文解释当前流水线中 AST 指标的含义、它们在程序源码中对应什么、在 Clang AST 文本中通常表现为什么形式，以及理论上为什么可能影响程序执行时间。

AST 是 Abstract Syntax Tree，即抽象语法树。它把源代码按照语法结构拆成树状节点。程序中的函数声明、变量声明、循环语句、if 语句、函数调用、数组访问、二元运算等，都可以成为 AST 上的一个节点。

例如源码：

```c
for (i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
}
```

在 AST 中通常会出现 `ForStmt`、`BinaryOperator`、`ArraySubscriptExpr`、`DeclRefExpr`、`IntegerLiteral` 等节点。当前项目中的 AST 文件来自 Clang 的 AST dump，节点通常以类似下面的文本出现：

```text
|-ForStmt
| |-BinaryOperator '<'
| |-UnaryOperator '++'
| `-CompoundStmt
|   `-BinaryOperator '='
|     |-ArraySubscriptExpr
|     `-BinaryOperator '+'
```

当前脚本会尽量只统计属于当前源文件的 AST 内容，避免把系统头文件、库声明等大量无关节点混进来。但由于 Clang AST dump 本身包含宏展开和部分隐式声明，AST 指标仍然更适合作为静态结构特征，而不是动态执行次数。

## `ast_node_count`

`ast_node_count` 表示 AST 文本中被计入的节点行数。这里的 node 不是图论意义上的运行时节点，而是源码语法结构中的一个语法节点。

程序里的什么可以看成 node：一个函数声明是 node，一个 `for` 循环是 node，一个 `if` 语句是 node，一个 `a[i]` 数组访问是 node，一个 `x + y` 二元表达式也是 node。Clang AST dump 中通常每一行对应一个语法节点，例如 `FunctionDecl`、`ForStmt`、`BinaryOperator`、`ArraySubscriptExpr`。

它的意义是衡量源码静态复杂度。`ast_node_count` 越大，通常说明程序声明、语句、表达式越多，静态代码规模越大。但它不是执行时间的直接等价物：一个只有十几行的三重循环可能比几百行只执行一次的初始化代码更慢。因此它需要和循环深度、数组访问、IR 指令以及输入规模一起看。

理论影响执行时间的方式：节点多往往意味着更多潜在操作、更多控制结构和更大的编译后指令体积，可能增加执行路径、指令 cache 压力和函数调用机会。但如果这些节点位于冷路径或只执行一次，它们对运行时间影响很小。

## `function_count`

`function_count` 统计 AST 中的 `FunctionDecl` 数量，即函数声明或函数定义数量。

源码中的函数，例如：

```c
void kernel_gemm(...) { ... }
int main(void) { ... }
```

在 AST 中通常表现为：

```text
FunctionDecl ... kernel_gemm 'void (...)'
FunctionDecl ... main 'int (void)'
```

理论影响执行时间的方式：函数越多，程序结构越模块化，可能带来更多函数调用、参数传递、栈帧维护和跨函数优化问题。不过函数数量本身不等于慢。如果函数被内联，调用开销会消失；如果很多函数只是工具函数且不在热路径，影响也很小。它更适合作为程序结构复杂度和潜在调用关系复杂度的特征。

## `call_expr_count`

`call_expr_count` 统计 `CallExpr`，即函数调用表达式数量。

源码中的：

```c
foo(x);
printf("%d\n", value);
malloc(n * sizeof(int));
```

在 AST 中通常表现为：

```text
CallExpr
  `-DeclRefExpr ... 'foo'
```

理论影响执行时间的方式：函数调用可能带来直接调用开销，也可能隐藏大量实际工作。比如 `printf`、`malloc`、`memcpy`、BLAS 函数、CUDA runtime 调用都可能远比普通算术表达式昂贵。对 PolyBench 这类 kernel 来说，热循环内部的调用尤其重要；循环外的初始化调用影响相对较小。

需要注意：当前指标只统计静态出现次数，不知道调用发生多少次。一个位于三重循环内部的 `CallExpr` 和一个只在 `main` 里执行一次的 `CallExpr`，在这个指标里都只加 1。

## `recursive_call_flag`

`recursive_call_flag` 表示当前脚本是否检测到函数调用自身的情况。值通常是 0 或 1。

源码示例：

```c
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
```

AST 中会有 `FunctionDecl fib`，函数体内部还会有指向 `fib` 的 `CallExpr`。当前脚本主要通过源码文本和函数名做近似检测。

理论影响执行时间的方式：递归会引入栈帧开销，且执行次数往往由输入值决定。递归深度和递归分支数可能导致线性、对数甚至指数级运行时间。这个字段只是提示“存在递归风险”，不能单独表示递归复杂度。

## `for_count`

`for_count` 统计 AST 中的 `ForStmt` 数量。

源码中的：

```c
for (i = 0; i < n; i++) {
    sum += a[i];
}
```

在 AST 中表现为：

```text
ForStmt
```

理论影响执行时间的方式：`for` 循环通常是数值程序的主要耗时来源。尤其在 PolyBench 中，矩阵乘、stencil、分解算法的主要计算都在嵌套 `for` 中。`for_count` 越多，说明程序中存在更多静态循环区域，但真正的时间成本还要看每个循环的边界、嵌套深度和循环体内容。

## `while_count`

`while_count` 统计 AST 中的 `WhileStmt` 数量。

源码示例：

```c
while (x > 0) {
    x = update(x);
}
```

理论影响执行时间的方式：`while` 循环常常比 `for` 更依赖数据状态，迭代次数可能不容易从源码常量直接看出。因此它不仅代表重复执行，还可能代表运行时间不确定性。搜索、收敛迭代、链表遍历、队列处理等程序常见这种结构。

## `do_while_count`

`do_while_count` 统计 AST 中的 `DoStmt` 数量。

源码示例：

```c
do {
    x = step(x);
} while (x > eps);
```

理论影响执行时间的方式：它和 `while` 类似，但循环体至少执行一次。它常见于迭代求解、输入处理、状态推进等场景。运行时间取决于退出条件和输入数据。

## `max_loop_depth`

`max_loop_depth` 表示 AST 中循环语句的最大嵌套深度。当前脚本通过 AST 文本缩进和 `ForStmt`、`WhileStmt`、`DoStmt` 的相对层级做近似计算。

例如：

```c
for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
            c[i][j] += a[i][k] * b[k][j];
        }
    }
}
```

最大循环深度是 3。

理论影响执行时间的方式：循环深度是时间复杂度的重要静态信号。单层循环常对应 `O(N)`，双层循环常对应 `O(N^2)`，三层循环常对应 `O(N^3)`。但这只是粗略推断，因为循环边界可能不是同一个变量，也可能有三角形边界、步长大于 1、提前退出或数据依赖条件。

## `loop_count_total`

`loop_count_total` 是 `for_count + while_count + do_while_count`。

它表示程序中循环语法结构的总数量。它可以反映程序重复计算区域的密度，但不能区分循环是否嵌套，也不能区分循环体是否重。因此它通常应和 `max_loop_depth`、`array_subscript_count`、IR 中的 `branch_count`、`phi_count` 一起使用。

理论影响执行时间的方式：循环越多，潜在热区越多，运行时间越可能由重复执行主导。但如果多个循环是顺序排列，复杂度可能是 `O(N)+O(N)`；如果它们是嵌套关系，则可能变成 `O(N^2)` 或 `O(N^3)`。

## `if_count`

`if_count` 统计 AST 中的 `IfStmt` 数量。

源码示例：

```c
if (a[i] > threshold) {
    count++;
}
```

AST 中表现为：

```text
IfStmt
```

理论影响执行时间的方式：`if` 表示控制流分裂。分支越多，程序路径越多，CPU 分支预测失败的风险越高。如果分支条件依赖随机数据或输入数据，运行时间方差可能变大。对于 GPU 程序，分支还可能造成 warp divergence，使同一个 warp 内不同线程走不同路径。

## `switch_count`

`switch_count` 统计 AST 中的 `SwitchStmt` 数量。

源码示例：

```c
switch (op) {
case 0: ...
case 1: ...
}
```

理论影响执行时间的方式：`switch` 是多路分支，可能被编译为跳转表，也可能被编译成比较链。分支数量和输入分布会影响跳转预测、代码局部性和路径长度。对解释器、状态机、分类处理程序尤其重要。

## `conditional_operator_count`

`conditional_operator_count` 统计三目运算符 `?:`，在 AST 中通常是 `ConditionalOperator`。

源码示例：

```c
x = flag ? a : b;
```

理论影响执行时间的方式：三目运算可能被编译为分支，也可能被编译为 LLVM IR 的 `select`。如果编译为 `select`，它可以减少显式跳转，但会形成数据依赖；如果编译为分支，则影响控制流复杂度。它对执行时间的影响通常小于循环和大规模访存，但能辅助解释分支密度。

## `binary_operator_count`

`binary_operator_count` 统计 `BinaryOperator` 节点，即二元运算表达式。

源码中的 `a + b`、`i < n`、`x = y`、`p && q` 等都可能对应二元运算节点。

AST 中表现为：

```text
BinaryOperator ... '+'
BinaryOperator ... '<'
BinaryOperator ... '='
```

理论影响执行时间的方式：二元运算多说明表达式计算多，可能增加整数/浮点运算、比较、赋值和逻辑操作。但 AST 的 `BinaryOperator` 比较粗，赋值、比较和算术都在其中，所以需要进一步看 `arithmetic_operator_count` 和 `comparison_operator_count`。

## `arithmetic_operator_count`

`arithmetic_operator_count` 统计算术类二元运算，如 `+`、`-`、`*`、`/`、`%` 以及部分复合赋值。

源码示例：

```c
c[i][j] += alpha * a[i][k] * b[k][j];
```

理论影响执行时间的方式：算术运算是 CPU/FPU 计算负载的直接来源。加减通常较便宜，乘法略重，除法和取余通常更慢。对矩阵乘、卷积、stencil、数值模拟程序来说，算术运算数量和循环动态次数相乘后，常常决定计算密集型部分的执行时间。

## `comparison_operator_count`

`comparison_operator_count` 统计 `<`、`>`、`==`、`!=`、`<=`、`>=` 等比较运算。

源码中的循环条件和分支条件都经常产生比较：

```c
i < n
a[i] > threshold
```

理论影响执行时间的方式：比较本身通常不贵，但它常常服务于分支、循环退出条件或选择逻辑。比较越多，控制判断越密集，可能对应更多 `icmp/fcmp`、`br` 或 `select`。如果比较位于热循环中，它们会被大量动态执行。

## `array_subscript_count`

`array_subscript_count` 统计 `ArraySubscriptExpr`，即数组下标访问。

源码示例：

```c
a[i]
b[i][j]
c[i][j][k]
```

在 AST 中通常表现为：

```text
ArraySubscriptExpr
```

多维数组通常会出现嵌套的 `ArraySubscriptExpr`。例如 `a[i][j]` 可能对应两层数组下标节点。

理论影响执行时间的方式：数组访问是内存读写的高层语法信号。数组访问越多，通常意味着更多 load/store、地址计算和 cache 行访问。顺序访问可能被 cache 和预取器很好地处理；跨步访问、间接访问、不连续访问则可能造成 cache miss。这个指标应和 LLVM IR 的 `load_count`、`store_count`、`getelementptr_count` 一起看。

## `pointer_deref_count`

`pointer_deref_count` 统计指针解引用形式的 `UnaryOperator '*'`。

源码示例：

```c
value = *p;
*dst = value;
```

理论影响执行时间的方式：指针解引用意味着间接内存访问。它可能让编译器更难判断别名关系，从而影响向量化、循环优化和寄存器分配。不规则指针访问还可能导致 cache miss。对链表、图、树、稀疏结构程序，这个指标通常比数组下标更能提示不规则访存风险。

## `malloc_free_count`

`malloc_free_count` 统计 `malloc`、`calloc`、`realloc`、`free` 等动态内存管理调用。

源码示例：

```c
int *a = malloc(n * sizeof(int));
free(a);
```

理论影响执行时间的方式：动态内存分配会调用运行时 allocator，可能产生锁、元数据维护、页分配和 cache/TLB 影响。一次大分配可能不明显，但循环内反复分配会非常昂贵。它还影响数据布局，从而间接影响后续访存局部性。

## `integer_literal_count`

`integer_literal_count` 统计 AST 中整数常量节点 `IntegerLiteral` 的数量。

源码示例：

```c
for (i = 0; i < 100; i++)
```

其中 `0`、`100` 都可能是整数常量。

理论影响执行时间的方式：整数常量本身不耗时，但它们可能暗示固定循环边界、数组大小、步长、分支阈值。对 benchmark 程序，较大的整数字面量有时对应问题规模。但对 PolyBench 这类通过头文件宏控制规模的程序，仅靠 `.c` 文件中的整数常量不可靠。

## `max_integer_literal`

`max_integer_literal` 是 AST 中出现的最大整数常量。

理论影响执行时间的方式：如果最大整数刚好是循环上界或数组规模，它可能和运行时间强相关。例如固定循环 `for (i = 0; i < 1000000; i++)` 中的 `1000000`。但如果最大整数只是格式化常量、随机种子、魔数或宏展开副产物，它就可能误导模型。因此它应作为辅助规模线索，而不是正式输入规模字段。

## `avg_integer_literal`

`avg_integer_literal` 是整数常量的平均值。

理论影响执行时间的方式：它可以粗略区分以小常量控制逻辑的程序和含有较大固定规模常量的程序。但平均值容易被少数大常量影响，也容易被大量 `0`、`1`、`2` 稀释。建模时它的解释性弱于显式输入规模字段。

## `float_literal_count`

`float_literal_count` 统计 `FloatingLiteral`，即浮点字面量数量。

源码示例：

```c
x = 0.25 * a[i] + 0.5 * b[i];
```

理论影响执行时间的方式：浮点常量多说明程序可能包含数值公式、系数、滤波器权重或物理模型参数。它不直接表示浮点运算次数，但能提示程序属于数值计算类型。真正的浮点计算强度更应看 LLVM IR 中的 `fadd_count`、`fmul_count`、`fdiv_count`。

## `pragma_omp_count`

`pragma_omp_count` 统计源码中的 OpenMP pragma，例如：

```c
#pragma omp parallel for
```

理论影响执行时间的方式：OpenMP 指令表示程序可能并行执行。它会影响线程创建、调度、同步、负载均衡和内存带宽竞争。对运行时间模型来说，看到 OpenMP 后必须同时记录线程数、调度策略、chunk size 等环境和输入字段，否则同一静态程序在不同线程数下运行时间会完全不同。

## `cuda_kernel_decl_count`

`cuda_kernel_decl_count` 统计 CUDA kernel 声明中的 `__global__`。

源码示例：

```c
__global__ void kernel(float *a) { ... }
```

理论影响执行时间的方式：CUDA kernel 数量关系到 GPU kernel launch 次数、设备端计算划分和同步边界。每个 kernel launch 都有固定开销，kernel 内部的线程块配置、访存合并、共享内存、寄存器占用又会影响实际执行时间。这个字段只是 CUDA 结构入口，后续还需要采集 grid/block 规模等 GPU 专用指标。

## 使用 AST 指标时的注意点

AST 指标最适合解释“源码写成了什么样”：有多少循环、分支、数组访问、函数调用和表达式。它对算法结构很直观，但它不是动态执行画像。

比如 `for_count = 1` 的程序可能只循环 10 次，也可能循环 10 亿次；`array_subscript_count = 1` 的数组访问如果在三重循环里，动态访存量仍然巨大。因此 AST 指标必须和输入规模、循环边界、CFG 回边、LLVM IR load/store 以及运行轮次数据结合起来使用。
