## CUDA 是什么？​​

​​CUDA​​（Compute Unified Device Architecture，统一计算设备架构）是 ​​NVIDIA 公司推出的并行计算平台和编程模型​​，允许开发者使用 ​​GPU（图形处理器）​​ 进行通用计算（GPGPU）。它通过扩展 C/C++ 语言，让开发者能够利用 GPU 的大规模并行计算能力加速计算密集型任务。

## ​异构计算（Heterogeneous Computing）详解​​

异构计算是指在一个系统中使用 ​​不同类型的处理器​​（如 CPU + GPU、FPGA、AI 加速器等）协同工作，以提高计算效率。CUDA 就是一种典型的异构计算技术，它利用 ​​CPU（主机）​​ 和 ​​GPU（设备）​​ 的并行计算能力加速任务。

## CUDA开发环节
从宏观上我们可以从以下几个环节完成CUDA应用开发：
1、领域层
2、逻辑层
3、硬件层

分析数据和函数 -> 如何进行并发 -> 理解线程如何映射到机器


## CUDA编程结构

主机：CPU及其内存
设备：GPU及其内存

## 内存管理

```cuda
cudaMemcpyHostToHost
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
```
意思即为英文翻译。

分配设备端的内存空间，为了区分设备和主机端内存，我们可以给变量加后缀或者前缀h_表示host，d_表示device.

## 线程管理
一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程
__不同块内线程不能相互影响！他们是物理隔离的！__

==依靠下面两个内置结构体确定线程标号：==

- blockIdx（线程块在线程网格内的位置索引）
- threadIdx（线程在线程块内的位置索引）

这两个内置结构体基于 uint3 定义，包含三个无符号整数的结构:

- blockIdx.x
- blockIdx.y
- blockIdx.z
- threadIdx.x
- threadIdx.y
- threadIdx.z

同样对应的两个结构体来保存其范围，也就是blockIdx中三个字段的范围threadIdx中三个字段的范围：

- blockDim
- gridDim

网格和块的维度存在几个限制因素，块大小主要与可利用的计算资源有关，如寄存器共享内存。
分成网格和块的方式可以使得我们的CUDA程序可以在任意的设备上执行。

## 核函数概述

**写CUDA程序就是写核函数**，第一步我们要确保核函数能正确的运行产生正切的结果，第二优化CUDA程序的部分，无论是优化算法，还是调整内存结构，线程结构都是要调整核函数内的代码，来完成这些优化的。

### 启动核函数
启动核函数，通过的以下的ANSI C 扩展出的CUDA C指令:
```cpp
kernel_name<<<grid,block>>>(argument list);
```

三个尖括号’<<<grid,block>>>’内是对设备代码执行的线程结构的配置（或者简称为对内核进行配置），也就是的线程结构中的网格，块。

通过指定grid和block的维度，我们可以配置：

- 内核中线程的数目
- 内核中使用的线程布局

想要主机等待设备端执行可以用下面这个指令：
```cpp
cudaError_t cudaDeviceSynchronize(void);
```

当核函数启动后的下一条指令就是从设备复制数据回主机端，那么主机端必须要等待设备端计算完成。

**所有CUDA核函数的启动都是异步的(不等待，继续干别的，等好了再回来处理)，这点与C语言是完全不同的**

### 编写核函数

限定符`__global__` `__device__` `__host__` 

有些函数可以同时定义为 device 和 host ，这种函数可以同时被设备和主机端的代码调用，主机端代码调用函数很正常，设备端调用函数与C语言一致，但是要声明成设备端代码，告诉nvcc编译成设备机器码，同时声明主机端设备端函数，那么就要告诉编译器，生成两份不同设备的机器码。

Kernel核函数编写有以下限制

- 只能访问设备内存
- 必须有void返回类型
- 不支持可变数量的参数
- 不支持静态变量
- 显示异步行为

### 验证核函数

可以使用`glog`中的check对其中的代码处理。
比如：
```cpp
CHECK(cudaMalloc((float**)&a_d,nByte));
CHECK(cudaMalloc((float**)&b_d,nByte));
CHECK(cudaMalloc((float**)&res_d,nByte));
```

## 错误处理

也是使用`glog`中的的`LOG(ERROR)`进行输出问题，防御性编程。

## 给核函数计时

想获得最高的效率，需要反复的优化，以及对硬件和编程细节的详细了解，怎么评估效率，时间是个很直观的测量方式

### 用CPU计时

使用gettimeofday() (Linux下的库函数)函数:

```cpp
#include <sys/time.h>
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}
```

需要注意的是当数据不能被完整切块的时候性能滑铁卢了，这个我们可以使用一点小技巧，比如只传输可完整切割数据块，然后剩下的1，2个使用cpu计算。更灵活一点，最大限度的发挥``GPU``的性能。

### 用nvprof计时

CUDA 5.0后有一个工具叫做nvprof的命令行分析工具,使用方法：
```bash
$ nvprof [nvprof_args] <application>[application_args]
```

汉化板：
```bash
$ nvprof [选项] <可执行程序> [程序参数]
```

## 组织并行线程

讨论：**不同的线程组织形式是怎样影响性能**

### 使用块和线程建立矩阵索引
```cpp
int ix=threadIdx.x+blockIdx.x*blockDim.x;
int iy=threadIdx.y+blockIdx.y*blockDim.y;
```
这里(ix,iy)就是整个线程模型中任意一个线程的索引，或者叫做全局地址，局部地址当然就是(threadIdx.x,threadIdx.y)了

计算出了线程的全局坐标，用线程的全局坐标对应矩阵的坐标，也就是说，线程的坐标(ix,iy)对应矩阵中(ix,iy)的元素，这样就形成了**一一对应，不同的线程处理矩阵中不同的数据**，举个具体的例子，ix=10,iy=10的线程去处理矩阵中(10,10)的数据.

简单的二维矩阵相加：
```cpp
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*ny;
    if (ix<nx && iy<ny)
    {
      MatC[idx]=MatA[idx]+MatB[idx];
    }
}
```

通过不同的维度的``网格``与``块``布局可以得到更好的效果.

## GPU设备信息

**NVIDIA-SMI**命令打印即可。

## CUDA执行模型概述

### 概述

了解CUDA的执行模型，可以帮助我们优化指令吞吐量，和内存使用来获得极限速度

### GPU架构概述

详细组件：
- CUDA核心
- 共享内存/一级缓存
- 寄存器文件
- 加载/存储单元
- 特殊功能单元
- 线程束调度器

#### SM

概念：SM（Streaming Multiprocessor）​​ 是 NVIDIA GPU 的核心计算单元，负责执行 CUDA 核函数（Kernel）中的并行线程。你可以把它想象成 GPU 的 "计算大脑"，每个 SM 包含多个 CUDA 核心（CUDA Cores）、寄存器、共享内存、缓存等资源，用于高效执行并行计算任务。

GPU中每个SM都能支持数百个线程并发执行，每个GPU通常有多个SM，当一个核函数的网格被启动的时候，多个block会被同时分配给可用的SM上执行。

**注意:** 当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。

#### 线程束

CUDA 采用单指令多线程SIMT架构管理执行线程，不同设备有不同的线程束大小，但是到目前为止基本所有设备都是**维持在32**，也就是说每个SM上有多个block，一个block有多个线程（可以是几百个，但不会超过某个最大值），但是从机器的角度，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行，线程束中的每个线程执行同一条指令

#### SIMD vs SIMT

| 特性                | SIMD（Single Instruction, Multiple Data） | SIMT（Single Instruction, Multiple Threads） |
|---------------------|-------------------------------------------|---------------------------------------------|
| 定义            | 单条指令同时操作多个数据（向量化计算）       | 单条指令由多个线程执行，每个线程处理不同数据  |
| 硬件代表        | CPU 的 SSE/AVX 指令集、GPU 的早期架构       | NVIDIA GPU 的 CUDA 架构（SM 和 Warp 机制）   |
| 并行粒度        | 数据级并行（操作向量寄存器）               | 线程级并行（以 Warp 为单位调度）             |
| 分支处理        | 所有数据必须执行相同操作（严格同步）         | 允许线程分化（Warp Divergence，但性能下降）  |
| 灵活性          | 低（需显式向量化代码）                     | 高（线程可独立执行不同逻辑）                 |

### Fermi 架构 vs Kepler 架构

详情链接：
[链接](expand_concept/Fermi架构vsKepler架构.md)

