# 模块概览与学习路线

## 一、项目文件模块概览

### 1. `include/` — 公共头文件（接口层）

| 文件 | 功能 |
|---|---|
| [config.hpp](include/config.hpp) | `modelConfig` 结构体，承载所有 CLI 参数和转换配置 |
| [cli.hpp](include/cli.hpp) | `Cli` 类，声明参数解析和核心转换入口 |
| [onnxConverter.hpp](include/onnxConverter.hpp) | `onnx2MNNNet()` 函数声明，ONNX→MNN 转换入口 |
| [writeFb.hpp](include/writeFb.hpp) | `writeFb()` 将 MNN 内部结构序列化为 .mnn 文件 |
| [addBizCode.hpp](include/addBizCode.hpp) | 读取已有 .mnn 并修改 bizCode |
| [ConverterScope.hpp](include/ConverterScope.hpp) | 作用域管理 — tensor 声明/查找、子图依赖（转换过程的"符号表"） |
| [PostConverter.hpp](include/PostConverter.hpp) | `optimizeNet()` 优化入口 |
| [OpCount.hpp](include/OpCount.hpp) | 框架支持的算子统计 |
| [logkit.h](include/logkit.h) | 日志宏（`LOG(INFO)`、`DCHECK` 等） |
| [cxxopts.hpp](include/cxxopts.hpp) | 第三方命令行解析库（header-only） |

### 2. `source/onnx/` — ONNX 解析与算子转换（"前端"）

| 文件 | 功能 |
|---|---|
| [onnx.proto](source/onnx/onnx.proto) | ONNX 协议的 protobuf 定义 |
| `generated/onnx.pb.h` | protobuf 生成的 C++ 代码（读写 ONNX 二进制） |
| [OnnxUtils.cpp/hpp](source/onnx/OnnxUtils.cpp) | protobuf 二进制读写封装 |
| [onnxOpConverter.hpp](source/onnx/onnxOpConverter.hpp) | **核心框架** — `onnxOpConverter` 基类 + `OnnxScope` + `REGISTER_CONVERTER` 自动注册宏 |
| [onnxOpConverter.cpp](source/onnx/onnxOpConverter.cpp) | 转换器工厂 + `DefaultonnxOpConverter`（兜底）+ 数据/Blob 类型转换 |
| [onnxConverter.cpp](source/onnx/onnxConverter.cpp) | **主循环** — 读 ONNX graph → 拓扑排序 → 遍历节点 → 调用 converter → 构建 MNN 图 |
| `*Onnx.cpp`（40+ 文件） | 具体算子转换实现（Conv、Gemm、Reshape、Loop、If 等） |
| [onnx_model_graph_opt.py](source/onnx/onnx_model_graph_opt.py) | Python ONNX 图预处理脚本 |

### 3. `source/optimizer/` — 图优化（"中端"）

这是最复杂的模块，承担将朴素的 MNN 图优化为高效计算图的任务。

| 子目录 | 功能 |
|---|---|
| `postconvert/` | 后转换 Pass：FuseDupOp、RemoveInplace、MergeBNToConv、TransformGroupConv 等 ~25 个 |
| `onnxextra/` | ONNX 专用 Pass：OnnxGemm、OnnxEinsum、OnnxLSTMMerge、OnnxPooling 等 |
| `merge/` | 通用融合 Pass：LayerNorm/Attention/GeLU 融合、Conv 折叠、常量折叠等 |
| `passes/` | Pass 基础设施 — `Pass` 基类 + `PassRegistry` 注册系统 |

顶层文件：

| 文件 | 功能 |
|---|---|
| [PostConverter.cpp](source/optimizer/PostConverter.cpp) | 优化主循环 — `optimizeNet()` 编排所有 Pass 执行顺序 |
| [PostTreatUtils.cpp/hpp](source/optimizer/PostTreatUtils.cpp) | Pass 管理器：注册/查找 + 图遍历工具 |
| [TemplateMerge.cpp/hpp](source/optimizer/TemplateMerge.cpp) | 模板匹配融合引擎 |
| [Program.cpp/hpp](source/optimizer/Program.cpp) | MNN 图 → Expression 程序转换 |

### 4. `source/common/` — 公共工具（"后端"）

| 文件 | 功能 |
|---|---|
| [cli.cpp](source/common/cli.cpp) | CLI 解析 + `convertModel` 主流程 + 模型测试 + MNN↔JSON |
| [writeFb.cpp](source/common/writeFb.cpp) | FlatBuffer 序列化（含压缩、量化、外部存储） |
| [ConverterScope.cpp](source/common/ConverterScope.cpp) | Scope 实现 — tensor 声明、Const/Input op 构建、子图依赖 |
| [CommonUtils.cpp/hpp](source/common/CommonUtils.cpp) | 压缩配置加载、文件存在检查、protobuf↔JSON |
| [WeightQuantAndCoding.cpp](source/common/WeightQuantAndCoding.cpp) | 权重量化编码 |
| [FullQuantAndCoding.cpp](source/common/FullQuantAndCoding.cpp) | 全精度量化（权重+激活） |
| [SaveHalfFloat.cpp](source/common/SaveHalfFloat.cpp) | float32→float16 半精度转换 |
| [AlignDenormalizedValue.cpp](source/common/AlignDenormalizedValue.cpp) | 非规格化浮点数对齐为 0 |
| [Json2Flatbuffer.cpp/hpp](source/common/Json2Flatbuffer.cpp) | JSON→FlatBuffer 转换 |
| convertToStaticModel / AddSparseInfo / AddUUID / ChannelPruneConvert 等 | 各自独立的工具函数 |

### 5. `source/MNN/` — MNN 原生格式

| 文件 | 功能 |
|---|---|
| [addBizCode.cpp](source/MNN/addBizCode.cpp) | 读取 .mnn → 反序列化 → 修改 bizCode → 输出 NetT |

### 6. `source/compression/` — 压缩协议

| 文件 | 功能 |
|---|---|
| [MNN_compression.proto](source/compression/MNN_compression.proto) | 压缩管线 protobuf 定义 |
| `generated/MNN_compression.pb.h` | protobuf 生成代码 |
| [quantization.hpp](source/compression/quantization.hpp) | 量化工具 |

### 7. 顶层文件

| 文件 | 功能 |
|---|---|
| [source/MNNConverter.cpp](source/MNNConverter.cpp) | `main()` 入口（11 行） |
| [CMakeLists.txt](CMakeLists.txt) | 构建脚本 |
| `source/MNNRevert2Buffer.cpp` | 独立工具：MNN→FlatBuffer 还原 |
| `source/MNNDump2Json.cpp` | 独立工具：MNN→JSON 导出 |
| `source/TestConvertResult.cpp` | 独立工具：转换结果精度测试 |
| [forward.json](forward.json) | 测试后端配置 |

### 8. `tools/` — 辅助 Python 脚本

| 文件 | 功能 |
|---|---|
| [auto_quant.py](tools/auto_quant.py) | 自动量化 |
| [user_quant_modify_demo.py](tools/user_quant_modify_demo.py) | 自定义量化参数示例 |
| [genInput.py](tools/genInput.py) | 生成测试输入 |
| `testConvertor.py / test4Caffe.py / test4TF.py` | 批量转换脚本 |

---

## 二、学习路线（由浅入深）

### 第一阶段：背景知识（1–2 天）

1. 了解 ONNX 是什么：https://onnx.ai
2. 了解 Protobuf 基本概念（message、序列化/反序列化）
3. 了解 FlatBuffers（MNN 存储格式）：schema → 序列化
4. 导出一个小 PyTorch 模型为 ONNX，用 [Netron](https://netron.app) 可视化其图结构

**阅读：**
- [onnx.proto](source/onnx/onnx.proto)（前 100 行即可，理解 ModelProto > GraphProto > NodeProto 层级）
- [MNNConverter.cpp](source/MNNConverter.cpp)（`main()` 只有 11 行）

### 第二阶段：核心数据流（2–3 天）

**CLI 解析：**
- [cli.cpp](source/common/cli.cpp) `initializeMNNConvertArgs()`（L138–541）— 参数如何映射到 `modelConfig`
- [cli.cpp](source/common/cli.cpp) `convertModel()`（L649–746）— 格式分派逻辑

**ONNX 解析（核心中的核心）：**
- [OnnxUtils.cpp](source/onnx/OnnxUtils.cpp)（45 行）— protobuf 二进制读取
- [onnxConverter.cpp](source/onnx/onnxConverter.cpp)（**重点**）：
  - L24–38：读取 ONNX 文件
  - L55–93：ONNX input → MNN Input Op
  - L96：拓扑排序
  - L116–165：**主循环** — 遍历 ONNX node → 查 converter → 建 MNN Op
  - L166–184：输出名称、bizCode、Meta

**Op 转换框架：**
- [onnxOpConverter.hpp](source/onnx/onnxOpConverter.hpp) — `onnxOpConverter` 基类 + `OnnxScope` + `REGISTER_CONVERTER` 宏
- [onnxOpConverter.cpp](source/onnx/onnxOpConverter.cpp) — `onnxOpConverterSuit` 工厂 + `DefaultonnxOpConverter`（兜底）+ 数据类型转换

### 第三阶段：阅读具体算子转换（3–4 天）

**由简到繁：**

| 顺序 | 文件 | 要点 |
|---|---|---|
| 1 | IdentityOnnx.cpp | 理解 `DECLARE_OP_CONVERTER` + `REGISTER_CONVERTER` 模板 |
| 2 | ReluOnnx.cpp / SigmoidOnnx.cpp | 单属性激活函数 |
| 3 | UnaryOnnx.cpp | "一表多用"设计（一个 converter 处理 Abs/Cos/Sin/Exp/Log...） |
| 4 | BinaryOpOnnx.cpp | Add/Sub/Mul/Div 等二元运算 |
| 5 | ReshapeOnnx.cpp | 需要访问 input tensor 推断 shape |
| 6 | ConcatOnnx.cpp / SplitOnnx.cpp | 变长输入输出 |
| 7 | ConstantOnnx.cpp | TensorProto→Blob 权重转换 |
| 8 | MatMulOnnx.cpp / LayerNormOnnx.cpp | 带 attribute 的复杂算子 |
| 9 | LoopOnnx.cpp / IfOnnx.cpp | **子图处理**（最复杂） |

### 第四阶段：图优化管线（2–3 天）

**主循环：**
- [PostConverter.cpp](source/optimizer/PostConverter.cpp) — `optimizeNetImpl()`（L265–397）编排 Pass 执行顺序：
  - `postConvertPass` → `ExtraPass` → `MergePass(5 个优先级)` → `afterPass`

**选读 3–5 个典型 Pass：**
1. [MergeBNToConvolution.cpp](source/optimizer/postconvert/MergeBNToConvolution.cpp) — BN→Conv 折叠
2. [TransformGroupConvolution.cpp](source/optimizer/postconvert/TransformGroupConvolution.cpp) — Group Conv 拆分
3. [FuseLayerNorm.cpp](source/optimizer/merge/FuseLayerNorm.cpp) — LayerNorm 模式融合
4. [RemoveUnusefulOp.cpp](source/optimizer/postconvert/RemoveUnusefulOp.cpp) — 清理冗余算子
5. [OnnxGemm.cpp](source/optimizer/onnxextra/OnnxGemm.cpp) — ONNX 独有 Gemm 优化

**Pass 基础设施：**
- [passes/Pass.hpp](source/optimizer/passes/Pass.hpp) + [PassRegistry.hpp](source/optimizer/passes/PassRegistry.hpp)
- [PostTreatUtils.hpp](source/optimizer/PostTreatUtils.hpp)

### 第五阶段：输出与后处理（1–2 天）

**FlatBuffer 序列化：**
- [writeFb.cpp](source/common/writeFb.cpp) `writeFb()`（L177–321）— 不支持算子检测、Meta Op 合并、FlatBufferBuilder 使用

**后处理：**
- [WeightQuantAndCoding.cpp](source/common/WeightQuantAndCoding.cpp) — 权重量化
- [SaveHalfFloat.cpp](source/common/SaveHalfFloat.cpp) — 半精度保存

### 第六阶段：辅助功能收尾（1 天）

- [cli.cpp](source/common/cli.cpp) L1055–1181 — `mnn2json()` / `json2mnn()` 互转
- [ConverterScope.cpp](source/common/ConverterScope.cpp) — Scope 系统完整实现
- [Json2Flatbuffer.cpp](source/common/Json2Flatbuffer.cpp)

---

## 三、核心流程图

```
main() [MNNConverter.cpp]
  │
  ├─ Cli::initializeMNNConvertArgs()  [cli.cpp]     ← 解析 CLI → modelConfig
  │
  └─ Cli::convertModel()              [cli.cpp]
       │
       ├─ onnx2MNNNet()               [onnxConverter.cpp]
       │    ├─ onnx_read_proto()      [OnnxUtils.cpp]   ← 读 ONNX 二进制
       │    ├─ topoSort()             [onnxOpConverter.cpp]  ← 拓扑排序
       │    └─ for each node:
       │         ├─ onnxOpConverterSuit::search()  ← 查 converter
       │         ├─ converter->run()               ← 转换为 MNN Op
       │         └─ scope->declareTensor()         ← 声明 tensor
       │
       ├─ optimizeNet()               [PostConverter.cpp]
       │    ├─ postConvertPass        (RemoveInplace, FuseDupOp, ...)
       │    ├─ ExtraPass              (OnnxExtra 融合)
       │    ├─ MergePass ×5           (模板匹配融合)
       │    └─ afterPass              (ReIndexTensor, ...)
       │
       └─ writeFb()                   [writeFb.cpp]
            ├─ 权重量化 / 半精度 / 稀疏化
            ├─ FlatBufferBuilder 构建
            └─ 写入 .mnn 文件
```
