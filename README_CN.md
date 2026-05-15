[English Version](README.md)

# MNNConvert (ONNX 版本)

将 ONNX 模型转换为 MNN 格式。这是 MNN 转换器的精简版本，专注于 ONNX 模型转换。

## 依赖

- gcc >= 4.9（或 Clang）
- protobuf >= 3.0
- MNN 框架（此转换器作为 MNN 项目的子组件构建）

```bash
# macOS
brew install protobuf
# Ubuntu/Debian
sudo apt-get install libprotobuf-dev protobuf-compiler
```

其他平台请参考 [protobuf 官方安装指南](https://github.com/protocolbuffers/protobuf/tree/master/src)。

## 编译

此转换器是 MNN 项目的子组件，需要从 MNN 根目录编译：

```bash
cd MNN
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make
```

## 使用方法

```bash
Usage:
  MNNConvert [OPTION...]

  -h, --help                 模型转换工具

  -v, --version              显示当前版本
  -f, --framework arg        模型类型，可选: [ONNX, MNN, JSON]
      --modelFile arg        ONNX 模型文件，例如: *.onnx
      --MNNModel arg         MNN 模型，例如: *.mnn
      --benchmarkModel       不保存大规模数据，如 Conv 的 weight、BN 的
                             gamma, beta, mean, variance 等。仅用于测试
                             模型的性能
      --bizCode arg          MNN 模型标识，例如: MNN
      --debug                启用调试模式
      --forTraining          是否保留训练用算子 BN 和 Dropout，默认: false
      --fp16                 以半精度浮点数保存 Conv 的 weight/bias
      --weightQuantBits arg  将 conv/matmul/LSTM 的 float 权重量化为
                             int8，2-8 位，默认: 0（不量化）
      --optimizeLevel arg    图优化等级:
                               0 - 不优化 (仅 MNN 源)
                               1 - 安全优化 (默认)
                               2 - 激进优化 (某些情况可能出错)
      --keepInputFormat      是否保持输入维度格式，默认: true
      --saveStaticModel      保存固定 shape 的静态模型，默认: false
      --inputConfigFile arg  静态模型输入配置文件，例如: ~/config.txt
      --compressionParamsFile arg
                             压缩参数文件路径（int8 校准表或稀疏参数）
      --info                 导出 MNN 模型信息
      --OP                   打印框架支持的操作列表
      --dumpPass             输出每个优化 Pass 的详细信息
```

> 说明: `--benchmarkModel` 选项会将模型中 Conv 的 weight、BN 的 mean/var 等参数移除，减小转换后的模型文件大小，在运行时随机初始化参数，以方便测试模型性能。

### ONNX 转 MNN

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN
```

三个选项（`-f`、`--modelFile`、`--MNNModel`）是必须的。

示例:

```bash
# 基本转换
./MNNConvert -f ONNX --modelFile path/to/model.onnx --MNNModel model.mnn --bizCode MNN

# 使用半精度浮点存储权重（减小模型体积）
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --fp16

# 使用 int8 权重量化（进一步减小模型体积）
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --weightQuantBits 8

# 保留训练算子（BatchNorm, Dropout）
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --forTraining
```

### MNN 转 MNN（重新编码/修改 bizCode）

```bash
./MNNConvert -f MNN --modelFile source.mnn --MNNModel output.mnn --bizCode NEWCODE
```

### MNN 转 JSON（导出模型结构供检查）

```bash
./MNNConvert -f MNN --modelFile model.mnn --JsonFile model.json
```

### JSON 转 MNN

```bash
./MNNConvert -f JSON --modelFile model.json --MNNModel model.mnn
```

### 查看版本号

```bash
./MNNConvert --version
```

### 查看模型信息

```bash
./MNNConvert -f MNN --modelFile model.mnn --info
```

### 列出支持的 ONNX 算子

```bash
./MNNConvert -f ONNX --OP
```

## PyTorch 模型转换

1. 用 PyTorch 的 onnx.export 导出 ONNX 模型（参考: https://pytorch.org/docs/stable/onnx.html）:

```python
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "alexnet.onnx",
                  verbose=True, input_names=input_names,
                  output_names=output_names, do_constant_folding=True)
```

2. 将 ONNX 模型转换为 MNN:

```bash
./MNNConvert -f ONNX --modelFile alexnet.onnx --MNNModel alexnet.mnn --bizCode MNN
```

## TensorFlow / Keras 模型转换

1. 使用 [tf2onnx](https://github.com/onnx/tensorflow-onnx) 将 TF/Keras 模型转换为 ONNX 格式
2. 然后按照上述方法将 ONNX 模型转换为 MNN

## 权重量化

启用仅权重量化以减小模型体积:

```bash
# 对称量化（兼容旧版 MNN）
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --weightQuantBits 8

# 非对称量化（精度更好，需要新版 MNN）
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN \
    --weightQuantBits 8 --weightQuantAsymmetric
```

## MNNDump2Json

将 MNN 二进制模型文件导出为可读的 JSON 格式，方便与原始模型参数进行对比。

## 外部数据模型

对于大型模型（>2GB），权重会自动存储在外部 `.weight` 文件中。
你也可以使用 `--saveExternalData` 强制启用外部存储。
