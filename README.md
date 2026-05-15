[中文版本](README_CN.md)

# MNNConvert (ONNX Edition)

Convert ONNX models to MNN format. This is a streamlined version of the MNN converter that focuses exclusively on ONNX model conversion.

## Dependencies

- gcc >= 4.9 (or Clang)
- protobuf >= 3.0
- MNN framework (this converter is built as part of the MNN project)

```bash
# macOS
brew install protobuf
# Ubuntu/Debian
sudo apt-get install libprotobuf-dev protobuf-compiler
```

For other platforms, refer to the [official protobuf installation guide](https://github.com/protocolbuffers/protobuf/tree/master/src).

## Build

This converter is a sub-component of the MNN project. Build it from the MNN root directory:

```bash
cd MNN
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make
```

## Usage

```bash
Usage:
  MNNConvert [OPTION...]

  -h, --help                 Convert Other Model Format To MNN Model

  -v, --version              show current version
  -f, --framework arg        model type, ex: [ONNX, MNN, JSON]
      --modelFile arg        ONNX model file, ex: *.onnx
      --MNNModel arg         MNN model, ex: *.mnn
      --benchmarkModel       Do NOT save big size data, such as Conv's weight, BN's
                             gamma, beta, mean and variance etc. Only used to test
                             the cost of the model
      --bizCode arg          MNN Model Flag, ex: MNN
      --debug                Enable debugging mode.
      --forTraining          whether or not to save training ops BN and Dropout,
                             default: false
      --fp16                 save Conv's weight/bias in half_float data type
      --weightQuantBits arg  save conv/matmul/LSTM float weights to int8 type,
                             2-8 bits, default: 0 (no weight quant)
      --optimizeLevel arg    graph optimize level:
                               0 - no optimize (MNN source only)
                               1 - safe optimize (default)
                               2 - aggressive optimize (may be wrong in some cases)
      --keepInputFormat      keep input dimension format or not, default: true
      --saveStaticModel      save static model with fixed shape, default: false
      --inputConfigFile arg  input config file for static model, ex: ~/config.txt
      --compressionParamsFile arg
                             path to compression parameters file (int8 calibration
                             table or sparsity params)
      --info                 dump MNN model info
      --OP                   print framework supported ops
      --dumpPass             verbose output for each optimization pass
```

> Note: The `--benchmarkModel` option removes parameters (Conv weights, BN mean/var) to reduce model size. Parameters are initialized randomly at runtime. Useful for performance testing.

### ONNX to MNN Conversion

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN
```

These three options (`-f`, `--modelFile`, `--MNNModel`) are required.

Examples:

```bash
# Basic conversion
./MNNConvert -f ONNX --modelFile path/to/model.onnx --MNNModel model.mnn --bizCode MNN

# With half-float weight storage (reduces model size)
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --fp16

# With int8 weight quantization (for smaller model size)
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --weightQuantBits 8

# Preserve training ops (BatchNorm, Dropout)
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --forTraining
```

### MNN to MNN (re-encode with new bizCode)

```bash
./MNNConvert -f MNN --modelFile source.mnn --MNNModel output.mnn --bizCode NEWCODE
```

### MNN to JSON (dump model structure for inspection)

```bash
./MNNConvert -f MNN --modelFile model.mnn --JsonFile model.json
```

### JSON to MNN

```bash
./MNNConvert -f JSON --modelFile model.json --MNNModel model.mnn
```

### Show Version

```bash
./MNNConvert --version
```

### Dump Model Info

```bash
./MNNConvert -f MNN --modelFile model.mnn --info
```

### List Supported ONNX Ops

```bash
./MNNConvert -f ONNX --OP
```

## How to Convert PyTorch Model

1. Export PyTorch model to ONNX (https://pytorch.org/docs/stable/onnx.html):

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

2. Convert ONNX to MNN:

```bash
./MNNConvert -f ONNX --modelFile alexnet.onnx --MNNModel alexnet.mnn --bizCode MNN
```

## How to Convert TensorFlow / Keras Model

1. Convert TF/Keras model to ONNX using [tf2onnx](https://github.com/onnx/tensorflow-onnx)
2. Then convert the ONNX model to MNN as described above

## Weight Quantization

Enable weight-only quantization to reduce model size:

```bash
# Symmetric weight quantization (compatible with older MNN versions)
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN --weightQuantBits 8

# Asymmetric weight quantization (better accuracy, requires newer MNN)
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN \
    --weightQuantBits 8 --weightQuantAsymmetric
```

## MNNDump2Json

Dump MNN binary model file to readable JSON format for inspection and comparison with the original model parameters.

## External Data Models

For large models (>2GB), weights are automatically stored in an external `.weight` file.
You can also force external storage with `--saveExternalData`.
