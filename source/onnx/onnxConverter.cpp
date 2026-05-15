//
//  onnxConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <queue>

#include "MNN_generated.h"
#include "OnnxUtils.hpp"
#include "logkit.h"

#include "OnnxTmpGraph.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnxOpConverter.hpp"

// MNNConverter.cpp 的 main() 先创建 modelConfig，
// 然后调用 initializeMNNConvertArgs() 解析命令行参数，把 -f ONNX、--modelFile、--MNNModel 等参数保存进 modelConfig。
// 随后 convertModel() 根据 modelConfig::ONNX 分支调用 onnx2MNNNet()。
// onnx2MNNNet() 先通过 onnx_read_proto_from_binary() 把 .onnx 文件反序列化成 onnx::ModelProto，再取出其中的 GraphProto。
// 然后 converter 创建 OnnxScope 管理 tensor 名称到 MNN tensor index 的映射，
// 先把 graph input 转成 MNN Input Op，再把 initializer 转成 MNN Const Op，
// 最后遍历每个 ONNX NodeProto，根据 op_type 查找对应的 onnxOpConverter，并调用 converter->run() 把该节点翻译成 MNN::OpT，
// 加入 netT->oplists。这样 ONNX 的 GraphProto 就被转换成了 MNN 内部的 NetT。

// ONNX 前端转换的主入口
int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT, MNN::OpT* meta, std::vector<std::string>& inputNames) {
                // MNN netT -> MNN 内部的整张图，大致对应 ONNX 的 GraphProto
                // netT->oplists -> MNN 图中的算子列表，对应 ONNX 的 NodeProto
    std::string modelDir;
    size_t pos = inputModel.find_last_of("\\/");
    if (pos != std::string::npos) {
        modelDir = inputModel.substr(0, pos + 1);
    }

    onnx::ModelProto onnxModel;  // 定义 onnx::ModelProto 对象，onnx::ModelProto 是 protobuf 生成的类，代表一个 ONNX 模型
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;
    if (!success) {
        MNN_ERROR("[ERROR] Model file is not onnx model.\n");
        return 1;
    }

    int opsetVersion = 13;
    auto opsetInfo = onnxModel.opset_import();
    if (!opsetInfo.empty()) {
        opsetVersion = static_cast<int>(opsetInfo.begin()->version());
    }
    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();
    LOG(INFO) << "ONNX Model opset version: " << opsetVersion;

    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();
    if (0 == nodeCount) {
        MNN_ERROR("[ERROR] Invalid ONNX Model:%s\n", inputModel.c_str());
        return 1;
    }
    for (int i=0; i<onnxGraph.input_size(); ++i) {
        inputNames.emplace_back(onnxGraph.input(i).name());
    }

    // OnnxScope 是 onnx 转换过程中保存转换状态的类，包含了当前已经转换的 MNN NetT、已经转换的 ONNX 节点等信息
    std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get(), modelDir));
    scope->mOpsetVersion = opsetVersion;
    // find the inputs which do not have initializer
    const auto& initializers         = scope->mInitializers;
    const auto& inputs               = scope->mInputs;
    const auto& outputs              = scope->mOutputs;
    // set input node to MNN net, 把 ONNX input 转成 MNN Input Op, GraphProto.input -> MNN::OpType_Input
    for (const auto& iter : inputs) {
        // 如果这个输入没有 initializer（权重），说明它是一个真正的model输入 input，需要转换成 MNN 的 Input Op
        bool notHaveInitializer = initializers.find(iter.first) == initializers.end();
        if (notHaveInitializer) {
            MNN::OpT* MNNOp  = new MNN::OpT;
            MNNOp->name      = iter.first;
            MNNOp->type      = MNN::OpType_Input;
            MNNOp->main.type = MNN::OpParameter_Input;
            auto inputParam  = new MNN::InputT;
            const auto it    = inputs.find(iter.first);
            //FUNC_PRINT_ALL(iter.first.c_str(), s);
            DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
            const auto& tensorInfo = (it->second)->type().tensor_type();
            const int inputDimSize = tensorInfo.shape().dim_size();
            inputParam->dims.resize(inputDimSize);
            for (int i = 0; i < inputDimSize; ++i) {
                const auto& dim = tensorInfo.shape().dim(i);
                if (dim.has_dim_value()) {
                    inputParam->dims[i] = static_cast<int32_t>(dim.dim_value());
                } else {
                    inputParam->dims[i] = -1;
                }
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(iter.first));
            MNNOp->main.value = inputParam;
            netT->oplists.emplace_back(MNNOp);
        }
    }

    // onnx model not all topo sort graph, sort it -> onnx::GraphProto 的节点顺序不一定是拓扑排序的，先进行拓扑排序，得到 idxMap
    // 拓扑排序：一个节点被转换之前，它依赖的前驱节点应该已经转换完
    std::vector<int> idxMap = OnnxScope::topoSort(onnxGraph);

    // 把 ONNX 的 initializer 转成 MNN 的 Const Op，GraphProto.initializer -> MNN::OpType_Const
    auto makeConst = [&](const std::string& inputName) {
        const auto it         = initializers.find(inputName);
        if (it != initializers.end() && scope->lookupTensor(it->first) == -1) {
            // Create const Op
            MNN::OpT* constOp   = new MNN::OpT;
            constOp->type       = MNN::OpType_Const;
            constOp->main.type  = MNN::OpParameter_Blob;
            constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second, modelDir, constOp);
            constOp->name    = it->first;
            constOp->outputIndexes.push_back(scope->declareTensor(it->first));
            netT->oplists.emplace_back(constOp);
            scope->insertConstant(inputName, constOp);
        }
    };
    for (int i=0; i<onnxGraph.output_size(); ++i) {
        makeConst(onnxGraph.output(i).name());
    }
    // Declare all outputs
    for (int idx = 0; idx < nodeCount; ++idx) {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.output_size(); k++) {
            scope->declareTensor(onnxNode.output(k));
        }
    }

    // onnx node ==> MNN node， 主循环 遍历ONNX node
    for (int idx = 0; idx < nodeCount; ++idx) {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();

        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        auto opConverter = onnxOpConverterSuit::get()->search(opType);

        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();

        // convert initializer to be Constant node(op)
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            makeConst(inputName);
        }

        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) {
            int inputIdx = scope->lookupTensor(onnxNode.input(k));
            if (inputIdx < 0) {
                LOG(INFO) << "Check it out ==> " << MNNOp->name << " has empty input, the index is " << k;
            }
            MNNOp->inputIndexes.push_back(inputIdx);
        }
        for (int k = onnxNode.input_size() - 1; k >= 0 && MNNOp->inputIndexes[k] < 0; --k) {
            MNNOp->inputIndexes.pop_back();
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        // build op
        opConverter->run(MNNOp, &onnxNode, scope.get());
        if (MNNOp->type == MNN::OpType_Const) {
            scope->insertConstant(name, MNNOp);
        }
        netT->oplists.emplace_back(MNNOp);
    }
    // ONNX NodeProto -> MNN OpT 的转换流程：
        // 拿到 ONNX NodeProto
        //   ↓
        // 读取 node.op_type()         
        //   ↓
        // 根据 op_type 查找对应 converter
        //   ↓
        // 创建 MNN::OpT
        //   ↓
        // 构建 inputIndexes / outputIndexes
        //   ↓
        // 调用 converter->run()
        //   ↓
        // 把 MNNOp 放进 netT->oplists

    netT->tensorNumber = netT->tensorName.size();
    // set MNN net output name
    for (int i = 0; i < onnxGraph.output_size(); ++i) {
        const auto& output = onnxGraph.output(i);
        netT->outputName.emplace_back(output.name());
    }

    netT->sourceType = MNN::NetSource_ONNX;
    netT->bizCode    = bizCode;
    auto metaSize = onnxModel.metadata_props_size();
    for (int i=0; i<metaSize; ++i) {
        std::unique_ptr<MNN::AttributeT> dstMeta(new MNN::AttributeT);
        auto srcMeta = onnxModel.metadata_props(i);
        dstMeta->key = srcMeta.key();
        dstMeta->s = srcMeta.value();
        meta->main.AsExtra()->attr.emplace_back(std::move(dstMeta));
    }

    return 0;
}
