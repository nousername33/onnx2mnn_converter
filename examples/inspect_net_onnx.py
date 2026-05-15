import onnx

model = onnx.load("tinynet.onnx")

print("========== ModelProto ==========")
print("IR version:", model.ir_version)
print("Opset imports:", model.opset_import)

graph = model.graph

print("\n========== GraphProto ==========")
print("Graph name:", graph.name)

print("\nGraph inputs:")
for item in graph.input:
    print("  ", item.name)

print("\nGraph outputs:")
for item in graph.output:
    print("  ", item.name)

print("\nInitializers / weights:")
for initializer in graph.initializer:
    print("  ", initializer.name, initializer.dims)

print("\n========== NodeProto ==========")
for i, node in enumerate(graph.node):
    print(f"\nNode {i}")
    print("  name:", node.name)
    print("  op_type:", node.op_type)
    print("  inputs:", list(node.input))
    print("  outputs:", list(node.output))

    if len(node.attribute) > 0:
        print("  attributes:")
        for attr in node.attribute:
            print("    ", attr.name, "=", onnx.helper.get_attribute_value(attr))