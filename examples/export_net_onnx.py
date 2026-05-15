import torch
import torch.nn as nn
from pathlib import Path


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = TinyNet()
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    output_path = Path("tinynet.onnx")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )

    print(f"Exported ONNX model to: {output_path.resolve()}")


if __name__ == "__main__":
    main()