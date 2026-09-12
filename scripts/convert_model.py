import torch
import torch.nn as nn

class ASLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ASLNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = ASLNet(input_dim=63, hidden_dim=128, num_classes=29)
model.load_state_dict(torch.load("asl_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 63)
torch.onnx.export(
    model,
    dummy_input,
    "asl_model.onnx",
    input_names=["landmarks"],
    output_names=["prediction"],
    opset_version=11
)

print("Successfully exported asl_model.onnx!")