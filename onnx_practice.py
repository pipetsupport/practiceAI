import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime

# Define the SuperResolution model
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Initialize model and load pre-trained weights
torch_model = SuperResolutionNet(upscale_factor=3)
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
torch_model.load_state_dict(model_zoo.load_url(model_url))
torch_model.eval()

# Convert to ONNX
x = torch.randn(1, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)
torch.onnx.export(torch_model, x, "super_resolution.onnx", export_params=True, opset_version=10,
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Load and check ONNX model
onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

# Run model using ONNX Runtime
ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: x.detach().numpy()}
ort_outs = ort_session.run(None, ort_inputs)

# Compare PyTorch and ONNX Runtime results
np.testing.assert_allclose(torch_out.detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Tested with ONNXRuntime, results match!")
