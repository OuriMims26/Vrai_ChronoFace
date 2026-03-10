import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# --- ARCHITECTURE (COPIED) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self, x): return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

def test_gan(model_path):
    print(f"\n--- Testing Model: {model_path} ---")
    model = GeneratorResNet(input_shape=(3, 256, 256), num_residual_blocks=9)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        print(f"   Weights loaded successfully.")
    else:
        print(f"   ERROR: {model_path} not found!")
        return

    # Use a realistic input (normalized random)
    input_tensor = torch.randn(1, 3, 256, 256)
    print(f"   Input Stats - Min: {input_tensor.min().item():.3f}, Max: {input_tensor.max().item():.3f}, Mean: {input_tensor.mean().item():.3f}")
    
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"   Success! Output shape: {output.shape}")
        print(f"   Output Stats - Min: {output.min().item():.3f}, Max: {output.max().item():.3f}, Mean: {output.mean().item():.3f}")
        
        # Denormalize and save
        out_img = output * 0.5 + 0.5
        out_img = torch.clamp(out_img, 0, 1)
        save_image(out_img, f"test_output_{os.path.basename(model_path)}.png")
        print(f"   Saved output to test_output_{os.path.basename(model_path)}.png")
    except Exception as e:
        print(f"   CRASH DURING INFERENCE: {e}")

if __name__ == "__main__":
    for m in ["best_G_AB.pth", "G_BA_45.pth"]:
        test_gan(m)
