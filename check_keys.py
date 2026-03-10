import torch
import os

def check_keys(model_path):
    print(f"\n--- Checking Keys: {model_path} ---")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        keys = list(state_dict.keys())
        print(f"   Total keys: {len(keys)}")
        # Count residual blocks
        res_blocks = set()
        for k in keys:
            if 'model.' in k:
                parts = k.split('.')
                # usually model.N.block.M...
                if parts[1].isdigit():
                    idx = int(parts[1])
                    # In our model, residual blocks start after downsampling
                    # Downsampling is 0 to 6 (roughly)
                    # Let's just print the keys to be sure
                    res_blocks.add(idx)
        print(f"   Indices in Sequential: {sorted(list(res_blocks))}")
    else:
        print(f"   ERROR: {model_path} not found!")

if __name__ == "__main__":
    for m in ["best_G_AB.pth", "G_BA_45.pth"]:
        check_keys(m)
