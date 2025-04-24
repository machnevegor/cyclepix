import torch
import torch.nn.functional as F

def score_cam(model, target_layer, input_tensor):
    model.eval()
    activations = []

    # Hook function to capture the output of the target layer during forward pass
    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register the forward hook on the target layer
    handle = target_layer.register_forward_hook(forward_hook)

    with torch.no_grad():
        output = model(input_tensor)

        feature_maps = activations[0][0]
        num_maps = feature_maps.shape[0]

        cam = torch.zeros(input_tensor.shape[2:], dtype=torch.float32)

        for i in range(num_maps):
            fmap = feature_maps[i]

            # Upsample the feature map to the input size
            fmap_resized = F.interpolate(
                fmap.unsqueeze(0).unsqueeze(0), 
                size=input_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)  # Shape: [H, W]

            # Normalize the upsampled feature map to [0, 1]
            fmap_norm = (fmap_resized - fmap_resized.min()) / (fmap_resized.max() - fmap_resized.min() + 1e-8)

            # Multiply the normalized feature map with the input image (as a mask)
            masked_input = input_tensor * fmap_norm.unsqueeze(0)

            # Get the model's output score for the masked input
            score = model(masked_input).mean().item()

            # Accumulate the weighted feature map into the CAM
            cam += fmap_resized * score

    # Apply ReLU and normalize the final CAM to [0, 1]
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle.remove()
    return cam.cpu().numpy()
