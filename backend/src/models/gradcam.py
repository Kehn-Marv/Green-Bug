import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import cv2

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN interpretability"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        try:
            # Find target layer
            target_module = None
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                logger.warning(f"Target layer {self.target_layer} not found")
                return
            
            # Register hooks
            def forward_hook(module, input, output):
                self.activations = output
            
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_backward_hook(backward_hook))
            
        except Exception as e:
            logger.error(f"Hook registration failed: {e}")
    
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """Generate Grad-CAM heatmap"""
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Get class index
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward()
            
            # Generate CAM
            if self.gradients is not None and self.activations is not None:
                # Global average pooling of gradients
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                
                # ReLU to keep only positive influences
                cam = F.relu(cam)
                
                # Normalize
                cam = cam / (torch.max(cam) + 1e-8)
                
                # Resize to input size
                cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
                
                return cam.squeeze()
            else:
                logger.warning("No gradients or activations captured")
                return torch.zeros((224, 224))
                
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return torch.zeros((224, 224))
    
    def __del__(self):
        """Clean up hooks"""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass