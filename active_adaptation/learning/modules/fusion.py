import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformations to features based on context:
    output = gamma * features + beta
    
    where gamma and beta are generated from the context input.
    
    Args:
        feature_dim: Dimension of the features to be modulated.
        context_dim: Dimension of the context input.
    """
    def __init__(self,
        feature_dim: int, 
        context_dim: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # Generate gamma (scale) and beta (shift) from context
        # Output dimension is 2 * feature_dim: first half for gamma, second half for beta
        self.context_proj = nn.Linear(context_dim, 2 * feature_dim)
        
        # Initialize so that gamma starts at 1.0 and beta starts at 0.0 (identity)
        # This ensures the layer starts as an identity operation
        with torch.no_grad():
            # Initialize bias: first half (gamma) = 1.0, second half (beta) = 0.0
            self.context_proj.bias.data[:feature_dim] = 1.0  # gamma starts at 1
            self.context_proj.bias.data[feature_dim:] = 0.0  # beta starts at 0
            # Initialize weights to be small so context has minimal effect initially
            nn.init.xavier_uniform_(self.context_proj.weight, gain=0.01)
        
    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to features.
        
        Args:
            features: Input features to modulate, shape (..., feature_dim)
            context: Context input, shape (..., context_dim)
            
        Returns:
            Modulated features, shape (..., feature_dim)
        """
        # Generate gamma and beta from context
        film_params = self.context_proj(context)  # (..., 2 * feature_dim)
        
        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each: (..., feature_dim)
        
        # Apply feature-wise linear modulation
        output = gamma * features + beta
        
        return output

