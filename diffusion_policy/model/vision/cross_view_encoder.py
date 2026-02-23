import math
import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class SpatialResNet(nn.Module):
    """Wrapper that returns spatial feature maps from a ResNet, skipping avgpool/fc."""
    def __init__(self, resnet):
        super().__init__()
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, x):
        return self.features(x)  # (B, C, H, W)


class SpatialViT(nn.Module):
    """Wrapper that returns spatial feature maps (B, C, H, W) from a timm ViT.
    Handles CLS and register tokens via num_prefix_tokens."""
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.embed_dim = vit.embed_dim
        self.num_prefix = getattr(vit, 'num_prefix_tokens', 0)

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # (B, prefix+N, D)
        tokens = tokens[:, self.num_prefix:]   # drop CLS + register tokens
        B, N, D = tokens.shape
        H = W = int(math.sqrt(N))
        return tokens.transpose(1, 2).reshape(B, D, H, W)


class CrossViewFusion(nn.Module):
    """Transformer-based cross-view fusion with learnable view and positional embeddings."""
    def __init__(self, feat_dim, proj_dim, n_views, spatial_size,
                 n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(feat_dim, proj_dim)

        n_spatial = spatial_size * spatial_size
        self.view_embed = nn.Embedding(n_views, proj_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_spatial, proj_dim) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, proj_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, spatial_features_list):
        """
        Args:
            spatial_features_list: list of (B, C, H, W) tensors, one per view
        Returns:
            (B, proj_dim) fused representation
        """
        B = spatial_features_list[0].shape[0]
        all_tokens = []

        for view_idx, feat_map in enumerate(spatial_features_list):
            # (B, C, H, W) → (B, H*W, C) → (B, H*W, proj_dim)
            tokens = feat_map.flatten(2).transpose(1, 2)
            tokens = self.proj(tokens)
            tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
            tokens = tokens + self.view_embed.weight[view_idx]
            all_tokens.append(tokens)

        # (B, N_views * H*W, proj_dim)
        tokens = torch.cat(all_tokens, dim=1)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.transformer(tokens)
        return self.norm(tokens[:, 0])


class CrossViewImageObsEncoder(MultiImageObsEncoder):
    """
    MultiImageObsEncoder with spatial features and cross-view transformer fusion.
    Instead of avgpool → concat, extracts spatial feature maps from each view
    and fuses them via transformer self-attention with view/position embeddings.
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model,
            proj_dim: int = 128,
            n_xattn_heads: int = 4,
            n_xattn_layers: int = 2,
            **kwargs):
        kwargs.pop('feature_dim', None)
        super().__init__(shape_meta=shape_meta, rgb_model=rgb_model,
                         feature_dim=None, **kwargs)

        # Wrap RGB models to return spatial features instead of pooled vectors
        def _wrap_spatial(model):
            if hasattr(model, 'forward_features'):
                return SpatialViT(model)
            return SpatialResNet(model)

        if self.share_rgb_model:
            self.key_model_map['rgb'] = _wrap_spatial(self.key_model_map['rgb'])
        else:
            for key in self.rgb_keys:
                self.key_model_map[key] = _wrap_spatial(self.key_model_map[key])

        # Probe spatial output dimensions
        with torch.no_grad():
            first_key = self.rgb_keys[0]
            input_shape = self.key_shape_map[first_key]
            dummy = torch.zeros(1, *input_shape)
            model_key = 'rgb' if self.share_rgb_model else first_key
            spatial_out = self.key_model_map[model_key](dummy)
            feat_dim = spatial_out.shape[1]
            spatial_size = spatial_out.shape[2]

        self.cross_view_fusion = CrossViewFusion(
            feat_dim=feat_dim,
            proj_dim=proj_dim,
            n_views=len(self.rgb_keys),
            spatial_size=spatial_size,
            n_heads=n_xattn_heads,
            n_layers=n_xattn_layers
        )
        self._proj_dim = proj_dim

    def forward(self, obs_dict):
        batch_size = None
        spatial_features = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img)

            model_key = 'rgb' if self.share_rgb_model else key
            feat_map = self.key_model_map[model_key](img)
            spatial_features.append(feat_map)

        # Cross-view fusion → (B, proj_dim)
        fused = self.cross_view_fusion(spatial_features)

        # Append low_dim features
        features = [fused]
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        return torch.cat(features, dim=-1)
