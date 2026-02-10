import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import math
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from .dinov2_layers import Attention, Mlp
from typing import Callable, Optional, Tuple, Union


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class SparsePriorDA(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        max_depth=20.0
    ):
        super(SparsePriorDA, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.max_depth = max_depth
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_embedder = DepthEmbedding()
        self.depth_self_att = DepthSelfBlock()
        self.depth_cross_att = DepthCrossBlock()
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x, depth_prior):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = list(self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)) #make it mutable
        depth_prior = self.depth_embedder(depth_prior, self.pretrained.pos_embed)
        depth_prior = self.depth_self_att(depth_prior)
        depth_prior = depth_prior[:, 1:] # remove cls token
        tokens, cls = features[-1] #only use last layer as it is first layer in fusion and has highest abstraction
        tokens = self.depth_cross_att(tokens, depth_prior) #cls token is not used in DPT
        features[-1] = (tokens, cls)
        features = tuple(features)

        
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, raw_depth_prior, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        depth_prior, (h_d, w_d) = self.depth2tensor(raw_depth_prior, input_size)
        assert(h == h_d and w == w_d), f"Image and Depth Prior must have same size"
        print(depth_prior.shape)
        
        depth = self.forward(image, depth_prior)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    
    def depth2tensor(self, raw_depth_prior, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])
        
        h, w = raw_depth_prior.shape[:2]
        
        
        depth_prior = transform({'depth_prior': raw_depth_prior})['depth_prior']
        depth_prior = torch.from_numpy(depth_prior).unsqueeze(0) / self.max_depth # dpt outputs depth between 0 and 1

        if depth_prior.ndim == 3: # ensure B, 1, H, W
            depth_prior = depth_prior.unsqueeze(1)
        if depth_prior.shape[1] != 1:
            depth_prior = depth_prior.mean(dim=1, keepdim= True)

        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        depth_prior = depth_prior.to(DEVICE)
        
        return depth_prior, (h, w)




class DepthEmbedding(nn.Module):
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 518, patch_size: Union[int, Tuple[int, int]] = 14, in_chans: int = 1, embed_dim: int = 768):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False
        num_patches = self.patch_embed.num_patches
        self.num_tokens = 1
        self.pos_embeddings_initialized = False
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    

    
    def init_weights(self, pos_embed: torch.Tensor):
        """
        Call after weights of Dino are loaded into model 
        If pos_embeds are already initialized the function does nothing. 
               
        """
        if torch.all(self.pos_embed == 0):
            with torch.no_grad():
                self.pos_embed.copy_(pos_embed)

        self.pos_embeddings_initialized = True


    def forward(self, x, pos_embed):
        if not self.pos_embeddings_initialized:
            self.init_weights(pos_embed)
            

        B, nc, w, h = x.shape
        x = self.patch_embed(x) # don't mask patches because we add positional embeddings so never zero


        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # we throw this away later but need it to match pos embedding dim
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x


class DepthSelfBlock(nn.Module):
    def __init__(self, dim = 768, num_heads = 8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=True)


        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    

class DepthCrossBlock(nn.Module):
    def __init__(self, dim = 768, num_heads = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

        nn.init.zeros_(self.proj.weight) 
        nn.init.zeros_(self.proj.bias) # when attention zero and bias zero all is zero

    def forward(self, rgb, depth):
        out = self.attn(
            self.norm_q(rgb),
            self.norm_kv(depth),
            self.norm_kv(depth),
        )[0]
        return rgb + self.proj(out)




# class DepthPatchEmbed(nn.Module):
#     def __init__(self, patch_size=14, embed_dim=1024):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Linear(1, embed_dim)

#     def forward(self, depth):
#         """
#         depth: [B, 1, H, W]
#         returns:
#           depth_tokens: [B, N, C]
#           depth_mask:   [B, N]  (1 = valid, 0 = missing)
#         """
#         B, _, H, W = depth.shape
#         ps = self.patch_size

#         # patchify
#         depth = depth.unfold(2, ps, ps).unfold(3, ps, ps)
#         # [B, 1, H_p, W_p, ps, ps]

#         # validity mask: at least one valid pixel in patch
#         valid = (depth > 0).float()
#         depth_mask = (valid.sum(dim=(-1, -2)) > 0).float()
#         # [B, 1, H_p, W_p]

#         # aggregate depth per patch (mean of valid pixels)
#         depth_sum = (depth * valid).sum(dim=(-1, -2))
#         depth_cnt = valid.sum(dim=(-1, -2)).clamp(min=1)
#         depth_patch = depth_sum / depth_cnt
#         # [B, 1, H_p, W_p]

#         depth_tokens = depth_patch.flatten(2).transpose(1, 2)
#         depth_mask = depth_mask.flatten(2).squeeze(1)

#         depth_tokens = self.proj(depth_tokens)  # [B, N, C]

#         return depth_tokens, depth_mask
    
#     class DepthSelfAttention(nn.Module):
#         def __init__(self, dim, num_heads=8):
#             super().__init__()
#             self.attn = nn.MultiheadAttention(
#                 dim, num_heads, batch_first=True
#             )
#             self.norm = nn.LayerNorm(dim)

#             # zero-init output projection
#             nn.init.zeros_(self.attn.out_proj.weight)
#             nn.init.zeros_(self.attn.out_proj.bias)

#         def forward(self, x, mask):
#             """
#             x:    [B, N, C]
#             mask: [B, N] (1 = valid)
#             """
#             key_padding_mask = (mask == 0)
#             out, _ = self.attn(
#                 x, x, x,
#                 key_padding_mask=key_padding_mask,
#                 need_weights=False
#             )
#             return self.norm(x + out)
        
#     class DepthCrossAttention(nn.Module):
#         def __init__(self, dim, num_heads=8):
#             super().__init__()
#             self.attn = nn.MultiheadAttention(
#                 dim, num_heads, batch_first=True
#             )
#             self.norm = nn.LayerNorm(dim)

#             # zero-init so initial behavior == original model
#             nn.init.zeros_(self.attn.out_proj.weight)
#             nn.init.zeros_(self.attn.out_proj.bias)

#         def forward(self, rgb_tokens, depth_tokens, depth_mask):
#             """
#             rgb_tokens:   [B, N, C]
#             depth_tokens: [B, N, C]
#             depth_mask:   [B, N]
#             """
#             key_padding_mask = (depth_mask == 0)

#             out, _ = self.attn(
#                 query=rgb_tokens,
#                 key=depth_tokens,
#                 value=depth_tokens,
#                 key_padding_mask=key_padding_mask,
#                 need_weights=False
#             )

#             return self.norm(rgb_tokens + out)
    
#     class DepthConditioner(nn.Module):
#         def __init__(self, embed_dim, patch_size=14, num_heads=8):
#             super().__init__()
#             self.depth_embed = DepthPatchEmbed(patch_size, embed_dim)
#             self.depth_self_attn = DepthSelfAttention(embed_dim, num_heads)
#             self.cross_attn = DepthCrossAttention(embed_dim, num_heads)

#         def forward(self, features, depth):
#             """
#             features: list of tuples from DINOv2
#                     [(patch_tokens, cls_token), ...]
#             depth:    [B, 1, H, W]

#             returns updated features list
#             """
#             depth_tokens, depth_mask = self.depth_embed(depth)

#             # depth self-attention (global propagation)
#             depth_tokens = self.depth_self_attn(depth_tokens, depth_mask)

#             # apply ONLY to deepest layer
#             patch_tokens, cls = features[-1]

#             patch_tokens = self.cross_attn(
#                 patch_tokens,
#                 depth_tokens,
#                 depth_mask
#             )

#             features = list(features)
#             features[-1] = (patch_tokens, cls)

#             return features
        
# features = self.pretrained.get_intermediate_layers(
#     x, self.intermediate_layer_idx[self.encoder],
#     return_class_token=False
#     )

# features = self.depth_conditioner(features, depth_prior)

# depth = self.depth_head(features, patch_h, patch_w)
