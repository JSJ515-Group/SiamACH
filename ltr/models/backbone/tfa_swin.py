import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights
from .module import (
    LayerNorm,
    TriContributionAwareAggregation,
    PairConfidenceAwareAggregation,
)


# =========================================================
class TriComponentModalFusion(nn.Module):

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        spatial_kernel: int = 1,
    ):
        super().__init__()
        C = channels
        self.C = C
        self.eps = eps

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                2 * C, 2 * C,
                kernel_size=1,
                stride=1,
                groups=2 * C,
                bias=False
            ),
            nn.BatchNorm2d(2 * C),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                2 * C, 2 * C,
                kernel_size=1,
                stride=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        assert spatial_kernel in (1, 3), "spatial_kernel 通常选 1 或 3"
        padding = 0 if spatial_kernel == 1 else 1
        self.spatial_rel = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=spatial_kernel,
            stride=1,
            padding=padding,
            bias=True
        )

        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_rgb: torch.Tensor, F_tir: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_rgb.shape
        assert C == self.C and F_tir.shape == F_rgb.shape

        #
        C_rgbt = torch.cat([F_rgb, F_tir], dim=1)  # [B, 2C, H, W]

        # 2) 通道权重
        W_all = self.channel_attn(C_rgbt)
        Wc_rgb, Wc_tir = torch.chunk(W_all, 2, dim=1)

        # 3) 空间权重
        M_avg = C_rgbt.mean(dim=1, keepdim=True)
        M_max = C_rgbt.max(dim=1, keepdim=True).values
        M = torch.cat([M_avg, M_max], dim=1)

        r_logits = self.spatial_rel(M)
        r_prob = torch.softmax(r_logits, dim=1)

        r_rgb = r_prob[:, 0:1, :, :]
        r_tir = r_prob[:, 1:2, :, :]

        # 4) 双模态门控
        R_rgb = Wc_rgb * r_rgb
        R_tir = Wc_tir * r_tir

        # 5) 归一化
        R_sum = R_rgb + R_tir + self.eps
        R_rgb = R_rgb / R_sum
        R_tir = R_tir / R_sum

        # 6) 门控融合
        F_fusion = R_rgb * F_rgb + R_tir * F_tir

        # 7) 残差增强
        F_catrgb = torch.cat([F_fusion, F_rgb], dim=1)
        F_cattir = torch.cat([F_fusion, F_tir], dim=1)

        Funsion_rgb = self.fuse_conv1(F_catrgb)
        Funsion_tir = self.fuse_conv2(F_cattir)

        RGB_funsion = Funsion_rgb + F_rgb
        tir_funsion = Funsion_tir + F_tir

        out = RGB_funsion + tir_funsion
        return out


# =========================================================
# =========================================================
class StageAlign(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(StageAlign, self).__init__()
        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, out_size):
        x = self.align(x)
        x = F.adaptive_avg_pool2d(x, out_size)
        return x


class JcfaMoudle(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(JcfaMoudle, self).__init__()

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=stride, stride=stride)
        )

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.tcmf = TriComponentModalFusion(out_channels)

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.pair_caa = PairConfidenceAwareAggregation(
            channels=out_channels,
            reduction=8
        )

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.norm_last_fus = LayerNorm(
            out_channels,
            eps=1e-6,
            data_format="channels_first"
        )

        self.norm_new_fus = LayerNorm(
            out_channels,
            eps=1e-6,
            data_format="channels_first"
        )

        self.norm_out = LayerNorm(
            out_channels,
            eps=1e-6,
            data_format="channels_first"
        )

        self.norm_extra = LayerNorm(
            out_channels,
            eps=1e-6,
            data_format="channels_first"
        )


    def forward(self, rgb_feat, aux_feat, fus_feat, extra_feat=None):

        # ===========================================================
        new_fus_feat = self.tcmf(rgb_feat, aux_feat)
        new_fus_feat = self.norm_new_fus(new_fus_feat)

        # ===========================================================
        # ===========================================================
        last_fus_feat = self.conv1(fus_feat)

        # ===========================================================
        # ===========================================================
        last_fus_feat = self.norm_last_fus(last_fus_feat)

        # ===========================================================
        # ===========================================================
        out = self.pair_caa(new_fus_feat, last_fus_feat)

        out = self.norm_out(out)

        # ===========================================================
        # ===========================================================
        if extra_feat is not None:
            if extra_feat.shape != out.shape:
                raise ValueError(
                    f"extra_feat shape must match out shape, but got "
                    f"extra_feat={extra_feat.shape}, out={out.shape}"
                )
            out = out + extra_feat
            out = self.norm_extra(out)

        return out


class TfaSwinBackbone(nn.Module):
    def __init__(self, weights=None):
        super(TfaSwinBackbone, self).__init__()

        feature_extractor_rgb = swin_b(weights=weights).features
        self.rgb_stage_list = nn.ModuleList(
            [feature_extractor_rgb[i: i + 2] for i in range(0, 8, 2)]
        )

        feature_extractor_aux = swin_b(weights=weights).features
        self.aux_stage_list = nn.ModuleList(
            [feature_extractor_aux[i: i + 2] for i in range(0, 8, 2)]
        )

        in_channels_list = [6, 128, 256, 512]
        out_channels_list = [128, 256, 512, 1024]
        stride_list = [4, 2, 2, 2]
        self.fus_stage_list = nn.ModuleList([
            JcfaMoudle(in_channels, out_channels, stride)
            for in_channels, out_channels, stride
            in zip(in_channels_list, out_channels_list, stride_list)
        ])

        # =========================================================
        # =========================================================
        self.rgb_align_stage1 = StageAlign(256, 1024)
        self.rgb_align_stage2 = StageAlign(512, 1024)

        self.aux_align_stage1 = StageAlign(256, 1024)
        self.aux_align_stage2 = StageAlign(512, 1024)

        # =========================================================
        # =========================================================
        self.rgb_s1_norm = LayerNorm(1024, data_format="channels_first")
        self.rgb_s2_norm = LayerNorm(1024, data_format="channels_first")
        self.rgb_s3_norm = LayerNorm(1024, data_format="channels_first")

        self.aux_s1_norm = LayerNorm(1024, data_format="channels_first")
        self.aux_s2_norm = LayerNorm(1024, data_format="channels_first")
        self.aux_s3_norm = LayerNorm(1024, data_format="channels_first")

        # =========================================================
        self.rgb_tricca = TriContributionAwareAggregation(channels=1024)
        self.aux_tricca = TriContributionAwareAggregation(channels=1024)

        # =========================================================
        # =========================================================
        self.rgbt_com_norm = LayerNorm(1024, data_format="channels_first")

    def forward(self, rgb_img, aux_img):
        rgb_feat = rgb_img
        aux_feat = aux_img
        fus_feat = torch.cat((rgb_img, aux_img), dim=1)  # (N, C, H, W)

        # =========================================================
        # =========================================================
        rgb_stage_feats = []
        aux_stage_feats = []

        for i in range(len(self.fus_stage_list)):
            rgb_feat = self.rgb_stage_list[i](rgb_feat)   # (N, H, W, C)
            aux_feat = self.aux_stage_list[i](aux_feat)   # (N, H, W, C)

            rgb_feat_nchw = rgb_feat.permute(0, 3, 1, 2)  # (N, C, H, W)
            aux_feat_nchw = aux_feat.permute(0, 3, 1, 2)  # (N, C, H, W)

            rgb_stage_feats.append(rgb_feat_nchw)
            aux_stage_feats.append(aux_feat_nchw)

            # =====================================================
            # =====================================================
            extra_feat = None
            if i == len(self.fus_stage_list) - 1:
                target_size = rgb_stage_feats[3].shape[-2:]

                # ---------------- ----------------
                rgb_s1 = self.rgb_s1_norm(self.rgb_align_stage1(rgb_stage_feats[1], target_size))
                rgb_s2 = self.rgb_s2_norm(self.rgb_align_stage2(rgb_stage_feats[2], target_size))
                rgb_s3 = self.rgb_s3_norm(rgb_stage_feats[3])

                rgb_w = self.rgb_tricca(rgb_s1, rgb_s2, rgb_s3)

                # ---------------- ----------------
                aux_s1 = self.aux_s1_norm(self.aux_align_stage1(aux_stage_feats[1], target_size))
                aux_s2 = self.aux_s2_norm(self.aux_align_stage2(aux_stage_feats[2], target_size))
                aux_s3 = self.aux_s3_norm(aux_stage_feats[3])

                tir_w = self.aux_tricca(aux_s1, aux_s2, aux_s3)

                # ---------------- ----------------
                rgbt_com = self.rgbt_com_norm(rgb_w + tir_w)

                extra_feat = rgbt_com

            fus_feat = self.fus_stage_list[i](
                rgb_feat_nchw,
                aux_feat_nchw,
                fus_feat,
                extra_feat
            )  # input and output: (N, C, H, W)

        return fus_feat


def tfa_swin_backbone(weights=Swin_B_Weights.IMAGENET1K_V1):
    return TfaSwinBackbone(weights)