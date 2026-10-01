import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .util import initialize_weights
import os
import sys
from sklearn.cluster._k_means_common import CHUNK_SIZE


def set_dynamic_omp_threads(X):
    """根据输入数据动态计算并设置OMP_NUM_THREADS"""
    if not sys.platform.startswith("win"):
        return

    n_samples = X.shape[0]
    n_active_threads = max(1, int(np.ceil(n_samples / CHUNK_SIZE)))
    os.environ["OMP_NUM_THREADS"] = str(n_active_threads)


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=256):
        super(SelfAttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.fc = nn.Linear(input_dim, input_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_proj = self.fc(x)
        attn_scores = torch.matmul(x_proj, self.query.transpose(-2, -1))
        attn_scores = attn_scores.squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        x_pool = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)
        return x_pool


class PrototypeSimilarityClustering(nn.Module):
    """
    Stage 1: Global Feature Fusion
    """

    def __init__(self, num_prototypes, feature_dim, embedding_dim, temperature=0.1):
        super(PrototypeSimilarityClustering, self).__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.W1 = nn.Parameter(torch.randn(feature_dim, embedding_dim))
        self.W2 = nn.Parameter(torch.randn(feature_dim, embedding_dim))

        if feature_dim != embedding_dim:
            self.feature_projection = nn.Linear(feature_dim, embedding_dim)
        else:
            self.feature_projection = nn.Identity()

        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)

    def forward(self, x):
        # x: [1, num_instances, feature_dim]
        if x.size(0) != 1:
            raise ValueError("Batch size must be 1")

        features = x.squeeze(0)

        prototypes_proj = torch.matmul(self.prototypes, self.W1)
        features_proj = torch.matmul(features, self.W2)

        prototypes_norm = F.normalize(prototypes_proj, p=2, dim=1)
        features_norm = F.normalize(features_proj, p=2, dim=1)

        similarity_matrix = torch.matmul(features_norm, prototypes_norm.T) / self.temperature

        attention_weights = F.softmax(similarity_matrix, dim=1)

        semi_global_features = torch.matmul(attention_weights, self.prototypes)

        local_features_projected = self.feature_projection(features)

        guided_embedding = local_features_projected + semi_global_features

        return guided_embedding.unsqueeze(0)


class AttentionBackbone(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBackbone, self).__init__()
        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [1, N, D]
        h = torch.tanh(self.w1(x))
        scores = self.w2(h)
        weights = F.softmax(scores, dim=1)
        return weights.squeeze(0).squeeze(-1)


def run_kmeans(features, k=2):
    """
    Wrapper for scikit-learn's KMeans.
    """
    set_dynamic_omp_threads(features)

    N = features.size(0)

    if N < k:
        return (
            torch.zeros(N, dtype=torch.long, device=features.device),
            features.mean(dim=0, keepdim=True).repeat(k, 1),
            torch.tensor([N, 0], device=features.device)
        )

    features_np = features.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(features_np)

    labels = torch.from_numpy(kmeans.labels_).to(features.device)
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(features.device)

    counts = torch.zeros(k, dtype=torch.long, device=features.device)
    for i in range(k):
        counts[i] = (labels == i).sum()

    return labels, centroids, counts


class PRPH(nn.Module):
    def __init__(
        self,
        n_classes=2,
        model_size="small",
        input_dim=1024,
        top_major_proto_ratio=0.1,
        top_major_proto_min=1
    ):
        super(PRPH, self).__init__()

        self.n_classes = n_classes

        # 高注意力区域中，额外从大簇中选取靠近大类原型的patch
        # top_major_proto_ratio=0.5 表示额外选取数量 = 小簇数量 * 0.5
        # top_major_proto_min=1 表示至少额外选1个
        self.top_major_proto_ratio = top_major_proto_ratio
        self.top_major_proto_min = top_major_proto_min

        self.size_dict = {
            "pathomics": {
                "small": [input_dim, 256, 256],
                "large": [input_dim, 512, 256]
            }
        }

        hidden = self.size_dict["pathomics"][model_size]

        # 1. 病理组学嵌入网络
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        # 2. 第一阶段：可学习原型聚类与特征融合
        self.proto_clustering = PrototypeSimilarityClustering(
            num_prototypes=12,
            feature_dim=hidden[-1],
            embedding_dim=hidden[-1]
        )

        # 3. 第二阶段：硬实例挖掘注意力网络
        self.attention_net = AttentionBackbone(hidden[-1])

        # 4. 最终池化与分类器
        self.final_pooling = SelfAttentionPooling(input_dim=hidden[-1])
        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]

        # --- 第一阶段：特征提取与全局融合 ---
        # [N, 1024] -> [N, 256] -> [1, N, 256]
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # [1, N, 256] -> [1, N, 256]
        fused_features = self.proto_clustering(pathomics_features)

        # --- 第二阶段：困难实例挖掘 ---
        features_flat = fused_features.squeeze(0)
        N = features_flat.size(0)

        if N < 20:
            all_hard_instances = features_flat

        else:
            # 1. 计算注意力分数
            attn_scores = self.attention_net(fused_features)

            # 按注意力从高到低排序
            sorted_scores, sorted_indices = torch.sort(attn_scores, descending=True)

            # 划分高注意力、低注意力、中间注意力patch
            n_top = max(1, int(N * 0.02))
            n_bot = max(1, int(N * 0.02))

            if n_top + n_bot >= N:
                n_top = N // 2
                n_bot = N - n_top

            top_indices = sorted_indices[:n_top]
            bottom_indices = sorted_indices[-n_bot:]
            middle_indices = sorted_indices[n_top: N - n_bot]

            hard_instances_list = []

            # ============================================================
            # 2. 高注意力部分困难实例挖掘
            # 原逻辑：
            #   top高注意力patch -> KMeans分成2类 -> 只取小簇
            #
            # 新逻辑：
            #   top高注意力patch -> KMeans分成2类
            #   hard instance = 小簇patch + 大簇中最靠近大类原型的一定数量patch
            # ============================================================
            if len(top_indices) > 0:
                top_features = features_flat[top_indices]

                labels_top, centroids_top, counts_top = run_kmeans(top_features, k=2)

                # 大簇：样本数较多的簇
                # 小簇：样本数较少的簇
                if counts_top[0] >= counts_top[1]:
                    major_cluster_id = 0
                    minor_cluster_id = 1
                else:
                    major_cluster_id = 1
                    minor_cluster_id = 0

                # 大类原型：大簇中心
                pos_prototype = centroids_top[major_cluster_id]

                major_mask = labels_top == major_cluster_id
                minor_mask = labels_top == minor_cluster_id

                selected_top_hard_list = []

                # 2.1 小簇patch全部加入hard instance
                if minor_mask.sum() > 0:
                    minor_features = top_features[minor_mask]
                    selected_top_hard_list.append(minor_features)

                # 2.2 大簇中选取最靠近大类原型的一定数量patch
                if major_mask.sum() > 0:
                    major_features = top_features[major_mask]

                    # 使用欧氏距离，因为KMeans本身也是基于欧氏距离
                    distances = torch.norm(
                        major_features - pos_prototype.unsqueeze(0),
                        p=2,
                        dim=1
                    )

                    num_minor = int(minor_mask.sum().item())

                    if num_minor > 0:
                        n_major_select = max(
                            self.top_major_proto_min,
                            int(num_minor * self.top_major_proto_ratio)
                        )
                    else:
                        n_major_select = self.top_major_proto_min

                    n_major_select = min(n_major_select, major_features.size(0))

                    if n_major_select > 0:
                        _, nearest_order = torch.sort(distances, descending=False)
                        nearest_indices = nearest_order[:n_major_select]
                        nearest_major_features = major_features[nearest_indices]
                        selected_top_hard_list.append(nearest_major_features)

                # 2.3 合并高注意力部分的hard instances
                if len(selected_top_hard_list) > 0:
                    top_hard_instances = torch.cat(selected_top_hard_list, dim=0)
                    hard_instances_list.append(top_hard_instances)

            else:
                pos_prototype = features_flat.mean(dim=0)

            # ============================================================
            # 3. 低注意力部分困难实例挖掘
            # 这一部分保持你原来的逻辑不变
            # ============================================================
            if len(bottom_indices) > 0:
                bottom_features = features_flat[bottom_indices]

                labels_bot, centroids_bot, counts_bot = run_kmeans(bottom_features, k=2)

                if counts_bot[0] >= counts_bot[1]:
                    neg_prototype = centroids_bot[0]
                    hard_mask = labels_bot == 1
                else:
                    neg_prototype = centroids_bot[1]
                    hard_mask = labels_bot == 0

                if hard_mask.sum() > 0:
                    hard_instances_list.append(bottom_features[hard_mask])

            else:
                neg_prototype = features_flat.mean(dim=0)

            # ============================================================
            # 4. 中间注意力部分困难实例挖掘
            # 这一部分保持你原来的逻辑不变
            # ============================================================
            if len(middle_indices) > 0:
                middle_features = features_flat[middle_indices]

                mid_norm = F.normalize(middle_features, p=2, dim=1)
                pos_proto_norm = F.normalize(pos_prototype.unsqueeze(0), p=2, dim=1)
                neg_proto_norm = F.normalize(neg_prototype.unsqueeze(0), p=2, dim=1)

                v_p = torch.round(
                    torch.matmul(mid_norm, pos_proto_norm.T).squeeze(1) * 100
                ) / 100

                v_n = torch.round(
                    torch.matmul(mid_norm, neg_proto_norm.T).squeeze(1) * 100
                ) / 100

                hard_mask_mid = v_p >= v_n

                if hard_mask_mid.sum() > 0:
                    hard_instances_list.append(middle_features[hard_mask_mid])

            # ============================================================
            # 5. 合并所有困难实例
            # ============================================================
            if len(hard_instances_list) > 0:
                all_hard_instances = torch.cat(hard_instances_list, dim=0)
            else:
                if len(top_indices) > 0:
                    all_hard_instances = features_flat[top_indices]
                else:
                    all_hard_instances = features_flat

        # --- 聚合与预测 ---
        hard_instances_batch = all_hard_instances.unsqueeze(0)

        path_fusion = self.final_pooling(hard_instances_batch)

        logits = self.classifier(path_fusion)

        return logits
