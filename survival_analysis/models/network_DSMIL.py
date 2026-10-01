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
    """
    根据输入数据动态计算并设置 OMP_NUM_THREADS。
    主要用于 Windows 系统，避免 KMeans 相关线程问题。
    """
    if not sys.platform.startswith("win"):
        return

    n_samples = X.shape[0]
    n_active_threads = max(1, int(np.ceil(n_samples / CHUNK_SIZE)))
    os.environ["OMP_NUM_THREADS"] = str(n_active_threads)


class SelfAttentionPooling(nn.Module):
    """
    原来的 SelfAttentionPooling。
    现在保留该类，但 PRPH 中不再使用它。
    """
    def __init__(self, input_dim=256):
        super(SelfAttentionPooling, self).__init__()

        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.fc = nn.Linear(input_dim, input_dim)

        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_dim]
        """
        x_proj = self.fc(x)

        attn_scores = torch.matmul(
            x_proj,
            self.query.transpose(-2, -1)
        )

        attn_scores = attn_scores.squeeze(-1)

        attn_weights = F.softmax(attn_scores, dim=-1)

        x_pool = torch.matmul(
            attn_weights.unsqueeze(1),
            x
        ).squeeze(1)

        return x_pool


class DSAttention(nn.Module):
    """
    DSMIL-style Dual-Stream Attention Aggregator.

    参考 DSMIL 的 BClassifier 思路：
    1. 先对每个实例生成 instance-level class score；
    2. 根据每一类得分最高的实例选择 critical instance；
    3. 用所有实例的 query 与 critical instance 的 query 计算 attention；
    4. 用 attention 聚合 value 特征；
    5. 用 Conv1d 得到 bag-level logits。

    输入:
        x: [1, N, D] 或 [N, D]

    输出:
        logits: [1, n_classes]
    """

    def __init__(
        self,
        input_size,
        output_class,
        dropout_v=0.0,
        nonlinear=True,
        passing_v=False
    ):
        super(DSAttention, self).__init__()

        self.input_size = input_size
        self.output_class = output_class

        # instance-level classifier
        # 对每个实例产生 n_classes 个分数
        self.instance_classifier = nn.Linear(input_size, output_class)

        # Query branch
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)

        # Value branch
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # DSMIL bag classifier
        # 输入: [1, output_class, input_size]
        # 输出: [1, output_class, 1]
        self.fcc = nn.Conv1d(
            output_class,
            output_class,
            kernel_size=input_size
        )

        nn.init.xavier_normal_(self.fcc.weight)
        if self.fcc.bias is not None:
            self.fcc.bias.data.zero_()

    def forward(self, x, return_attn=False):
        """
        x:
            [1, N, D] 或 [N, D]

        return:
            logits: [1, output_class]
        """

        if x.dim() == 3:
            feats = x.squeeze(0)
        elif x.dim() == 2:
            feats = x
        else:
            raise ValueError(
                f"DSAttention expects input shape [1, N, D] or [N, D], but got {x.shape}"
            )

        device = feats.device

        # --------------------------------------------------
        # 1. Instance-level prediction
        # --------------------------------------------------
        classes = self.instance_classifier(feats)  # [N, C]

        # --------------------------------------------------
        # 2. Query / Value
        # --------------------------------------------------
        V = self.v(feats)  # [N, D]
        Q = self.q(feats).view(feats.shape[0], -1)  # [N, 128]

        # --------------------------------------------------
        # 3. 对每个类别选择得分最高的 critical instance
        # --------------------------------------------------
        _, m_indices = torch.sort(
            classes,
            dim=0,
            descending=True
        )  # [N, C]

        m_feats = torch.index_select(
            feats,
            dim=0,
            index=m_indices[0, :]
        )  # [C, D]

        q_max = self.q(m_feats)  # [C, 128]

        # --------------------------------------------------
        # 4. 计算 DSMIL attention
        # --------------------------------------------------
        A = torch.mm(
            Q,
            q_max.transpose(0, 1)
        )  # [N, C]

        scale = torch.sqrt(
            torch.tensor(
                Q.shape[1],
                dtype=torch.float32,
                device=device
            )
        )

        A = F.softmax(A / scale, dim=0)  # [N, C]

        # --------------------------------------------------
        # 5. 聚合得到 bag representation
        # --------------------------------------------------
        B = torch.mm(
            A.transpose(0, 1),
            V
        )  # [C, D]

        B = B.view(
            1,
            B.shape[0],
            B.shape[1]
        )  # [1, C, D]

        # --------------------------------------------------
        # 6. Bag-level prediction
        # --------------------------------------------------
        logits = self.fcc(B)  # [1, C, 1]
        logits = logits.view(1, -1)  # [1, C]

        if return_attn:
            return logits, A, B, classes

        return logits


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

        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, feature_dim)
        )

        self.W1 = nn.Parameter(
            torch.randn(feature_dim, embedding_dim)
        )

        self.W2 = nn.Parameter(
            torch.randn(feature_dim, embedding_dim)
        )

        if feature_dim != embedding_dim:
            self.feature_projection = nn.Linear(
                feature_dim,
                embedding_dim
            )
        else:
            self.feature_projection = nn.Identity()

        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)

    def forward(self, x):
        """
        x: [1, num_instances, feature_dim]
        """

        if x.size(0) != 1:
            raise ValueError("Batch size must be 1")

        features = x.squeeze(0)

        prototypes_proj = torch.matmul(
            self.prototypes,
            self.W1
        )

        features_proj = torch.matmul(
            features,
            self.W2
        )

        prototypes_norm = F.normalize(
            prototypes_proj,
            p=2,
            dim=1
        )

        features_norm = F.normalize(
            features_proj,
            p=2,
            dim=1
        )

        similarity_matrix = torch.matmul(
            features_norm,
            prototypes_norm.T
        ) / self.temperature

        attention_weights = F.softmax(
            similarity_matrix,
            dim=1
        )

        semi_global_features = torch.matmul(
            attention_weights,
            self.prototypes
        )

        local_features_projected = self.feature_projection(features)

        guided_embedding = local_features_projected + semi_global_features

        return guided_embedding.unsqueeze(0)


class AttentionBackbone(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBackbone, self).__init__()

        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x: [1, N, D]

        return:
            weights: [N]
        """

        h = torch.tanh(self.w1(x))

        scores = self.w2(h)

        weights = F.softmax(scores, dim=1)

        return weights.squeeze(0).squeeze(-1)


def run_kmeans(features, k=2):
    """
    Wrapper for scikit-learn KMeans.

    features: PyTorch Tensor [N, D]
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

    kmeans = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    ).fit(features_np)

    labels = torch.from_numpy(
        kmeans.labels_
    ).to(features.device)

    centroids = torch.from_numpy(
        kmeans.cluster_centers_
    ).to(features.device)

    counts = torch.zeros(
        k,
        dtype=torch.long,
        device=features.device
    )

    for i in range(k):
        counts[i] = (labels == i).sum()

    return labels, centroids, counts


class PRPH(nn.Module):
    def __init__(self, n_classes=4, model_size="small", input_dim=1024):
        super(PRPH, self).__init__()

        self.n_classes = n_classes

        self.size_dict = {
            "pathomics": {
                "small": [input_dim, 256, 256],
                "large": [input_dim, 512, 256]
            }
        }

        hidden = self.size_dict["pathomics"][model_size]

        # --------------------------------------------------
        # 1. Pathomics feature embedding
        # --------------------------------------------------
        fc = []

        for idx in range(len(hidden) - 1):
            fc.append(
                nn.Linear(
                    hidden[idx],
                    hidden[idx + 1]
                )
            )

            fc.append(nn.ReLU())

            fc.append(nn.Dropout(0.25))

        self.pathomics_fc = nn.Sequential(*fc)

        # --------------------------------------------------
        # 2. Stage 1: Prototype similarity global fusion
        # --------------------------------------------------
        self.proto_clustering = PrototypeSimilarityClustering(
            num_prototypes=6,
            feature_dim=hidden[-1],
            embedding_dim=hidden[-1]
        )

        # --------------------------------------------------
        # 3. Stage 2: Hard instance mining attention
        # --------------------------------------------------
        self.attention_net = AttentionBackbone(hidden[-1])

        # --------------------------------------------------
        # 4. Final pooling
        #
        # 原来:
        #   self.final_pooling = SelfAttentionPooling(input_dim=hidden[-1])
        #   self.classifier = nn.Linear(hidden[-1], self.n_classes)
        #
        # 现在:
        #   使用 DSMIL-style DSAttention。
        #   DSAttention 内部已经包含 instance classifier 和 bag classifier，
        #   因此不再需要额外的 self.classifier。
        # --------------------------------------------------
        self.final_pooling = DSAttention(
            input_size=hidden[-1],
            output_class=self.n_classes,
            dropout_v=0.0,
            nonlinear=True,
            passing_v=False
        )

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]

        # --------------------------------------------------
        # Stage 1: feature extraction and global fusion
        # --------------------------------------------------
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        fused_features = self.proto_clustering(pathomics_features)

        # --------------------------------------------------
        # Stage 2: hard instance mining
        # --------------------------------------------------
        features_flat = fused_features.squeeze(0)

        N = features_flat.size(0)

        if N < 20:
            all_hard_instances = features_flat

        else:
            # 1. attention scores for hard instance mining
            attn_scores = self.attention_net(fused_features)

            sorted_scores, sorted_indices = torch.sort(
                attn_scores,
                descending=True
            )

            # 这里保持你当前代码中的比例：0.5%
            n_top = max(1, int(N * 0.01))
            n_bot = max(1, int(N * 0.01))

            if n_top + n_bot >= N:
                n_top = N // 2
                n_bot = N - n_top

            top_indices = sorted_indices[:n_top]
            bottom_indices = sorted_indices[-n_bot:]
            middle_indices = sorted_indices[n_top: N - n_bot]

            hard_instances_list = []

            # --------------------------------------------------
            # 2. top instances clustering
            # --------------------------------------------------
            if len(top_indices) > 0:
                top_features = features_flat[top_indices]

                labels_top, centroids_top, counts_top = run_kmeans(
                    top_features,
                    k=2
                )

                if counts_top[0] >= counts_top[1]:
                    pos_prototype = centroids_top[0]
                    hard_mask = labels_top == 1
                else:
                    pos_prototype = centroids_top[1]
                    hard_mask = labels_top == 0

                if hard_mask.sum() > 0:
                    hard_instances_list.append(
                        top_features[hard_mask]
                    )

            else:
                pos_prototype = features_flat.mean(dim=0)

            # --------------------------------------------------
            # 3. bottom instances clustering
            # --------------------------------------------------
            if len(bottom_indices) > 0:
                bottom_features = features_flat[bottom_indices]

                labels_bot, centroids_bot, counts_bot = run_kmeans(
                    bottom_features,
                    k=2
                )

                if counts_bot[0] >= counts_bot[1]:
                    neg_prototype = centroids_bot[0]
                    hard_mask = labels_bot == 1
                else:
                    neg_prototype = centroids_bot[1]
                    hard_mask = labels_bot == 0

                if hard_mask.sum() > 0:
                    hard_instances_list.append(
                        bottom_features[hard_mask]
                    )

            else:
                neg_prototype = features_flat.mean(dim=0)

            # --------------------------------------------------
            # 4. middle instances selection
            # --------------------------------------------------
            if len(middle_indices) > 0:
                middle_features = features_flat[middle_indices]

                mid_norm = F.normalize(
                    middle_features,
                    p=2,
                    dim=1
                )

                pos_proto_norm = F.normalize(
                    pos_prototype.unsqueeze(0),
                    p=2,
                    dim=1
                )

                neg_proto_norm = F.normalize(
                    neg_prototype.unsqueeze(0),
                    p=2,
                    dim=1
                )

                v_p = torch.matmul(
                    mid_norm,
                    pos_proto_norm.T
                ).squeeze(1)

                v_n = torch.matmul(
                    mid_norm,
                    neg_proto_norm.T
                ).squeeze(1)

                hard_mask_mid = v_p > v_n

                if hard_mask_mid.sum() > 0:
                    hard_instances_list.append(
                        middle_features[hard_mask_mid]
                    )

            # --------------------------------------------------
            # 5. merge hard instances
            # --------------------------------------------------
            if len(hard_instances_list) > 0:
                all_hard_instances = torch.cat(
                    hard_instances_list,
                    dim=0
                )
            else:
                if len(top_indices) > 0:
                    all_hard_instances = features_flat[top_indices]
                else:
                    all_hard_instances = features_flat

        # --------------------------------------------------
        # Final prediction with DSMIL-style DSAttention
        # --------------------------------------------------
        hard_instances_batch = all_hard_instances.unsqueeze(0)

        logits = self.final_pooling(hard_instances_batch)

        hazards = torch.sigmoid(logits)

        S = torch.cumprod(
            1 - hazards,
            dim=1
        )

        return hazards, S
