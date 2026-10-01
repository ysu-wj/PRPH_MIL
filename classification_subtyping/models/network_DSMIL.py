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
    现在保留该类，方便后续做消融实验；
    但当前 PRPH 中不再使用它。
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


class BClassifier(nn.Module):
    """
    DSMIL 的 bag-level classifier。

    输入:
        feats: [N, D]
        c:     [N, C]，instance-level class logits

    输出:
        C: [1, C]，bag-level logits
        A: [N, C]，类别特异性注意力
        B: [1, C, D]，类别特异性 bag representation
    """

    def __init__(
        self,
        input_size,
        output_class,
        dropout_v=0.0,
        nonlinear=True,
        passing_v=True
    ):
        super(BClassifier, self).__init__()

        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # Conv1d 的 kernel_size 覆盖完整特征维度 D。
        # 输入 B: [1, C, D]
        # 输出 C: [1, C, 1]
        self.fcc = nn.Conv1d(
            output_class,
            output_class,
            kernel_size=input_size
        )

    def forward(self, feats, c, no_norm=False):
        """
        feats: [N, D]
        c:     [N, C]
        """

        device = feats.device

        # Value / Query branches.
        V = self.v(feats)                              # [N, D]
        Q = self.q(feats).view(feats.shape[0], -1)     # [N, 128]

        # 每个类别选取得分最高的 critical instance。
        _, m_indices = torch.sort(
            c,
            dim=0,
            descending=True
        )                                              # [N, C]

        m_feats = torch.index_select(
            feats,
            dim=0,
            index=m_indices[0, :]
        )                                              # [C, D]

        q_max = self.q(m_feats)                        # [C, 128]

        # DSMIL dual-stream attention.
        A = torch.mm(
            Q,
            q_max.transpose(0, 1)
        )                                              # [N, C]

        A = A / torch.sqrt(
            torch.tensor(
                Q.shape[1],
                dtype=torch.float32,
                device=device
            )
        )

        if no_norm:
            _A = A

        A = F.softmax(A, dim=0)

        if not no_norm:
            _A = A

        # Class-specific bag representations.
        B = torch.mm(
            A.transpose(0, 1),
            V
        )                                              # [C, D]

        B = B.view(
            1,
            B.shape[0],
            B.shape[1]
        )                                              # [1, C, D]

        C = self.fcc(B)                                # [1, C, 1]
        C = C.view(1, -1)                              # [1, C]

        return C, _A, B


class DSMIL(nn.Module):
    """
    用于分类任务的 DSMIL 聚合器。

    默认返回:
        logits_pair = [bag_logits, instance_logits]
        B           = [1, n_classes, feature_dim]

    return_attn=True 时返回:
        logits_pair, B, attn
    """

    def __init__(
        self,
        n_classes=2,
        mlp_dim=256,
        cls_attn=True,
        attn_index="max"
    ):
        super(DSMIL, self).__init__()

        # Instance-level classifier.
        self.i_classifier = nn.Linear(
            mlp_dim,
            n_classes
        )

        # Bag-level classifier.
        self.b_classifier = BClassifier(
            input_size=mlp_dim,
            output_class=n_classes
        )

        self.cls_attn = cls_attn
        self.attn_index = attn_index

    def attention(
        self,
        x,
        no_norm=False,
        return_attn=False,
        return_cam=False,
        **kwargs
    ):
        """
        x: [1, N, D]
        """

        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(
                "DSMIL expects input shape [1, N, D], "
                f"but got {tuple(x.shape)}"
            )

        feats = x.squeeze(0)                           # [N, D]

        classes = self.i_classifier(feats)             # [N, C]

        prediction_bag, A, B = self.b_classifier(
            feats,
            classes,
            no_norm=no_norm
        )

        # Instance stream:
        # 对每个类别取所有 instance 中的最大分类分数。
        classes_bag, _ = torch.max(
            classes,
            dim=0
        )                                              # [C]

        if return_attn:
            if self.attn_index == "max":
                if self.cls_attn:
                    attn, _ = torch.max(classes, dim=-1)
                else:
                    attn, _ = torch.max(A, dim=-1)
            else:
                index = int(self.attn_index)
                if self.cls_attn:
                    attn = classes[:, index]
                else:
                    attn = A[:, index]

            attn = attn.unsqueeze(0)                   # [1, N]

            if return_cam:
                cam = classes
                attn = [attn, cam.unsqueeze(0)]
        else:
            attn = None

        return (
            prediction_bag,                            # [1, C]
            classes_bag.unsqueeze(0),                  # [1, C]
            attn,
            B
        )

    def forward(
        self,
        x,
        return_attn=False,
        no_norm=False,
        **kwargs
    ):
        logits, logits_ins, attn, B = self.attention(
            x,
            no_norm=no_norm,
            return_attn=return_attn,
            **kwargs
        )

        if return_attn:
            return [logits, logits_ins], B, attn

        return [logits, logits_ins], B


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

        features = x.squeeze(0)                        # [N, D]

        prototypes_proj = torch.matmul(
            self.prototypes,
            self.W1
        )                                              # [P, E]

        features_proj = torch.matmul(
            features,
            self.W2
        )                                              # [N, E]

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
        ) / self.temperature                           # [N, P]

        attention_weights = F.softmax(
            similarity_matrix,
            dim=1
        )                                              # [N, P]

        semi_global_features = torch.matmul(
            attention_weights,
            self.prototypes
        )                                              # [N, D]

        local_features_projected = self.feature_projection(features)

        guided_embedding = local_features_projected + semi_global_features

        return guided_embedding.unsqueeze(0)            # [1, N, D]


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
            torch.tensor([N, 0], dtype=torch.long, device=features.device)
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
    ).to(features.device).type_as(features)

    counts = torch.zeros(
        k,
        dtype=torch.long,
        device=features.device
    )

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

        # 高注意力区域中，额外从大簇中选取靠近大类原型的 patch。
        # top_major_proto_ratio=0.5 表示额外选取数量 = 小簇数量 * 0.5。
        # top_major_proto_min=1 表示至少额外选 1 个。
        self.top_major_proto_ratio = top_major_proto_ratio
        self.top_major_proto_min = top_major_proto_min

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
        #   使用 DSMIL 聚合器。
        #   DSMIL 内部已经包含 instance classifier 和 bag classifier，
        #   因此不再需要额外的 self.classifier。
        # --------------------------------------------------
        self.final_pooling = DSMIL(
            n_classes=self.n_classes,
            mlp_dim=hidden[-1],
            cls_attn=True,
            attn_index="max"
        )

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]

        # --------------------------------------------------
        # 兼容 DataLoader 输出:
        #   batch_size=1 时，x_path 可能是 [1, N, D]
        #   直接调用模型时，x_path 可能是 [N, D]
        #
        # 后续统一转成 [N, D]。
        # --------------------------------------------------
        if x_path.dim() == 3 and x_path.size(0) == 1:
            x_path = x_path.squeeze(0)
        elif x_path.dim() != 2:
            raise ValueError(
                "PRPH expects x_path shape [N, D] or [1, N, D], "
                f"but got {tuple(x_path.shape)}"
            )

        # --------------------------------------------------
        # Stage 1: feature extraction and global fusion
        # --------------------------------------------------
        # [N, input_dim] -> [N, 256] -> [1, N, 256]
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # [1, N, 256] -> [1, N, 256]
        fused_features = self.proto_clustering(pathomics_features)

        # --------------------------------------------------
        # Stage 2: hard instance mining
        # --------------------------------------------------
        features_flat = fused_features.squeeze(0)       # [N, 256]

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

            # 高注意力 patch 和低注意力 patch 的比例。
            n_top = max(1, int(N * 0.05))
            n_bot = max(1, int(N * 0.05))

            if n_top + n_bot >= N:
                n_top = N // 2
                n_bot = N - n_top

            top_indices = sorted_indices[:n_top]
            bottom_indices = sorted_indices[-n_bot:]
            middle_indices = sorted_indices[n_top: N - n_bot]

            hard_instances_list = []

            # ============================================================
            # 2. 高注意力部分困难实例挖掘
            #
            # top 高注意力 patch -> KMeans 分成 2 类
            # hard instances = 小簇 patch + 大簇中最靠近大类原型的一定数量 patch
            # ============================================================
            if len(top_indices) > 0:
                top_features = features_flat[top_indices]

                labels_top, centroids_top, counts_top = run_kmeans(
                    top_features,
                    k=2
                )

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

                # 2.1 小簇 patch 全部加入 hard instances
                if minor_mask.sum() > 0:
                    minor_features = top_features[minor_mask]
                    selected_top_hard_list.append(minor_features)

                # 2.2 大簇中选取最靠近大类原型的一定数量 patch
                if major_mask.sum() > 0:
                    major_features = top_features[major_mask]

                    # 使用欧氏距离，因为 KMeans 本身基于欧氏距离。
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

                    n_major_select = min(
                        n_major_select,
                        major_features.size(0)
                    )

                    if n_major_select > 0:
                        _, nearest_order = torch.sort(
                            distances,
                            descending=False
                        )

                        nearest_indices = nearest_order[:n_major_select]

                        nearest_major_features = major_features[
                            nearest_indices
                        ]

                        selected_top_hard_list.append(
                            nearest_major_features
                        )

                # 2.3 合并高注意力部分的 hard instances
                if len(selected_top_hard_list) > 0:
                    top_hard_instances = torch.cat(
                        selected_top_hard_list,
                        dim=0
                    )

                    hard_instances_list.append(
                        top_hard_instances
                    )

            else:
                pos_prototype = features_flat.mean(dim=0)

            # ============================================================
            # 3. 低注意力部分困难实例挖掘
            # ============================================================
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

            # ============================================================
            # 4. 中间注意力部分困难实例挖掘
            # ============================================================
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

                v_p = torch.round(
                    torch.matmul(
                        mid_norm,
                        pos_proto_norm.T
                    ).squeeze(1) * 100
                ) / 100

                v_n = torch.round(
                    torch.matmul(
                        mid_norm,
                        neg_proto_norm.T
                    ).squeeze(1) * 100
                ) / 100

                hard_mask_mid = v_p >= v_n

                if hard_mask_mid.sum() > 0:
                    hard_instances_list.append(
                        middle_features[hard_mask_mid]
                    )

            # ============================================================
            # 5. 合并所有困难实例
            # ============================================================
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
        # Final prediction with DSMIL
        # --------------------------------------------------
        hard_instances_batch = all_hard_instances.unsqueeze(0)  # [1, M, D]

        # DSMIL 返回:
        #   logits_pair = [bag_logits, instance_logits]
        #   dsmil_bag_features = [1, n_classes, feature_dim]
        logits_pair, dsmil_bag_features = self.final_pooling(
            hard_instances_batch
        )

        bag_logits, instance_logits = logits_pair

        # 当前 engine.py 使用 CrossEntropyLoss，
        # 因此这里必须返回分类 logits，而不是 hazards / S。
        #
        # DSMIL 双流融合:
        #   bag_logits      来自 bag-level classifier
        #   instance_logits 来自 max-instance classifier
        logits = 0.5 * (bag_logits + instance_logits)

        return logits
