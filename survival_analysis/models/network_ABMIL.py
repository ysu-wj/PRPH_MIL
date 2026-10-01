from einops import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from models.nystrom_attention import NystromAttention
from .util import initialize_weights
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_common import CHUNK_SIZE  # 导入KMeans的默认数据块大小


def set_dynamic_omp_threads(X):
    """根据输入数据动态计算并设置OMP_NUM_THREADS"""
    if not sys.platform.startswith("win"):
        return  # 非Windows系统不处理
    n_samples = X.shape[0]
    n_active_threads = max(1, int(np.ceil(n_samples / CHUNK_SIZE)))  # 计算有效数据块数
    os.environ["OMP_NUM_THREADS"] = str(n_active_threads)
    # 可选：打印日志
    # print(f"动态设置OMP_NUM_THREADS={n_active_threads}（当前样本数：{n_samples}）")


# class TransLayer(nn.Module):
#     def __init__(self, norm_layer=nn.LayerNorm, dim=256, head=8):
#         super().__init__()
#         self.norm = norm_layer(dim)
#         self.attn = NystromAttention(
#             dim=dim,
#             dim_head=dim // 8,
#             heads=head,
#             num_landmarks=dim // 2,
#             pinv_iterations=6,
#             residual=True,
#             dropout=0.1,
#         )

#     def forward(self, x, need_attn=False, need_v=False, no_norm=False):
#         if need_attn:
#             z, attn, v = self.attn(self.norm(x), return_attn=need_attn, no_norm=no_norm)
#             x = x + z
#             if need_v:
#                 return x, attn, v
#             else:
#                 return x, attn
#         else:
#             x = x + self.attn(self.norm(x))
#             return x


# class SAttention(nn.Module):
#     def __init__(self, mlp_dim=768, head=8):
#         """
#         移除了所有位置编码相关的参数 (pos, pos_pos, peg_k)
#         """
#         super(SAttention, self).__init__()
#         self.norm = nn.LayerNorm(mlp_dim)

#         # 定义可学习的全局 token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

#         # 两层 Transformer 结构
#         self.layer1 = TransLayer(dim=mlp_dim, head=head)
#         self.layer2 = TransLayer(dim=mlp_dim, head=head)

#     def forward(self, x, return_attn=False, return_act=False, no_norm=False, **kwargs):
#         batch, num_patches, C = x.shape
#         attn = []

#         # 1. 扩展并拼接 cls_token -> [batch, num_patches + 1, mlp_dim]
#         cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # 2. 通过第一层 Transformer
#         if return_attn:
#             x, _attn, v = self.layer1(x, need_attn=True, need_v=True, no_norm=no_norm)
#             attn.append(_attn.clone())
#         else:
#             x = self.layer1(x)

#         # 3. 通过第二层 Transformer
#         if return_attn:
#             x, _attn, _ = self.layer2(x, need_attn=True, need_v=True, no_norm=no_norm)
#             attn.append(_attn.clone())
#         else:
#             x = self.layer2(x)

#         # 4. 归一化并提取 cls_token 作为聚合后的全局特征
#         x = self.norm(x)
#         logits = x[:, 0, :]  # 提取第0位特征

#         if return_attn:
#             output = [logits, attn]
#             if return_act:
#                 output.append(v)
#             return output
#         else:
#             return logits
class Attention(nn.Module):
    def __init__(self,mlp_dim=768,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = mlp_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A
        

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
        # 两层线性网络用于计算注意力分数
        self.w1 = nn.Linear(input_dim, input_dim)  # 第一层（增加非线性）
        self.w2 = nn.Linear(input_dim, 1)  # 第二层（输出分数）

    def forward(self, x):
        # x: [1, N, D]（批次1，N个实例，D维特征）
        h = torch.tanh(self.w1(x))  # 非线性变换，形状：[1, N, D]
        scores = self.w2(h)  # 计算原始分数，形状：[1, N, 1]
        weights = F.softmax(scores, dim=1)  # 归一化权重，形状：[1, N, 1]
        return weights.squeeze(0).squeeze(-1)  # 移除冗余维度，形状：[N]


def run_kmeans(features, k=2):
    """
    Wrapper for scikit-learn's KMeans.
    """
    # 新增：在调用KMeans前动态设置OMP线程数
    set_dynamic_omp_threads(features)  # 传入特征数据用于计算线程数

    # features: PyTorch Tensor [N, D]
    N = features.size(0)
    if N < k:
        # Not enough samples, return all as cluster 0
        return (
            torch.zeros(N, dtype=torch.long, device=features.device),
            features.mean(dim=0, keepdim=True).repeat(k, 1),  # 质心（重复均值k次）
            torch.tensor([N, 0], device=features.device)
        )

    # 转换为numpy数组（sklearn需要numpy输入）
    features_np = features.detach().cpu().numpy()

    # Run KMeans# 运行K-Means（n_init=10：多次初始化取最优）
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(features_np)

    # 转换回PyTorch张量（保持设备一致）
    labels = torch.from_numpy(kmeans.labels_).to(features.device)
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(features.device)
    # 统计每个簇的样本数
    counts = torch.zeros(k, dtype=torch.long, device=features.device)
    for i in range(k):
        counts[i] = (labels == i).sum()  # 第i类的样本数

    return labels, centroids, counts


class PRPH(nn.Module):
    def __init__(self, n_classes=4, model_size="small", input_dim=1024):
        super(PRPH, self).__init__()
        self.n_classes = n_classes  # 输出类别数
        # 模型大小配置（病理组学特征的隐藏层维度）
        self.size_dict = {
            "pathomics": {"small": [input_dim, 256, 256], "large": [input_dim, 512, 256]}
        }
        hidden = self.size_dict["pathomics"][model_size]  # 隐藏层维度列表

        # 1. 病理组学嵌入网络（将原始特征映射到低维特征）
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))  # 线性层（降维）
            fc.append(nn.ReLU())  # 激活函数（增加非线性）
            fc.append(nn.Dropout(0.25))  # dropout（防止过拟合）
        self.pathomics_fc = nn.Sequential(*fc)  # 串联层

        # 2. 第一阶段：可学习聚类与特征融合
        self.proto_clustering = PrototypeSimilarityClustering(
            num_prototypes=12,   # 8个原型
            feature_dim=hidden[-1],  # 输入特征维度（最后一层隐藏层维度）
            embedding_dim=hidden[-1]  # 嵌入维度（与特征维度相同）
        )

        # 3. 第二阶段：硬实例挖掘组件（关注难分样本）
        self.attention_net = AttentionBackbone(hidden[-1])  # 注意力网络

        # 4. 最终池化与分类器（聚合特征并预测）
        self.final_pooling = Attention(mlp_dim=hidden[-1])
  # 自注意力池化
        self.classifier = nn.Linear(hidden[-1], self.n_classes)  # 分类器（输出logits）

        self.apply(initialize_weights)  # 应用自定义权重初始化

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]  # 输入病理组学特征：[N, 1024]（N个实例，原始特征维度1024）

      
        # --- 第一阶段：特征提取与全局融合 ---
        # 病理组学特征通过嵌入网络：[N, 1024] → [N, 256]（以small模型为例）
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)  # 增加批次维度：[1, N, 256]
        # 全局融合（结合原型信息）：[1, N, 256] → [1, N, 256]
        fused_features = self.proto_clustering(pathomics_features)

        # --- 第二阶段：硬实例挖掘（关注难分样本，提升模型鲁棒性） ---
        features_flat = fused_features.squeeze(0)  # 移除批次维度：[N, 256]
        N = features_flat.size(0)  # 实例数

        # 样本数较少时（<20），直接使用所有实例
        if N < 20:
            all_hard_instances = features_flat
        else:
            # 1. 注意力评分（区分实例重要性）
            attn_scores = self.attention_net(fused_features)  # [N]（每个实例的注意力分数）

            # 按分数排序（降序）
            sorted_scores, sorted_indices = torch.sort(attn_scores, descending=True)

            # 划分top 10%、bottom 10%、middle 80%实例
            n_top = max(1, int(N * 0.02))  # top 10%的数量（至少1个）
            n_bot = max(1, int(N * 0.02))  # bottom 10%的数量（至少1个）

            # 避免top和bottom重叠（若总数不足，均分）
            if n_top + n_bot >= N:
                n_top = N // 2
                n_bot = N - n_top

            # 索引划分
            top_indices = sorted_indices[:n_top]  # top 10%的索引
            bottom_indices = sorted_indices[-n_bot:]  # bottom 10%的索引
            middle_indices = sorted_indices[n_top: N - n_bot]  # middle 80%的索引

            hard_instances_list = []  # 存储硬实例

            # 2. 对top 10%聚类（提取“难分的重要样本”）
            if len(top_indices) > 0:
                top_features = features_flat[top_indices]  # [n_top, 256]
                # K=2聚类
                labels_top, centroids_top, counts_top = run_kmeans(top_features, k=2)

                # 确定“正原型”（样本数多的簇，代表主要模式）
                if counts_top[0] >= counts_top[1]:
                    pos_prototype = centroids_top[0]
                    hard_mask = (labels_top == 1)  # 样本少的簇为“硬实例”（偏离主要模式）
                else:
                    pos_prototype = centroids_top[1]
                    hard_mask = (labels_top == 0)

                # 若有硬实例，加入列表
                if hard_mask.sum() > 0:
                    hard_instances_list.append(top_features[hard_mask])
            else:
                # 无top实例时，用所有特征的均值作为正原型
                pos_prototype = features_flat.mean(dim=0)

            # 3. 对bottom 10%聚类（提取“难分的次要样本”）
            if len(bottom_indices) > 0:
                bottom_features = features_flat[bottom_indices]  # [n_bot, 256]
                labels_bot, centroids_bot, counts_bot = run_kmeans(bottom_features, k=2)

                # 确定“负原型”（样本数多的簇）
                if counts_bot[0] >= counts_bot[1]:
                    neg_prototype = centroids_bot[0]
                    hard_mask = (labels_bot == 1)  # 样本少的簇为“硬实例”
                else:
                    neg_prototype = centroids_bot[1]
                    hard_mask = (labels_bot == 0)

                if hard_mask.sum() > 0:
                    hard_instances_list.append(bottom_features[hard_mask])
            else:
                neg_prototype = features_flat.mean(dim=0)  # 无bottom实例时的负原型

            # 4. 对middle 80%筛选（基于与正/负原型的相似度）
            if len(middle_indices) > 0:
                middle_features = features_flat[middle_indices]  # [n_mid, 256]

                # 归一化特征和原型（计算余弦相似度）
                mid_norm = F.normalize(middle_features, p=2, dim=1)
                pos_proto_norm = F.normalize(pos_prototype.unsqueeze(0), p=2, dim=1)
                neg_proto_norm = F.normalize(neg_prototype.unsqueeze(0), p=2, dim=1)

                # 与正/负原型的相似度
                v_p = torch.matmul(mid_norm, pos_proto_norm.T).squeeze(1)  # [n_mid]（与正原型的相似度）
                v_n = torch.matmul(mid_norm, neg_proto_norm.T).squeeze(1)  # [n_mid]（与负原型的相似度）

                # 筛选：与正原型相似度 > 与负原型相似度的样本（难分的中间样本）
                hard_mask_mid = (v_p > v_n)

                if hard_mask_mid.sum() > 0:
                    hard_instances_list.append(middle_features[hard_mask_mid])

            # 合并所有硬实例
            if len(hard_instances_list) > 0:
                all_hard_instances = torch.cat(hard_instances_list, dim=0)  # 拼接硬实例
            else:
                # 若无硬实例， fallback到top实例或所有实例
                if len(top_indices) > 0:
                    all_hard_instances = features_flat[top_indices]
                else:
                    all_hard_instances = features_flat

        # --- 聚合与预测 ---
        # 恢复批次维度：[1, N_hard, 256]（N_hard为硬实例数量）
        hard_instances_batch = all_hard_instances.unsqueeze(0)

        # 自注意力池化（聚合硬实例特征）：[1, N_hard, 256] → [1, 256]
        # path_fusion = self.final_pooling(hard_instances_batch)
        fusion_output = self.final_pooling(hard_instances_batch)
        path_fusion = fusion_output[0]  # 提取元组第一个元素（聚合特征张量）
        path_fusion = path_fusion.squeeze(1)  # 移除第1维，变为[1, 256]（2维）
        # 分类器输出logits，再转换为风险值（hazards）和生存函数（S）
        logits = self.classifier(path_fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)  # 风险值（sigmoid激活，范围0-1）
        S = torch.cumprod(1 - hazards, dim=1)  # 生存函数（累积乘积，代表存活概率）

        # 记录注意力分数
        # 当实例数小于20时，未计算注意力分数，因此返回None
        # attn_scores_out = attn_scores if N >= 20 else None
        #
        return hazards, S
        # , attn_scores_out
