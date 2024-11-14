import torch
import math
from torch.nn.functional import softmax

class TransformerAttention:

    @staticmethod
    def scaled_dot_product_attention(Q, K, V):
        scores = TransformerAttention._calculate_scores(Q, K)
        scaled_scores = TransformerAttention._reduce_dim(K, scores)
        masked_scaled_scores = TransformerAttention._apply_mask(scaled_scores)
        attention = TransformerAttention._softmax(masked_scaled_scores)
        attention_vectors = TransformerAttention._calculate_attention(attention, V)
        return attention_vectors

    def _calculate_scores(Q, K):
        return torch.matmul(Q, K.t())
    
    def _reduce_dim(K, scores):
        d_k = K.size(-1)
        return scores / math.sqrt(d_k)
    
    def _apply_mask(scaled_scores):
        size = scaled_scores.size(-1)
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask == 1
        masked_scaled_scores = scaled_scores.masked_fill(mask, float('-inf'))
        return masked_scaled_scores
    
    def _softmax(masked_scaled_scores):
        return softmax(masked_scaled_scores, dim=1)
    
    def _calculate_attention(attention_softmax, V):
        return torch.matmul(attention_softmax, V)

