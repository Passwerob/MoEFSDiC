"""
Loss functions for MoE-FSNL-DiC
包含扩散损失和负载均衡损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    """
    MoE 损失函数
    结合扩散模型的 MSE 损失和负载均衡损失
    """
    def __init__(self, lambda_load=0.01):
        super(MoELoss, self).__init__()
        self.lambda_load = lambda_load
        self.mse = nn.MSELoss()
    
    def forward(self, noise_pred, target_noise, all_moe_logits):
        """
        计算总损失
        
        Args:
            noise_pred: 预测的噪声 [B, C, H, W]
            target_noise: 真实噪声 [B, C, H, W]
            all_moe_logits: MoE 路由 logits 列表，每个元素为 [B, num_experts]
        
        Returns:
            total_loss: 总损失
            dm_loss_item: 扩散损失标量值
            load_loss_item: 负载均衡损失标量值
        """
        # 1. 扩散模型损失 (MSE)
        dm_loss = self.mse(noise_pred, target_noise)
        
        # 2. 负载均衡损失
        load_loss = 0.0
        if len(all_moe_logits) > 0:
            for logits in all_moe_logits:
                # logits: [B, num_experts]
                # 计算 softmax 概率分布
                probs = F.softmax(logits, dim=1)  # [B, num_experts]
                
                # 计算每个批次样本的专家概率方差
                # 方差越大说明负载越不均衡
                prob_var = torch.var(probs, dim=1)  # [B]
                
                # 累加所有样本的方差
                load_loss += prob_var.mean()
            
            # 平均所有 MoE 块的负载损失
            load_loss = load_loss / len(all_moe_logits)
        else:
            # 如果没有 MoE logits（MoE 禁用），负载损失为 0
            load_loss = torch.tensor(0.0, device=noise_pred.device)
        
        # 3. 总损失
        total_loss = dm_loss + self.lambda_load * load_loss
        
        # 返回标量值用于记录
        dm_loss_item = dm_loss.item()
        load_loss_item = load_loss.item() if isinstance(load_loss, torch.Tensor) else 0.0
        
        return total_loss, dm_loss_item, load_loss_item

