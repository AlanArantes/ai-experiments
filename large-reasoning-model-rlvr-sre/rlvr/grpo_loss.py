import torch
import torch.nn.functional as F

def compute_grpo_loss(
    policy_model,
    ref_model,
    batch_input_ids: torch.Tensor,       # [G, SeqLen]
    batch_loss_mask: torch.Tensor,       # [G, SeqLen]
    batch_attention_mask: torch.Tensor,  # [G, SeqLen]
    rewards: list[float],                # [G]
    epsilon: float = 0.2,
    beta_kl: float = 0.04
) -> torch.Tensor:
    """
    Standard Group Relative Policy Optimization (GRPO) loss computation across rollout group size G.
    Eliminates the separate critic/value network required by PPO.
    """
    device = batch_input_ids.device
    r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # 1. Compute group-relative standardized advantages
    std = r_tensor.std()
    if std < 1e-6:
        advantages = r_tensor - r_tensor.mean()
    else:
        advantages = (r_tensor - r_tensor.mean()) / (std + 1e-8)
        
    # advantages shape: [G, 1]
    adv = advantages.unsqueeze(-1)

    # 2. Get current policy logits vs reference policy logits
    with torch.no_grad():
        ref_logits = ref_model(batch_input_ids, attention_mask=batch_attention_mask).logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_selected = torch.gather(ref_log_probs[:, :-1, :], 2, batch_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    curr_logits = policy_model(batch_input_ids, attention_mask=batch_attention_mask).logits
    curr_log_probs = F.log_softmax(curr_logits, dim=-1)
    curr_selected = torch.gather(curr_log_probs[:, :-1, :], 2, batch_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Shift mask by 1 to align with next-token prediction
    active_mask = batch_loss_mask[:, 1:]

    # 3. Ratio and Clipped Surrogate Loss
    log_ratio = curr_selected - ref_selected.detach()
    ratio = torch.exp(log_ratio)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
    policy_loss = -torch.min(surr1, surr2)

    # 4. Per-token KL Penalty (Schulman approximation)
    kl_div = torch.exp(ref_selected.detach() - curr_selected) - (ref_selected.detach() - curr_selected) - 1.0

    total_loss_per_token = policy_loss + (beta_kl * kl_div)
    
    # 5. Masked loss reduction
    masked_loss = (total_loss_per_token * active_mask).sum() / (active_mask.sum() + 1e-8)
    return masked_loss
