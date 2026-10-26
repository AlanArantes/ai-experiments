import torch

class SRETrajectoryFormatter:
    """
    Formats multi-turn agent interactions into token sequences with precise loss masking.
    Only tokens emitted by the agent ('assistant') are assigned loss_mask=1,
    while system instructions, environmental observations, and tool outputs have loss_mask=0.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def format_and_mask(self, trajectory_turns: list[dict], max_length: int = 4096) -> dict[str, torch.Tensor]:
        """
        trajectory_turns format:
        [
            {"role": "system", "content": "..."},
            {"role": "assistant", "content": "<think>...</think><action>...</action>"},
            {"role": "environment", "content": "<observation>...</observation>"},
            ...
        ]
        """
        full_token_ids = []
        loss_mask = []
        
        for turn in trajectory_turns:
            role = turn["role"]
            content = turn["content"]
            
            # Encode each turn separately to map exact token boundaries
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            full_token_ids.extend(tokens)
            
            if role == "assistant":
                # Model generated tokens -> compute policy gradient
                loss_mask.extend([1] * len(tokens))
            else:
                # System prompt or Environment observation -> MASK OUT
                loss_mask.extend([0] * len(tokens))
                
        # Truncate or Pad
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

        if len(full_token_ids) > max_length:
            full_token_ids = full_token_ids[:max_length]
            loss_mask = loss_mask[:max_length]
        else:
            pad_len = max_length - len(full_token_ids)
            full_token_ids.extend([pad_token_id] * pad_len)
            loss_mask.extend([0] * pad_len)
            
        return {
            "input_ids": torch.tensor(full_token_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
            "attention_mask": (torch.tensor(full_token_ids) != pad_token_id).long()
        }
