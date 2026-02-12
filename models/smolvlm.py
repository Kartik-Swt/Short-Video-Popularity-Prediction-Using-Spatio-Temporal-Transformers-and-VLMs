import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class SmolVLMForVideoClassification(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.model = base_model.model 
        self.config = base_model.config
        self.num_labels = num_labels
        
        # Get hidden size dynamically
        if hasattr(self.config, "text_config"):
            hidden_size = self.config.text_config.hidden_size
            init_range = getattr(self.config.text_config, "initializer_range", 0.02)
        else:
            hidden_size = getattr(self.config, "hidden_size", 2048) 
            init_range = getattr(self.config, "initializer_range", 0.02)

        self.score = nn.Linear(hidden_size, num_labels, bias=False).to(base_model.device)
        self.score.weight.data.normal_(mean=0.0, std=init_range)

    def forward(self, input_ids, attention_mask, pixel_values=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        
        # Pool last token
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1 
        pooled_logits = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        logits = self.score(pooled_logits)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        torch.save(self.score.state_dict(), f"{save_directory}/score_head.pt")

    @classmethod
    def from_pretrained(cls, model_id, num_labels=2, use_qlora=True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ) if use_qlora else None

        base_model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
        )
        
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            task_type="CAUSAL_LM", bias="none"
        )
        base_model = prepare_model_for_kbit_training(base_model)
        base_model = get_peft_model(base_model, lora_config)
        
        model = cls(base_model, num_labels=num_labels)
        for param in model.score.parameters():
            param.requires_grad = True
            param.data = param.data.to(torch.bfloat16)
            
        return model
