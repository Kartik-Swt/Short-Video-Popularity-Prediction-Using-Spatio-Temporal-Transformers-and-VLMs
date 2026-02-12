import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def get_timesformer_model(model_id, label2id, id2label):
    print(f"Loading TimeSformer: {model_id}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = TimesformerForVideoClassification.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Cast norms to float32
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    # Hook for gradients on embeddings
    def make_inputs_require_grad(module, input, output):
        if isinstance(output, tuple):
            output[0].requires_grad_(True) 
        else:
            output.requires_grad_(True)

    model.timesformer.embeddings.patch_embeddings.register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    # Re-init classifier
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(label2id))
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True
    
    # LoRA Config
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["qkv", "dense", "temporal_dense"],
        lora_dropout=0.1, bias="none",
        modules_to_save=["classifier"],
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(model, lora_config)
    return model
