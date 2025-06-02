import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchcrf import CRF

class AdvancedNERModel(nn.Module):
    """
    Production-grade NER model with CRF layer for sequence consistency
    """
    def __init__(self, model_checkpoint, num_labels, dropout_rate=0.3, real_world_training=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.config.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_checkpoint, config=self.config)
        
        # Stronger dropout for real-world training to combat overfitting
        if real_world_training:
            dropout_rate = 0.4  # Increase from 0.3 to 0.4
            print(f"🔧 Real-world training mode: Using higher dropout rate {dropout_rate}")
        
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.config.hidden_size
        
        # More aggressive dropout in classifier for real-world training
        classifier_dropout = dropout_rate * 1.2 if real_world_training else dropout_rate
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.loss_weights = nn.Parameter(torch.ones(num_labels), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Check if the model supports token_type_ids (DistilBERT doesn't)
        transformer_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Only add token_type_ids if the model supports it
        if token_type_ids is not None and hasattr(self.transformer, 'embeddings') and hasattr(self.transformer.embeddings, 'token_type_embeddings'):
            transformer_inputs['token_type_ids'] = token_type_ids
        
        outputs = self.transformer(**transformer_inputs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        outputs_dict = {"logits": logits}
        if labels is not None:
            valid_labels_mask = (labels != -100)
            crf_labels = labels.clone()
            crf_labels[~valid_labels_mask] = 0
            # Aggressive clamping & diagnostic to avoid CUDA assert if any label is out-of-bounds
            if crf_labels.max().item() >= self.config.num_labels or crf_labels.min().item() < 0:
                print("!!! WARNING: CRF labels out of bounds. Clamping to valid range !!!")
                print(f"    Min label: {crf_labels.min().item()}, Max label: {crf_labels.max().item()}, Num labels: {self.config.num_labels}")
                crf_labels = torch.clamp(crf_labels, 0, self.config.num_labels - 1)
            mask = attention_mask.bool() if attention_mask is not None else None
            if mask is not None:
                mask = mask & valid_labels_mask
            else:
                mask = valid_labels_mask
            # Ensure mask has the right shape before touching first column
            if mask is not None and mask.ndim > 1 and mask.shape[1] > 0:
                mask[:, 0] = True
            elif mask is not None and mask.ndim == 1 and mask.shape[0] > 0:
                mask[0] = True
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            outputs_dict["loss"] = loss
        return outputs_dict

    def decode(self, logits, attention_mask=None):
        mask = attention_mask.bool() if attention_mask is not None else None
        return self.crf.decode(logits, mask=mask)