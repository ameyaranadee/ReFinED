from typing import Optional, Tuple
from torch import nn
from torch import Tensor
import torch

class SalienceLayer(nn.Module):
    """
    Binary classification layer to predict the salience of a mention span.
    """
    def __init__(self, dropout: float):
        """
        :param dropout: dropout rate
        :param encoder_hidden_size: hidden size of the encoder
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(encoder_hidden_size=, 1) # 1 because binary classification
        # self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the layer.
        """
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def forward(
        self, mention_embeddings: Tensor, doc_embeddings: Tensor, salience_targets: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass of Salience layer.
        :param mention_embeddings: mention embeddings (num_entities, encoder_hidden_size)
        :param salience_targets: binary salience targets for training
            shape: (num_entities,) with values in {0.0, 1.0} where 1.0 = salient
        :return: loss tensor (if salience_targets is provided), salience probabilities
            shape: (num_entities,) with values in [0, 1] representing probability of being salient
        """
        mention_embeddings = self.dropout(mention_embeddings)
        doc_embeddings = self.dropout(doc_embeddings)
        # logits = self.linear(mention_embeddings)  # (num_entities, 1)
        logits = torch.sum(mention_embeddings*doc_embeddings, dim=1) # (num_entities,)
        
        if salience_targets is not None:
            # loss_function = nn.BCEWithLogitsLoss()
            # loss = loss_function(logits.squeeze(-1), salience_targets) # (num_entities,)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, salience_targets)
            return loss, torch.sigmoid(logits)
            
        #     return loss, logits.sigmoid().squeeze(-1) # return probs
        
        # return None, logits.sigmoid().squeeze(-1)
        return None, torch.sigmoid(logits)