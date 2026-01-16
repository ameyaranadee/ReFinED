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
        
    def forward(
        self,
        mention_embeddings: Tensor,
        doc_embeddings: Tensor,
        salience_targets: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass of Salience layer.
        :param mention_embeddings: mention embeddings (num_entities, encoder_hidden_size)
        :param doc_embeddings: document embeddings (num_entities, encoder_hidden_size)
        :param salience_targets: binary salience targets for training
            shape: (num_entities,) with values in {0.0, 1.0} where 1.0 = salient
        :return: loss tensor (if salience_targets is provided), salience probabilities
            shape: (num_entities,) with values in [0, 1] representing probability of being salient
        """
        # print(f"mention_embeddings.shape: {mention_embeddings.shape}", flush=True)
        # print(f"doc_embeddings.shape: {doc_embeddings.shape}", flush=True)

        mention_embeddings = self.dropout(mention_embeddings)
        doc_embeddings = self.dropout(doc_embeddings)
        
        logits = torch.sum(mention_embeddings * doc_embeddings, dim=1) # (num_entities,)
        salience_probs = torch.sigmoid(logits)

        if salience_targets is not None:
            # loss_function = nn.BCEWithLogitsLoss()
            # loss = loss_function(logits.squeeze(-1), salience_targets) # (num_entities,)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, salience_targets)
            print(f"SalienceLayer forward - loss: {loss.item():.4f}", flush=True)
            
            if salience_probs.numel() > 0:
                predicted_salient = (salience_probs > 0.5).sum().item()
                print(f"SalienceLayer predictions - predicted_salient: {predicted_salient}, "
                      f"predicted_non_salient: {salience_probs.shape[0] - predicted_salient}, "
                      f"mean_prob: {salience_probs.mean().item():.4f}", flush=True)
            
            return loss, salience_probs
            
        
        return None, salience_probs