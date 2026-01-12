from dataclasses import dataclass, field
# add weak match for QA and EL
from typing import List, Any


@dataclass
class Metrics:
    el: bool  # flags whether the metrics are for entity linking (EL) or entity disambiguation (ED)
    num_gold_spans: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_md: int = 0
    fp_md: int = 0
    fn_md: int = 0
    gold_entity_in_cand: int = 0
    num_docs: int = 0
    example_errors: List[Any] = field(default_factory=list)
    example_errors_md: List[Any] = field(default_factory=list)
    # salience_predictions: List[float] = field(default_factory=list)
    # gold_salience_labels: List[float] = field(default_factory=list)

    def __add__(self, other: 'Metrics'):
        return Metrics(
            el=self.el,
            num_gold_spans=self.num_gold_spans + other.num_gold_spans,
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tp_md=self.tp_md + other.tp_md,
            fp_md=self.fp_md + other.fp_md,
            fn_md=self.fn_md + other.fn_md,
            gold_entity_in_cand=self.gold_entity_in_cand + other.gold_entity_in_cand,
            num_docs=self.num_docs + other.num_docs,
            example_errors=self.example_errors + other.example_errors,
            example_errors_md=self.example_errors_md + other.example_errors_md,
            # salience_predictions=self.salience_predictions + other.salience_predictions,
            # gold_salience_labels=self.gold_salience_labels + other.gold_salience_labels
        )

    def get_summary(self):
        p = self.get_precision()
        r = self.get_recall()
        f1 = self.get_f1()
        accuracy = self.get_accuracy()
        gold_recall = self.get_gold_recall()
        result = f"\n****************\n" \
                 f"************\n" \
                 f"f1: {f1:.4f}\naccuracy: {accuracy:.4f}\ngold_recall: {gold_recall:.4f}\np: {p:.4f}\nr: " \
                 f"{r:.4f}\nnum_gold_spans: {self.num_gold_spans}\n" \
                 f"************\n"
        if self.el:
            # MD results only make sense for when EL mode is enabled
            result += f"*******MD*****\n" \
                      f"MD_f1: {self.get_f1_md():.4f}, (p: {self.get_precision_md():.4f}," \
                      f" r: {self.get_recall_md():.4f})" \
                      f"\n*****************\n"

        # if self.salience_predictions:
        #     salience_stats = self.get_salience_stats()
        #     result += f"*******SALIENCE*****\n" \
        #               f"num_predictions: {len(self.salience_predictions)}\n" \
        #               f"mean_salience: {salience_stats['mean']:.4f}\n" \
        #               f"median_salience: {salience_stats['median']:.4f}\n" \
        #               f"min_salience: {salience_stats['min']:.4f}\n" \
        #               f"max_salience: {salience_stats['max']:.4f}\n" \
        #               f"std_salience: {salience_stats['std']:.4f}\n"
            
            # Add binary classification metrics if we have gold labels
            # if self.gold_salience_labels:
            #     salience_corr = self.get_salience_correlation()
            #     result += f"salience_correlation: {salience_corr:.4f}\n"
                
            #     # Binary classification metrics with threshold 0.5
            #     cls_metrics = self.get_salience_classification_metrics(threshold=0.5)
            #     result += f"salience_classification (threshold=0.5):\n" \
            #               f"  accuracy: {cls_metrics['accuracy']:.4f}\n" \
            #               f"  precision: {cls_metrics['precision']:.4f}\n" \
            #               f"  recall: {cls_metrics['recall']:.4f}\n" \
            #               f"  f1: {cls_metrics['f1']:.4f}\n" \
            #               f"  num_matched: {cls_metrics['num_matched']}\n" \
            #               f"  (tp={cls_metrics['tp']}, fp={cls_metrics['fp']}, fn={cls_metrics['fn']}, tn={cls_metrics['tn']})\n"
                
            #     # ROC-AUC
            #     auc = self.get_salience_auc()
            #     if auc > 0:
            #         result += f"  roc_auc: {auc:.4f}\n"
            
            result += f"*****************\n"

        return result

    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-8 * 1.0)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + 1e-8 * 1.0)

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_precision_md(self):
        return self.tp_md / (self.tp_md + self.fp_md + 1e-8 * 1.0)

    def get_recall_md(self):
        return self.tp_md / (self.tp_md + self.fn_md + 1e-8 * 1.0)

    def get_f1_md(self):
        # Note that MD results only make sense for when EL mode is enabled as gold `spans` and `md_spans` may differ.
        p = self.get_precision_md()
        r = self.get_recall_md()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_accuracy(self):
        return 1.0 * self.tp / (self.num_gold_spans + 1e-8)

    def get_gold_recall(self):
        return 1.0 * self.gold_entity_in_cand / (self.num_gold_spans + 1e-8)

    # def get_salience_stats(self):
    #     """Calculate statistics for salience predictions."""
    #     import numpy as np
    #     if not self.salience_predictions:
    #         return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
    #     preds = np.array(self.salience_predictions)
    #     return {
    #         'mean': float(np.mean(preds)),
    #         'median': float(np.median(preds)),
    #         'min': float(np.min(preds)),
    #         'max': float(np.max(preds)),
    #         'std': float(np.std(preds))
    #     }

    # def get_salience_correlation(self):
    #     """Calculate correlation between predicted and gold salience scores."""
    #     import numpy as np
    #     if not self.salience_predictions or not self.gold_salience_labels:
    #         return 0.0
    #     if len(self.salience_predictions) != len(self.gold_salience_labels):
    #         return 0.0
    #     preds = np.array(self.salience_predictions)
    #     golds = np.array(self.gold_salience_labels)
    #     # Filter out None values
    #     valid_mask = ~(np.isnan(preds) | np.isnan(golds))
    #     if valid_mask.sum() < 2:
    #         return 0.0
    #     corr = np.corrcoef(preds[valid_mask], golds[valid_mask])[0, 1]
    #     return float(corr) if not np.isnan(corr) else 0.0

    # def get_salience_classification_metrics(self, threshold: float = 0.5):
    #     """Calculate binary classification metrics for salience (using threshold)."""
    #     import numpy as np
    #     if not self.salience_predictions or not self.gold_salience_labels:
    #         return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_matched': 0}
        
    #     # Match predictions with gold labels by index
    #     min_len = min(len(self.salience_predictions), len(self.gold_salience_labels))
    #     if min_len == 0:
    #         return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_matched': 0}
        
    #     preds = np.array(self.salience_predictions[:min_len])
    #     golds = np.array(self.gold_salience_labels[:min_len])
        
    #     # Filter out NaN values
    #     valid_mask = ~(np.isnan(preds) | np.isnan(golds))
    #     if valid_mask.sum() == 0:
    #         return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_matched': 0}
        
    #     preds_valid = preds[valid_mask]
    #     golds_valid = golds[valid_mask]
        
    #     # Convert probabilities to binary predictions
    #     preds_binary = (preds_valid >= threshold).astype(int)
    #     golds_binary = golds_valid.astype(int)
        
    #     # Calculate metrics
    #     tp = np.sum((preds_binary == 1) & (golds_binary == 1))
    #     fp = np.sum((preds_binary == 1) & (golds_binary == 0))
    #     fn = np.sum((preds_binary == 0) & (golds_binary == 1))
    #     tn = np.sum((preds_binary == 0) & (golds_binary == 0))
        
    #     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    #     precision = tp / (tp + fp + 1e-8)
    #     recall = tp / (tp + fn + 1e-8)
    #     f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
    #     return {
    #         'accuracy': float(accuracy),
    #         'precision': float(precision),
    #         'recall': float(recall),
    #         'f1': float(f1),
    #         'num_matched': int(valid_mask.sum()),
    #         'tp': int(tp),
    #         'fp': int(fp),
    #         'fn': int(fn),
    #         'tn': int(tn)
    #     }

    # def get_salience_auc(self):
    #     """Calculate ROC-AUC for salience predictions."""
    #     try:
    #         from sklearn.metrics import roc_auc_score
    #         import numpy as np
            
    #         if not self.salience_predictions or not self.gold_salience_labels:
    #             return 0.0
            
    #         min_len = min(len(self.salience_predictions), len(self.gold_salience_labels))
    #         if min_len < 2:
    #             return 0.0
            
    #         preds = np.array(self.salience_predictions[:min_len])
    #         golds = np.array(self.gold_salience_labels[:min_len])
            
    #         # Filter out NaN values
    #         valid_mask = ~(np.isnan(preds) | np.isnan(golds))
    #         if valid_mask.sum() < 2:
    #             return 0.0
            
    #         preds_valid = preds[valid_mask]
    #         golds_valid = golds[valid_mask].astype(int)
            
    #         # Check if we have both classes
    #         if len(np.unique(golds_valid)) < 2:
    #             return 0.0
            
    #         auc = roc_auc_score(golds_valid, preds_valid)
    #         return float(auc) if not np.isnan(auc) else 0.0
    #     except ImportError:
    #         return 0.0
    #     except Exception:
    #         return 0.0

    @classmethod
    def zeros(cls, el: bool):
        return Metrics(num_gold_spans=0, tp=0, fp=0, fn=0, el=el)
