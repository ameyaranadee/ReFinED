import os
import random
from typing import List, Iterable
from itertools import islice
from refined.utilities.general_utils import get_logger
from refined.inference.processor import Refined
from refined.evaluation.evaluation import get_datasets_obj
from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Span, Entity

logger = get_logger(__name__)

def calculate_class_imbalance(train_docs: List[Doc]):
    total_spans = 0
    salient_count = 0
    non_salient_count = 0
    no_label_count = 0
    
    for doc in train_docs:
        if doc.spans is not None:
            for span in doc.spans:
                total_spans += 1
                if span.gold_salience is not None:
                    if span.gold_salience == 1.0:
                        salient_count += 1
                    elif span.gold_salience == 0.0:
                        non_salient_count += 1
                    else:
                        logger.warning(f"Unexpected salience value: {span.gold_salience}")
                else:
                    no_label_count += 1
    
    labeled_spans = salient_count + non_salient_count
    
    logger.info(f"Total spans: {total_spans}")
    logger.info(f"  - Spans with salience labels: {labeled_spans}")
    logger.info(f"Salient entities (label=1.0): {salient_count}")
    logger.info(f"Non-salient entities (label=0.0): {non_salient_count}")
    
    if labeled_spans > 0:
        salient_pct = (salient_count / labeled_spans) * 100
        non_salient_pct = (non_salient_count / labeled_spans) * 100
        
        logger.info("")
        logger.info(f"Percentage distribution:")
        logger.info(f"  - Salient: {salient_pct:.2f}%")
        logger.info(f"  - Non-salient: {non_salient_pct:.2f}%")
        logger.info("")
    else:
        logger.error("No labeled spans found! Cannot calculate class imbalance.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    logger.info("Loading datasets...")
    random.seed(42)
    logger.info("Loading model...")
    refined = Refined.from_pretrained(
        model_name='wikipedia_model',
        entity_set='wikipedia',
        use_precomputed_descriptions=True
    )
    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    train_docs = list(datasets.get_wns_train_docs("train", include_gold_label=True))
    logger.info(f"Loaded {len(train_docs)} training documents")

    calculate_class_imbalance(train_docs)