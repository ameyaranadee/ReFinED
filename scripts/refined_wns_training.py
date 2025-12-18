import os
import random
from typing import List, Iterable
from itertools import islice
from refined.utilities.general_utils import get_logger
from refined.inference.processor import Refined
from refined.training.fine_tune.fine_tune import fine_tune_on_docs
from refined.training.fine_tune.fine_tune_args import FineTuningArgs
from refined.evaluation.evaluation import get_datasets_obj
from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Span, Entity

logger = get_logger(__name__)

def create_sample_training_data(refined, num_samples=1000):
    logger.info("Loading datasets...")
    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    train_docs = list(datasets.get_wns_train_docs("train", include_gold_label=True))
    logger.info(f"Loaded {len(train_docs)} training documents")

    # sample_docs = random.sample(train_docs, min(num_samples, len(train_docs)))
    # logger.info(f"Sampled {len(sample_docs)} training documents")

    # eval_docs = list(datasets.get_wns_train_docs("val", include_gold_label=True))
    # eval_sample = random.sample(eval_docs, min(20, len(eval_docs)))
    # logger.info(f"Sampled {len(eval_sample)} evaluation documents")

    return train_docs

def main():
    random.seed(42)
    logger.info("Loading model...")
    refined = Refined.from_pretrained(
        model_name='wikipedia_model',
        entity_set='wikipedia',
        use_precomputed_descriptions=False
    )

    train_docs = create_sample_training_data(refined)

    ft_args = FineTuningArgs(
        experiment_name="full_wns_training",
        device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        el=True,
        epochs=3,
        batch_size=2,
        lr=1e-5,
        gradient_accumulation_steps=2,
        num_warmup_steps=10,
        checkpoint_every_n_steps=1000000,
        output_dir="full_wns_trained_models"
    )
    fine_tune_on_docs(
        refined=refined,
        fine_tuning_args=ft_args,
        train_docs=train_docs,
        eval_docs=None
    )
    logger.info("Training complete!")

if __name__ == "__main__":
    main()