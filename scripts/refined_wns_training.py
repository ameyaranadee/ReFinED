import os
import random
import argparse
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

def load_data(refined):
    logger.info("Loading dataset...")
    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    train_docs = list(datasets.get_wns_train_docs("train", include_gold_label=True))
    logger.info(f"Loaded {len(train_docs)} training documents")

    eval_docs = list(datasets.get_wns_train_docs("val", include_gold_label=True))
    logger.info(f"Loaded {len(eval_docs)} evaluation documents")

    return train_docs, eval_docs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ReFinED model on WNS dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Path to model")
    parser.add_argument("--entity_set", type=str, required=True, help="Entity set")
    args = parser.parse_args()

    random.seed(42)
    logger.info("Loading model...")

    refined = Refined.from_pretrained(
        model_name=args.model_name,
        entity_set=args.entity_set,
        use_precomputed_descriptions=False
    )

    train_docs, eval_docs = load_data(refined)

    ft_args = FineTuningArgs(
        experiment_name="ReFinED-Wikipedia_EL-SSP-FT-on-WNS_Mention-Article-CE-SSP-1.1_260124",
        device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        el=True,
        epochs=5,
        batch_size=4,
        lr=3e-5,
        gradient_accumulation_steps=4,
        num_warmup_steps=100,
        checkpoint_every_n_steps=500,
        output_dir="../finetuned_models"
    )
    fine_tune_on_docs(
        refined=refined,
        fine_tuning_args=ft_args,
        train_docs=train_docs,
        eval_docs=eval_docs
    )
    logger.info("Training complete!")

if __name__ == "__main__":
    main()