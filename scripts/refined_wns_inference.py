import csv
import json
import argparse
import pandas as pd
from refined.inference.processor import Refined
from refined.utilities.general_utils import get_logger

class RefinedInference:
    def __init__(self, model_name="wikipedia_model", entity_set="wikipedia"):
        self.logger = get_logger(__name__)
        self.logger.info(f"Loading model from {model_name}")
        self.refined = Refined.from_pretrained(
            model_name=model_name,
            entity_set=entity_set,
            use_precomputed_descriptions=False
        )

    def process_articles(self, input_path, output_path):
        test_data = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(test_data)} articles")
            
        test_data_df = test_data.drop_duplicates(subset=['text']).reset_index(drop=True)
        spans_df = []
        
        for index, row in test_data_df.iterrows():
            article_spans = self.refined.process_text(row['text'])
            
            for span in article_spans:
                candidate_entities_str = None
                if span.candidate_entities:
                    candidate_entities_str = str(span.candidate_entities)

                # salience_score = span.predicted_salience_score if span.predicted_salience_score is not None else 0.0
                # salience_label = 1 if salience_score > 0.5 else 0
                
                span_dict = {
                    'text': row['text'],
                    'date': row['date'],
                    'title': row['title'],
                    'predicted_entity': span.text,
                    'predicted_wiki_ID': span.predicted_entity.wikidata_entity_id if span.predicted_entity else None, # wikidata_Q_ID
                    'predicted_wiki_title': span.predicted_entity.wikipedia_entity_title if span.predicted_entity else None,
                    'entity_linking_confidence': span.entity_linking_model_confidence_score,  # Confidence score
                    'candidate_entities': candidate_entities_str, # list oftuples (wikidata_Q_ID, confidence score)
                    'start': span.start,
                    'end': span.start + span.ln,
                    # 'predicted_salience_score': salience_score,
                    # 'predicted_salience_label': salience_label,
                }
                spans_df.append(span_dict)
        self.logger.info(f"Processed {len(spans_df)} spans")
        spans_df = pd.DataFrame(spans_df)
        spans_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--entity_set", type=str, required=True)
    args = parser.parse_args()

    refined_inference = RefinedInference(model_name=args.model_name, entity_set=args.entity_set)
    refined_inference.process_articles(args.input_path, args.output_path)

if __name__ == "__main__":
    main()