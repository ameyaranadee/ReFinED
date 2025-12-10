import os
from refined.inference.processor import Refined

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "~/.cache/refined")

refined = Refined.from_pretrained(
    model_name="wikipedia_model",
    entity_set="wikipedia",
    # data_dir=data_dir,
    # download_files=True,
    use_precomputed_descriptions=True
)

preprocessor = refined.preprocessor

qcode_to_idx = preprocessor.qcode_to_idx
qcode_idx_to_class_idx = preprocessor.qcode_idx_to_class_idx
class_to_idx = preprocessor.class_to_idx
index_to_class = preprocessor.index_to_class

class_indices_tensor = preprocessor.class_handler.get_classes_idx_for_qcode_batch(
    ["Q76"], shape=(-1, preprocessor.max_num_classes_per_ent)
)

class_indices = class_indices_tensor[0].tolist()  # Get first row and convert to list
class_qcodes = [
    preprocessor.index_to_class[idx] 
    for idx in class_indices 
    if idx != 0  # 0 is padding
]

print(f"Entity types for Q76: {class_qcodes}")

# Entity types for Q76: ['Q482980', 'Q36180', 'Q5', 'Q372436', 'Q215627', 'Q15980158', 'Q3778211', 'Q82955', 'Q15253558', 'Q57735705', 'Q28813620', 'Q702269', 'Q2500638']