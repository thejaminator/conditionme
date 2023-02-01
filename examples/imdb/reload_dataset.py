from typing import Optional

import datasets
from datasets import Dataset

preprocessed_dataset_path = "dataset_tokenized_imdb.hf"


def try_load_preprocessed_dataset() -> Optional[Dataset]:
    try:
        dataset_tokenized: Dataset = datasets.load_from_disk(preprocessed_dataset_path)  # type: ignore [assignment]
        return dataset_tokenized
    except FileNotFoundError:
        print("Preprocessed dataset not found.")
        return None
