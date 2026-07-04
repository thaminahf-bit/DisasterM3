from abc import ABC, abstractmethod
from typing import Dict, List


class BaseDataset(ABC):
    """
    Common interface every dataset adapter must implement.

    A dataset adapter's ONLY job is translating a dataset's on-disk format
    into a list of normalized records. It must not know which model will
    consume the records, and it must not build model-specific prompts or
    chat messages -- that stays out of this layer entirely (it belongs to
    the model runner), so a dataset can be swapped without touching any
    prompting logic, and vice versa.
    """

    def __init__(self, data_root: str, subset: str):
        self.data_root = data_root
        self.subset = subset
        self._records: List[Dict] = []

    @abstractmethod
    def load(self) -> List[Dict]:
        """
        Load and normalize this dataset's subset into a list of records.

        Each record must be a dict with (at minimum) these keys:
            - "id":       unique string id for the sample
            - "task":     the subset/task name this record belongs to
            - "prompt":   raw question/instruction text (unformatted --
                          no task-specific templating applied here)
            - "options":  list/string of answer choices, or None if the
                          task is open-ended (e.g. captioning)
            - "images":   list of one or more absolute image file paths
            - "reference": ground-truth answer/label if available, else
                           None (many benchmarks, including the current
                           DisasterM3 release, don't ship this)

        Returns:
            List[Dict]: the normalized records for this subset.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        return self._records[idx]
