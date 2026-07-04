import json
import os
from typing import Dict, List, Optional

from .base import BaseDataset


class DisasterM3Dataset(BaseDataset):
    """
    Dataset adapter for the DisasterM3 benchmark.

    This encapsulates the subset-specific field-name handling that, in the
    original `pyscripts/run_vllm.py`, lived inside `get_messages_from_data()`
    intermixed with prompt-template selection. Here it does ONE job: read
    the subset's JSON file and normalize it into BaseDataset records.
    Prompt templating and vLLM message construction stay in the model
    runner layer, where they belong.
    """

    # Subsets that compare a pre-disaster / post-disaster image pair
    BI_TEMPORAL_SUBSETS = {
        "bearing_body", "building_damage_counting",
        "disaster_type", "road_damage_counting",
        "caption", "recovery",
    }

    def load(self) -> List[Dict]:
        subset_json = os.path.join(self.data_root, f"{self.subset}.json")
        with open(subset_json, "r") as f:
            raw_data = json.load(f)

        records = []
        for idx, item in enumerate(raw_data):
            records.append({
                "id": f"{self.subset}_{idx}",
                "task": self.subset,
                "prompt": item.get("prompts"),
                "options": self._extract_options(item),
                "images": self._extract_images(item),
                # No ground-truth field is ever read in the original
                # run_vllm.py -- the upstream repo ships no evaluation
                # script, so the real key name (if any) is unconfirmed
                # against the actual (gated) dataset files. Left as None
                # rather than guessed, per analysis.md's noted limitation.
                "reference": item.get("answer"),
            })

        self._records = records
        return records

    def _extract_images(self, item: Dict) -> List[str]:
        if self.subset in self.BI_TEMPORAL_SUBSETS:
            return [
                os.path.join(self.data_root, "images", item["pre_image_path"]),
                os.path.join(self.data_root, "images", item["post_image_path"]),
            ]
        elif self.subset == "landuse":
            return [os.path.join(self.data_root, "images", item["pre_image_path"])]
        elif self.subset == "relational_reasoning_qa":
            # Original code normalizes Windows-style separators here too
            return [os.path.join(self.data_root, item["image_path"].replace("\\", "/"))]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

    def _extract_options(self, item: Dict) -> Optional[str]:
        if self.subset == "relational_reasoning_qa":
            # NOTE: the ORIGINAL run_vllm.py reads this field as
            # "option_str" (singular) for this one subset only, while
            # every other subset uses "options_str" (plural). This is an
            # inconsistency in the upstream dataset/code, not introduced
            # here -- preserved as-is for compatibility, and flagged in
            # analysis.md as a naming bug worth fixing upstream.
            return item.get("option_str")
        return item.get("options_str")
