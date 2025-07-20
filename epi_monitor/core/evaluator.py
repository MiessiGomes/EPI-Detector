from typing import Any, Dict, List, Set


class Evaluator:
    def __init__(self, required_ppe_ids: List[int], class_names: Dict[int, str]):
        self.required_ppe_ids: Set[int] = set(required_ppe_ids)
        self.class_names: Dict[int, str] = class_names

    def check_compliance(self, ppe_detections: List[Dict[str, Any]]) -> List[str]:
        """
        Checks compliance for a single person based on the EPIs detected on them.
        """
        worn_ppe_ids = {det["class_id"] for det in ppe_detections}

        missing_ppe_ids = self.required_ppe_ids - worn_ppe_ids

        if not missing_ppe_ids:
            return []

        missing_ppe_names = sorted([self.class_names[id] for id in missing_ppe_ids])

        return [f"Person is missing: {', '.join(missing_ppe_names)}"]
