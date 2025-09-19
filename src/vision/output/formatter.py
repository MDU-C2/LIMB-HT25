from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tags import TagDetectionResult
from cup import CupDetectionResult


class OutputFormatter:
    """Formats results into structured Python dicts suitable for JSON serialization."""

    @staticmethod
    def build_output(
        tag_result: TagDetectionResult,
        cup_result: CupDetectionResult,
    ) -> Dict[str, Any]:
        return {
           "tag_result": tag_result,
           "cup_result": cup_result,
        }