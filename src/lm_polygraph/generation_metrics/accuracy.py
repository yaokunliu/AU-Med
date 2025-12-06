import re
import string
import numpy as np
import logging
from typing import List, Dict
from .generation_metric import GenerationMetric

log = logging.getLogger("lm_polygraph")

# Updated pattern to robustly detect A–D options, avoiding false positives like "A cat..."
CHOICE_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])                             
    (?:\b(?:answer|option|choice|response|the\s+correct\s+answer\s+is)\b[\s\S]{0,50})?
    ([A-D])
    (?=[\.\):\s<,\n\|]|$)                         
    """,
    re.IGNORECASE | re.VERBOSE
)

# YNM_PATTERN = re.compile(r"^\s*(yes|no|maybe)\b", re.IGNORECASE)
YNM_PATTERN = re.compile(
    r"(?:\b(?:answer|option|choice|response|decision)\b[\s\S]{0,50})?\b(yes|no|maybe)\b",
    re.IGNORECASE
)


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    For 'choice' mode, it extracts the most likely multiple-choice letter (A–D)
    from model output, tolerating extra explanations.
    """

    def __init__(
        self,
        target_ignore_regex=None,
        output_ignore_regex=None,
        normalize=False,
        mode: str = None,
    ):
        super().__init__(["greedy_texts"], "sequence")
        self.target_ignore_regex = (
            re.compile(target_ignore_regex) if target_ignore_regex else None
        )
        self.output_ignore_regex = (
            re.compile(output_ignore_regex) if output_ignore_regex else None
        )
        self.normalize = normalize
        self.mode = mode

        if self.target_ignore_regex or self.output_ignore_regex or self.normalize:
            log.warning(
                "Specifying ignore_regex or normalize in AccuracyMetric is deprecated. "
                "Use output and target processing scripts instead."
            )

    def __str__(self):
        return "Accuracy"

    def _extract_answer(self, text: str) -> str:
        """
        Extracts the answer letter (A–D) or yes/no/maybe from text.
        Handles outputs with explanations (e.g., 'Answer is B. Surgery ...').
        """
        if self.mode == "choice":
            match = CHOICE_PATTERN.search(text)
            if match:
                return match.group(1).upper()
        elif self.mode == "ynm":
            match = YNM_PATTERN.search(text)
            if match:
                return match.group(1).capitalize()
        return text.strip()

    def _score_single(self, output: str, target: str) -> int:
        return int(output.strip() == target.strip())

    def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
        return ignore_regex.sub("", text) if ignore_regex else text

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates accuracy between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): includes 'greedy_texts'
            target_texts (List[str]): list of correct answers (e.g., ['B', 'A', ...])

        Returns:
            np.ndarray: binary array (1 = correct, 0 = incorrect)
        """
        greedy_texts = stats["greedy_texts"]
        result = []

        for hyp, ref in zip(greedy_texts, target_texts):
            ref = self._filter_text(ref, self.target_ignore_regex)
            hyp = self._filter_text(hyp, self.output_ignore_regex)

            ref = self._extract_answer(ref)
            hyp = self._extract_answer(hyp)

            if self.normalize:
                ref = self._normalize_text(ref)
                hyp = self._normalize_text(hyp)

            result.append(self._score_single(hyp, ref))

        return np.array(result)


# ----------------original code below----------------
# import re
# import string
# import numpy as np
# import logging

# from typing import List, Dict
# from .generation_metric import GenerationMetric

# log = logging.getLogger("lm_polygraph")

# class AccuracyMetric(GenerationMetric):
#     """
#     Calculates accuracy between model-generated texts and ground-truth.
#     Two texts are considered equal if theis string representation is equal.
#     """

#     def __init__(
#         self, target_ignore_regex=None, output_ignore_regex=None, normalize=False,
#     ):
#         super().__init__(["greedy_texts"], "sequence")
#         self.target_ignore_regex = (
#             re.compile(target_ignore_regex) if target_ignore_regex else None
#         )
#         self.output_ignore_regex = (
#             re.compile(output_ignore_regex) if output_ignore_regex else None
#         )
#         self.normalize = normalize

#         if self.target_ignore_regex or self.output_ignore_regex or self.normalize:
#             log.warning(
#                 "Specifying ignore_regex or normalize in AccuracyMetric is deprecated. Use output and target processing scripts instead."
#             )

#     def __str__(self):
#         return "Accuracy"

#     def _score_single(self, output: str, target: str) -> int:
#         if output.strip() == target.strip():
#             return 1
#         return 0

#     def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
#         text = ignore_regex.sub("", text) if ignore_regex else text

#         return text

#     def _normalize_text(self, text: str) -> str:
#         text = text.strip().lower()
#         text = text.translate(str.maketrans("", "", string.punctuation))

#         return text

#     def __call__(
#         self,
#         stats: Dict[str, np.ndarray],
#         target_texts: List[str],
#     ) -> np.ndarray:
#         """
#         Calculates accuracy between stats['greedy_texts'] and target_texts.

#         Parameters:
#             stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
#                 * model-generated texts in 'greedy_texts'
#             target_texts (List[str]): ground-truth texts
#         Returns:
#             np.ndarray: list of accuracies: 1 if generated text is equal to ground-truth and 0 otherwise.
#         """
#         greedy_texts = stats["greedy_texts"]

#         result = []

#         for hyp, ref in zip(greedy_texts, target_texts):
#             ref = self._filter_text(ref, self.target_ignore_regex)
#             hyp = self._filter_text(hyp, self.output_ignore_regex)

#             if self.normalize:
#                 ref = self._normalize_text(ref)
#                 hyp = self._normalize_text(hyp)

#             result.append(self._score_single(hyp, ref))

#         return np.array(result)

