from dataclasses import dataclass
from typing import List, Sequence, Set, Dict


@dataclass
class SelectionResult:
    selected_indices: List[int]
    overlap_indices: List[int]
    relevance_only: List[int]
    persistence_only: List[int]
    protected_indices: List[int]
    budget: int
    diagnostics: Dict[str, float]


class HybridKVSelector:
    """
    CPU-only prototype for hybrid KV selection.

    Combines:
      1. receiver-conditioned relevance (kNN-style ranking)
      2. sender-derived persistent importance (H2O-style ranking)
      3. optional protected structural tokens (sink/recent)

    No weighted combination of scores is required.

    Selection principle:
      - protect structurally required tokens
      - prioritize agreement between relevance and persistence
      - alternate remaining selections between both rankings
      - enforce a strict total KV budget
    """

    def __init__(
        self,
        overlap_pool_fraction: float = 0.50,
    ):
        if not 0.0 < overlap_pool_fraction <= 1.0:
            raise ValueError(
                "overlap_pool_fraction must be in (0, 1]."
            )

        self.overlap_pool_fraction = overlap_pool_fraction

    @staticmethod
    def _validate_scores(
        relevance_scores: Sequence[float],
        persistence_scores: Sequence[float],
    ) -> None:

        if len(relevance_scores) != len(persistence_scores):
            raise ValueError(
                "relevance_scores and persistence_scores "
                "must have equal length."
            )

        if len(relevance_scores) == 0:
            raise ValueError("Score arrays cannot be empty.")

    @staticmethod
    def _rank_descending(
        scores: Sequence[float],
        excluded: Set[int],
    ) -> List[int]:

        candidates = [
            i
            for i in range(len(scores))
            if i not in excluded
        ]

        return sorted(
            candidates,
            key=lambda i: scores[i],
            reverse=True,
        )

    def select(
        self,
        relevance_scores: Sequence[float],
        persistence_scores: Sequence[float],
        budget: int,
        protected_indices: Sequence[int] = (),
    ) -> SelectionResult:

        self._validate_scores(
            relevance_scores,
            persistence_scores,
        )

        n_tokens = len(relevance_scores)

        if budget <= 0:
            raise ValueError("budget must be positive.")

        budget = min(int(budget), n_tokens)

        protected = {
            int(i)
            for i in protected_indices
            if 0 <= int(i) < n_tokens
        }

        # If protected tokens already exceed the requested budget,
        # retain only protected tokens up to the budget.
        if len(protected) >= budget:

            selected = sorted(protected)[:budget]

            return SelectionResult(
                selected_indices=selected,
                overlap_indices=[],
                relevance_only=[],
                persistence_only=[],
                protected_indices=selected,
                budget=budget,
                diagnostics={
                    "overlap_count": 0,
                    "protected_count": len(selected),
                    "relevance_only_count": 0,
                    "persistence_only_count": 0,
                },
            )

        relevance_rank = self._rank_descending(
            relevance_scores,
            protected,
        )

        persistence_rank = self._rank_descending(
            persistence_scores,
            protected,
        )

        selected: Set[int] = set(protected)

        overlap_selected: List[int] = []
        relevance_only: List[int] = []
        persistence_only: List[int] = []

        remaining_budget = budget - len(selected)

        # ------------------------------------------------------
        # STAGE 1: AGREEMENT / OVERLAP
        # ------------------------------------------------------
        #
        # Look at the highest-ranked region of BOTH criteria.
        # Tokens appearing in both are supported by two
        # independent notions of importance.
        # ------------------------------------------------------

        pool_size = max(
            1,
            int(
                len(relevance_rank)
                * self.overlap_pool_fraction
            ),
        )

        relevance_pool = set(
            relevance_rank[:pool_size]
        )

        persistence_pool = set(
            persistence_rank[:pool_size]
        )

        overlap = (
            relevance_pool
            & persistence_pool
        )

        # Rank overlap candidates by the sum of their RANKS,
        # rather than combining raw scores.
        relevance_position = {
            token: rank
            for rank, token
            in enumerate(relevance_rank)
        }

        persistence_position = {
            token: rank
            for rank, token
            in enumerate(persistence_rank)
        }

        overlap_ranked = sorted(
            overlap,
            key=lambda token: (
                relevance_position[token]
                + persistence_position[token]
            ),
        )

        for token in overlap_ranked:

            if len(selected) >= budget:
                break

            selected.add(token)
            overlap_selected.append(token)

        # ------------------------------------------------------
        # STAGE 2: BALANCED FILL
        # ------------------------------------------------------
        #
        # Alternate between relevance and persistence.
        #
        # This prevents either signal from completely dominating
        # without requiring hand-tuned numerical weights.
        # ------------------------------------------------------

        r_pointer = 0
        p_pointer = 0

        turn = "relevance"

        while len(selected) < budget:

            added = False

            if turn == "relevance":

                while r_pointer < len(relevance_rank):

                    token = relevance_rank[r_pointer]
                    r_pointer += 1

                    if token not in selected:

                        selected.add(token)
                        relevance_only.append(token)
                        added = True
                        break

                turn = "persistence"

            else:

                while p_pointer < len(persistence_rank):

                    token = persistence_rank[p_pointer]
                    p_pointer += 1

                    if token not in selected:

                        selected.add(token)
                        persistence_only.append(token)
                        added = True
                        break

                turn = "relevance"

            # Safety fallback.
            if not added:

                remaining = [
                    i
                    for i in range(n_tokens)
                    if i not in selected
                ]

                if not remaining:
                    break

                selected.add(remaining[0])

        selected_indices = sorted(selected)

        diagnostics = {
            "overlap_count": len(overlap_selected),
            "protected_count": len(protected),
            "relevance_only_count": len(relevance_only),
            "persistence_only_count": len(
                persistence_only
            ),
            "overlap_fraction_of_selected": (
                len(overlap_selected)
                / len(selected_indices)
                if selected_indices
                else 0.0
            ),
        }

        return SelectionResult(
            selected_indices=selected_indices,
            overlap_indices=sorted(
                overlap_selected
            ),
            relevance_only=sorted(
                relevance_only
            ),
            persistence_only=sorted(
                persistence_only
            ),
            protected_indices=sorted(
                protected
            ),
            budget=budget,
            diagnostics=diagnostics,
        )