from __future__ import annotations

from ..exceptions import ProtocolViolation


def assert_postprocess_selection_source(dataset_id: str) -> None:
    if dataset_id != "LS_SEG_200":
        raise ProtocolViolation(
            "Threshold and postprocessing selection are restricted to LS_SEG_200 internal data"
        )
