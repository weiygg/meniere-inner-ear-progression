from __future__ import annotations

from collections.abc import Mapping

from ..exceptions import ProtocolBlocker


def assert_primary_fit_ready(config: Mapping[str, object]) -> None:
    gates = config.get("blocking_gates", {})
    unresolved = [name for name, ready in gates.items() if not ready]
    if unresolved:
        raise ProtocolBlocker(f"Final P-EBM blocked: {', '.join(unresolved)}")
