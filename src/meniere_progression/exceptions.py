class ProtocolViolation(ValueError):
    """Raised when an operation violates a locked study-design boundary."""


class ProtocolBlocker(RuntimeError):
    """Raised when required clinical or data-governance information is unresolved."""
