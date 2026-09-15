"""Version signal shown to the agent before it decides whether to read.

The signal is prompt context, not a charged schema read. It names the release
the agent is talking to and, when a baseline version is known, lists the
query-side breaking changes since that baseline so the policy can decide what
to look up.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from restock.schema.diff import Change, ChangeKind

# Kinds that mean a memorised field or argument may no longer work.
BREAKING_FOR_SIGNAL = frozenset(
    {
        ChangeKind.TYPE_REMOVED,
        ChangeKind.TYPE_KIND_CHANGED,
        ChangeKind.FIELD_REMOVED,
        ChangeKind.FIELD_TYPE_CHANGED,
        ChangeKind.ARG_REMOVED,
        ChangeKind.ARG_TYPE_CHANGED,
        ChangeKind.REQUIRED_ARG_ADDED,
        ChangeKind.REQUIRED_INPUT_FIELD_ADDED,
        ChangeKind.INTERFACE_REMOVED,
        ChangeKind.UNION_MEMBER_REMOVED,
        ChangeKind.ENUM_VALUE_REMOVED,
    }
)


@dataclass(frozen=True)
class VersionSignal:
    """What the agent is told about the API version it faces."""

    version: str
    """Release (or simulated id) the store is serving."""
    trained_on: str | None = None
    """Release the weights were trained against, if any."""
    changes: tuple[Change, ...] = ()
    """Query-side breaking changes from ``trained_on`` to ``version``."""

    def text(self) -> str:
        """Prompt text for the version signal."""
        lines = [f"API version: {self.version}"]
        if self.trained_on is None:
            return lines[0]
        lines.append(f"Trained on: {self.trained_on}")
        breaking = [c for c in self.changes if c.query_side and c.kind in BREAKING_FOR_SIGNAL]
        if not breaking:
            lines.append("Query-side breaking changes since training: none")
            return "\n".join(lines)
        lines.append(f"Query-side breaking changes since training ({len(breaking)}):")
        for change in sorted(breaking, key=lambda c: c.coordinate):
            lines.append(f"  {change.kind.value}  {change.coordinate}")
        return "\n".join(lines)


def version_signal(
    version: str,
    *,
    trained_on: str | None = None,
    changes: Iterable[Change] = (),
) -> VersionSignal:
    """Build a version signal from a release id and an optional schema diff."""
    return VersionSignal(version=version, trained_on=trained_on, changes=tuple(changes))
