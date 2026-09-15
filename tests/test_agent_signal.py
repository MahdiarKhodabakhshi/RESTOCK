from restock.agent.signal import VersionSignal, version_signal
from restock.schema.diff import Change, ChangeKind


def test_signal_without_training_version_is_just_the_release():
    assert version_signal("3.23.0").text() == "API version: 3.23.0"


def test_signal_reports_no_breaking_changes_when_the_diff_is_empty():
    text = version_signal("3.23.0", trained_on="3.20.0").text()
    assert "Trained on: 3.20.0" in text
    assert "none" in text


def test_signal_lists_only_query_side_breaking_coordinates():
    changes = (
        Change(ChangeKind.FIELD_REMOVED, "Product", "slug", query_side=True),
        Change(ChangeKind.FIELD_ADDED, "Product", "rating", query_side=True),
        Change(ChangeKind.FIELD_REMOVED, "Token", "value", query_side=False),
        Change(ChangeKind.DEPRECATED, "Product", "name", query_side=True),
    )
    text = VersionSignal("3.23.0", "3.20.0", changes).text()
    assert "field_removed  Product.slug" in text
    assert "Token" not in text
    assert "rating" not in text
    assert "deprecated" not in text
