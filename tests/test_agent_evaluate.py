from restock.agent.episode import EpisodeResult
from restock.agent.evaluate import (
    BucketStats,
    EvalReport,
    expected_answer,
    format_report,
    task_bucket,
)
from restock.tasks.focus import Focus
from restock.tasks.task import Later, Task


def task(**kwargs) -> Task:
    defaults = dict(
        id="t1",
        question="q",
        query="{ shop { name } }",
        answer={"shop": {"name": "S"}},
        focus=Focus("shop", ("Shop",), ()),
        coordinates=["Query.shop", "Shop.name"],
    )
    defaults.update(kwargs)
    return Task(**defaults)


def test_task_bucket_prefers_broken_over_deprecated():
    assert task_bucket(task(broken=["Shop.name"])) == "changed"
    assert task_bucket(task(deprecated=["Shop.name"])) == "deprecated"
    assert task_bucket(task()) == "untouched"


def test_expected_answer_skips_gone_and_unresolved_on_later_release():
    base = task(
        later={
            "3.23": Later("gone"),
            "3.22": Later("same"),
            "3.21": Later("ported", query="{ shop { name } }", answer={"shop": {"name": "S"}}),
        }
    )
    assert expected_answer(base, "3.20") == base.answer
    assert expected_answer(base, "3.22") == base.answer
    assert expected_answer(base, "3.21") == {"shop": {"name": "S"}}
    assert expected_answer(base, "3.23") is None


def test_format_report_lists_buckets_and_overall():
    report = EvalReport(policy="memorised", release="3.20")
    changed = BucketStats()
    changed.add(EpisodeResult("memorised", "q", "{ x }", 0, True, 1.0))
    untouched = BucketStats()
    untouched.add(EpisodeResult("memorised", "q", None, 0, False, 0.0, error="no query"))
    report.by_bucket["changed"] = changed
    report.by_bucket["untouched"] = untouched
    text = format_report(report)
    assert "policy=memorised" in text
    assert "changed" in text and "untouched" in text and "overall" in text
    assert "acc=0.500" in text
