import ast

import httpx

from restock.saleor import build
from restock.saleor.seed import EXTENSION_APPS, GROW_ACCOUNTS, GROW_VOLATILE

# Mount points and targets of app extensions in Saleor 3.20 (saleor/app/types.py).
MOUNTS = {
    "customer_overview_create",
    "customer_overview_more_actions",
    "customer_details_more_actions",
    "product_overview_create",
    "product_overview_more_actions",
    "product_details_more_actions",
    "navigation_catalog",
    "navigation_orders",
    "navigation_customers",
    "navigation_discounts",
    "navigation_translations",
    "navigation_pages",
    "order_details_more_actions",
    "order_overview_create",
    "order_overview_more_actions",
}
TARGETS = {"popup", "app_page"}


class FakeRuntime:
    """Records what a build asks of the runtime, in order."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, *args, *(kwargs.items())))
            return {"create_app_token": "token", "start_api": "http://saleor.test/"}.get(name)

        return record


def run_build(monkeypatch, versions) -> FakeRuntime:
    runtime = FakeRuntime()
    monkeypatch.setattr(build, "read_store", lambda client: None)
    monkeypatch.setattr(build, "plan_products", lambda store, count, seed: [])
    monkeypatch.setattr(build, "create_products", lambda client, plans: None)
    monkeypatch.setattr(
        build, "SaleorClient", lambda url, token: httpx.Client(transport=httpx.MockTransport(None))
    )
    build.build_store(runtime, versions, seed=3, log=lambda message: None)
    return runtime


def test_scripts_are_valid_python_once_filled_in():
    ast.parse(GROW_ACCOUNTS.format(seed=0, customers=5, orders=5))
    ast.parse(GROW_VOLATILE.format(seed=0, digital=5, apps=EXTENSION_APPS))


def test_extension_apps_use_mount_points_and_targets_saleor_knows():
    extensions = [extension for _, extensions in EXTENSION_APPS for extension in extensions]

    assert len({name for name, _ in EXTENSION_APPS}) == len(EXTENSION_APPS)
    assert {mount for _, mount, _ in extensions} == MOUNTS
    assert {target for _, _, target in extensions} == TARGETS


def test_the_first_release_is_grown_before_any_copy_is_made(monkeypatch):
    runtime = run_build(monkeypatch, ["3.20", "3.21", "3.22"])

    scripts = [call[4] for call in runtime.calls if call[:3] == ("manage", "3.20", "shell")]
    assert scripts == [
        GROW_ACCOUNTS.format(seed=3, customers=180, orders=400),
        GROW_VOLATILE.format(seed=3, digital=60, apps=EXTENSION_APPS),
    ]

    steps = [call[:3] for call in runtime.calls if call[0] in ("manage", "create_database")]
    grown = steps.index(("manage", "3.20", "update_search_indexes"))
    assert steps[grown + 1 :] == [
        ("create_database", "3.21", ("copy_of", "3.20")),
        ("manage", "3.21", "migrate"),
        ("create_database", "3.22", ("copy_of", "3.21")),
        ("manage", "3.22", "migrate"),
    ]
