"""Build one store and serve it from several Saleor releases."""

from collections.abc import Callable, Sequence

from restock.saleor.client import SaleorClient
from restock.saleor.runtime import Runtime
from restock.saleor.seed import (
    EXTENSION_APPS,
    GROW_ACCOUNTS,
    GROW_VOLATILE,
    create_products,
    plan_products,
    read_store,
)


def _progress(message: str) -> None:
    print(message, flush=True)


def build_store(
    runtime: Runtime,
    versions: Sequence[str],
    *,
    products: int = 240,
    customers: int = 180,
    orders: int = 400,
    digital: int = 60,
    seed: int = 0,
    log: Callable[[str], None] = _progress,
) -> None:
    """Create the store on the first release and upgrade a copy of it for each later one.

    ``versions`` must be in release order, because every database is migrated
    from the one before it. Existing databases of these releases are replaced.
    """
    first, *later = versions
    runtime.pull()
    for version in versions:
        runtime.pull(version)
        runtime.stop_api(version)
    runtime.start_postgres()

    log(f"{first}: creating the sample store")
    runtime.create_database(first)
    runtime.manage(first, "migrate")
    runtime.manage(first, "populatedb", "--createsuperuser")
    token = runtime.create_app_token(first)

    log(f"{first}: adding {products} products")
    with SaleorClient(runtime.start_api(first), token) as client:
        create_products(client, plan_products(read_store(client), products, seed))
    runtime.stop_api(first)

    log(f"{first}: adding {customers} customers and {orders} orders")
    script = GROW_ACCOUNTS.format(seed=seed, customers=customers, orders=orders)
    runtime.manage(first, "shell", "-c", script)

    log(f"{first}: adding digital content, customer history and app extensions")
    script = GROW_VOLATILE.format(seed=seed, digital=digital, apps=EXTENSION_APPS)
    runtime.manage(first, "shell", "-c", script)
    runtime.manage(first, "update_search_indexes")

    previous = first
    for version in later:
        log(f"{version}: upgrading a copy of {previous}")
        runtime.create_database(version, copy_of=previous)
        runtime.manage(version, "migrate")
        previous = version
