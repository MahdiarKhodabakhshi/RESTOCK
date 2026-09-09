from pathlib import Path

from restock.saleor.runtime import Runtime, api_port, api_url, database_name, database_url


def test_each_release_has_its_own_port_and_database():
    assert api_port("3.20") == 8020
    assert api_port("3.23") == 8023
    assert api_url("3.21") == "http://127.0.0.1:8021/graphql/"
    assert database_name("3.20") == "saleor_3_20"


def test_database_url_points_at_the_socket_directory():
    url = database_url(Path("/srv/restock/run"), "3.22")

    assert url == "postgres://saleor@%2Fsrv%2Frestock%2Frun/saleor_3_22"


def test_runtime_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runtime = Runtime(Path("data"))

    assert runtime.run == tmp_path / "data" / "run"
    assert runtime.running() == []


def test_spawned_process_is_tracked_until_stopped(tmp_path):
    runtime = Runtime(tmp_path)

    runtime._spawn("sleeper", ["sleep", "60"])

    assert runtime.is_running("sleeper")
    assert runtime.running() == ["sleeper"]
    # A second runtime sees it through the PID file, as another CLI call would.
    assert Runtime(tmp_path).is_running("sleeper")

    runtime._stop("sleeper")

    assert not runtime.is_running("sleeper")
    assert Runtime(tmp_path).running() == []


def test_process_that_exited_is_not_reported(tmp_path):
    runtime = Runtime(tmp_path)

    runtime._spawn("short", ["true"])
    runtime._children["short"].wait()

    assert runtime.running() == []
    runtime._stop("short")
    assert not (runtime.run / "short.pid").exists()


def test_runtime_directory_is_private(tmp_path):
    runtime = Runtime(tmp_path)

    runtime._spawn("short", ["true"])

    assert runtime.run.stat().st_mode & 0o777 == 0o700
