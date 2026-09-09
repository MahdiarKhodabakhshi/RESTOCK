"""Run Saleor releases locally with Apptainer.

One Postgres server holds a database per release and every release runs its
own API process. The database of a later release is a copy of an earlier one
with that release's migrations applied, so all versions serve the same store.
"""

import os
import secrets
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import httpx

SALEOR_IMAGE = "docker://ghcr.io/saleor/saleor:{version}"
POSTGRES_IMAGE = "docker://postgres:15-alpine"
DB_USER = "saleor"


def api_port(version: str) -> int:
    """Port of a 3.x release: 3.20 is served on 8020."""
    return 8000 + int(version.split(".")[1])


def api_url(version: str) -> str:
    return f"http://127.0.0.1:{api_port(version)}/graphql/"


def database_name(version: str) -> str:
    return "saleor_" + version.replace(".", "_")


def database_url(socket_dir: Path, version: str) -> str:
    """Connection URL over the Unix socket in ``socket_dir``."""
    return f"postgres://{DB_USER}@{quote(str(socket_dir), safe='')}/{database_name(version)}"


class Runtime:
    def __init__(self, root: Path = Path("data")):
        self.root = root.resolve()
        self.images = self.root / "images"
        self.pgdata = self.root / "pg"
        self.media = self.root / "media"
        self.logs = self.root / "logs"
        # Postgres trusts every connection on its socket, so keep the socket private.
        self.run = self.root / "run"
        self._children: dict[str, subprocess.Popen] = {}

    # -- images ------------------------------------------------------------------------

    def pull(self, version: str | None = None) -> Path:
        """Fetch the image of a Saleor release, or of Postgres when no version is given."""
        image = self._saleor_image(version) if version else self._postgres_image
        if not image.exists():
            source = SALEOR_IMAGE.format(version=version) if version else POSTGRES_IMAGE
            self.images.mkdir(parents=True, exist_ok=True)
            _run(["apptainer", "pull", str(image), source])
        return image

    # -- postgres ----------------------------------------------------------------------

    def start_postgres(self) -> None:
        if self.is_running("postgres"):
            return
        self._prepare_dirs()
        if not (self.pgdata / "PG_VERSION").exists():
            self._postgres("initdb", "-D", str(self.pgdata), "-U", DB_USER, "--auth=trust")
        server = [
            "postgres",
            "-D",
            str(self.pgdata),
            "-k",
            str(self.run),
            "-c",
            "listen_addresses=",
        ]
        self._spawn("postgres", self._postgres_argv(*server))
        self._wait_for("postgres", lambda: self._postgres("pg_isready", "-q", check=False) == 0)

    def stop_postgres(self) -> None:
        # SIGINT asks Postgres for a fast shutdown instead of waiting for clients.
        self._stop("postgres", signal.SIGINT)

    def create_database(self, version: str, *, copy_of: str | None = None) -> None:
        """Create the database of a release, replacing any existing one.

        ``copy_of`` names a release whose database is used as the starting
        point. Its API must be stopped, since Postgres cannot copy a database
        that has open connections.
        """
        name = database_name(version)
        self._postgres("dropdb", "-U", DB_USER, "--if-exists", name)
        template = ["-T", database_name(copy_of)] if copy_of else []
        self._postgres("createdb", "-U", DB_USER, *template, name)

    # -- saleor ------------------------------------------------------------------------

    def manage(self, version: str, *args: str) -> str:
        """Run a Django management command of a release and return its output."""
        return _run(self._saleor_argv(version, "python", "manage.py", *args), self._env(version))

    def start_api(self, version: str, workers: int = 4) -> str:
        """Serve a release and return its GraphQL endpoint."""
        name = f"api_{version}"
        if not self.is_running(name):
            server = ["uvicorn", "saleor.asgi:application", "--host", "127.0.0.1"]
            server += ["--port", str(api_port(version)), "--workers", str(workers)]
            self._spawn(name, self._saleor_argv(version, *server), self._env(version))
            self._wait_for(name, lambda: _answers(api_url(version)))
        return api_url(version)

    def stop_api(self, version: str) -> None:
        self._stop(f"api_{version}")

    # -- processes ---------------------------------------------------------------------

    def is_running(self, name: str) -> bool:
        child = self._children.get(name)
        if child is not None:
            return child.poll() is None
        pid = self._pid(name)
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def running(self) -> list[str]:
        names = sorted(path.stem for path in self.run.glob("*.pid"))
        return [name for name in names if self.is_running(name)]

    def _spawn(self, name: str, argv: list[str], env: dict[str, str] | None = None) -> None:
        self._prepare_dirs()
        with (self.logs / f"{name}.log").open("ab") as log:
            child = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        self._children[name] = child
        (self.run / f"{name}.pid").write_text(str(child.pid))

    def _stop(self, name: str, sig: signal.Signals = signal.SIGTERM, timeout: float = 60) -> None:
        pid = self._pid(name)
        if pid is not None and self.is_running(name):
            os.killpg(pid, sig)
            deadline = time.monotonic() + timeout
            while self.is_running(name):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} did not stop")
                time.sleep(0.2)
        self._children.pop(name, None)
        (self.run / f"{name}.pid").unlink(missing_ok=True)

    def _pid(self, name: str) -> int | None:
        path = self.run / f"{name}.pid"
        return int(path.read_text()) if path.exists() else None

    def _wait_for(self, name: str, ready: Callable[[], bool], timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            if not self.is_running(name):
                raise RuntimeError(f"{name} exited, see {self.logs / name}.log")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name} is not ready, see {self.logs / name}.log")
            time.sleep(0.5)

    # -- containers --------------------------------------------------------------------

    @property
    def _postgres_image(self) -> Path:
        return self.images / "postgres_15.sif"

    def _saleor_image(self, version: str) -> Path:
        return self.images / f"saleor_{version}.sif"

    def _postgres_argv(self, *command: str) -> list[str]:
        return ["apptainer", "exec", str(self._postgres_image), *command]

    def _postgres(self, *command: str, check: bool = True) -> int:
        env = {**os.environ, "APPTAINERENV_PGHOST": str(self.run)}
        result = subprocess.run(self._postgres_argv(*command), env=env, capture_output=True)
        if check and result.returncode:
            raise RuntimeError(result.stderr.decode()[-2000:])
        return result.returncode

    def _saleor_argv(self, version: str, *command: str) -> list[str]:
        bind = f"{self.media}:/app/media"
        image = str(self._saleor_image(version))
        return ["apptainer", "exec", "--pwd", "/app", "--bind", bind, image, *command]

    def _env(self, version: str) -> dict[str, str]:
        self._prepare_dirs()
        settings = {
            "DATABASE_URL": database_url(self.run, version),
            "SECRET_KEY": self._secret_key(),
            "RSA_PRIVATE_KEY": self._rsa_key(),
            "ALLOWED_HOSTS": "127.0.0.1,localhost",
            "ALLOWED_CLIENT_HOSTS": "127.0.0.1,localhost",
            "PUBLIC_URL": f"http://127.0.0.1:{api_port(version)}/",
            "DEBUG": "False",
            "PLAYGROUND_ENABLED": "False",
        }
        return {**os.environ, **{f"APPTAINERENV_{key}": value for key, value in settings.items()}}

    def _secret_key(self) -> str:
        path = self.run / "secret_key"
        if not path.exists():
            _write_private(path, secrets.token_urlsafe(48))
        return path.read_text()

    def _rsa_key(self) -> str:
        path = self.run / "rsa_key.pem"
        if not path.exists():
            _write_private(path, _run(["openssl", "genrsa", "2048"]))
        return path.read_text()

    def _prepare_dirs(self) -> None:
        for directory in (self.media, self.logs, self.run):
            directory.mkdir(parents=True, exist_ok=True)
        self.run.chmod(0o700)


def _run(argv: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(argv[:6])} failed:\n{result.stderr[-2000:]}")
    return result.stdout


def _write_private(path: Path, content: str) -> None:
    path.touch(mode=0o600)
    path.write_text(content)


def _answers(url: str) -> bool:
    try:
        response = httpx.post(url, json={"query": "{ shop { name } }"}, timeout=5)
    except httpx.TransportError:
        return False
    return response.status_code == 200
