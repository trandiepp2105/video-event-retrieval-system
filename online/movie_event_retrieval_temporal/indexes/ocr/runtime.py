from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


def _healthcheck(base_url: str, timeout_sec: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/health"
    try:
        with request.urlopen(url, timeout=timeout_sec) as response:
            return int(getattr(response, "status", 0)) == 200
    except Exception:
        return False


@dataclass
class MeiliSearchRuntime:
    base_url: str
    process: subprocess.Popen | None
    started_by_us: bool

    def shutdown(self) -> None:
        if self.process is None or not self.started_by_us:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class MeiliSearchRuntimeManager:
    def ensure_running(
        self,
        *,
        base_url: str,
        api_key: str | None,
        binary_path: Path | None,
        db_path: Path | None,
        wait_timeout_sec: float = 30.0,
    ) -> MeiliSearchRuntime:
        if _healthcheck(base_url):
            return MeiliSearchRuntime(base_url=base_url, process=None, started_by_us=False)

        if binary_path is None:
            raise RuntimeError(
                f"Meilisearch chua chay tai {base_url} va khong co binary_path de auto-start."
            )
        if not binary_path.exists():
            raise FileNotFoundError(f"Khong tim thay meilisearch binary: {binary_path}")

        env = os.environ.copy()
        if api_key:
            env["MEILI_MASTER_KEY"] = api_key

        args = [str(binary_path)]
        http_addr = base_url.removeprefix("http://").removeprefix("https://")
        args.extend(["--http-addr", http_addr])
        if db_path is not None:
            db_path.mkdir(parents=True, exist_ok=True)
            args.extend(["--db-path", str(db_path)])

        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        started_at = time.time()
        while time.time() - started_at < wait_timeout_sec:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Meilisearch process da thoat som voi ma {process.returncode}. "
                    f"Khong the start local server tai {base_url}."
                )
            if _healthcheck(base_url):
                return MeiliSearchRuntime(base_url=base_url, process=process, started_by_us=True)
            time.sleep(0.5)

        process.terminate()
        raise RuntimeError(f"Het thoi gian doi Meilisearch san sang tai {base_url}.")
