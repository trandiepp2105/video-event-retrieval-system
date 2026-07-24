from __future__ import annotations

import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


@dataclass
class MeiliSearchRuntime:
    base_url: str
    process: subprocess.Popen[str] | None = None
    started_by_this_process: bool = False

    def shutdown(self) -> None:
        if not self.started_by_this_process or self.process is None:
            return
        print(f"[OCR] Shutting down Meilisearch started by this process at {self.base_url}")
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
        timeout_sec: float = 60.0,
    ) -> MeiliSearchRuntime:
        print(f"[OCR] Checking Meilisearch health at {base_url} ...")
        if self._is_healthy(base_url, api_key):
            print(f"[OCR] Meilisearch already running at {base_url}")
            return MeiliSearchRuntime(base_url=base_url, process=None, started_by_this_process=False)
        if binary_path is None:
            raise RuntimeError(f"Meilisearch chua chay tai {base_url} va khong co binary_path de auto-start.")
        if not binary_path.is_file():
            raise FileNotFoundError(f"Khong tim thay meilisearch binary: {binary_path}")

        host_port = base_url.removeprefix("http://").removeprefix("https://")
        cmd = [str(binary_path), "--http-addr", host_port]
        if db_path is not None:
            db_path.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--db-path", str(db_path)])
        if api_key:
            cmd.extend(["--master-key", api_key])

        print(f"[OCR] Starting Meilisearch from binary: {binary_path}")
        print(f"[OCR] Meilisearch db path: {db_path if db_path is not None else '(default internal path)'}")
        print(f"[OCR] Meilisearch http addr: {host_port}")
        startup_log_path = None
        if db_path is not None:
            startup_log_path = db_path / "meilisearch.log"
            print(f"[OCR] Meilisearch log path: {startup_log_path}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            close_fds=True,
            env=os.environ.copy(),
        )
        selector = selectors.DefaultSelector()
        if process.stdout is not None:
            selector.register(process.stdout, selectors.EVENT_READ)
        startup_logs: list[str] = []

        started = time.time()
        while (time.time() - started) < timeout_sec:
            for key, _mask in selector.select(timeout=0):
                line = key.fileobj.readline()
                if line:
                    startup_logs.append(line.rstrip("\n"))
            if process.poll() is not None:
                if process.stdout is not None:
                    remainder = process.stdout.read()
                    if remainder:
                        startup_logs.extend(remainder.splitlines())
                if startup_log_path is not None and startup_logs:
                    startup_log_path.write_text("\n".join(startup_logs) + "\n", encoding="utf-8")
                excerpt = "\n".join(startup_logs[-20:]) if startup_logs else "(khong co startup log)"
                raise RuntimeError(
                    "Meilisearch process da thoat som voi ma "
                    f"{process.returncode}. Startup log:\n{excerpt}"
                )
            if self._is_healthy(base_url, api_key):
                if startup_log_path is not None and startup_logs:
                    startup_log_path.write_text("\n".join(startup_logs) + "\n", encoding="utf-8")
                selector.close()
                print(f"[OCR] Meilisearch is healthy at {base_url}")
                return MeiliSearchRuntime(base_url=base_url, process=process, started_by_this_process=True)
            time.sleep(0.5)

        process.terminate()
        if process.stdout is not None:
            remainder = process.stdout.read()
            if remainder:
                startup_logs.extend(remainder.splitlines())
        if startup_log_path is not None and startup_logs:
            startup_log_path.write_text("\n".join(startup_logs) + "\n", encoding="utf-8")
        selector.close()
        raise RuntimeError(f"Het thoi gian doi Meilisearch san sang tai {base_url}.")

    @staticmethod
    def _is_healthy(base_url: str, api_key: str | None) -> bool:
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = request.Request(base_url.rstrip("/") + "/health", headers=headers)
            with request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (error.URLError, error.HTTPError):
            return False
