from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEARCH_URL = "https://cf53-34-44-113-239.ngrok-free.app/search"
QUERY = (
    "Hai người mang các hộp có chữ CAKE tới cổng khu doanh trại, "
    "trình giấy tờ với lính gác rồi được cho phép đi vào"
)
TOP_K = 10
TIMEOUT_SEC = 60


def main() -> None:
    request_body = json.dumps(
        {
            "query": QUERY,
            "top_k": TOP_K,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        SEARCH_URL,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
        method="POST",
    )

    print(f"Sending request to: {SEARCH_URL}", flush=True)
    print(f"Query: {QUERY}", flush=True)
    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=TIMEOUT_SEC) as response:
            payload = json.loads(response.read().decode("utf-8"))
            print(f"HTTP status: {response.status}")
            print(f"Elapsed: {time.perf_counter() - started_at:.3f} seconds")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP error: {error.code} {error.reason}")
        print(response_body)
        raise
    except URLError as error:
        print(f"Connection error: {error.reason}")
        raise


if __name__ == "__main__":
    main()
