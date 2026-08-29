"""Minimal JSON-over-HTTPS helper built on the standard library.

Deliberately dependency-free: the harness must install and run from a clean
Python with no wheels available, because a benchmark that cannot be
reinstalled years later cannot be reproduced.  ``urllib`` also picks up
``HTTPS_PROXY`` from the environment on its own, which matters in sandboxed CI.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from .base import LLMError, RetryableError

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        message = f"HTTP {exc.code} from {url}: {detail}"
        if exc.code in _RETRYABLE_STATUS:
            raise RetryableError(message) from exc
        raise LLMError(message) from exc
    except urllib.error.URLError as exc:
        raise RetryableError(f"network error contacting {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RetryableError(f"timeout contacting {url}") from exc
    except json.JSONDecodeError as exc:
        raise RetryableError(f"malformed JSON from {url}: {exc}") from exc
