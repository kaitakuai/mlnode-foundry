"""Content-Type injector — S4 form of patches/0001-content-type-middleware.

Same change the Stage-3 patch makes, expressed as a runner-style Python
patch so it can also be applied on top of a PUBLISHED mlnode image (mode
"upstream-overlay"), where no Stage-3 build exists to carry patches/.

Why the change is needed: the network DAPI (Go-http-client) POSTs JSON to
/api/v1/inference/* without a Content-Type header; the FastAPI/Starlette
pin then refuses the body. The injector adds the header for POST /api/*
when it is missing, wrapping ProxyMiddleware from the outside.

Idempotent: skips if the class is already present (e.g. the image was
built from our Stage 3, which applies the patch already).
"""

import sys

APP_PY = "/app/packages/api/src/api/app.py"
PROXY_PY = "/app/packages/api/src/api/proxy.py"

IMPORT_OLD = "from api.proxy import ProxyMiddleware,"
IMPORT_NEW = "from api.proxy import ContentTypeInjector, ProxyMiddleware,"

ADD_MW_ANCHOR = "app.add_middleware(ProxyMiddleware)"
ADD_MW_NEW = """app.add_middleware(ProxyMiddleware)
# Wrap ProxyMiddleware so Content-Type is injected before downstream parsing.
# Last add_middleware = outermost = runs first on each request.
app.add_middleware(ContentTypeInjector)"""

PROXY_ANCHOR = "health_check_task: Optional[asyncio.Task] = None"

INJECTOR = '''
class ContentTypeInjector:
    """Plain ASGI middleware: inject ``Content-Type: application/json`` for
    ``POST /api/*`` requests when the header is missing.

    Network DAPI (Go-http-client/1.1) does not set ``Content-Type`` on JSON
    POSTs to ``/api/v1/inference/up`` and ``/api/v1/inference/pow/init/generate``.
    Pydantic v2, shipped with the FastAPI/Starlette pin used by the vllm 0.20
    base, requires ``Content-Type: application/json`` to interpret the body
    as JSON; without it the bytes are surfaced to ``model_validate`` as a
    string value, producing a 422 ``model_attributes_type`` error
    (``Input should be a valid dictionary or object``).

    The vllm 0.15.1 base predates Pydantic v2's strictness, which is why
    production mlnode-009 accepts the same DAPI request shape unchanged and
    only the vllm 0.20 mlnode image hits the regression.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path", "").startswith("/api/")
            and not any(h[0] == b"content-type" for h in scope["headers"])
        ):
            scope = dict(scope)
            scope["headers"] = list(scope["headers"]) + [
                (b"content-type", b"application/json")
            ]
        await self.app(scope, receive, send)
'''


def main() -> int:
    try:
        proxy = open(PROXY_PY).read()
        app = open(APP_PY).read()
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: content-type-injector: {e}\n")
        return 1

    if "class ContentTypeInjector" in proxy and "ContentTypeInjector" in app:
        print("content-type-injector: already present; no-op")
        return 0

    if "class ContentTypeInjector" not in proxy:
        if PROXY_ANCHOR not in proxy:
            sys.stderr.write(
                "ERROR: content-type-injector: anchor not found in proxy.py; "
                "upstream may have been refactored\n"
            )
            return 1
        idx = proxy.index(PROXY_ANCHOR) + len(PROXY_ANCHOR)
        proxy = proxy[:idx] + "\n\n\n" + INJECTOR.strip("\n") + proxy[idx:]
        open(PROXY_PY, "w").write(proxy)
        print("content-type-injector: ContentTypeInjector added to proxy.py")

    if "ContentTypeInjector" not in app:
        if IMPORT_OLD not in app or ADD_MW_ANCHOR not in app:
            sys.stderr.write(
                "ERROR: content-type-injector: app.py anchors not found; "
                "upstream may have been refactored\n"
            )
            return 1
        app = app.replace(IMPORT_OLD, IMPORT_NEW, 1)
        app = app.replace(ADD_MW_ANCHOR, ADD_MW_NEW, 1)
        open(APP_PY, "w").write(app)
        print("content-type-injector: middleware wired in app.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
