#!/usr/bin/env python
"""Transparent Anthropic-format proxy that pins OpenRouter to a single provider.

Claude Code speaks the Anthropic Messages protocol but cannot set OpenRouter's
`provider` routing field. This proxy sits in front of OpenRouter's Anthropic-skin
endpoint, injects `provider: {only: [<PROVIDER>], allow_fallbacks: false}` into
every request body, and streams the response back unchanged. With fallbacks off,
a request either runs on the pinned provider or fails, so the served provider and
its quantization are guaranteed rather than merely likely.

Env:
  OPENROUTER_API_KEY  required, the real key (never logged)
  DS_PROXY_PORT       listen port (default 8788)
  DS_PROXY_PROVIDER   provider slug to pin (default "deepinfra")
  DS_PROXY_LOG        provenance log path (default deepseek_proxy.log)
"""
import os, json, asyncio, datetime, aiohttp
from aiohttp import web

UPSTREAM  = "https://openrouter.ai/api"
KEY       = os.environ["OPENROUTER_API_KEY"]
PROVIDER  = os.environ.get("DS_PROXY_PROVIDER", "deepinfra")
# Let OpenRouter load-balance across all providers at or above a quantization floor, so
# routing never lands on a degraded low-bit build; quality is guaranteed, provider may
# vary. Floor is configurable per lane (DS_PROXY_QUANT), e.g. "fp8" or "bf16,fp8".
QUANT     = [q.strip() for q in os.environ.get("DS_PROXY_QUANT", "fp8").split(",") if q.strip()]
PIN       = {"quantizations": QUANT, "allow_fallbacks": True}
# Optional provider preference order (DS_PROXY_ORDER, comma-separated), tried first
# before other providers at the quant floor. Used to prefer a host observed to be
# reliable (e.g. BaseTen) while still allowing fp8 fallback.
_ORDER    = [p.strip() for p in os.environ.get("DS_PROXY_ORDER", "").split(",") if p.strip()]
if _ORDER:
    PIN["order"] = _ORDER
# Optional reasoning-effort injection for models whose effort Claude Code cannot set
# (non-Anthropic models via the skin). e.g. DS_PROXY_REASONING="high".
REASONING = os.environ.get("DS_PROXY_REASONING") or None
LOGPATH   = os.environ.get("DS_PROXY_LOG", "deepseek_proxy.log")
# Transient upstream statuses to retry. DeepInfra capacity blips surface as 404
# model_not_found under a hard provider pin; retrying a few times with backoff
# absorbs them so a momentary blip does not kill a whole agent run.
RETRY_STATUS = {404, 408, 409, 425, 429, 500, 502, 503, 504}
MAX_TRIES    = int(os.environ.get("DS_PROXY_RETRIES", "6"))

def log(msg):
    with open(LOGPATH, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")

async def handle(request):
    raw = await request.read()
    model = "?"
    body = raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "model" in obj:
            model = obj.get("model", "?")
            obj["provider"] = PIN           # the pin
            if REASONING:
                obj["reasoning_effort"] = REASONING
            body = json.dumps(obj).encode()
    except Exception:
        pass
    hdrs = {}
    for k, v in request.headers.items():
        if k.lower() in ("host", "authorization", "x-api-key", "content-length"):
            continue
        hdrs[k] = v
    hdrs["Authorization"] = f"Bearer {KEY}"
    hdrs["Content-Type"]  = "application/json"
    url = UPSTREAM + request.path_qs
    session = request.app["session"]
    for attempt in range(1, MAX_TRIES + 1):
        try:
            up = await session.post(url, data=body, headers=hdrs)
        except Exception as e:
            log(f"{request.path} model={model} attempt={attempt} CONNERR {type(e).__name__}: {e}")
            if attempt < MAX_TRIES:
                await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 3))
                continue
            return web.json_response({"type": "error", "error": {"message": str(e)}}, status=502)
        # Retry transient upstream errors (e.g. DeepInfra 404 capacity blips) before
        # the client ever sees them, so a momentary blip does not kill an agent run.
        if up.status in RETRY_STATUS and attempt < MAX_TRIES:
            await up.read()
            up.release()
            log(f"{request.path} model={model} attempt={attempt} retry status={up.status}")
            await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 3))
            continue
        served = up.headers.get("x-provider-name", "?")
        log(f"{request.path} model={model} served={served} status={up.status} attempt={attempt}")
        resp = web.StreamResponse(
            status=up.status,
            headers={"Content-Type": up.headers.get("Content-Type", "application/json")},
        )
        await resp.prepare(request)
        try:
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
        finally:
            up.release()
        return resp

async def health(request):
    return web.Response(text=f"ok pin={PROVIDER}\n")

async def on_start(app):
    app["session"] = aiohttp.ClientSession()

async def on_clean(app):
    await app["session"].close()

def main():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_clean)
    port = int(os.environ.get("DS_PROXY_PORT", "8788"))
    log(f"start pin={PROVIDER} port={port}")
    web.run_app(app, host="127.0.0.1", port=port, print=None)

if __name__ == "__main__":
    main()
