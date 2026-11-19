# tokio-data: the web terminal's data proxy

A Cloudflare Worker that lets `docs/terminal/` read free market data from
sources that don't send CORS headers. GET only, a fixed allowlist of upstream
URLs, strict parameter checks, edge caching, CORS only for the TokIO site and
localhost.

| Route | Upstream |
|---|---|
| `/yahoo/chart/{symbol}?range=10y&interval=1d` | Yahoo Finance chart API |
| `/cboe/options/{symbol}` | Cboe delayed option chain |
| `/cboe/putcall/{equity,index,total}` | Cboe put/call history CSV |
| `/cboe/daily/{YYYY-MM-DD}` | Cboe daily options statistics |

```
npm install
npx wrangler dev      # local, no account needed
npx wrangler login    # once
npx wrangler deploy   # -> https://tokio-data.<your-subdomain>.workers.dev
```

Deployed at https://tokio-data.tokio-data.workers.dev (set as `PROXY` in `docs/terminal/index.html`). For local
testing, open the terminal with `?proxy=http://127.0.0.1:8787`.
