// TokIO data proxy: a Cloudflare Worker that lets the web terminal read free
// market data from sources that don't send CORS headers (Yahoo, Cboe).
//
// It is deliberately narrow: GET only, a fixed list of upstream URL shapes,
// strict parameter checks, responses cached at the edge, and CORS granted only
// to the TokIO site and localhost. It is not a general-purpose proxy.

const ALLOWED_ORIGINS = [
  "https://kuzi-dev.github.io",
  /^http:\/\/localhost:\d+$/,
  /^http:\/\/127\.0\.0\.1:\d+$/,
];

const SYMBOL = /^[A-Za-z0-9^_.=\-]{1,15}$/;
const RANGES = new Set(["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "max", "ytd"]);
const INTERVALS = new Set(["1d", "1wk", "1mo"]);
const PC_KINDS = new Set(["equity", "index", "total"]);
const DATE = /^\d{4}-\d{2}-\d{2}$/;

// Each route: path pattern -> upstream URL builder (returns null to reject) and cache seconds.
const ROUTES = [
  {
    // /yahoo/chart/SPY?range=10y&interval=1d
    re: /^\/yahoo\/chart\/([^/]+)$/,
    ttl: 3600,
    build: (m, q) => {
      const sym = decodeURIComponent(m[1]);
      const range = q.get("range") || "10y";
      const interval = q.get("interval") || "1d";
      if (!SYMBOL.test(sym) || !RANGES.has(range) || !INTERVALS.has(interval)) return null;
      return `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?interval=${interval}&range=${range}&includeAdjustedClose=true`;
    },
  },
  {
    // /cboe/options/SPY  (delayed option chain; indices like _SPX)
    re: /^\/cboe\/options\/([^/]+)$/,
    ttl: 900,
    build: (m) => {
      const sym = decodeURIComponent(m[1]);
      if (!SYMBOL.test(sym)) return null;
      return `https://cdn.cboe.com/api/global/delayed_quotes/options/${encodeURIComponent(sym.toUpperCase())}.json`;
    },
  },
  {
    // /cboe/putcall/equity  (2006-2019 history CSV)
    re: /^\/cboe\/putcall\/([a-z]+)$/,
    ttl: 86400,
    build: (m) => (PC_KINDS.has(m[1]) ? `https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/${m[1]}pc.csv` : null),
  },
  {
    // /cboe/daily/2024-05-01  (daily options market statistics, 2019-10 onward)
    re: /^\/cboe\/daily\/(\d{4}-\d{2}-\d{2})$/,
    ttl: 86400,
    build: (m) => (DATE.test(m[1]) ? `https://cdn.cboe.com/data/us/options/market_statistics/daily/${m[1]}_daily_options` : null),
  },
];

function allowedOrigin(origin) {
  if (!origin) return null;
  return ALLOWED_ORIGINS.some((o) => (typeof o === "string" ? o === origin : o.test(origin))) ? origin : null;
}

function corsHeaders(origin) {
  const h = { "Vary": "Origin" };
  if (origin) {
    h["Access-Control-Allow-Origin"] = origin;
    h["Access-Control-Allow-Methods"] = "GET, OPTIONS";
    h["Access-Control-Max-Age"] = "86400";
  }
  return h;
}

function reply(status, body, origin, extra = {}) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin), ...extra },
  });
}

// ---------------------------------------------------------------------------
// Free model: visitors without their own OpenRouter key get a few questions a
// day on a free model, paid for by nobody (OpenRouter ":free" models) but sent
// with TokIO's key. Limits: allowed origins only, our two tools only, a burst
// limit and a per-IP daily quota, capped sizes and output tokens.
// ---------------------------------------------------------------------------
const TOOL_NAMES = new Set(["get_prices", "run_python"]);
const MAX_BODY = 400_000;
const MAX_MESSAGES = 80;

async function freeModel(request, env, origin) {
  const json = (status, obj) => reply(status, JSON.stringify(obj), origin);
  if (request.method !== "POST") return json(405, { error: "POST only" });
  if (!origin) return json(403, { error: "the free model is only available from the TokIO terminal" });
  if (!env.OPENROUTER_KEY) return json(503, { error: "the free model isn't set up yet; add your own OpenRouter key" });

  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  if (env.LLM_BURST) {
    const { success } = await env.LLM_BURST.limit({ key: ip });
    if (!success) return json(429, { error: "too many requests; wait a minute" });
  }

  const raw = await request.text();
  if (raw.length > MAX_BODY) return json(413, { error: "conversation too long for the free model; start a new one" });
  let body;
  try { body = JSON.parse(raw); } catch { return json(400, { error: "bad JSON" }); }
  const messages = body.messages, tools = body.tools || [];
  if (!Array.isArray(messages) || !messages.length || messages.length > MAX_MESSAGES) return json(400, { error: "bad messages" });
  if (messages[0].role !== "system" || !String(messages[0].content || "").startsWith("You are TokIO")) return json(400, { error: "not a TokIO conversation" });
  if (!Array.isArray(tools) || tools.some((t) => !TOOL_NAMES.has(t?.function?.name))) return json(400, { error: "unsupported tools" });

  // Per-IP daily quota (KV; eventually consistent, so it is a soft cap).
  const perDay = Number(env.FREE_CALLS_PER_DAY || 60);
  const day = new Date().toISOString().slice(0, 10);
  const qkey = `q:${day}:${ip}`;
  let used = 0;
  if (env.LLM_QUOTA) {
    used = Number(await env.LLM_QUOTA.get(qkey)) || 0;
    if (used >= perDay) return json(429, { error: `free limit reached for today (${perDay} model calls). Add your own OpenRouter key to keep going.` });
  }

  const up = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.OPENROUTER_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://kuzi-dev.github.io/Tokio-ai/",
      "X-Title": "TokIO Terminal (free)",
    },
    body: JSON.stringify({ model: env.FREE_MODEL, messages, tools, max_tokens: 3000 }),
  });
  const text = await up.text();
  if (up.ok && env.LLM_QUOTA) await env.LLM_QUOTA.put(qkey, String(used + 1), { expirationTtl: 2 * 86400 });
  return new Response(text, {
    status: up.status,
    headers: { "Content-Type": "application/json", "X-Free-Calls-Left": String(Math.max(0, perDay - used - 1)), "Access-Control-Expose-Headers": "X-Free-Calls-Left", ...corsHeaders(origin) },
  });
}

export default {
  async fetch(request, env, ctx) {
    const origin = allowedOrigin(request.headers.get("Origin"));
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      const h = corsHeaders(origin);
      if (origin && url.pathname === "/llm/chat") {
        h["Access-Control-Allow-Methods"] = "POST, OPTIONS";
        h["Access-Control-Allow-Headers"] = "Content-Type";
      }
      return new Response(null, { status: 204, headers: h });
    }
    if (url.pathname === "/llm/chat") return freeModel(request, env, origin);
    if (url.pathname === "/llm/info") {
      return reply(200, JSON.stringify({ model: env.FREE_MODEL, per_day: Number(env.FREE_CALLS_PER_DAY || 60), enabled: Boolean(env.OPENROUTER_KEY) }), origin);
    }
    if (request.method !== "GET") return reply(405, '{"error":"GET only"}', origin);

    if (url.pathname === "/" || url.pathname === "/health") {
      return reply(200, JSON.stringify({ ok: true, routes: ["/yahoo/chart/{symbol}", "/cboe/options/{symbol}", "/cboe/putcall/{equity|index|total}", "/cboe/daily/{YYYY-MM-DD}"] }), origin);
    }

    let upstream = null, ttl = 0;
    for (const r of ROUTES) {
      const m = url.pathname.match(r.re);
      if (m) { upstream = r.build(m, url.searchParams); ttl = r.ttl; break; }
    }
    if (!upstream) return reply(404, '{"error":"unknown or invalid route"}', origin);

    // Edge cache keyed on the upstream URL, so every visitor shares one fetch.
    const cache = caches.default;
    const key = new Request(upstream, { method: "GET" });
    let res = await cache.match(key);
    if (!res) {
      const up = await fetch(upstream, { headers: { "User-Agent": "Mozilla/5.0 (TokIO data proxy)", "Accept": "*/*" } });
      if (!up.ok) return reply(up.status === 404 ? 404 : 502, JSON.stringify({ error: `upstream returned ${up.status}` }), origin);
      const body = await up.arrayBuffer();
      if (body.byteLength > 20e6) return reply(502, '{"error":"upstream response too large"}', origin);
      res = new Response(body, {
        headers: { "Content-Type": up.headers.get("Content-Type") || "application/octet-stream", "Cache-Control": `public, max-age=${ttl}` },
      });
      ctx.waitUntil(cache.put(key, res.clone()));
    }
    const out = new Response(res.body, res);
    for (const [k, v] of Object.entries(corsHeaders(origin))) out.headers.set(k, v);
    return out;
  },
};
