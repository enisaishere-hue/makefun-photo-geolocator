// photo-geolocator-01 entry for @aqist (make.fun).
// One claude-sonnet-5-5 call per photo through the gateway, with a short
// GeoGuessr-style reasoning prompt, then the answers are posted in photo order.
// A photo whose call is refused or whose answer holds no coordinates falls back
// to a fixed prior point: the median of the 75 public practice photos
// (30.253517, 21.031961), which misses by ~7,294 km on average instead of the
// 20,015 km an unanswered photo counts. The fallback is one constant, not a
// table of photos. Deterministic given the model's answers: every request is a
// function of the photo's bytes and the fixed prompt, calls run four at a time
// and answers are posted in photo order, no wall clock, no unseeded randomness.
//
// Runs as the grader runs an entry, `node /bot/main.ts` on Node 22 (erasable
// TypeScript, the standard library only), with MAKEFUN_ENV_URL, ANTHROPIC_BASE_URL
// and ANTHROPIC_API_KEY from the trial. Model calls run on the Anthropic key
// pushed to the competition before submitting; with no key pushed every call is
// answered 401 and every photo takes the fallback point.

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 1024;
const CONCURRENCY = 4;
const PROMPT =
  'Look at this street-level photo and work out where on Earth it was taken. ' +
  'Use the visible clues: language and script on signs, which side of the road vehicles drive on, ' +
  'road markings and sign shapes, vegetation and climate, architecture, vehicles and license plates, ' +
  'and any landmark. Reason briefly, then give your best estimate as one line of JSON on its own last line: ' +
  '{"lat": <degrees north>, "lon": <degrees east>}.';
/** Retried with a fixed backoff: a provider's transient errors. A gateway refusal (4xx) is final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;
/** Fallback prior: median of the 75 public practice photos (see header). */
const FALLBACK = { lat: 30.253517, lon: 21.031961 };

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** The model's answer to one photo, as text; null when the gateway refused the call (final) or the request failed. */
async function ask(jpeg: Buffer): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${anthropicUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body,
      });
    } catch (err) {
      process.stderr.write(`geolocator: model request failed: ${(err as Error).message}\n`);
      return null;
    }
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geolocator: the model call was refused (HTTP ${res.status}): ${text.slice(0, 300)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let refused = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      // One re-ask when the call succeeded but the answer held no coordinates:
      // the same prompt and photo, so the replay makes the same calls.
      let c: { lat: number; lon: number } | null = null;
      for (let attempt = 0; attempt < 2 && c === null; attempt += 1) {
        const text = await ask(jpeg);
        if (text === null) {
          refused += 1;
          break;
        }
        c = coordinates(text);
        if (c === null) process.stderr.write(`geolocator: no coordinates in the answer for ${p.id} (attempt ${attempt + 1})\n`);
      }
      guesses[i] = c === null ? { id: p.id, lat: FALLBACK.lat, lon: FALLBACK.lon } : { id: p.id, lat: c.lat, lon: c.lon };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return refused > 0 ? 1 : 0;
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  process.stdout.write(`geolocator: ${answers.length} of ${photos.length} photos answered (${refused} model calls refused): HTTP ${res.status} ${await res.text()}`);
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`geolocator: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
