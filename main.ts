// photo-geolocator-01 entry for @aqist (make.fun).
// One claude-sonnet-5-5 call per photo through the gateway, with a short
// GeoGuessr-style reasoning prompt, then the answers are posted in photo order.
// The prompt distils the deduction order and clue tables of the geolocation
// and geoguessr methodology references (hierarchical elimination, bollard /
// pole / plate / road-line tables, verification gate) into vision-only
// reasoning: no web fetch, no reverse image search and no photo table run
// inside the entry. A photo whose call is refused or whose answer holds no
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
  'Geolocate this street-level photo. Work in this order, briefly:\n' +
  '1. Environmental read first: continent, biome (vegetation, soil, terrain), road quality, driving side. ' +
  'Left-hand traffic means the UK, Ireland, Japan, Australia, New Zealand, India, Thailand, Indonesia, Malaysia, ' +
  'South Africa, Kenya or another former British territory; everywhere else drives on the right.\n' +
  '2. Script and language on signs: Cyrillic = Russia, Ukraine, Belarus, Bulgaria (word endings in -ът), Serbia, North Macedonia, Mongolia; ' +
  'Ukrainian Cyrillic has ї, є, ґ. Hangul = South Korea. Kanji + Hiragana + Katakana = Japan. Simplified Chinese = mainland China, ' +
  'traditional = Taiwan, Hong Kong. Thai, Khmer, Lao, Myanmar scripts are distinct. Devanagari = India, Nepal. Arabic script = Middle East, ' +
  'North Africa, Pakistan, Iran; Urdu adds extra dotted letters. Latin with ñ = Spanish-speaking; ç, ã, õ = Portugal, Brazil; ' +
  'ø, å = Denmark, Norway; ä, ö, ü = Germany, Austria, Switzerland, Turkey (also ş, ğ, ı); ą, ę, ł, ź = Poland; ř, ě, ů = Czech. ' +
  'Ge\'ez = Ethiopia, Eritrea. Georgian and Armenian scripts are unique. Greek = Greece, Cyprus.\n' +
  '3. Country narrowers. Bollards: France white post with a red strip; Poland one broad red diagonal stripe; Czech fluorescent orange stripes; ' +
  'Hungary red front / white back reflector; Austria black hat on top; Germany tall narrow white front + red back, rectangular reflectors; ' +
  'Netherlands red/white striped; Sweden, Norway, Finland short white posts with yellow reflector tops; UK large red reflector on top; ' +
  'Denmark white with red reflector; Spain black/white with a yellow rectangle; Thailand obelisk black/white; Malaysia two red rectangles; ' +
  'Estonia cylindrical with rectangular front and two circle rear reflectors; Latvia thin plank; Lithuania wedge, orange front. ' +
  'Utility poles: France small blue stickers; Japan dense wiring and yellow/black striped poles; Hungary, Romania, Poland have concrete poles ' +
  'with holes ("holey poles") — holes to the ground plus a yellow voltage marker and white base paint = Romania, holes stopping above ground = ' +
  'Hungary or Poland; Taiwan yellow/black stripes touching the ground; Ukraine and Russia T-shaped crossbars. ' +
  'License plates: EU blue strip on the left with a country code (D, F, I, E, P, NL, B, A, PL, CZ, SK, H, RO, BG, HR, SLO; Switzerland red strip); ' +
  'UK wide plates, yellow rear; Japan small plates, yellow for kei cars, white otherwise, green text; South Korea wide; China blue, green for EVs, ' +
  'with Chinese characters; India white front, yellow rear; Brazil and Argentina Mercosul white with a blue band on top; US and Canada state designs; ' +
  'Mexico colourful state plates; Thailand Thai script; Kenya white plates with a K prefix; South Africa province-coded. ' +
  'Road lines: yellow center line = the Americas (except Chile, all white), Iceland, Norway; white center = Europe, Asia, Africa, Oceania; ' +
  'double white center = Poland, Latvia; double yellow center = Brazil; yellow outer lines = South Africa, Israel, Jordan, UAE, Oman. ' +
  'Direction signs: green = most of Europe and the Americas, blue = motorways in the EU, white = UK, yellow = Australia.\n' +
  '4. Sub-country placement: architecture (colourful colonial = Latin America; half-timbered = Germany, Alsace, England, Scandinavia; ' +
  'brutalist concrete = former Soviet states and Eastern Europe; terracotta roofs = Mediterranean and Latin America; tin roofs = Sub-Saharan Africa, ' +
  'Caribbean, Southeast Asia; flat roofs with satellite dishes = Middle East, North Africa; red wooden houses = Sweden; whitewashed walls = Greece, ' +
  'southern Spain, Morocco, Tunisia), vegetation (red soil = Brazil, Sub-Saharan Africa, India, Southeast Asia; birch = northern Europe, Russia, Canada; ' +
  'eucalyptus = Australia; baobab = Sub-Saharan Africa; rice paddies = Southeast Asia, China, Japan, India; vineyards = Mediterranean, California, ' +
  'Chile, Argentina, South Africa, Australia; palm oil = Malaysia, Indonesia, West Africa; cacti = Mexico, southwestern US, North Africa, Middle East; ' +
  'savanna = East Africa, Brazil, northern Australia; dense conifer forest = Scandinavia, Russia, Canada, Pacific Northwest), terrain, ' +
  'domain names and phone prefixes on signs (.br, .pl, .co.za, .co.uk, .de, .fr, .com.au, .co.jp, .ru, .nl; +1 US/Canada, +44 UK, +33 France, ' +
  '+49 Germany, +55 Brazil, +91 India, +81 Japan, +86 China), gas station brands (Pemex = Mexico, Petrobras = Brazil, Orlen = Poland, ' +
  'Sinopec = China, Lukoil = Russia, Total = France and Africa), postbox colours (red = UK, yellow = Germany, blue = US, green = Ireland). ' +
  'Satellite dishes point south in the northern hemisphere and north in the southern hemisphere; sun in the north also means southern hemisphere. ' +
  'Jollibee = Philippines, pink taxis = Mexico City, rickshaws = Bangladesh, cobblestone = Portugal, Brazil, Central European town centres.\n' +
  'Verification gate before answering: name two or three alternative countries that fit the clues, eliminate each with one specific visible clue, ' +
  'then check the winner against the tiers above for contradictions. If nothing distinguishes, give your best single estimate anyway.\n' +
  'Keep the reasoning under about 200 words, then give the estimate as one line of JSON on its own last line: ' +
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
