# Functions Reference — har function ki detail

Project ke **teeno files** ke saare functions, constants aur module-level code ka line-by-line explanation.
Overall flow samajhna ho to pehle [`PROJECT_GUIDE_HINGLISH.md`](./PROJECT_GUIDE_HINGLISH.md) padho.

**Naming convention:** jis function ke naam ke aage `_` hai (jaise `_plate_score`) wo **private** hai — sirf usi file ke andar use hota hai. Baaki public hain.

---

## Index

**`main.py`** — [module-level setup](#mainpy--module-level-setup) · [`root`](#root) · [`health`](#health) · [`read_plate_azure_cv_api`](#read_plate_azure_cv_api)

**`azure_cv.py`** — [module-level setup](#azure_cvpy--module-level-setup) · [`_normalize_for_azure_ocr`](#_normalize_for_azure_ocr) · [`extract_text_from_image_azure_cv`](#extract_text_from_image_azure_cv) · [`_region_of`](#_region_of) · [`_azure_candidate_confidence`](#_azure_candidate_confidence) · [`read_plate_azure_cv`](#read_plate_azure_cv)

**`utils.py`** — [constants](#utilspy--constants) · [`extract_state`](#extract_state) · [`_is_plausible_plate`](#_is_plausible_plate) · [`confident_region`](#confident_region) · [`_plate_score`](#_plate_score) · [`extract_plate_candidate`](#extract_plate_candidate) · [`extract_plate_number`](#extract_plate_number) · [`_normalize_plate`](#_normalize_plate) · [`_min_confidence`](#_min_confidence) · [`reconcile_plate_number`](#reconcile_plate_number) · [`create_plate_response`](#create_plate_response)

---
---

# `main.py`

## `main.py` — module-level setup

Ye code **server start hone par ek hi baar** chalta hai, kisi request ke andar nahi.

```python
alpr = ALPR(detector_model="yolo-v9-s-608-license-plate-end2end")
```
YOLO-v9 plate detector load karta hai (`608` = input resolution). Iske saath fast_alpr ka **default OCR model bhi** load hota hai — yahi hamara "Candidate 1" banta hai.

```python
_all_states = []
for country in get_countries():
    _all_states.extend(get_states_of_country(country.iso2))

ALL_STATE_NAMES = {state.name.upper(): state.name for state in _all_states}
```
Duniya ke saare countries ghoom ke unke states nikalta hai, aur ek dict banata hai:
`{"WASHINGTON": "Washington", "NEW YORK": "New York", ...}` — total **~5,300 entries**.

- **Key uppercase kyun?** OCR ka text uppercase hota hai, isliye match karna easy.
- **Value proper case kyun?** Kyunki API response mein `"Washington"` dikhana hai, `"WASHINGTON"` nahi.
- **Startup par kyun?** 5,300 states har request par load karna bahut slow hota. Ek baar memory mein, phir sabhi requests use karti hain.

> ⚠️ **Note:** `from azure_cv import read_plate_azure_cv` line file ke **beech mein** (line 48) hai, top par nahi. Kaam karta hai, par convention ke against hai — ideally sabhi imports top par hone chahiye.

---

## `root`

```python
@app.get("/")
def root() -> dict
```

**Kaam:** API zinda hai ya nahi, ye batata hai.
**Returns:** `{"message": "ANPR API is running 🚗", "version": "1.0.0"}`

---

## `health`

```python
@app.get("/health")
def health() -> dict
```

**Kaam:** Health check endpoint. Load balancer / Docker / Kubernetes isse check karte hain ki container theek chal raha hai.
**Returns:** `{"status": "ok"}`

---

## `read_plate_azure_cv_api`

```python
@app.post("/read-plate-azure-cv")
async def read_plate_azure_cv_api(file: UploadFile = File(...)) -> JSONResponse
```

**Kaam:** Main endpoint. Uploaded image ko temp file mein rakh ke pipeline chalata hai.

**Steps:**

1. **Extension check** — sirf `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`. Kuch aur → `HTTPException 400`.
2. **Temp file** — `tempfile.NamedTemporaryFile(delete=False)` se disk par save karta hai, original extension ke saath.
   - *Temp file kyun?* Kyunki `read_plate_azure_cv()` **file path** leta hai (`cv2.imread` ke liye), raw bytes nahi.
   - *`delete=False` kyun?* Windows par ek khuli file ko doosra process nahi padh sakta. Isliye pehle close karte hain, baad mein manually delete.
3. **Pipeline chalao** — `read_plate_azure_cv(tmp_path, alpr, ALL_STATE_NAMES)`.
4. **Filename theek karo** — `result["file"] = file.filename`, kyunki andar temp ka random naam (`tmpqjv877oi.jpg`) set hota hai. User ko apna filename dikhna chahiye.
5. **`finally: os.unlink(tmp_path)`** — temp file **hamesha** delete hoti hai, error aaye ya na aaye. Warna disk bharti jaati.

**Returns:** `JSONResponse` — poora result dict.

---
---

# `azure_cv.py`

## `azure_cv.py` — module-level setup

```python
load_dotenv()
endpoint = os.getenv("VISION_ENDPOINT")
key = os.getenv("VISION_KEY")

if not endpoint or not key:
    raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment variables")

vision_client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
```

`.env` se credentials padhta hai aur Azure client banata hai.

**Yahi par `raise` kyun?** Ye **fail-fast** design hai. Credentials missing hain to app **start hone se hi mana** kar degi, bajaye ki 100 requests baad pata chale. Import time par error aata hai, isliye galti turant pakadti hai.

**Client ek hi baar kyun banta hai?** HTTP connection pooling — har request par naya client banane se latency badhti.

---

## `_normalize_for_azure_ocr`

```python
def _normalize_for_azure_ocr(image_crop, min_dim: int = 50, max_dim: int = 16000)
```

**Kaam:** Crop ka size Azure ki limits ke andar laata hai. Azure Image Analysis READ API demand karta hai ki **dono dimensions 50 se 16000 pixels ke beech** hon.

**Logic:**

```python
min_scale = max(min_dim / w, min_dim / h)    # itna scale karo ki dono dims >= 50
max_scale = min(max_dim / w, max_dim / h)    # itne se zyada nahi, warna dims > 16000

if min_scale > max_scale:
    return None                               # koi valid scale hi nahi (impossible aspect ratio)

scale = min(max(1.0, min_scale), max_scale)
```

Us aakhri line ka matlab:

| Crop ki halat | Kya hota hai |
|---|---|
| Dono dims already 50–16000 ke beech | `scale = 1.0` — **kuch nahi badalta** |
| Bahut chhota (jaise 47×53 px) | Upscale, sirf utna ki 50 tak pahunche |
| Bahut bada (>16000) | Downscale |
| Ek dim bahut chhoti + doosri bahut badi | `None` — is crop ko bhejna hi possible nahi |

**Interpolation:** upscale karte time `INTER_CUBIC` (smooth), downscale mein `INTER_AREA` (aliasing kam) — standard OpenCV practice.

**Returns:** resized image, ya `None` agar image khaali/invalid ho.

> **Interesting finding:** Testing mein pata chala ki crop ko zyada bada karne se OCR accuracy **improve nahi hoti**. Ek plate ko 300px aur 600px pe bhi try kiya — Azure ne wahi galat `CHF` hi diya. Isliye yahan sirf Azure ki *limit* satisfy ki jaati hai, accuracy ke liye upscaling nahi ki jaati.

---

## `extract_text_from_image_azure_cv`

```python
def extract_text_from_image_azure_cv(image_crop) -> tuple[str, dict[str, float]]
```

**Kaam:** Crop ko Azure OCR ko bhejta hai aur **do cheezein** wapas laata hai — poora text, aur har word ki confidence.

**Steps:**

1. `_normalize_for_azure_ocr()` se size theek karo → `None` mile to `("", {})` return.
2. `cv2.imencode('.jpg', ...)` se JPEG bytes banao (Azure bytes leta hai, numpy array nahi).
3. `vision_client.analyze(..., visual_features=[VisualFeatures.READ])` — actual API call.
4. Response ka structure nested hai: **blocks → lines → words**. Isko flatten karte hain:

```python
lines = [line for block in result.read.blocks for line in block.lines] if result.read else []
```

5. Har word ki confidence ek dict mein daalte hain, key ko normalize kar ke (sirf letters/digits, uppercase):

```python
key = re.sub(r'[^A-Z0-9]', '', word.text.upper())   # ':2025' → '2025'
word_confidences[key] = min(word.confidence, word_confidences.get(key, 1.0))
```

- *Key normalize kyun?* Kyunki baad mein `extract_plate_candidate` ke tokens bhi alphanumeric-only hote hain. Dono same format mein hon to match ho paayenge.
- *`min(...)` kyun?* Ek hi word do baar aa sakta hai (jaise `.` do baar). Aise case mein **kamzor** confidence rakhte hain — safe side.
- *Default `1.0` kyun?* Kyunki confidence kabhi 1.0 se zyada nahi hoti, to `min(conf, 1.0)` = `conf` — pehli baar ke liye neutral value.

**Returns:**
```python
("WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE .",
 {"WASHINGTON": 0.994, "4": 0.989, "2025": 0.665, "CHF": 0.993, "4653": 0.988, ...})
```

**Error handling:** poora function `try/except` mein hai. Azure down ho, timeout ho, kuch bhi — `("", {})` return hota hai aur pipeline crash nahi hoti. fast_alpr ka read phir bhi kaam kar jaata hai.

---

## `_region_of`

```python
def _region_of(ocr) -> str | None
```

**Kaam:** fast_alpr ke OCR object se region nikalta hai, par sirf tab jab confidence bharosemand ho. Andar `confident_region()` call karta hai.

```python
return confident_region(ocr.region, ocr.region_confidence) if ocr else None
```

**Ye chhota wrapper kyun?** Do jagah region chahiye tha (`read_plate_azure_cv` mein normal path aur invalid-bbox path). Dono jagah `ocr` ka `None` check + confidence check repeat na karna pade, isliye ek line mein wrap kar diya.

---

## `_azure_candidate_confidence`

```python
def _azure_candidate_confidence(parts: list[str], word_confidences: dict[str, float]) -> float | None
```

**Kaam:** Azure ke plate candidate ki confidence nikalta hai — jin words se wo bana hai, unme se **sabse kamzor** wali.

**Example:** candidate `CHF4653` do parts se bana — `['CHF', '4653']`.
`CHF` ki confidence 0.993, `4653` ki 0.988 → answer **0.988** (kamzor wali).

**Kamzor wali kyun?** Kyunki plate number poora sahi hona chahiye. Ek character galat = poora number galat. Isliye pessimistic (sabse kharab) value hi sach batati hai.

```python
matched = [conf for part in parts
           for word, conf in word_confidences.items()
           if part == word or part in word]
return min(matched) if matched else None
```

`part in word` (substring) ka check bhi hai — kyunki kabhi Azure `ABC-1234` ko ek hi word mein de deta hai jabki hamare parts `ABC` aur `1234` alag hote hain.

**`None` kyun return karte hain (0.0 nahi)?** Ye important hai. `None` matlab "**maapa nahi ja saka**", jabki `0.0` matlab "**bilkul bharosa nahi**". Do bilkul alag baatein hain. `reconcile_plate_number` `None` ko "unmeasured" treat karta hai.

---

## `read_plate_azure_cv`

```python
def read_plate_azure_cv(image_path: str, alpr, ALL_STATE_NAMES: dict[str, str]) -> dict
```

**Ye poore project ka main orchestrator hai.** Baaki sab functions yahin se coordinate hote hain.

**Parameters:**
| Param | Kya hai |
|---|---|
| `image_path` | Image ka path (temp file) |
| `alpr` | `main.py` mein load kiya hua ALPR instance — **inject** kiya jaata hai, yahan load nahi hota |
| `ALL_STATE_NAMES` | Startup par bana state names ka dict |

> **Dependency injection kyun?** Model ko parameter mein lene se ye function testable ban jaata hai — test mein ek fake `alpr` object de sakte ho jo `predict()` par jo chahe return kare. Agar function apne andar model load karta, to test karna mushkil hota.

### Flow

**1. Image padho**
```python
img = cv2.imread(image_path)
if img is None:
    return {"error": f"Failed to read image: {image_path}", "plates": []}
```
> ⚠️ Ye **early return** hai — is path mein `processing_time_sec` field nahi aati. Response ka shape baaki cases se alag hai, client code mein iska dhyan rakhna.

**2. Plates detect karo**
```python
results = alpr.predict(img)
if not results:
    → message: "No license plate detected in the image"
```

**3. Har detection par loop** (ek photo mein multiple gaadiyan ho sakti hain):

**3a. Bounding box nikalo aur safe banao**
```python
x1, x2 = sorted([int(round(bbox.x1)), int(round(bbox.x2))])   # sorted: x1 < x2 guarantee
pad = 6                                                        # thoda extra context
x1 = max(0, x1 - pad);  x2 = min(img_w, x2 + pad)              # image ke bahar na jaaye
```
- **`sorted` kyun?** Detector kabhi ulta box de sakta hai (x2 < x1). Sort karne se guarantee ho jaata hai.
- **`pad = 6` kyun?** Plate ke thoda bahar ka hissa lene se OCR ko context milta hai (border, state ka naam upar).
- **clip kyun?** Negative index ya image se bada index dena crash/galat crop deta hai.

Agar box phir bhi invalid ho (`x2 <= x1`) → `"Invalid plate bounding box from detector"`.

**3b. Candidate 1 — fast_alpr ka apna OCR** (free hai, already compute ho chuka)
```python
alpr_text = result.ocr.text            # 'CHF4653'
alpr_conf = result.ocr.confidence      # [1.0, 1.0, 0.993, ...] per-character list
region    = _region_of(result.ocr)     # 'United States' (low confidence par None)
```

**3c. Candidate 2 — Azure OCR**
```python
raw_text, word_confidences = extract_text_from_image_azure_cv(plate_crop)
```

Agar **dono** khaali → `"OCR failed to extract text from detected plate"`.
> Dhyan do: `and` hai, `or` nahi. Azure fail ho jaaye par fast_alpr ne padh liya ho, to hum aage badhte hain.

**3d. Parse karo** — order important hai:
```python
state = extract_state(raw_text, ALL_STATE_NAMES)                       # 1. state pehle
azure_text, azure_parts = extract_plate_candidate(raw_text, state) or (None, [])   # 2. state hata ke plate
azure_conf = _azure_candidate_confidence(azure_parts, word_confidences)            # 3. uski confidence
```
**State pehle kyun?** Kyunki uska naam plate ke candidates se hatana hai — warna `WASHINGTON` bhi ek plate candidate ban jaata.

`or (None, [])` ka trick: `extract_plate_candidate` `None` return kar sakta hai; usse seedha unpack karne par crash hota. Ye default de deta hai.

**3e. Reconcile aur response**
```python
reconciled = reconcile_plate_number(alpr_text, alpr_conf, azure_text, azure_conf)
plates.append(create_plate_response(..., **reconciled fields))
```

**4. Exception safety** — poora loop `try/except` mein hai. Koi bhi error aaye → `"Processing error: {e}"` wala response, API 500 nahi deta.

**5. Timing aur logging**
```python
elapsed = round(time.time() - start_time, 3)
print(f"{os.path.basename(image_path)} -> {plates} | Time: {elapsed}s")
```
Yahi wo console line hai jo aap server logs mein dekhte ho.

**Returns:**
```python
{"file": "image8.jpg", "plates": [ ... ], "processing_time_sec": 0.594}
```

---
---

# `utils.py`

> **Ye file poori tarah pure hai** — koi image nahi, koi network call nahi, koi credential nahi. Sirf string in → string out. Isi wajah se ye poora logic bina Azure key aur bina images test kiya ja sakta hai.

## `utils.py` — constants

```python
_NOISE_TOKENS = {"READ", "TEXT", "ALPR", "STATE", "AUTO", "USA", "DMV", "EXP"}
```
Aise words jo plate jaise dikhte hain par kabhi plate number nahi hote. Slogans aur stickers se aate hain.

```python
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')
```
Registration year sticker pakadta hai (`2025`, `1998`). Ye 4-digit hota hai aur plate number jaisa lagta hai, isliye explicitly reject karna padta hai.

```python
_TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)
```
Words todne ke liye. `[^\W_]` ka matlab "word character but not underscore" — **accented letters bhi rakhta hai**, jo `FÈS` ya `MUŞ` jaise state names ke liye zaroori hai. Simple `[A-Z]+` use karte to ye naam toot jaate.

```python
_PLATE_SHAPE = re.compile(r'^([A-Z]{1,4}[0-9]{2,5}|[0-9]{1,4}[A-Z]{2,4})$')
```
Asli plate ka shape: pehle letters phir digits (`CHF4653`), **ya** pehle digits phir letters (`6022RE`). Ye match ho to bonus score milta hai.

```python
_MAX_STATE_WORDS = 4        # "New South Wales" jaise multi-word names ke liye
_MIN_PLATE_LEN = 5          # sample set mein sabse chhota genuine read 6 char ka tha
_MAX_PLATE_LEN = 10         # international layouts (MH12AB1234) 10 tak jaate hain
_MIN_REGION_CONFIDENCE = 0.5  # isse neeche region prediction sirf guess hai
```

---

## `extract_state`

```python
def extract_state(text: str, ALL_STATE_NAMES: dict[str, str]) -> str | None
```

**Kaam:** OCR text mein se state/region ka naam dhoondhta hai.

**Logic — n-gram matching, lambe se chhote ki taraf:**

```python
words = _TOKEN_RE.findall(text.upper())        # ['NEW', 'YORK', 'ABC', '1234']

for size in range(min(4, len(words)), 0, -1):  # pehle 4-word, phir 3, 2, 1
    for i in range(len(words) - size + 1):
        chunk = words[i:i + size]
        for phrase in (" ".join(chunk), "-".join(chunk)):
            if phrase in ALL_STATE_NAMES:
                return ALL_STATE_NAMES[phrase]
```

**Do critical properties:**

**1. Whole-word matching** — words todke match karte hain, `in` substring se nahi.
Isse `CHF4653` ke andar `"RA"` (Fiji ka state) match nahi hota.

**2. Longest-first** — 4-word phrases pehle try hote hain, phir 3, 2, 1.
Isse `NEW YORK` mil jaata hai `YORK` se pehle.

**`-` wala variant kyun?** Kuch state names hyphenated hote hain (`Al-Qahirah`), par OCR unhe alag words mein deta hai. Dono variants try karte hain.

**Examples:**
| Input | Output | Notes |
|---|---|---|
| `TEXAS FNR*8034` | `Texas` | simple |
| `NEW YORK ABC-1234` | `New York` | longest-first ne bachaya (pehle `York` aata tha ❌) |
| `OHIO VAN 4321` | `Ohio` | `Van` (Turkey) nahi jeeta ✅ |
| `CHF4653` | `None` | koi state naam nahi |
| `VICTORIA MOTORS WASHINGTON ABC1234` | `Victoria` | ⚠️ **known limitation** — dealer text jeet gaya |

**Performance:** ~5,300 entries wala dict hai par lookup **O(1)** hai (dict hai, list nahi). Sirf words ke n-grams par loop hota hai, jo bahut kam hote hain.

---

## `_is_plausible_plate`

```python
def _is_plausible_plate(clean: str | None) -> bool
```

**Kaam:** "Kya ye string plate number **ho bhi sakti hai**?" Ye wo **minimum bar hai jo har candidate ko paar karni padti hai** — chahe wo Azure se aaya ho ya fast_alpr se.

```python
return bool(clean) and (_MIN_PLATE_LEN <= len(clean) <= _MAX_PLATE_LEN
                        and clean not in _NOISE_TOKENS)
```

Teen checks: khaali na ho, length 5–10 ho, noise word na ho.

**Ye function kyun bana?** Ek real bug ki wajah se. Pehle Azure ke candidate par strict validation thi par **fast_alpr ke read par koi validation hi nahi thi**. Ek blacked-out Texas plate par fast_alpr ne `"TEXAS"` lettering se `"TE"` padh liya, aur wo 2-character kachra seedha `plate_number` ban gaya. Ab dono reads yahi se guzarte hain.

| Input | Result | Kyun |
|---|---|---|
| `TE` | `False` | bahut chhota (2 < 5) |
| `STATE` | `False` | noise word |
| `ABCDEFGHIJKL` | `False` | bahut lamba (12 > 10) |
| `6022RE` | `True` | ✅ |
| `MH12AB1234` | `True` | ✅ 10 chars |

---

## `confident_region`

```python
def confident_region(region: str | None, region_confidence: float | None) -> str | None
```

**Kaam:** fast_alpr ka region prediction return karta hai, par **sirf tab jab wo guess na kar raha ho**.

```python
if not region or (region_confidence or 0.0) < _MIN_REGION_CONFIDENCE:
    return None
return region
```

**Threshold 0.5 kahan se aaya?** Real data se. Testing mein saare genuine plates ka region confidence **0.98 se zyada** tha. Ek redacted Texas plate par model ne `"Belgium"` bola — confidence sirf **0.224**. Beech mein 0.5 rakhna safe hai.

**`(region_confidence or 0.0)` kyun?** Confidence `None` bhi aa sakti hai; usse compare karne par `TypeError` aata. `or 0.0` se `None` bhi reject ho jaata hai.

| Input | Output |
|---|---|
| `("United States", 0.999)` | `"United States"` |
| `("Belgium", 0.224)` | `None` |
| `("Belgium", None)` | `None` |
| `(None, 0.999)` | `None` |

---

## `_plate_score`

```python
def _plate_score(clean: str, parts: list[str]) -> int
```

**Kaam:** Candidate ko score deta hai. **0 = plate nahi hai.** Jitna zyada score, utna zyada chance ki ye asli plate hai.

**Reject rules (score 0):**
```python
if not _is_plausible_plate(clean):                                    return 0  # length/noise
if any(part in _NOISE_TOKENS or _YEAR_RE.match(part) for part in parts): return 0  # year sticker
if not re.search(r'[A-Z]', clean) or not re.search(r'[0-9]', clean):  return 0  # letter+digit dono chahiye
```

**Bonus rules:**
```python
score = 1
if _PLATE_SHAPE.match(clean):  score += 2   # letters/digits saaf-saaf grouped
if len(parts) == 1:            score += 1   # OCR ne ek hi token mein padha = strong evidence
```

**Max score 4** (1 + 2 + 1).

**`parts` bhi kyun check karte hain, sirf `clean` nahi?** Kyunki `2025CHF` mein `clean` mein letter+digit dono hain aur length bhi theek hai — par uska ek part (`2025`) year sticker hai. Sirf jude hue string ko dekhne se ye pakda nahi jaata.

**Score comparison example** (text: `WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE .`):

| Candidate | parts | Score | Reason |
|---|---|---|---|
| `CHF4653` | `['CHF','4653']` | **3** | 1 + shape 2 |
| `CHE4653` (single token) | `['CHE4653']` | **4** | 1 + shape 2 + single 1 |
| `2025CHF` | `['2025','CHF']` | 0 | part year hai |
| `4653` | `['4653']` | 0 | letter nahi |
| `EVERGREEN` | `['EVERGREEN']` | 0 | digit nahi |
| `STATE` | `['STATE']` | 0 | noise word |

> **Ye known limitation ki jad hai.** `MH 12 AB 1234` par do problem hain:
> 1. `MH12AB1234` **candidate hi nahi banta** — candidates sirf 2 adjacent tokens tak jodte hain, ye 4 tokens ka hai.
> 2. Agar zabardasti bana bhi dein, to uska score sirf **1** hai — kyunki `_PLATE_SHAPE` mein Indian layout (`LL DD LL DDDD`) ka pattern hi nahi hai, to +2 bonus nahi milta. Jabki `AB1234` ko score **3** milta hai.
>
> Isliye `AB1234` jeet jaata hai aur `MH12` chhoot jaata hai. Poora fix ke liye dono cheezein chahiye: 4-token joins **aur** international shapes `_PLATE_SHAPE` mein add karna.

---

## `extract_plate_candidate`

```python
def extract_plate_candidate(text: str, state: str | None = None) -> tuple[str, list[str]] | None
```

**Kaam:** Azure ke gande OCR text se plate number nikalta hai, **aur ye bhi batata hai ki kaunse tokens se bana** (taaki caller unki confidence dekh sake).

**Steps:**

**1. State ka naam hatao**
```python
if state:
    text_upper = re.sub(re.escape(state.upper()), " ", text_upper)
```
`re.escape` zaroori hai — state names mein `.` ya `-` ho sakta hai jo regex mein special meaning rakhta hai.

**2. Alphanumeric tokens todo**
```python
tokens = re.findall(r'[A-Z0-9]+', text_upper)
```
Ye automatically **saare separators handle kar deta hai** — space, `*`, `-`, `·`, `.` sab. `FNR*8034` → `['FNR', '8034']`.

**3. Candidates banao — singles + adjacent pairs**
```python
candidates = ([(t, [t]) for t in tokens]
              + [(a + b, [a, b]) for a, b in zip(tokens, tokens[1:])])
```
`zip(tokens, tokens[1:])` ek classic trick hai — paas-paas ke jode banata hai:
`['A','B','C']` → `[('A','B'), ('B','C')]`.

**4. Score karo aur best chuno**
```python
scored = [(_plate_score(clean, parts), clean, parts) for clean, parts in candidates]
score, clean, parts = max(scored, key=lambda c: c[0], default=(0, "", []))
return (clean, parts) if score else None
```
`max()` barabar score mein **pehla** wala leta hai (leftmost). `default=` khaali list ke case ke liye hai (khaali text par crash na ho).

**Returns:** `("CHF4653", ["CHF", "4653"])` ya `None`.

| Input | Output |
|---|---|
| `CHE4653` | `("CHE4653", ["CHE4653"])` |
| `WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE .` (state=`Washington`) | `("CHF4653", ["CHF", "4653"])` |
| `FNR*8034` | `("FNR8034", ["FNR", "8034"])` |
| `EVERGREEN STATE` | `None` |
| `MH 12 AB 1234` | `("AB1234", ["AB","1234"])` ⚠️ galat — `MH12` chhoot gaya |

---

## `extract_plate_number`

```python
def extract_plate_number(text: str, state: str | None = None) -> str | None
```

**Kaam:** `extract_plate_candidate` ka **thin wrapper** — sirf number deta hai, tokens nahi.

```python
candidate = extract_plate_candidate(text, state)
return candidate[0] if candidate else None
```

**Alag function kyun?** Kuch callers ko sirf number chahiye hota hai, tokens ki zaroorat nahi. Ye unke liye clean API hai (aur backward compatible bhi — pehle sirf yahi function tha).

---

## `_normalize_plate`

```python
def _normalize_plate(text: str | None) -> str | None
```

**Kaam:** Do plate strings ko compare karne se pehle same format mein laata hai.

```python
return re.sub(r'[^A-Z0-9]', '', text.upper()) or None if text else None
```

Ye one-liner aise padho: `(re.sub(...) or None) if text else None` — conditional expression ki precedence sabse kam hoti hai.

- `None`/khaali input → `None`
- `"CHF 4653"` → `"CHF4653"`
- `"chf-4653"` → `"CHF4653"`
- `"---"` → `None` (sab strip hone ke baad khaali bacha)

**Zaroori kyun?** Isi wajah se `reconcile_plate_number` samajh paata hai ki `"CHF4653"` aur `"CHF 4653"` **same hi jawab** hai, alag nahi.

---

## `_min_confidence`

```python
def _min_confidence(confidence: float | list[float] | None) -> float | None
```

**Kaam:** Confidence ko ek single pessimistic number mein badalta hai.

```python
values = ([confidence] if isinstance(confidence, (int, float))
          else [float(c) for c in confidence or []])
return round(min(values), 3) if values else None
```

Teen input types handle karta hai:

| Input | Output | Notes |
|---|---|---|
| `0.993` (single float) | `0.993` | Azure aise deta hai |
| `[1.0, 1.0, 0.993, 1.0]` (per-char list) | `0.993` | fast_alpr aise deta hai |
| `None` ya `[]` | `None` | "maapa nahi gaya" |

**`min` kyun, `average` nahi?** Kyunki plate number **poora** sahi hona chahiye. Agar 6 character 100% confident hain aur 1 character 40%, to average 91% aayega — jo jhooth hai. Wo ek kamzor character poore number ko galat kar dega. Isliye weakest link hi sach hai.

> **Note:** fast_alpr apni list ko fixed 10 slots tak pad karta hai (blank slots included), isliye `min` un blanks par bhi lagta hai. Practically wo blanks high confidence par hote hain, to answer conservative rehta hai.

---

## `reconcile_plate_number`

```python
def reconcile_plate_number(
    alpr_text: str | None,
    alpr_confidence: float | list[float] | None,
    azure_text: str | None,
    azure_confidence: float | None,
) -> dict
```

**Ye project ka sabse important function hai.** Do independent OCR sources mein se final answer chunta hai.

**Step 1 — dono reads normalize karo, aur plausibility gate lagao**
```python
reads = [(_normalize_plate(alpr_text), _min_confidence(alpr_confidence), "alpr"),
         (_normalize_plate(azure_text), _min_confidence(azure_confidence), "azure")]
present = [r for r in reads if _is_plausible_plate(r[0])]
```

> **fast_alpr list mein pehle kyun hai?** Kyunki `sorted()` Python mein **stable** hai — barabar confidence par original order bana rehta hai. Isliye exact tie hone par fast_alpr jeetta hai (kyunki plates padhna hi uska kaam hai). Ye order **jaan-boojh ke** hai, accident nahi.

**Step 2 — kitne valid reads bache?**

```python
if not present:            return result(None, None, None)      # koi nahi → null
if len(present) == 1:      return result(*present[0])           # ek hi → wahi
```

**Step 3 — dono valid hain: same hain ya alag?**

```python
if present[0][0] == present[1][0]:
    confidences = [c for _, c, _ in present if c is not None]
    return result(present[0][0], max(confidences) if confidences else None, "both")
```
Dono agree karte hain → `source: "both"`, aur confidence mein **`max`** (zyada wali).
**`max` kyun, `min` nahi?** Kyunki do independent engines ka same jawab dena **corroboration** hai — bharosa badhta hai, ghatta nahi.

**Step 4 — dono alag: confidence se decide karo**
```python
winner, loser = sorted(present, key=lambda r: r[1] or 0.0, reverse=True)
return result(*winner, loser[0])
```
Zyada confidence wala jeeta, haarne wala `alternate_plate_number` mein.

**Returns:**
```python
{"plate_number": "CHF4653", "confidence": 0.993,
 "source": "alpr", "alternate_plate_number": "CHE4653"}
```

**Complete behaviour table:**

| ALPR read | Azure read | Result | `source` |
|---|---|---|---|
| `CHF4653` @0.99 | `CHF4653` @0.988 | `CHF4653` @0.99 | `both` |
| `CHF4653` @0.993 | `CHE4653` @0.376 | `CHF4653`, alt=`CHE4653` | `alpr` |
| `CHF4653` @0.4 | `ABC1234` @0.97 | `ABC1234`, alt=`CHF4653` | `azure` |
| `CHF4653` @0.9 | `CHE4653` @0.9 | `CHF4653` (tie → specialist) | `alpr` |
| `CHF4653` @`None` | `CHE4653` @0.5 | `CHE4653` (unmeasured haar gaya) | `azure` |
| `CHF4653` @0.9 | *(kuch nahi)* | `CHF4653` | `alpr` |
| `TE` @0.493 | *(kuch nahi)* | `None` (plausibility gate) | `None` |
| *(kuch nahi)* | *(kuch nahi)* | `None` | `None` |

---

## `create_plate_response`

```python
def create_plate_response(plate_number, state, region=None, confidence=None,
                          source=None, alternate_plate_number=None,
                          message=None) -> dict
```

**Kaam:** Ek plate ka standard response dict banata hai. **Saare** response isi function se bante hain, isliye API ka shape hamesha consistent rehta hai — chahe success ho ya failure.

**Message ki logic:**
```python
if message is None:                        # caller ne override nahi diya
    if not plate_number:
        message = "No plate number could be extracted from OCR text"
    elif alternate_plate_number:
        message = (f"OCR sources disagreed; kept the more confident read "
                   f"({source}={plate_number}, rejected {alternate_plate_number})")
```

Teen possibilities:
1. Caller ne apna `message` diya → wahi use hoga (jaise `"No license plate detected in the image"`)
2. Plate number nahi mila → default "could not extract" message
3. Plate mila **par** dono sources alag the → disagreement warning
4. Sab theek → `message: null`

**`message` parameter kyun add kiya?** Pehle calling code aise karta tha:
```python
resp = create_plate_response(None, None, region)
resp["message"] = "Invalid plate bounding box"     # dict ko baahar se ched-chaad
plates.append(resp)
```
Ye 4 jagah repeat ho raha tha. Ab ek hi call mein ho jaata hai — cleaner aur kam lines.

**Saare defaults kyun hain?** Taaki purane/simple call sites (`create_plate_response(None, None)`) bina change kaam karein.

**Returns:** 7 keys wala dict — `plate_number`, `state`, `region`, `confidence`, `source`, `alternate_plate_number`, `message`.

---
---

## Quick summary — kaun kisko call karta hai

```
read_plate_azure_cv_api  (main.py — HTTP layer)
    │
    └─► read_plate_azure_cv  (azure_cv.py — orchestrator)
            │
            ├─► alpr.predict()                        ← fast_alpr: detect + OCR (Candidate 1)
            │
            ├─► extract_text_from_image_azure_cv      ← Azure OCR (Candidate 2)
            │       └─► _normalize_for_azure_ocr
            │
            ├─► _region_of  ──► confident_region
            │
            ├─► extract_state                          ← state ka naam
            │
            ├─► extract_plate_candidate                ← Azure text se plate
            │       └─► _plate_score
            │               └─► _is_plausible_plate
            │
            ├─► _azure_candidate_confidence            ← us candidate ki confidence
            │
            ├─► reconcile_plate_number                 ← WINNER decide karo
            │       ├─► _normalize_plate
            │       ├─► _min_confidence
            │       └─► _is_plausible_plate
            │
            └─► create_plate_response                  ← final JSON

extract_plate_number  ──► extract_plate_candidate      (standalone helper, pipeline mein use nahi)
```
