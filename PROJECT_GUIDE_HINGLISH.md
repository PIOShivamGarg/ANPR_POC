# ANPR POC — Complete Project Guide (Hinglish)

Ye document poore project ko samjhata hai — kya hai, kaise kaam karta hai, aur kyun aise banaya gaya hai.
Function-by-function detail ke liye [`FUNCTIONS_REFERENCE.md`](./FUNCTIONS_REFERENCE.md) dekho.

---

## 1. Project kya hai?

Ek **ANPR (Automatic Number Plate Recognition)** REST API hai. Aap ek gaadi ki photo upload karte ho, aur API aapko batati hai:

- **plate_number** — gaadi ka number (jaise `CHF4653`)
- **state** — plate par likha state ka naam (jaise `Washington`)
- **region** — kaunsa desh/region hai (jaise `United States`)
- **confidence** — kitna bharosa hai is read par (0 se 1 ke beech)

FastAPI par bana hai, ek endpoint hai jisme image POST karte ho aur JSON milta hai.

---

## 2. Tech stack — har cheez ka kaam

| Library | Kaam |
|---|---|
| **fast_alpr** | Do kaam karti hai: (1) photo mein plate **kahan** hai wo dhoondhti hai — YOLO-v9 detector se, (2) plate ka text bhi **khud padhti** hai apne OCR model se |
| **Azure Computer Vision** | Doosra OCR — plate ke crop ka poora text padhta hai (number + state ka naam + slogan, sab kuch) |
| **countrystatecity-countries** | Duniya ke saare (~5,300) state/region names ki list — isse state ka naam match karte hain |
| **FastAPI + uvicorn/gunicorn** | REST API server |
| **OpenCV (cv2)** | Image padhna, crop karna, resize karna |
| **onnxruntime** | fast_alpr ke models chalane ke liye engine |

---

## 3. Poora flow — step by step

```
   Image upload (POST /read-plate-azure-cv)
                │
                ▼
   [main.py]  File extension check (.jpg/.png/.bmp/.tiff)
                │
                ▼
   Temp file mein save karo
                │
                ▼
   [azure_cv.py]  read_plate_azure_cv()
                │
                ├──► cv2.imread()  ── image padha?  nahi → {"error": ...}
                │
                ▼
   alpr.predict(img)   ← YOLO detector: plate ka box dhoondho
                │
                ├──► koi plate nahi mila → message: "No license plate detected"
                │
                ▼
   Har detected plate ke liye (loop):
                │
                ├──► bbox nikalo + 6px padding + image ke andar clip karo
                │
                ├──► CANDIDATE 1: fast_alpr ka apna OCR
                │      result.ocr.text        → 'CHF4653'
                │      result.ocr.confidence  → [1.0, 1.0, 0.993, ...]  (per character)
                │      result.ocr.region      → 'United States'
                │
                ├──► CANDIDATE 2: Azure CV OCR (crop bhejte hain)
                │      raw text          → 'WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE .'
                │      word confidences  → {'WASHINGTON': 0.994, 'CHF': 0.993, '4653': 0.988, ...}
                │
                ▼
   [utils.py]  extract_state(raw_text)      → 'Washington'
                │
                ▼
   [utils.py]  extract_plate_candidate()    → ('CHF4653', ['CHF', '4653'])
                │       (state ka naam, year sticker, slogan sab hata ke)
                ▼
   [utils.py]  reconcile_plate_number()     → dono OCR ko compare karo, winner chuno
                │
                ▼
   [utils.py]  create_plate_response()      → final JSON banao
                │
                ▼
   {"file": ..., "plates": [...], "processing_time_sec": 0.59}
```

---

## 4. Sabse important design decision — **do OCR engines kyun?**

Ye project ka core idea hai, isliye dhyan se samjho.

### Problem
Ek hi OCR par bharosa karne se galat answer aata hai aur **pata bhi nahi chalta ki galat hai**.

Real example — ek hi plate ki do photos:

| Photo | Azure ne padha | Sahi kya hai? |
|---|---|---|
| `image7` | `CHE4653` | ❌ galat (`E` nahi, `F` hai) |
| `image8` | `CHF4653` | ✅ sahi |

Azure ne ek image mein `F` ko `E` padh liya. Agar sirf Azure use karte, to galat number return hota, bina kisi warning ke.

### Solution
Dono engines ka answer lo aur compare karo:

- **fast_alpr ka OCR** — sirf plates padhne ke liye bana hai, isliye number par zyada accurate hai. Lekin state ka naam ya slogan nahi padhta.
- **Azure CV** — general-purpose OCR hai, poora plate padhta hai (state, slogan, year sticker sab), isliye `state` field ke liye zaroori hai.

Phir 3 mein se ek situation hoti hai:

| Situation | Kya hota hai | `source` field |
|---|---|---|
| **Dono same** | Wahi number, confidence zyada wala | `"both"` |
| **Dono alag** | Jiski **confidence zyada** wo jeetega; haarne wala `alternate_plate_number` mein record ho jata hai | `"alpr"` ya `"azure"` |
| **Sirf ek ne padha** | Wahi use hoga | `"alpr"` ya `"azure"` |
| **Kisi ne bhi valid nahi padha** | `plate_number: null` | `null` |

**Ye kaam karta hai kyunki confidence sach batati hai.** Us galat `CHE4653` read par Azure ki apni confidence sirf **0.376** thi, jabki sahi reads par 0.98+ hoti hai. fast_alpr ki confidence 0.993 thi — isliye sahi wala `CHF4653` jeeta.

**Extra cost zero hai** — `alpr.predict()` already chal raha tha, uska OCR result hum pehle throw kar rahe the. Koi naya model load nahi, koi naya API call nahi.

---

## 5. Response ke saare fields

```json
{
  "file": "car.jpg",
  "plates": [
    {
      "plate_number": "CHF4653",
      "state": "Washington",
      "region": "United States",
      "confidence": 0.993,
      "source": "alpr",
      "alternate_plate_number": "CHE4653",
      "message": "OCR sources disagreed; kept the more confident read (alpr=CHF4653, rejected CHE4653)"
    }
  ],
  "processing_time_sec": 1.95
}
```

| Field | Matlab |
|---|---|
| `plate_number` | Final number, separators (space/`-`/`*`/`·`) hata ke. `null` = padha nahi ja saka |
| `state` | Plate par likha state ka naam, proper case mein. Tight crop mein `null` aa sakta hai (state ka naam crop mein hi nahi aaya) |
| `region` | fast_alpr ka desh prediction. Confidence 0.5 se kam ho to `null` (guess tha) |
| `confidence` | Jeetne wale read ki confidence (0–1). Per-character list ho to **sabse kamzor** character wali value |
| `source` | `"alpr"` / `"azure"` / `"both"` — kisne jeeta. `"both"` sabse strong hai |
| `alternate_plate_number` | Sirf jab dono alag padhein — reject kiya gaya answer. Normally `null` |
| `message` | `null` matlab sab theek. Warna reason batata hai (disagreement, no plate, error) |
| `plates` | **Array** hai — ek photo mein multiple gaadiyan hui to multiple entries |
| `processing_time_sec` | Total time. Azure API call isme included hai |

> **Note:** Agar image hi nahi padhi ja saki, response alag shape ka hota hai:
> `{"error": "Failed to read image: ...", "plates": []}` — isme `processing_time_sec` nahi aata.

---

## 6. Setup aur run

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Dependencies
pip install -r requirements.txt

# 3. Azure credentials — .env file banao
#    .env.example se copy karke apni values daalo
VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
VISION_KEY=your_key_here

# 4. Server chalao
uvicorn main:app --reload --port 8000

# Production mein:
gunicorn -w 5 -k uvicorn.workers.UvicornWorker main:app
```

Phir browser mein **http://127.0.0.1:8000/docs** kholo — Swagger UI se seedha image upload kar ke test kar sakte ho.

Docker se chalana ho:
```bash
docker build -t anpr-poc .
docker run -p 8000:8000 --env-file .env anpr-poc
```

> **Zaroori:** `VISION_ENDPOINT` aur `VISION_KEY` set nahi honge to app **start hi nahi hogi** — `azure_cv.py` import time par hi `ValueError` phenk deta hai. Ye jaan-boojh ke aisa hai, taaki galti pakadne mein der na ho.

---

## 7. API endpoints

| Method | Path | Kaam |
|---|---|---|
| `GET` | `/` | API zinda hai ya nahi + version |
| `GET` | `/health` | Health check (load balancer ke liye) |
| `POST` | `/read-plate-azure-cv` | Main endpoint — image upload karo, plate details lo |
| `GET` | `/docs` | Swagger UI (FastAPI khud deta hai) |

Allowed image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`. Kuch aur bhejo to **400** milega.

---

## 8. Plate number nikalne ka logic — ek example se samjho

Azure ka raw OCR text aata hai:

```
WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE .
```

Isme plate number `CHF4653` hai, baaki sab kachra hai. Code ye steps karta hai:

**Step 1 — State pehle nikalo.** `extract_state()` → `Washington`.
State pehle isliye nikalte hain ki uske baad wo naam text se hata sakein.

**Step 2 — State ka naam hatao.** Text bacha: `4 :2025 CHF 4653 . EVERGREEN STATE .`

**Step 3 — Tokens todo.** `['4', '2025', 'CHF', '4653', 'EVERGREEN', 'STATE']`

**Step 4 — Candidates banao.** Har single token + paas-paas ke **do tokens jod ke**:
```
singles : 4, 2025, CHF, 4653, EVERGREEN, STATE
pairs   : 42025, 2025CHF, CHF4653, 4653EVERGREEN, EVERGREENSTATE
```
> Pairs banane ka reason: Azure plate ko do hisso mein padhta hai — `CHF` aur `4653` alag-alag. Purana code sirf `CHF-4653` jaise separator samajhta tha, **space wala case fail ho jata tha**. Ab space bhi handle hota hai.

**Step 5 — Har candidate ko score do** (`_plate_score`):

| Rule | Result |
|---|---|
| Length 5–10 ke bahar? | score 0 (reject) |
| `_NOISE_TOKENS` mein hai (`STATE`, `AUTO`, `USA`...)? | score 0 |
| Koi part year sticker hai (`2025`, `1998`)? | score 0 |
| Letters **aur** digits dono nahi hain? | score 0 |
| Base score | 1 |
| Letters-then-digits shape hai (`CHF4653`)? | **+2** |
| Ek hi token se bana hai (jodna nahi pada)? | **+1** |

**Step 6 — Sabse zyada score wala jeeta.**

| Candidate | Score | Kyun |
|---|---|---|
| `2025` | 0 | year sticker |
| `4653` | 0 | koi letter nahi |
| `STATE` | 0 | noise word |
| `EVERGREEN` | 0 | koi digit nahi |
| `2025CHF` | 0 | part mein year hai |
| **`CHF4653`** | **3** | letter+digit ✓, shape ✓ (2 tokens se bana isliye +1 nahi) |

Winner: **`CHF4653`** ✅

---

## 9. Ab tak kaunse bugs fix hue (history)

Ye samajhna zaroori hai — code aise hi kyun likha hai:

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | `plate_number` `null` aa raha tha | Regex mein **space separator** nahi tha, aur Azure `CHF 4653` do tokens mein deta hai | Token + adjacent-pair candidates, phir scoring |
| 2 | Galat state — `NEW YORK` → **York**, `OHIO VAN 4321` → **Van** (Turkey!) | 5,300 names ko **plain substring** se match kar raha tha, bina word boundary, bina order | Whole-word matching + **lamba naam pehle** |
| 3 | Ek hi plate ke do alag jawab (`CHE` vs `CHF`) | Sirf ek OCR par bharosa, koi cross-check nahi | Do-engine reconciliation + confidence comparison |
| 4 | Blacked-out plate par `plate_number: "TE"` aa gaya | Azure ka candidate strictly validate hota tha, par **fast_alpr ka read bilkul validate hi nahi hota tha** — 2-character kachra pass ho gaya | `_is_plausible_plate()` — dono reads par same minimum bar |
| 5 | Texas plate par `region: "Belgium"` | `region_confidence` **0.224** thi (model guess kar raha tha), par hum use dekhte hi nahi the | `confident_region()` — 0.5 se kam confidence par `null` |

---

## 10. Kya nahi kaam karta (known limitations)

**1. Redacted / blacked-out plates — ye fix nahi ho sakta.**
Do sample images mein plate number **jaan-boojh ke kaale marker se chhupaya** gaya hai (privacy). Koi bhi OCR use nahi padh sakta. Ab code sahi tareeke se `null` deta hai — pehle kachra ("TE") return kar deta tha. Ye **expected behaviour hai, bug nahi**.

**2. Indian / EU multi-part plates aadhe kat jaate hain.** ⚠️ *Ye fix hona baaki hai*

| Input | Ab milta hai | Milna chahiye |
|---|---|---|
| `MH 12 AB 1234` | `AB1234` ❌ | `MH12AB1234` |
| `DL 3C AY 1234` | `AY1234` ❌ | `DL3CAY1234` |
| `KA 01 JZ 4030` | `JZ4030` ❌ | `KA01JZ4030` |

Reason: candidates sirf **2 adjacent tokens** tak jodte hain, Indian plates mein 4 tokens hote hain. Ye chup-chaap galat answer deta hai (jo `null` se bhi zyada khatarnaak hai). README "worldwide" claim karta hai, to ye gap real hai.

**3. Dealer-frame text state ko hijack kar sakta hai.**
`VICTORIA MOTORS WASHINGTON ABC1234` → state **Victoria** (Australia) aata hai, `Washington` nahi.
Fix ka idea: fast_alpr already `region: "United States"` batata hai — usse state search ko us desh tak limit kar do.

**4. Vanity aur numeric-only plates ka cross-check nahi hota.**
`GOFAST` (sirf letters) ya `12345` (sirf digits) — Azure path inhe reject karta hai kyunki letter **aur** digit dono chahiye. Ye fast_alpr path se aa jaate hain, par verify nahi hote.

**5. Sample set chhota hai.** Poora testing 18 detections par hua hai, jinme se zyadatar Washington ke plates hain. Rules solid hain par properly benchmarked nahi.

---

## 11. File structure

```
ANPR_POC/
├── main.py              # FastAPI app, routes, model + state-list loading (startup)
├── azure_cv.py          # Azure OCR client + main pipeline orchestration
├── utils.py             # Saara text-parsing logic (state, plate, reconciliation)
├── requirements.txt     # Dependencies
├── .env                 # Azure credentials (git mein nahi jaata)
├── .env.example         # Template
├── Dockerfile           # Container build
├── start.sh             # Production start (gunicorn, 5 workers)
├── README.md            # Original project readme
├── PROJECT_GUIDE_HINGLISH.md   # ← ye file
└── FUNCTIONS_REFERENCE.md      # Har function ki detail
```

**Layers ka logic:**

- **`main.py`** — sirf HTTP ka kaam. Models aur state-list **ek hi baar** startup par load hote hain (har request par nahi — warna bahut slow hota).
- **`azure_cv.py`** — Azure se baat karna + pipeline chalana. Yahi decide karta hai kaunsa crop bhejna hai.
- **`utils.py`** — **koi image, koi API nahi.** Sirf pure functions jo string in → string out karte hain. Isliye ye poora logic bina credentials, bina images test ho sakta hai — aur isi wajah se saare bugs yahan pakde gaye.
