# API Contract — FROZEN

This is the single source of truth shared by the backend and frontend repos. **Do not change anything
here without updating both sides.** Every agent working on this project builds against this document.

Base URL: `/api/v1`. All responses are JSON. All timestamps are ISO-8601 UTC strings.

---

## 1. Enums (exact string values — these are the wire format)

```
Category  : road | water | waste | electricity | drainage | safety | other
Priority  : low | medium | high | critical
Status    : open | assigned | in_progress | resolved | rejected
Role      : citizen | staff | admin
AISource  : llm | ml | rules        # which analyzer tier produced the result
```

Category display labels (frontend): Roads & Potholes, Water Supply & Leakage, Waste & Sanitation,
Electricity & Streetlights, Drainage & Sewerage, Public Safety, Other.

> Streetlight complaints map to `electricity`. There is no separate `streetlight` category — the spec's
> canonical list is Road / Water / Waste / Electricity / Drainage / Safety / Other.

---

## 2. Core objects

### Complaint
```jsonc
{
  "id": "uuid",
  "reference_code": "CIV-8F3K2M",       // public tracking handle, unique, human-typeable
  "title": "string",                     // short, AI-generated or first line of description
  "description": "string",               // the citizen's raw text
  "category": "road",
  "priority": "high",
  "status": "open",
  "location_text": "Block 5, Gulshan-e-Iqbal, Karachi",
  "area": "Gulshan-e-Iqbal",             // coarse bucket, used by area analytics
  "latitude": 24.9204,                   // nullable
  "longitude": 67.0971,                  // nullable
  "citizen_name": "string",              // nullable (anonymous allowed)
  "citizen_phone": "string",             // nullable
  "citizen_email": "string",             // nullable
  "image_url": "string",                 // nullable
  "department": { "id": "uuid", "name": "Roads & Infrastructure", "slug": "roads" }, // nullable
  "duplicate_of_id": "uuid",             // nullable
  "ai_status": "pending",                // pending | complete | failed
  "ai": { /* AIAnalysis, nullable until ai_status == complete */ },
  "created_at": "2026-08-08T10:00:00Z",
  "updated_at": "2026-08-08T10:00:00Z",
  "resolved_at": null,                   // nullable
  "resolution_hours": null               // nullable, computed
}
```

### AIAnalysis
```jsonc
{
  "category": "road",
  "priority": "high",
  "summary": "Large pothole on main road near school causing traffic hazard.",
  "department_suggestion": "Roads & Infrastructure",
  "confidence": 0.91,                    // 0..1
  "source": "llm",                       // llm | ml | rules  -> drives the UI badge
  "model_name": "deepseek-v4-flash",     // or "tfidf-linearsvc-v1" / "keyword-rules-v1"
  "reasoning": "string",                 // short human-readable justification, nullable
  "keywords": ["pothole", "school"],     // extracted signals, may be []
  "sentiment": "angry",                  // nullable: calm | concerned | angry
  "is_emergency": false,
  "latency_ms": 1840,
  "created_at": "2026-08-08T10:00:03Z"
}
```

### StatusEvent (timeline)
```jsonc
{ "id":"uuid", "from_status":"open", "to_status":"assigned",
  "note":"Assigned to roads team", "actor":"admin@civic.gov", "created_at":"..." }
```

### Department
```jsonc
{ "id":"uuid", "name":"Roads & Infrastructure", "slug":"roads",
  "categories":["road"], "contact_email":"roads@civic.gov.pk", "open_complaints": 12 }
```

---

## 3. Endpoints

### Public (no auth)
| Method | Path | Body / Query | Returns |
|---|---|---|---|
| `POST` | `/complaints` | `ComplaintCreate` | `201` `Complaint` with `ai_status:"pending"` |
| `GET` | `/complaints/track/{reference_code}` | — | `Complaint` (404 if unknown) |
| `POST` | `/complaints/analyze-preview` | `{description}` | `AIAnalysis` — analyse WITHOUT saving (powers the live "AI is reading your complaint" step) |
| `GET` | `/departments` | — | `Department[]` |
| `GET` | `/health` | — | `{status, database, ai_provider, version}` (root path, not under `/api/v1`) |

`ComplaintCreate`:
```jsonc
{ "description": "required, 15..5000 chars",
  "location_text": "required, 3..300",
  "area": "optional", "latitude": null, "longitude": null,
  "citizen_name": null, "citizen_phone": null, "citizen_email": null,
  "image_url": null,
  "category": null,   // optional citizen hint; AI still runs and may override
  "consent": true }
```

### Auth
| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/auth/login` | `{email, password}` | `{access_token, token_type:"bearer", user}` |
| `GET` | `/auth/me` | — | `User` |

`User`: `{id, email, full_name, role}`. Token is a JWT in `Authorization: Bearer <token>`.
Demo credentials seeded: `admin@civic.gov.pk` / `Admin@123` (role `admin`).

### Admin / staff (auth required)
| Method | Path | Notes |
|---|---|---|
| `GET` | `/complaints` | List with filters. Query: `q, category[], priority[], status[], department_id, area, date_from, date_to, sort (created_at\|priority\|status\|resolution_hours), order (asc\|desc), page (1-based), page_size (default 20, max 100)`. Returns `{items, total, page, page_size, pages}` |
| `GET` | `/complaints/{id}` | Full complaint incl. `timeline: StatusEvent[]` |
| `PATCH` | `/complaints/{id}` | `{status?, priority?, category?, department_id?, note?}` → updated `Complaint`, appends a `StatusEvent` |
| `POST` | `/complaints/{id}/reanalyze` | Re-runs the AI pipeline, returns updated `Complaint` |
| `GET` | `/complaints/{id}/duplicates` | `{candidates:[{complaint, similarity, reason}]}` |
| `DELETE` | `/complaints/{id}` | admin only, soft delete |

### Analytics (auth required except `/analytics/public-summary`)
| Method | Path | Returns |
|---|---|---|
| `GET` | `/analytics/overview` | KPI cards + headline insights |
| `GET` | `/analytics/categories` | frequency distribution + mode |
| `GET` | `/analytics/priorities` | priority distribution + cross-tab with category |
| `GET` | `/analytics/resolution-times` | full descriptive stats + quartiles + Tukey fences + outliers |
| `GET` | `/analytics/trends?days=90` | daily series + 7-day moving average |
| `GET` | `/analytics/departments` | per-department volume + median resolution + backlog |
| `GET` | `/analytics/areas` | per-area volume, top category, hotspot flag |
| `GET` | `/analytics/insights` | `Insight[]` — the plain-English narrative layer |
| `GET` | `/analytics/public-summary` | small subset for the public landing page |

All analytics endpoints accept optional `date_from`, `date_to`, `category`, `area`.

`Insight` (the "explain the statistics" deliverable):
```jsonc
{ "id":"resolution_skew", "severity":"info",      // info | warn | critical
  "title":"Half of complaints are resolved within 3.2 days",
  "detail":"The median is 3.2 days but the mean is 6.8 days. The gap means a minority of very slow cases is dragging the average up, so the median is the honest headline number.",
  "metric": 3.2, "unit":"days" }
```

`/analytics/resolution-times` response shape (statistics benchmark lives here):
```jsonc
{ "n": 412, "unit": "hours",
  "mean": 163.4, "median": 76.5, "mode": 48.0,
  "min": 2.0, "max": 1180.0, "range": 1178.0,
  "variance": 41230.5, "std_dev": 203.1, "ddof": 1,
  "q1": 31.0, "q2": 76.5, "q3": 190.0, "iqr": 159.0,
  "lower_fence": -207.5, "upper_fence": 428.5,
  "outliers": [{"reference_code":"CIV-...", "value": 1180.0}],
  "histogram": [{"bin_start":0,"bin_end":24,"count":58}, ...],
  "by_category": [{"category":"road","n":90,"median":88.0,"q1":40,"q3":210}],
  "interpretation": "…plain English…",
  "sample_warning": null   // set when n is too small to trust
}
```

### AI
| Method | Path | Notes |
|---|---|---|
| `POST` | `/assistant/chat` | `{message, history?[]}` → `{answer, citations:[{reference_code,id}], used_stats:{}, source}` |
| `GET` | `/ai/health` | `{llm_available, ml_model_loaded, rules_available, model_name, last_error}` |
| `GET` | `/ai/evaluation` | Serves the stored model-evaluation report (accuracy, macro-F1, per-class, confusion matrix) |

---

## 4. Error shape (every non-2xx)

```jsonc
{ "error": { "code": "validation_error",
             "message": "Human readable, safe to show a user",
             "details": [{"field":"description","issue":"too short"}],
             "request_id": "01J..." } }
```
Codes: `validation_error` (422) · `not_found` (404) · `unauthorized` (401) · `forbidden` (403) ·
`conflict` (409) · `rate_limited` (429) · `ai_unavailable` (503) · `internal_error` (500).

---

## 5. Non-negotiable behaviours

1. **`POST /complaints` never blocks on the LLM.** It saves, returns `201` with `ai_status:"pending"`,
   and enriches in a FastAPI `BackgroundTask`. The frontend polls the complaint until
   `ai_status != "pending"`. A DeepSeek outage can therefore never break complaint submission.
2. **`POST /complaints/analyze-preview` is the one synchronous AI call**, used on the submit screen so
   the citizen literally watches the AI work. It has a hard 25s timeout and falls back down the tiers.
3. **The analyzer tier is always recorded** in `ai.source` and surfaced in the UI as a badge.
   Never silently pretend a rules-based result came from the LLM.
4. **CORS** allows the Vercel origin and `http://localhost:5173`.
5. Every list endpoint is paginated. Never return an unbounded array of complaints.
