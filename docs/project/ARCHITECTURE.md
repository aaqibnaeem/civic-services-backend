# Architecture — AI Smart Civic Services

## The problem in one line

A citizen writes *"there is a large water leak near the main road and traffic is becoming difficult"*.
That sentence is useless to a service team until something turns it into: **category**, **priority**,
**responsible department**, and a **one-line actionable summary**. That transformation is what this system does.

## System overview

```mermaid
flowchart TB
    subgraph client["Citizen &amp; Admin — React SPA on Vercel"]
        C1["Report a complaint"]
        C2["Track by reference code"]
        A1["Triage inbox"]
        A2["Analytics dashboard"]
        A3["AI assistant"]
    end

    subgraph api["Python API — FastAPI on Render"]
        R["Routers / api/v1"]
        SVC["ComplaintManager<br/>(services)"]
        REPO["Repositories"]
        AN["AnalyticsService"]
        AIP["AI Pipeline"]
    end

    subgraph ai["AI layer — three tiers, in fallback order"]
        T1["1· DeepSeekAnalyzer<br/>deepseek-v4-flash"]
        T2["2· MLAnalyzer<br/>TF-IDF + LinearSVC"]
        T3["3· RuleBasedAnalyzer<br/>keyword + heuristics"]
    end

    DB[("PostgreSQL — Neon<br/>complaints · ai_analysis · status_events<br/>departments · users")]

    C1 --> R
    C2 --> R
    A1 --> R
    A2 --> R
    A3 --> R
    R --> SVC --> REPO --> DB
    R --> AN --> DB
    SVC -. "background task" .-> AIP
    AIP --> T1
    T1 -- "fails / no key / circuit open" --> T2
    T2 -- "model missing" --> T3
    AIP --> DB
```

## Where the AI sits, precisely

```mermaid
sequenceDiagram
    participant Citizen
    participant API as FastAPI
    participant DB as Postgres
    participant AI as AI Pipeline
    participant DS as DeepSeek

    Citizen->>API: POST /complaints {description, location}
    API->>DB: INSERT complaint (ai_status = "pending")
    API-->>Citizen: 201 {reference_code, ai_status:"pending"}
    Note over API,AI: submission NEVER waits on the network
    API->>AI: BackgroundTask: analyze_and_store(id)
    AI->>DS: chat.completions (json_object, thinking disabled)
    alt DeepSeek answers
        DS-->>AI: {category, priority, summary, department, confidence}
    else timeout / 5xx / no key / circuit open
        AI->>AI: MLAnalyzer -> else RuleBasedAnalyzer
    end
    AI->>DB: INSERT ai_analysis (source = llm|ml|rules) + route department
    AI->>DB: UPDATE complaint (category, priority, ai_status="complete")
    Citizen->>API: poll GET /complaints/track/{ref}
    API-->>Citizen: complaint + AI result + source badge
```

**AI input:** the raw complaint text plus light context (citizen-supplied location and optional category hint).
**AI processing:** a single JSON-mode chat completion against `deepseek-v4-flash` with a byte-stable system
prompt carrying the 7-category taxonomy, the 4 priority criteria and few-shot examples; the reply is parsed and
validated against a Pydantic model before it is trusted.
**AI output:** `category`, `priority`, `summary`, `department_suggestion`, `confidence`, `keywords`,
`is_emergency` — persisted to `ai_analysis` and used to set the complaint's own category, priority and
assigned department.
**Limitations:** documented in [AI_TESTING_EVIDENCE.md](../AI_TESTING_EVIDENCE.md).

## Why three AI tiers

A hackathon demo that depends on one API call is a demo that fails. Each tier degrades to the next:

| Tier | Technology | When it runs | Confidence | Cost |
|---|---|---|---|---|
| 1 | DeepSeek `deepseek-v4-flash` | Default | Highest — understands context, negation, Roman-Urdu | ~$0.0002/complaint |
| 2 | TF-IDF (word + char n-gram) → calibrated LinearSVC | No key, API error, or circuit breaker open | Good on typical phrasing | Free, local |
| 3 | Weighted keyword + emergency heuristics | Model artifacts missing | Coarse but never wrong-by-crash | Free, sub-millisecond |

The tier that produced each result is stored on the record and shown in the UI as a badge. The system never
pretends a rule-based guess came from the LLM.

## Layering rules

`Router → Service → Repository → Database`. A router never touches a session; a repository never contains
business logic. The AI pipeline is invoked by the service layer, never by a router, and it opens its own
session because it runs after the response has been sent.

## Class model (OOP benchmark)

```mermaid
classDiagram
    class AIAnalyzer {
        <<abstract>>
        +name: str
        +source: AISource
        +is_available() bool
        +analyze(text, context)* AnalysisResult
    }
    AIAnalyzer <|-- DeepSeekAnalyzer
    AIAnalyzer <|-- MLAnalyzer
    AIAnalyzer <|-- RuleBasedAnalyzer

    class StorageService {
        <<abstract>>
        +save(file) str
    }
    StorageService <|-- LocalStorageService
    StorageService <|-- NoopStorageService

    class NotificationService {
        <<abstract>>
        +send(event) None
    }
    NotificationService <|-- ConsoleNotificationChannel
    NotificationService <|-- EmailNotificationChannel

    class ComplaintManager {
        -_repo
        -_notifier
        +create(payload) Complaint
        +transition(id, status) Complaint
        +assign(id, department) Complaint
    }
    class AnalyticsService {
        -_frame
        +overview() dict
        +resolution_times() dict
        +insights() list
    }
    ComplaintManager --> AIAnalyzer : delegates via pipeline
    ComplaintManager --> NotificationService
```

Inheritance is used only where the subclasses are genuinely substitutable: the three analyzers are selected at
runtime by availability, and the storage/notification hierarchies are swapped by configuration. Nothing is a
class merely to satisfy a rubric.

## Statistics layer

Complaints are pulled once per request with SQL filters, loaded into a single pandas DataFrame, and every
metric is derived from that frame. Descriptive statistics use sample estimators (`ddof=1`). Resolution time is
reported by **median and IQR rather than mean**, because the distribution is strongly right-skewed — the
dashboard says so in words. Tukey fences (`Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`) identify abnormally slow cases, which
surface as an actionable list rather than a number. A chi-square test of independence checks whether category
and priority are related, and reports its own expected-frequency assumption.

Every number is paired with a plain-English `Insight` sentence generated by a **deterministic rules engine**,
never by the LLM — so a statistic can never be hallucinated.

## Deployment

| Layer | Platform | Notes |
|---|---|---|
| Frontend | Vercel | Static Vite build, SPA rewrite, `VITE_API_URL` env var |
| Backend | Render (free web service) | uvicorn, `/health` health check, sleeps after ~15 min idle |
| Database | Neon Postgres (free) | Permanent free tier; async driver via `asyncpg` |
| AI | DeepSeek API | Called from the backend only — the key never reaches the browser |

Secrets live only in platform environment variables and a gitignored `.env`. No key is ever committed, and the
frontend never holds an AI credential.
