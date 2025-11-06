# Parenting Assistant

> An AI-powered family planning platform with real-time streaming, multi-modal assistance, and production-grade infrastructure.

## 🚀 Deployment Status

**✅ LIVE IN PRODUCTION**

- **Backend**: Deployed on DigitalOcean App Platform
  - URL: https://parenting-assistant-platform-39b47.ondigitalocean.app
  - Status: Running with managed PostgreSQL + Redis

- **iOS App**: Distributed via TestFlight
  - Status: Available for beta testing
  - Platform: iOS 16+
  - Distribution: Apple TestFlight

---

## Table of Contents

- [Deployment Status](#-deployment-status)
- [Product Overview](#product-overview)
- [Technical Architecture](#technical-architecture)
- [AI/LLM Technologies](#aillm-technologies)
- [Backend Platform](#backend-platform)
- [iOS Application](#ios-application)
- [Infrastructure & Deployment](#infrastructure--deployment)
- [Security & Safety](#security--safety)
- [Learning Outcomes](#learning-outcomes)

---

## Product Overview

### What is Parenting Assistant?

Parenting Assistant is a comprehensive family planning platform that uses AI to help families with:

1. **Meal Planning** - AI-powered weekly meal plans with grocery lists organized by aisle
2. **Chore Scheduling** - Smart chore scheduling with notification reminders
3. **Routine Building** - Age-appropriate routines for children with step-by-step guidance

### Key Features

- **Real-time AI Streaming**: Server-Sent Events (SSE) for live response generation
- **Apple Sign In**: Secure authentication with JWT token management
- **Offline Mode**: Full functionality with sample data when AI is disabled
- **Family Profiles**: Customizable household preferences (ages, allergies, cuisines)
- **Smart Grocery Lists**: Automatic consolidation and aisle organization
- **Notification Integration**: Schedule reminders for chores and routines
- **Multi-Provider AI**: Fallback between Anthropic Claude and OpenAI GPT models

### User Experience

```
Sign In → Configure Family Profile → Choose Feature (Meals/Chores/Routines) →
AI Generates Plan in Real-Time → View Results → Schedule Notifications
```

---

## Technical Architecture

### System Overview

```
┌─────────────────┐
│   iOS App       │  SwiftUI, Apple Sign In, SSE Client
│  (Swift/iOS)    │
└────────┬────────┘
         │ HTTPS + SSE
         ▼
┌─────────────────┐
│  API Gateway    │  FastAPI, Rate Limiting, Auth
│   (Python)      │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬─────────┐
    ▼         ▼          ▼         ▼
┌────────┐ ┌─────┐  ┌────────┐ ┌─────────┐
│PostGres│ │Redis│  │Anthropic│ │ OpenAI │
│+pgvector│ │     │  │ Claude │ │  GPT   │
└────────┘ └─────┘  └────────┘ └─────────┘
```

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.115.0 with Uvicorn ASGI server
- **Language**: Python 3.11+ with async/await
- **Database**: PostgreSQL 16 with pgvector extension for vector search
- **Cache/Rate Limiting**: Redis 7
- **ORM**: SQLAlchemy 2.0.36 with Alembic migrations
- **Validation**: Pydantic 2.9.2
- **Observability**: OpenTelemetry (OTLP export)
- **Authentication**: Apple Sign In verification + RS256 JWT

#### iOS
- **Framework**: SwiftUI (iOS 16+)
- **Language**: Swift 5.9+
- **Architecture**: MVVM with Combine
- **Authentication**: AuthenticationServices (Apple Sign In)
- **Networking**: URLSession with custom SSE implementation
- **Storage**: Keychain Services for secure credential storage
- **Concurrency**: Swift async/await with @MainActor

#### Infrastructure
- **Orchestration**: Docker Compose
- **Deployment**: DigitalOcean App Platform
- **CI/CD**: GitHub Actions
- **Production Server**: Gunicorn with Uvicorn workers

---

## AI/LLM Technologies

### Multi-Provider Architecture

The platform implements a sophisticated **model router** with automatic failback chains:

#### Provider Configuration

**Anthropic Claude 3.5 Sonnet**
- Primary model for planning phase
- Use case: Query generation, intent extraction
- Temperature: 0.1 (focused, deterministic)
- Token limit: 32 tokens for planning queries
- Cost: ~$3 per million input tokens

**OpenAI GPT-4o-mini**
- Primary model for synthesis phase
- Use case: Human-readable summaries, creative text generation
- Temperature: 0.3 (moderate creativity)
- Token limit: 220 tokens for output
- Cost: ~$0.15 per million input tokens (20x cheaper)

**OpenAI text-embedding-3-large**
- Vector embeddings for RAG (Retrieval-Augmented Generation)
- Dimensions: 3072
- Use case: Recipe similarity search

### Prompt Engineering Strategy

#### Two-Phase Approach

**Phase 1: Planning (Low Temperature, Focused)**
```python
prompt = f"""
You are a meal planner for a family. Context: {json.dumps(profile)}
User goal: {goal}
Return a short search query string only, suitable for recipe retrieval
(e.g., "kid friendly dinners no nuts").
"""
# Temperature: 0.1, Max tokens: 32, Model: Claude 3.5 Sonnet
```

**Phase 2: Synthesis (Moderate Temperature, Creative)**
```python
prompt = f"""
Create a concise weekly meal outline using these recipe titles: {titles}.
Tone: family-friendly, encouraging.
"""
# Temperature: 0.3, Max tokens: 220, Model: GPT-4o-mini
```

#### Context Injection

Family profile embedded directly in prompts:
```json
{
  "adults": 2,
  "kidsAges": [4, 7],
  "allergies": ["peanuts", "shellfish"],
  "cuisines": ["italian", "mexican", "asian"]
}
```

### RAG (Retrieval-Augmented Generation)

#### Knowledge Base

**39 Total Embeddings** across 6 content types:
- **Recipes (5)**: Kid-friendly meals with dietary restriction support
- **Safety (5)**: Choking hazards, home childproofing, emergency preparedness
- **Discipline (5)**: Positive reinforcement strategies, time-outs, age-appropriate techniques
- **Routines (5)**: Bedtime routines for ages 0-10, sleep optimization tips
- **Development (12)**: Comprehensive milestones (0-5 years) with red flags
- **Tips (7)**: Picky eating solutions, nutritional guidance, parenting strategies

#### Vector Database Implementation

**Storage**: PostgreSQL with pgvector extension
```sql
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    kind VARCHAR(32),           -- 'recipe', 'safety', 'discipline', 'routines', 'development', 'tips'
    doc_id VARCHAR(128),         -- Document identifier
    embedding TEXT,              -- Vector as text: "[0.123, 0.456, ...]"
    meta JSONB                   -- Full document metadata
);
```

#### Embedding Pipeline

**Step 1: Text Preparation**
```python
# Concatenate title + ingredients for semantic richness
text = f"{recipe['title']}\n" + "\n".join(recipe['ingredients'])
```

**Step 2: Generate Embeddings**
```python
embeddings = await openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=[text]
)
vector = embeddings.data[0].embedding  # 3072 dimensions
```

**Step 3: Store in PostgreSQL**
```python
# Convert vector to pgvector format
embedding_str = f"[{','.join(map(str, vector))}]"
conn.execute(
    "INSERT INTO vectors (kind, doc_id, embedding, meta) VALUES (:k, :id, :emb, :m)",
    {"k": "recipe", "id": recipe_id, "emb": embedding_str, "m": json.dumps(recipe)}
)
```

#### Vector Similarity Search

**Cosine Distance Query**
```sql
SELECT meta,
       (embedding::vector <-> :query_vector::vector) AS distance
FROM vectors
WHERE kind = 'recipe'
ORDER BY embedding::vector <-> :query_vector::vector
LIMIT 8;
```

**Retrieval Flow**:
1. User goal → LLM generates search query
2. Search query → Embed with text-embedding-3-large
3. Query vector → Cosine similarity search in pgvector
4. Top-K recipes → Feed to synthesis LLM
5. LLM synthesizes final meal plan

### Streaming Architecture

#### Server-Sent Events (SSE) Protocol

**Event Types**:
- `context` - User profile and household data
- `token` - Progress messages ("Planning meals...", "Fetching recipes...")
- `tool_call` - AI tool invocation with arguments
- `tool_result` - Tool execution results
- `final` - Final structured response
- `refusal` - Safety layer rejection
- `quota_exceeded` - Monthly usage quota exceeded (Step 14)
- `done` - Completion with elapsed time
- `error` - Error occurred

**Backend Streaming Implementation**:
```python
async def event_gen():
    start = time.perf_counter()
    yield json.dumps({"event": "context", "data": context})
    yield json.dumps({"event": "token", "data": "Planning meals..."})

    # RAG retrieval
    yield json.dumps({"event": "tool_call", "data": {"name": "fetch_recipes", "args": {"query": q}}})
    recipes = await fetch_recipes(query=q, k=8)
    yield json.dumps({"event": "tool_result", "data": {"count": len(recipes)}})

    # LLM synthesis
    plan_text = await llm_generate(recipes)
    yield json.dumps({"event": "final", "data": {"plan_text": plan_text, "recipes": recipes}})

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    yield json.dumps({"event": "done", "elapsed_ms": elapsed_ms})

return StreamingResponse(event_gen(), media_type="text/event-stream")
```

**iOS SSE Client Implementation**:
```swift
func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    guard let text = String(data: data, encoding: .utf8) else { return }
    buffer += text
    parseBuffer()  // Parse complete JSON objects from stream
}

private func parseBuffer() {
    // Find complete JSON objects using brace depth tracking
    var depth = 0
    // Extract complete objects: {"event":"token","data":"Planning meals..."}
    // Emit events to UI via callback
    onEvent?(.token("Planning meals..."))
}
```

### Cost Tracking & Budget Enforcement

#### Token Tracking & Usage Analytics

**Real-time Token Tracking**:
```python
class ModelRouter:
    def __init__(self):
        # Token tracking for the current run
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.models_used = {}  # {provider/model: count}

    async def _gen_once(self, provider: str, model: str, params: GenerateParams):
        # Extract actual token usage from LLM API response
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Accumulate tokens across all LLM calls in this request
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens

        # Track which models were used
        model_key = f"{provider}/{model}"
        self.models_used[model_key] = self.models_used.get(model_key, 0) + 1
```

**Database Persistence**:
```python
# Persist actual token counts to agent_runs table
agent_run = AgentRun(
    user_id=user_id,
    graph=mode,
    cost_cents=cost_cents,
    duration_ms=elapsed,
    input_tokens=router_instance.input_tokens,
    output_tokens=router_instance.output_tokens,
    total_tokens=router_instance.total_tokens,
    models_used=router_instance.models_used  # {"openai/gpt-4o-mini": 2}
)
```

**Usage Quotas**:
```python
# Monthly request limits per user (default: 10 requests/month)
USER_MONTHLY_QUOTA = 10

def check_user_quota(db: Session, user_id: int):
    month_start = datetime(now.year, now.month, 1)
    runs_this_month = db.query(func.count(AgentRun.id)).filter(
        AgentRun.user_id == user_id,
        AgentRun.created_at >= month_start
    ).scalar()

    remaining = max(0, USER_MONTHLY_QUOTA - runs_this_month)
    return {"allowed": remaining > 0, "quota": USER_MONTHLY_QUOTA, "used": runs_this_month}
```

#### Token Estimation
```python
# Naive estimation: 4 characters ≈ 1 token
approx_tokens = len(prompt) // 4 + max_tokens
estimated_cost_cents = approx_tokens * 0.00005  # $0.05 per 1000 tokens
```

#### Budget Enforcement
```python
if not self._cost_ok(approx_tokens, params.budget_cents):
    # Skip to next provider in fallback chain
    continue

# Fallback chain: Claude (expensive) → GPT-4o-mini (cheap)
FALLBACKS = {
    "planning": ["anthropic/claude-3-5-sonnet-latest", "openai/gpt-4o-mini"],
    "synthesis": ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-latest"]
}
```

#### Per-Request Budget Caps
```python
RUN_BUDGET_CENTS_DEFAULT = 10  # $0.10 per request
RUN_TOKEN_LIMIT_DEFAULT = 2000  # Hard token cap
```

### LLM Observability

#### OpenTelemetry Tracing
```python
async with aspan("llm.generate", provider=provider, model=model):
    response = await anthropic_client.messages.create(...)

# Captured metrics:
# - Latency (per-call, end-to-end)
# - Model metadata (provider, name, version)
# - Token counts (actual from API response - Step 14)
# - Cost calculations (based on actual usage)
# - Model usage tracking (which models called how many times)
```

#### Metrics Persistence
```sql
CREATE TABLE run_metrics (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    latency_ms INTEGER,
    cost_cents INTEGER,
    model_meta JSONB,  -- {"mode": "meals", "provider": "anthropic", "model": "claude-3-5-sonnet"}
    created_at TIMESTAMP
);

-- Token tracking in agent_runs (Step 14)
-- Actual token counts from LLM API responses, not estimates
-- Enables per-user usage analytics and cost monitoring
```

---

## Backend Platform

### API Endpoints

#### Authentication
```
POST /v1/auth/apple
```
- Verifies Apple ID token with Apple's public keys
- Creates user if doesn't exist
- Mints RS256 JWT for app authentication

#### Streaming AI Assist
```
POST /v1/assist/stream
```
- Server-Sent Events endpoint
- Real-time AI response generation
- Modes: meals, chores, routines

#### User Management
```
POST /v1/users/upsert
GET  /v1/households/by_user/{user_id}
POST /v1/households/upsert
```

#### Feedback
```
POST /v1/feedback
```
- Captures user feedback (thumbs up/down, edits)
- Links to agent runs for quality tracking

#### RAG Search (Knowledge Retrieval)
```
GET /v1/rag/search/recipes?q=healthy%20dinner&k=3
GET /v1/rag/search/knowledge?q=bedtime%20routine&k=3
GET /v1/rag/search/knowledge?q=choking%20hazards&kind=safety&k=2
GET /v1/rag/search/all?q=positive%20parenting&k=5
```
- Semantic search across recipes and parenting knowledge base
- Filter by content type (safety, discipline, routines, development, tips)
- Returns ranked results with similarity scores

#### Usage Analytics
```
GET /v1/users/{user_id}/usage?days=30
```
- Per-user token usage and cost statistics
- Returns: total_runs, total_tokens, input/output tokens, cost, model breakdown

**Response**:
```json
{
  "user_id": 1,
  "period_days": 30,
  "total_runs": 5,
  "total_tokens": 1780,
  "input_tokens": 625,
  "output_tokens": 1155,
  "total_cost_cents": 5,
  "total_cost_usd": 0.05,
  "avg_tokens_per_run": 356.0,
  "models_used": {
    "openai/gpt-4o-mini": 10,
    "anthropic/claude-3-5-sonnet-latest": 2
  }
}
```

```
GET /v1/admin/stats?days=30
```
- Platform-wide usage statistics (admin only)
- Returns: aggregate stats, top users, model breakdown

**Response**:
```json
{
  "period_days": 30,
  "total_users": 15,
  "total_runs": 247,
  "total_tokens": 89234,
  "total_cost_cents": 247,
  "total_cost_usd": 2.47,
  "top_users": [
    {"user_id": 1, "total_tokens": 15620, "total_cost_cents": 42}
  ],
  "models_breakdown": {
    "openai/gpt-4o-mini": 380,
    "anthropic/claude-3-5-sonnet-latest": 114
  }
}
```

```
GET /v1/admin/dashboard
```
- Web-based admin dashboard for monitoring platform usage
- Real-time metrics: users, requests, tokens, costs
- Top users table by token usage
- Model usage breakdown
- Period selector (7/30/90 days)
- Auto-refresh functionality

### Database Schema

#### Core Tables

**users**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    apple_sub VARCHAR(128) UNIQUE NOT NULL,  -- Apple Sign In identifier
    created_at TIMESTAMP DEFAULT NOW()
);
```

**households**
```sql
CREATE TABLE households (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    prefs JSONB  -- {"adults": 2, "kidsAges": [4,7], "allergies": [...], "cuisines": [...]}
);
```

**agent_runs**
```sql
CREATE TABLE agent_runs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    graph VARCHAR(64),      -- e.g., "meals_v1", "chores_v1"
    state JSONB,            -- Full execution state
    cost_cents INTEGER,
    duration_ms INTEGER,
    input_tokens INTEGER,   -- Token tracking (Step 14)
    output_tokens INTEGER,
    total_tokens INTEGER,
    models_used JSONB,      -- {"openai/gpt-4o-mini": 2, "anthropic/claude-3-5-sonnet-latest": 1}
    created_at TIMESTAMP
);
```

**vectors (pgvector)**
```sql
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    kind VARCHAR(32),       -- 'recipe', 'memory'
    doc_id VARCHAR(128),    -- Document identifier
    embedding TEXT,         -- Vector stored as text
    meta JSONB              -- Full document
);
CREATE INDEX ON vectors (kind, doc_id);
```

**feedback**
```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    run_id INTEGER,         -- References agent_runs
    signal VARCHAR(16),     -- 'up', 'down', 'edit'
    comment TEXT,
    payload JSONB,          -- Arbitrary feedback data
    created_at TIMESTAMP
);
```

### Orchestrator - Simple Planner

The orchestrator coordinates AI calls, tool execution, and streaming:

```python
async def plan_meals_stream(goal: str, context: dict, router: ModelRouter):
    # Step 1: LLM generates search query
    yield {"event": "token", "data": "Planning meals..."}
    query = await router.generate(
        prompt=f"Generate recipe search query for: {goal}",
        max_tokens=32, temperature=0.1
    )

    # Step 2: RAG retrieval
    yield {"event": "tool_call", "data": {"name": "fetch_recipes", "args": {"query": query}}}
    recipes = await fetch_recipes(query=query, dietary=context["allergies"], k=8)
    yield {"event": "tool_result", "data": {"count": len(recipes)}}

    # Step 3: Consolidate groceries
    groceries = consolidate_groceries(recipes)

    # Step 4: LLM synthesis
    plan_text = await router.generate(
        prompt=f"Create weekly meal plan using: {[r.title for r in recipes]}",
        max_tokens=220, temperature=0.3
    )

    # Step 5: Return final payload
    yield {"event": "final", "data": {
        "query": query,
        "plan_text": plan_text,
        "grocery": groceries
    }}
```

### Tools Implementation

#### Recipe Fetching (RAG)
```python
async def fetch_recipes(inp: FetchRecipesIn, router: ModelRouter):
    # Embed query
    query_vector = (await router.embed([inp.query]))[0]

    # Vector similarity search
    hits = search_recipes(query_vector, k=inp.k)

    # Post-filter by dietary restrictions
    filtered = [h for h in hits if dietary_safe(h, inp.dietary)]

    return FetchRecipesOut(recipes=filtered)
```

#### Grocery Consolidation
```python
def consolidate_groceries(ingredients: list[str]):
    # Normalize (lowercase, strip whitespace)
    # Deduplicate
    # Guess aisle (produce, dairy, meat, pantry)
    # Group by aisle
    return {"aisles": {"produce": [...], "dairy": [...], ...}}
```

#### Chore Scheduling
```python
def schedule_chores(name: str, start_date: datetime, weekdays: Weekday, days: int):
    occurrences = []
    for day_offset in range(days):
        date = start_date + timedelta(days=day_offset)
        if weekdays[date.weekday()]:  # Check if day is enabled
            occurrences.append(Occurrence(name=name, at=date))
    return occurrences
```

### Safety & Policy Engine

#### Multi-Layered Safety

**1. Precheck (Before LLM Call)**
```python
def precheck(domain: str, profile: dict, goal: str):
    # PII redaction
    goal = redact_pii(goal)

    # Content filtering (regex-based)
    for rule in policies[domain]["disallow_content"]:
        if re.search(rule["pattern"], goal, re.I):
            return {"ok": False, "refusal_text": get_refusal(rule["refusal"])}

    # Age gates
    for gate in policies[domain]["age_gates"]:
        for age in profile["kidsAges"]:
            if gate["min_age"] <= age <= gate["max_age"]:
                for keyword in gate["disallow_keywords"]:
                    if keyword.lower() in goal.lower():
                        return {"ok": False, "refusal_text": gate["refusal"]}

    return {"ok": True, "goal_sanitized": goal}
```

**2. PII Redaction**
```python
PII_REGEXES = [
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone numbers
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # Emails
    r"\b\d{1,5}\s+[A-Za-z0-9.\s]+(?:Street|St|Road|Ave)\b"  # Addresses
]

def redact_pii(text: str):
    for regex in PII_REGEXES:
        text = re.sub(regex, "[redacted]", text)
    return text
```

**3. YAML-Based Policies**
```yaml
domains:
  meals:
    disallow_content:
      - pattern: "(?:diagnose|medicine|prescription)"
        refusal: "medical"

  routines:
    age_gates:
      - min_age: 0
        max_age: 2
        disallow_keywords: ["honey", "whole nuts"]
        refusal: "age_inappropriate"

refusals:
  medical: "I can't provide medical advice. Please consult a pediatrician."
  age_inappropriate: "This isn't appropriate for this age. I'll adjust the plan."
```

**4. Postcheck (After LLM Response)**
```python
def postcheck(domain: str, text_out: str):
    # Final PII redaction
    return {"ok": True, "text": redact_pii(text_out)}
```

### Apple Sign In + JWT Flow

#### Step 1: Apple ID Token Verification
```python
def verify_apple_identity_token(id_token: str) -> str:
    # Fetch Apple's public keys
    jwks = requests.get("https://appleid.apple.com/auth/keys").json()

    # Get key ID from token header
    header = jwt.get_unverified_header(id_token)
    kid = header["kid"]

    # Find matching key and convert to PEM
    key = next(k for k in jwks["keys"] if k["kid"] == kid)
    public_key = RSAAlgorithm.from_jwk(key)

    # Verify signature and decode
    decoded = jwt.decode(
        id_token, public_key, algorithms=["RS256"],
        audience="com.ParentingAssistant.ParentingAssistantiOS",
        issuer="https://appleid.apple.com"
    )

    return decoded["sub"]  # Apple user identifier
```

#### Step 2: User Creation/Lookup
```python
apple_sub = verify_apple_identity_token(id_token)
user = db.query(User).filter_by(apple_sub=apple_sub).first()
if not user:
    user = User(apple_sub=apple_sub)
    db.add(user)
    db.commit()
```

#### Step 3: JWT Minting (RS256)
```python
def mint_jwt(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iss": "parenting-assistant",
        "aud": "parenting-assistant-ios"
    }
    return jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
```

**Key Management**:
- Private key: `secrets/jwt_private.pem` (RS256)
- Public key: `secrets/jwt_public.pem` (for verification)
- Environment variables: `JWT_PRIVATE_PEM`, `JWT_PUBLIC_PEM`

### Rate Limiting & Usage Quotas

**Short-term Rate Limiting** (Abuse Prevention):
```python
# Redis-backed rate limiter
RATE_LIMIT_PER_MIN = 30  # 30 requests per minute per IP

@router.post("/assist/stream", dependencies=[Depends(rate_limiter)])
async def assist_stream(...):
    # Rate limit enforced before handler execution
    pass
```

**Monthly Usage Quotas** (Cost Management - Step 14):
```python
# Per-user monthly request limits
USER_MONTHLY_QUOTA = 10  # Default: 10 requests per month

@router.post("/assist/stream")
async def assist_stream(...):
    # Check quota before processing
    quota_status = check_user_quota(db, user_id)
    if not quota_status["allowed"]:
        # Return quota_exceeded event via SSE
        yield json.dumps({
            "event": "quota_exceeded",
            "data": {
                "message": f"Monthly quota exceeded. You've used {quota_status['used']}/{quota_status['quota']} requests.",
                "quota": quota_status["quota"],
                "used": quota_status["used"],
                "reset_date": quota_status["reset_date"]
            }
        })
        return
```

### Observability (OpenTelemetry)

```python
# Setup OTLP exporter
def setup_tracing():
    resource = Resource.create({
        SERVICE_NAME: "parenting-assistant-api",
        "deployment.environment": "production"
    })

    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        headers={"Authorization": f"Bearer {api_key}"}
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

# Instrument code
async with aspan("llm.generate", provider="anthropic", model="claude-3-5-sonnet"):
    response = await llm_call()

async with aspan("tool.fetch_recipes", k=8):
    recipes = await fetch_recipes()
```

**Captured Traces**:
- HTTP requests (endpoint, status, latency)
- LLM generation (provider, model, tokens, latency)
- Tool invocations (name, args, results, latency)
- Database queries (SQL, duration)
- Redis operations

---

## iOS Application

### SwiftUI Architecture (MVVM)

#### App Entry Point
```swift
@main
struct ParentingAssistantApp: App {
    @StateObject private var session = AppSession.shared

    var body: some Scene {
        WindowGroup {
            if session.isAuthenticated && session.userId != nil {
                TabView {
                    MealsView().tabItem { Label("Meals", systemImage: "fork.knife") }
                    ChoresView().tabItem { Label("Chores", systemImage: "checklist") }
                    RoutinesView().tabItem { Label("Routines", systemImage: "figure.walk") }
                    AboutView().tabItem { Label("Settings", systemImage: "gearshape") }
                }
            } else {
                SignInWithAppleButtonView()
            }
        }
    }
}
```

### State Management

#### AppSession (Global Singleton)
```swift
@MainActor
final class AppSession: ObservableObject {
    @Published var isAuthenticated: Bool = false
    @Published var userId: Int?
    @Published var jwt: String?

    static let shared = AppSession()

    private init() {
        // Restore from Keychain on app launch
        if let data = Keychain.get("jwt"), let token = String(data: data, encoding: .utf8) {
            self.jwt = token
            self.isAuthenticated = true
        }
        if let data = Keychain.get("userId"), let id = Int(String(data: data, encoding: .utf8)!) {
            self.userId = id
        }
    }

    func signIn(userId: Int, jwt: String) {
        self.userId = userId
        self.jwt = jwt
        self.isAuthenticated = true
        Keychain.set(Data("\(userId)".utf8), for: "userId")
        Keychain.set(Data(jwt.utf8), for: "jwt")
    }

    func signOut() {
        self.userId = nil
        self.jwt = nil
        self.isAuthenticated = false
        Keychain.remove("userId")
        Keychain.remove("jwt")
    }
}
```

**Critical Pattern**: Uses `@ObservedObject` (not `@StateObject`) in child views to reference singleton

#### Feature ViewModels (MealsVM Example)
```swift
@MainActor
final class MealsVM: ObservableObject {
    @Published var planText: String = ""
    @Published var groups: [GroceryAisleGroup] = []
    @Published var streaming: Bool = false
    @Published var status: String = ""

    private let sse = SSEClient()

    func startPlanning(userId: Int, jwt: String?) {
        guard !streaming else { return }
        streaming = true

        let body = AssistRequest(user_id: userId, goal: "Plan dinners...", mode: "meals")
        sse.postStream(path: "/v1/assist/stream", body: body, jwt: jwt) { [weak self] event in
            Task { @MainActor in
                guard let self = self else { return }
                switch event {
                case .token(let t):
                    self.status = t  // Real-time UI update
                case .final(let data):
                    // Decode and populate UI
                    self.planText = data["plan_text"]
                    self.groups = parseGroceries(data["grocery"])
                    self.streaming = false
                case .unauthorized:
                    AppSession.shared.signOut()
                // ... other cases
                }
            }
        }
    }
}
```

### SSE Client - Advanced Streaming

#### Complete Implementation
```swift
final class SSEClient: NSObject, URLSessionDataDelegate {
    private var onEvent: ((SSEEvent) -> Void)?
    private var task: URLSessionDataTask?
    private var session: URLSession?
    private var buffer = ""  // Accumulates partial JSON

    func postStream<T: Encodable>(path: String, body: T, jwt: String?,
                                   onEvent: @escaping (SSEEvent) -> Void) {
        self.onEvent = onEvent
        buffer = ""

        var req = URLRequest(url: AppConfig.baseURL.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.addValue("text/event-stream", forHTTPHeaderField: "Accept")
        if let jwt { req.addValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization") }
        req.httpBody = try? JSONEncoder().encode(body)

        let s = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        self.session = s
        self.task = s.dataTask(with: req)
        self.task?.resume()
    }

    // CRITICAL: Handle HTTP response codes before streaming starts
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        if let http = response as? HTTPURLResponse {
            if http.statusCode == 401 {
                onEvent?(.unauthorized)
                completionHandler(.cancel)
                return
            }
            if http.statusCode >= 400 {
                onEvent?(.error("HTTP \(http.statusCode)"))
                completionHandler(.cancel)
                return
            }
        }
        completionHandler(.allow)
    }

    // Receive streaming data incrementally
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let text = String(data: data, encoding: .utf8) else { return }
        buffer += text
        parseBuffer()
    }

    // Parse complete JSON objects from stream
    private func parseBuffer() {
        var startIndex = buffer.startIndex

        while startIndex < buffer.endIndex {
            // Skip whitespace
            while startIndex < buffer.endIndex && buffer[startIndex].isWhitespace {
                startIndex = buffer.index(after: startIndex)
            }

            guard startIndex < buffer.endIndex, buffer[startIndex] == "{" else { break }

            // Find matching closing brace using depth tracking
            var depth = 0
            var currentIndex = startIndex
            var foundEnd = false

            while currentIndex < buffer.endIndex {
                let char = buffer[currentIndex]
                if char == "{" { depth += 1 }
                else if char == "}" {
                    depth -= 1
                    if depth == 0 {
                        foundEnd = true
                        currentIndex = buffer.index(after: currentIndex)
                        break
                    }
                }
                currentIndex = buffer.index(after: currentIndex)
            }

            if !foundEnd { break }  // Incomplete JSON, wait for more data

            // Extract and parse JSON object
            let jsonString = String(buffer[startIndex..<currentIndex])
            if let jsonData = jsonString.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
               let event = obj["event"] as? String {
                handleEvent(event, obj)
            }

            startIndex = currentIndex
        }

        // Remove processed data from buffer
        if startIndex > buffer.startIndex {
            buffer.removeSubrange(buffer.startIndex..<startIndex)
        }
    }
}
```

**Algorithm Highlights**:
- **Depth tracking** for nested JSON object parsing
- **Incremental processing** - handles partial data chunks
- **Buffer management** - prevents unbounded memory growth
- **Thread safety** - URLSession delegates on background thread, wrapped in `Task { @MainActor }`

### Apple Sign In Flow

```swift
@MainActor
final class AuthService: NSObject, ObservableObject,
                         ASAuthorizationControllerDelegate {

    private var continuation: CheckedContinuation<(userId: Int, jwt: String), Error>?

    func signInWithApple() async throws -> (userId: Int, jwt: String) {
        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont

            let provider = ASAuthorizationAppleIDProvider()
            let request = provider.createRequest()
            request.requestedScopes = [.fullName, .email]

            let controller = ASAuthorizationController(authorizationRequests: [request])
            controller.delegate = self
            controller.presentationContextProvider = self
            controller.performRequests()
        }
    }

    func authorizationController(controller: ASAuthorizationController,
                                didCompleteWithAuthorization authorization: ASAuthorization) {
        Task {
            // Extract Apple ID token
            let credential = authorization.credential as! ASAuthorizationAppleIDCredential
            let idToken = String(data: credential.identityToken!, encoding: .utf8)!

            // Send to backend
            let url = AppConfig.baseURL.appendingPathComponent("/v1/auth/apple")
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.httpBody = try JSONEncoder().encode(["id_token": idToken])

            let (data, _) = try await URLSession.shared.data(for: req)
            let response = try JSONDecoder().decode(AuthResponse.self, from: data)

            // Resume continuation with result
            continuation?.resume(returning: (userId: response.user_id, jwt: response.jwt))
        }
    }
}
```

**Pattern**: Bridges delegate-based API to async/await using `CheckedContinuation`

### Keychain Storage (Secure Credentials)

```swift
enum Keychain {
    static func set(_ value: Data, for key: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: value
        ]
        SecItemDelete(query as CFDictionary)
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    static func get(_ key: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]
        var result: AnyObject?
        return SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess
            ? result as? Data : nil
    }
}
```

**Security**:
- Data encrypted at rest by iOS
- Protected by device passcode/biometrics
- Survives app restarts
- Cleared on app deletion (configurable)

### Configuration Management

#### AppConfig (API Endpoints)
```swift
enum AppConfig {
    private static let defaultHost = "https://parenting-assistant-platform-39b47.ondigitalocean.app"

    static var baseURL: URL {
        // Allow runtime override for local development
        if let override = UserDefaults.standard.string(forKey: "apiHostOverride"),
           !override.isEmpty {
            return URL(string: override)!
        }
        return URL(string: defaultHost)!
    }

    static let sseAssistPath = "/v1/assist/stream"
    static let authApplePath = "/v1/auth/apple"
}
```

#### FeatureFlags (Offline Mode)
```swift
enum FeatureFlags {
    static var aiAssistEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "aiAssistEnabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "aiAssistEnabled") }
    }

    static let sampleMeals = ["Pasta", "Chicken stir-fry", "Veggie tacos", ...]
}
```

### Notifications Integration

```swift
enum Notifier {
    static func requestPermission() async -> Bool {
        let center = UNUserNotificationCenter.current()
        return (try? await center.requestAuthorization(options: [.alert, .sound])) ?? false
    }

    static func schedule(title: String, at date: Date) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.sound = .default

        let components = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: date)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: trigger)

        UNUserNotificationCenter.current().add(request)
    }
}
```

---

## Infrastructure & Deployment

### Docker Compose (Local Development)

```yaml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: parenting
    ports: ["5432:5432"]
    volumes: [db_data:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  api:
    build: .
    env_file: .env
    depends_on: [db, redis]
    ports: ["8000:8000"]
    command: uvicorn gateway.main:app --host 0.0.0.0 --reload
```

### Production Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY gateway/pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY gateway/gateway ./gateway
COPY gateway/alembic.ini ./
COPY gateway/alembic ./alembic
COPY secrets ./secrets

ENV PYTHONUNBUFFERED=1

CMD ["bash", "docker/start.sh"]
```

### Production Start Script

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Applying database migrations..."
alembic upgrade head

echo "Starting Gunicorn..."
exec gunicorn -k uvicorn.workers.UvicornWorker gateway.main:app \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers ${WEB_CONCURRENCY:-2} \
  --timeout 120
```

**Features**:
- Automatic migrations on startup
- Gunicorn with Uvicorn workers (async support)
- Environment-based scaling
- 120s timeout for long LLM requests

### DigitalOcean Deployment

**App Platform Configuration**:
```yaml
name: parenting-assistant-platform
services:
  - name: api
    build_command: docker build -f Dockerfile -t api .
    run_command: bash docker/start.sh
    envs:
      - key: DATABASE_URL
        scope: RUN_TIME
      - key: REDIS_URL
        scope: RUN_TIME
      - key: ANTHROPIC_API_KEY
        scope: RUN_TIME
        type: SECRET
      - key: OPENAI_API_KEY
        scope: RUN_TIME
        type: SECRET
    http_port: 8000

databases:
  - name: db
    engine: PG
    version: "16"

  - name: redis
    engine: REDIS
    version: "7"
```

### iOS TestFlight Distribution

**Status**: ✅ Live on TestFlight

The iOS app is professionally packaged and distributed through Apple's official beta testing platform.

**Distribution Details**:
- **Platform**: iOS 16.0+
- **Distribution Method**: TestFlight (Apple's Beta Testing)
- **Signing**: Automatic signing with Xcode
- **App Icon**: 1024x1024 production-ready icon
- **Bundle ID**: `com.ParentingAssistant.ParentingAssistantiOS`

**TestFlight Features**:
- Beta testing with up to 10,000 external testers
- Automatic update notifications
- Crash reporting and analytics
- Pre-release testing before App Store submission

**Build Process**:
1. Archive app in Xcode (Product → Archive)
2. Distribute to App Store Connect
3. Wait for processing (5-15 minutes)
4. Configure build in TestFlight
5. Invite testers via email

**Current Build**:
- Features: AI meal planning, chore scheduling, routines
- Authentication: Apple Sign In with JWT
- Streaming: Real-time SSE responses from backend
- Offline Mode: Sample data fallback

### Makefile Commands

```makefile
up: ; docker compose up -d --build
down: ; docker compose down
logs: ; docker compose logs -f api
migrate: ; docker compose exec api alembic upgrade head
seed: ; docker compose exec api python -m scripts.seed
ingest: ; docker compose exec api python -m gateway.rag.ingest

test:
	curl -N -X POST localhost:8000/v1/assist/stream \
	  -H 'content-type: application/json' \
	  -d '{"user_id":1,"goal":"Plan dinners","mode":"meals"}'
```

### CI/CD Pipeline (GitHub Actions)

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
      redis:
        image: redis:7-alpine

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }

      - name: Install dependencies
        run: pip install -e gateway[dev]

      - name: Run migrations
        run: alembic upgrade head

      - name: Seed data
        run: python -m scripts.seed_context

      - name: RAG ingestion
        if: env.OPENAI_API_KEY != ''
        run: python -m gateway.rag.ingest

      - name: Run tests
        run: pytest -q
```

---

## Security & Safety

### Authentication Security

**Apple Sign In**:
- Industry-standard OAuth 2.0 flow
- Verifies signatures with Apple's public keys (RSA)
- No passwords stored
- Biometric authentication (Face ID/Touch ID)

**JWT Security**:
- RS256 asymmetric signing
- 24-hour expiration
- Audience validation
- Issuer validation
- Secure key storage (env vars for prod, files for dev)

### Data Security

**In Transit**:
- HTTPS/TLS for all API calls
- Certificate pinning (can be added)

**At Rest**:
- Keychain encryption (iOS)
- PostgreSQL encryption at rest (DigitalOcean managed)
- Environment variables for secrets

### Safety Features

**PII Redaction**:
- Regex-based detection (emails, phones, addresses)
- Applied to user input and LLM output
- Replaced with `[redacted]`

**Content Filtering**:
- Medical advice refusal
- Sensitive parenting topics (sleep training, etc.)
- Age-inappropriate content

**Age Gates**:
- 0-2 years: Block honey, whole nuts, screen time
- Customizable per domain

**Rate Limiting & Usage Quotas**:
- Short-term: 30 requests per minute per IP
- Long-term: 10 requests per month per user (Step 14)
- Redis-backed distributed rate limiting
- Prevents abuse and runaway costs
- Quota tracking with automatic reset on first of month

---

## Learning Outcomes

### AI/LLM Engineering

✅ **Multi-Model Orchestration**
- Implemented fallback chains between Anthropic Claude and OpenAI GPT
- Model selection based on task (planning vs synthesis)
- Budget enforcement and cost optimization

✅ **Prompt Engineering**
- Two-phase approach (low temp planning → moderate temp synthesis)
- Context injection with family profiles
- System vs user prompt separation

✅ **RAG Implementation**
- Vector embeddings with OpenAI text-embedding-3-large
- PostgreSQL pgvector for similarity search
- Hybrid search (vector + metadata filtering)

✅ **Streaming Responses**
- Server-Sent Events (SSE) protocol
- Real-time token emission
- Custom JSON parsing for partial data

✅ **LLM Observability**
- OpenTelemetry tracing
- Cost tracking and budget caps
- Latency monitoring per component
- Actual token tracking from API responses (Step 14)
- Per-user usage analytics and quotas
- Platform-wide cost monitoring with admin dashboard

### Backend Development

✅ **FastAPI Expertise**
- Async/await throughout
- Dependency injection
- Middleware (CORS, security headers, rate limiting)
- StreamingResponse for SSE

✅ **Database Design**
- SQLAlchemy ORM with async support
- Alembic migrations
- JSONB for flexible schemas
- pgvector extension for ML workloads

✅ **Authentication**
- Apple Sign In OAuth flow
- RS256 JWT implementation
- Public key cryptography
- Secure token lifecycle management

✅ **API Design**
- RESTful conventions
- Structured error responses
- API versioning (/v1/)
- Rate limiting best practices

### iOS Development

✅ **SwiftUI Mastery**
- MVVM architecture
- Property wrappers (@StateObject, @ObservedObject, @Published)
- Navigation patterns (TabView, NavigationStack)
- Declarative UI with reactive updates

✅ **Advanced Networking**
- Custom SSE client with URLSessionDataDelegate
- Incremental JSON parsing with depth tracking
- Buffer management for streaming
- 401 auto-logout handling

✅ **Async Programming**
- Swift async/await
- Continuation bridges for delegate APIs
- @MainActor for thread safety
- Task-based concurrency

✅ **Security**
- Keychain Services API
- Apple Sign In integration (AuthenticationServices)
- Secure credential storage
- JWT handling

### DevOps & Infrastructure

✅ **Containerization**
- Docker multi-stage builds
- Docker Compose orchestration
- Environment-based configuration

✅ **Cloud Deployment**
- DigitalOcean App Platform
- Managed PostgreSQL and Redis
- Auto-scaling with Gunicorn workers

✅ **Observability**
- OpenTelemetry instrumentation
- Distributed tracing
- Metrics collection
- Structured logging

✅ **CI/CD**
- GitHub Actions workflows
- Automated testing
- Database migrations in CI
- Secret management

### Software Engineering Best Practices

✅ **Clean Code**
- Separation of concerns
- Dependency injection
- Type safety (Pydantic, Swift type system)
- Error handling patterns

✅ **Testing**
- Unit tests (pytest)
- Integration tests (end-to-end API tests)
- RAG evaluation (RAGAS framework)

✅ **Documentation**
- Code comments
- API documentation
- Architecture diagrams
- This README!

---

## Repository Structure

```
ParentingAssistant/
├── parenting-assistant-platform/    # Backend (Python/FastAPI)
│   ├── gateway/
│   │   ├── gateway/
│   │   │   ├── main.py              # FastAPI app
│   │   │   ├── routes/              # API endpoints
│   │   │   ├── model_router/        # LLM abstraction
│   │   │   ├── orchestrator/        # Planning logic
│   │   │   ├── tools/               # AI tools (recipes, groceries, etc.)
│   │   │   ├── rag/                 # Vector search & ingestion
│   │   │   ├── data/                # RAG knowledge base
│   │   │   │   ├── seed/            # Parenting knowledge (markdown)
│   │   │   │   │   ├── safety/      # Choking hazards, home safety
│   │   │   │   │   ├── discipline/  # Positive strategies, time-outs
│   │   │   │   │   ├── routines/    # Bedtime, morning routines
│   │   │   │   │   ├── development/ # Milestones, red flags
│   │   │   │   │   └── tips/        # Picky eating, nutrition
│   │   │   │   └── recipes_seed.json
│   │   │   ├── safety/              # Policy engine
│   │   │   ├── auth/                # Apple Sign In + JWT
│   │   │   ├── db/                  # SQLAlchemy models
│   │   │   └── observability/       # OpenTelemetry
│   │   ├── alembic/                 # Database migrations
│   │   └── pyproject.toml
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── Makefile
│
└── parenting-assistant-ios/         # iOS App (Swift/SwiftUI)
    ├── ParentingAssistant/
    │   ├── ParentingAssistantApp.swift
    │   ├── AppState/                # Global state (AppSession)
    │   ├── Features/                # Meals, Chores, Routines, Auth, Settings
    │   ├── Services/                # SSEClient, APIClient, Auth, Keychain, Notifications
    │   ├── Models/                  # Data models
    │   └── Config/                  # AppConfig, FeatureFlags
    └── ParentingAssistant.xcodeproj
```

---

## Getting Started

### Backend Setup

```bash
cd parenting-assistant-platform

# Copy environment template
cp .env.example .env
# Edit .env with your API keys:
# - ANTHROPIC_API_KEY
# - OPENAI_API_KEY
# - JWT keys (generate with openssl)

# Start services
make up

# Run migrations
make migrate

# Seed sample data
make seed

# Ingest RAG knowledge base (39 embeddings: recipes + parenting knowledge)
docker compose exec api python -m gateway.rag.ingest

# Test streaming endpoint
make test
```

### iOS Setup

```bash
cd parenting-assistant-ios

# Open in Xcode
open ParentingAssistant.xcodeproj

# Update bundle ID in project settings:
# com.ParentingAssistant.ParentingAssistantiOS

# Run on Simulator or device (Cmd+R)
```

**Dev Mode**: In DEBUG builds, use "Dev Mode (Skip Sign In)" button to bypass authentication for testing.

---

## Future Enhancements

### AI/LLM
- [ ] Token-by-token streaming from LLM (vs orchestrator-level)
- [ ] Fine-tuned models for meal planning
- [ ] Multi-turn conversations with memory
- [ ] Image generation for meal photos
- [ ] Voice input/output

### RAG
- [ ] HNSW index for faster vector search
- [ ] Expand recipe dataset (100+ → 10,000+)
- [ ] Hybrid search (vector + BM25)
- [ ] Recipe image embeddings (CLIP)

### Features
- [ ] Pantry inventory tracking
- [ ] Nutrition information
- [ ] Shopping list sharing
- [ ] Recipe favorites
- [ ] Routine progress tracking
- [ ] Family calendar integration

### Infrastructure
- [ ] GraphQL API
- [ ] WebSocket alternative to SSE
- [ ] Redis caching layer
- [ ] CDN for static assets
- [ ] Multi-region deployment

### Observability
- [ ] APM (Datadog, New Relic)
- [ ] Error tracking (Sentry)
- [ ] User analytics
- [ ] A/B testing framework

---

## License

MIT License - See LICENSE file for details

---

## Contact

Built by Ahmed M. - Learning AI/LLM engineering through hands-on projects.

- GitHub: [ParentingAssistant](https://github.com/ParentingAssistant)
- Backend: https://parenting-assistant-platform-39b47.ondigitalocean.app

---

**Last Updated**: January 2025
