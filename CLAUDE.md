# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parenting Assistant is a production AI-powered family planning platform with:
- **Backend**: FastAPI with multi-agent LLM orchestrator, deployed on DigitalOcean App Platform
- **iOS App**: SwiftUI with custom SSE streaming, distributed via TestFlight
- **Production URL**: https://parenting-assistant-platform-39b47.ondigitalocean.app

## Repository Structure

This is a **monorepo** with two main submodules:
- `parenting-assistant-platform/` - Python/FastAPI backend
- `parenting-assistant-ios/` - Swift/SwiftUI iOS app

## Common Development Commands

### Backend (parenting-assistant-platform/)

**Local Development:**
```bash
make up              # Start all services (API, DB, Redis, Grafana)
make down            # Stop all services
make logs            # Follow API logs
make migrate         # Run database migrations
make autogen         # Generate new migration from model changes
```

**Database Operations:**
```bash
make seed            # Seed initial data
make seed_context    # Seed domain-specific context
make ingest_recipes  # Ingest recipes into RAG vector DB
make rag.ingest      # Ingest custom data: SRC=path/to/data make rag.ingest
```

**Testing & Debugging:**
```bash
make smoke_meals     # Test meal planning agent with SSE streaming
make smoke_chores    # Test chores agent
make smoke_routines  # Test routines agent
make tests           # Run pytest suite
make evals           # Run RAGAS evaluation on meal planning

# Usage tracking (requires USER_ID, DAYS, TOKEN)
make usage.user USER_ID=1 DAYS=30
make usage.admin DAYS=7
make usage.db        # Query usage from PostgreSQL directly

# Profile & memory (requires TOKEN)
make profile.get TOKEN=<jwt>
make memory.list TOKEN=<jwt> LIMIT=20
```

**Database Access:**
```bash
docker compose exec db psql -U app -d parenting
# Or use make targets:
make profile.db      # View user profiles
make memory.db USER_ID=1  # View user memories
```

**Design System (Figma → Swift):**
```bash
make figma.tokens         # Export design tokens from Figma API
make designsystem.swift   # Generate DesignSystem.swift from tokens
make designsystem.all     # Run both in sequence
```

### iOS (parenting-assistant-ios/)

**Build & Run:**
1. Open `parenting-assistant-ios/ParentingAssistant.xcodeproj` in Xcode
2. Select simulator or device
3. Cmd+R to build and run

**TestFlight Distribution:**
1. Product → Archive
2. Distribute to App Store Connect
3. Wait for processing (5-15 minutes)
4. Configure build in TestFlight

## Architecture Overview

### Multi-Agent Orchestrator

The backend uses intent-based routing to specialized agents:

```
User Request → Intent Classifier → {
  "meal_planning" → MealPlannerAgent (RAG + grocery lists)
  "routines"      → RoutineAgent (age-appropriate schedules)
  "chores"        → ChoresAgent (household task planning)
  *               → GenericParentingAgent (fallback)
}
```

**Key Files:**
- `gateway/agents/orchestrator.py` - Main orchestration logic
- `gateway/agents/intent.py` - Intent classification (keyword-based)
- `gateway/agents/meal_agent.py` - Meal planning with RAG retrieval
- `gateway/agents/routine_agent.py` - Bedtime/morning routines
- `gateway/agents/chores_agent.py` - Chore scheduling
- `gateway/agents/generic_agent.py` - General parenting advice

### Smart Model Routing (Step 38)

Dynamic LLM selection with fallback chains:

```python
# gateway/model_router/router.py
# Complexity-based routing: simple → cheap models, complex → powerful models
# Safety-critical (allergies) → always Claude
# Budget-aware with automatic provider failover
```

**Fallback Chain Example:**
```
1. Try: anthropic/claude-3-5-sonnet-latest (safety-capable)
2. Fallback: openai/gpt-4o (if Claude fails)
3. Fallback: openai/gpt-4o-mini (if budget exceeded)
4. Final: hf/llama-3.2-3B (free tier last resort)
```

### Server-Sent Events (SSE) Streaming

**Backend** (`gateway/routes/assist.py`):
```python
async def event_gen():
    yield json.dumps({"event": "context", "data": context})
    yield json.dumps({"event": "token", "data": "Planning meals..."})
    # ... RAG retrieval, LLM synthesis
    yield json.dumps({"event": "final", "data": result})
    yield json.dumps({"event": "done", "elapsed_ms": ms})
```

**iOS** (`ParentingAssistant/Services/SSEClient.swift`):
- Custom URLSessionDataDelegate with incremental buffer parsing
- Depth tracking for nested JSON objects
- Thread-safe UI updates via `Task { @MainActor in }`

### RAG (Retrieval-Augmented Generation)

**Pipeline:**
1. User prompt → LLM generates search query (temperature: 0.1)
2. Embed query with OpenAI text-embedding-3-large (3072 dimensions)
3. pgvector cosine similarity search in PostgreSQL
4. Top-K results → Feed to synthesis LLM (temperature: 0.3)

**Key Files:**
- `gateway/rag/ingest.py` - Embedding pipeline
- `gateway/rag/retrieval.py` - Vector search
- `gateway/db/models.py` - `vectors` table (kind, doc_id, embedding, meta)

### Database Migrations

**Location:** `gateway/alembic/versions/`

**Naming Convention:** `YYYY_MM_DD_NNNNNN_description.py`

**Process:**
1. Modify models in `gateway/db/models.py`
2. `make autogen` - Generate migration
3. Review generated migration file
4. `make migrate` - Apply locally
5. Commit migration file
6. Production applies automatically on deploy via `docker/start.sh`

### Authentication Flow

**Apple Sign In → JWT:**
1. iOS sends Apple ID token to `/v1/auth/apple`
2. Backend verifies with Apple's public keys
3. Backend mints RS256 JWT (24-hour expiry)
4. iOS stores JWT in Keychain
5. Automatic refresh via `/v1/auth/refresh-v2` (60 min before expiry)

**Key Files:**
- `gateway/auth/apple.py` - Apple ID token verification
- `gateway/auth/jwt_handler.py` - JWT minting/verification
- `ParentingAssistant/Services/AuthService.swift` - Apple Sign In flow
- `ParentingAssistant/Services/TokenRefreshService.swift` - Background refresh

### iOS Architecture (MVVM)

**State Management:**
- `AppSession.swift` - Global singleton for auth state (userId, jwt)
- Feature ViewModels - `@ObservedObject` pattern, not `@StateObject`
- `@MainActor` for thread safety

**Services:**
- `SSEClient.swift` - Streaming with URLSessionDataDelegate
- `APIClient.swift` - REST API calls with async/await
- `AssistClient.swift` - High-level wrapper for agent calls
- `ArtifactsClient.swift` - Save/load meal plans and routines
- `UsageClient.swift` - Usage tracking and tier info

**Design System:**
- `DesignSystem.swift` - Auto-generated from Figma tokens
- `DS.Colors.accent`, `DS.Spacing.md`, `DS.Radius.pill` namespace pattern
- Reusable components: `DSButton`, `DSCard`, `ChatBubble`

## Critical Deployment Rules

### ⚠️ DigitalOcean Secrets Management

**NEVER run:** `doctl apps update --spec .do/app.yaml`
- This **WIPES ALL SECRET VALUES** from production

**Instead:**
1. Use DigitalOcean web UI: Settings → App-Level Environment Variables
2. Or run backup script before changes: `.do/backup-secrets.sh "passphrase"`

**Safe Commands:**
```bash
doctl apps spec get <app-id>               # View current config with secrets
doctl apps list-deployments <app-id>       # View deployments
doctl apps create-deployment <app-id>      # Trigger deploy without config change
```

**Recovery:** See `.do/README-SECRETS.md` for full backup/restore procedures

### Router Prefix Pattern

When adding new API routers, include the route prefix in the router declaration:

```python
# CORRECT - prefix in router
router = APIRouter(prefix="/usage", tags=["usage"])
@router.get("/summary")  # Results in /v1/usage/summary

# WRONG - missing prefix
router = APIRouter(tags=["usage"])
@router.get("/summary")  # Results in /v1/summary (not /v1/usage/summary)
```

Then register in `gateway/main.py`:
```python
app.include_router(usage.router, prefix="/v1")
```

### iOS Error Handling Pattern

Use NSError pattern consistently across all clients:

```swift
// CORRECT
throw NSError(domain: "UsageClient", code: 401,
             userInfo: [NSLocalizedDescriptionKey: "User not authenticated"])

// WRONG - don't create custom error enums unless shared
enum APIError: Error { case unauthorized }
throw APIError.unauthorized
```

## AI/LLM Best Practices

### Two-Phase Strategy

All agents use this pattern:
1. **Planning Phase** (temp: 0.1, max_tokens: 32-200)
   - Deterministic query generation
   - Intent extraction
   - Use cheap models (gpt-4o-mini)

2. **Synthesis Phase** (temp: 0.3, max_tokens: 220-800)
   - Human-readable output
   - Family-friendly tone
   - Use appropriate model based on complexity

### Safety-Critical Personalization

For allergies and dietary restrictions:
```python
if allergies:
    prompt += f"\n\n⚠️ CRITICAL ALLERGY SAFETY:\n"
    prompt += f"Family has allergies to: {', '.join(allergies)}\n"
    prompt += "DO NOT suggest any recipes containing these ingredients!"
```

Always route safety-critical requests to Claude (configured in model router).

### Profile Normalization

User profiles must have canonical structure:
```python
{
  "adults": [{"name": "Me", "role": "parent"}],  # Default
  "kids": [{"name": "Emma", "age": 4, "dietary_restrictions": ["peanuts"]}],
  "cuisines": ["italian", "mexican"],
  "consent_ai": true,
  "consent_analytics": true
}
```

Normalize on read, not write (see `gateway/services/profile.py`).

## Testing in Production

**Model Routing Tests:**
```bash
cd gateway/scripts
./test_routing_curl.sh  # Tests all routing scenarios
```

**Usage API Tests:**
```bash
JWT_TOKEN="<valid-jwt>" python scripts/test_usage_api.py
```

**Meal Planning Smoke Test:**
```bash
JWT_TOKEN="<valid-jwt>" curl -H "Authorization: Bearer $JWT_TOKEN" \
  "https://parenting-assistant-platform-39b47.ondigitalocean.app/v1/assist/stream" \
  -d '{"user_id":1,"goal":"Plan dinners","mode":"meals"}'
```

## Observability

**Admin Dashboard:**
- URL: `https://parenting-assistant-platform-39b47.ondigitalocean.app/v1/admin/dashboard`
- Shows: System health, daily costs, top users, model usage
- No auth required for dashboard endpoints (public)

**Metrics Export:**
- Prometheus: `/v1/metrics`
- Grafana Agent scrapes every 15 seconds
- OpenTelemetry traces exported to configured endpoint

**Logs:**
```bash
make logs  # Local
doctl apps logs <app-id> --follow  # Production
```

## Common Pitfalls

1. **Missing Router Prefix**: Always add `prefix="/resource"` to APIRouter
2. **Expired JWT**: Use TokenRefreshService, not manual refresh
3. **Secrets Wiped**: Never use `doctl apps update --spec` without backup
4. **Migration Conflicts**: Always pull latest before `make autogen`
5. **iOS State Loss**: Use `@ObservedObject` for AppSession, not `@StateObject`
6. **SSE Buffer Overflow**: Parse incrementally, don't accumulate full response
7. **Allergy Safety**: Always check profile for dietary restrictions in meal agent
8. **Profile Format**: Use normalized structure, don't assume old format exists
