# Building Agents with AWS Bedrock AgentCore: A Comprehensive Guide

This guide walks you through every aspect of building, testing, deploying, and invoking AI agents using the [Bedrock AgentCore Python SDK](https://github.com/aws/bedrock-agentcore-sdk-python). It covers the underlying technology, multiple agent framework integrations, tool usage, memory systems, identity/auth, evaluation, and more — all with technical depth and source-level references.

---

## Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
  - [1.1 What is Bedrock AgentCore?](#11-what-is-bedrock-agentcore)
  - [1.2 SDK Architecture and Underlying Technology](#12-sdk-architecture-and-underlying-technology)
- [2. Writing Agent Code — Multiple Approaches](#2-writing-agent-code--multiple-approaches)
  - [2.1 Minimal Agent (Framework-Free)](#21-minimal-agent-framework-free)
  - [2.2 Strands Agent Framework](#22-strands-agent-framework)
  - [2.3 Streaming Agent](#23-streaming-agent)
  - [2.4 WebSocket Agent](#24-websocket-agent)
  - [2.5 LangGraph, CrewAI, Autogen, or Custom Frameworks](#25-langgraph-crewai-autogen-or-custom-frameworks)
- [3. Agent Powerfulness Dimensions](#3-agent-powerfulness-dimensions)
  - [3.1 Simple Chat Agent](#31-simple-chat-agent)
  - [3.2 Using Tools](#32-using-tools)
  - [3.3 Persistent Memory (Short-Term and Long-Term)](#33-persistent-memory-short-term-and-long-term)
  - [3.4 Identity and Authentication (OAuth2 & API Keys)](#34-identity-and-authentication-oauth2--api-keys)
  - [3.5 Cloud-Based Tools: Browser and Code Interpreter](#35-cloud-based-tools-browser-and-code-interpreter)
  - [3.6 Async Task Management and Health Tracking](#36-async-task-management-and-health-tracking)
- [4. Testing Agents](#4-testing-agents)
  - [4.1 Unit Testing](#41-unit-testing)
  - [4.2 Integration Testing](#42-integration-testing)
  - [4.3 Agent Evaluation](#43-agent-evaluation)
- [5. Deploying Agents](#5-deploying-agents)
  - [5.1 Local Development and Running](#51-local-development-and-running)
  - [5.2 Containerization](#52-containerization)
  - [5.3 Deploying to Bedrock AgentCore Runtime](#53-deploying-to-bedrock-agentcore-runtime)
- [6. Invoking Deployed Agents](#6-invoking-deployed-agents)
  - [6.1 HTTP Invocation](#61-http-invocation)
  - [6.2 WebSocket Invocation (SigV4 Auth)](#62-websocket-invocation-sigv4-auth)
  - [6.3 WebSocket with Presigned URLs (Frontend)](#63-websocket-with-presigned-urls-frontend)
  - [6.4 OAuth WebSocket Invocation](#64-oauth-websocket-invocation)
- [7. Observability and Monitoring](#7-observability-and-monitoring)
- [8. Security Considerations](#8-security-considerations)
- [Summary: End-to-End Agent Lifecycle](#summary-end-to-end-agent-lifecycle)

---

## 1. Architecture Overview

### 1.1 What is Bedrock AgentCore?

Amazon Bedrock AgentCore is an AWS service that provides production-ready infrastructure for deploying and operating AI agents at scale. Instead of managing servers, containers, and networking yourself, you wrap your agent logic in the SDK and deploy it to AgentCore — which then handles compute, scaling, auth, memory, and observability.

The SDK in this repository (`bedrock-agentcore`) is the Python client that:

1. **Wraps your agent code** into a production-ready HTTP/WebSocket server
2. **Provides client libraries** for AgentCore services (memory, tools, identity, evaluation)
3. **Connects to AWS data/control plane APIs** for managing resources

### 1.2 SDK Architecture and Underlying Technology

At its core, the SDK is built on top of [Starlette](https://www.starlette.io/) (an ASGI web framework) and served via [Uvicorn](https://www.uvicorn.org/) (an ASGI server). Understanding this is key to understanding how everything works.

**The `BedrockAgentCoreApp` class** ([`src/bedrock_agentcore/runtime/app.py`](../src/bedrock_agentcore/runtime/app.py)) extends `starlette.applications.Starlette` and defines three routes:

```python
# From src/bedrock_agentcore/runtime/app.py, lines 100-104
routes = [
    Route("/invocations", self._handle_invocation, methods=["POST"]),
    Route("/ping", self._handle_ping, methods=["GET"]),
    WebSocketRoute("/ws", self._handle_websocket),
]
```

- **`POST /invocations`** — The main entrypoint for invoking your agent. This is where your `@app.entrypoint` function gets called.
- **`GET /ping`** — Health check endpoint used by AgentCore Runtime to determine if the container is ready to accept requests.
- **`/ws`** — WebSocket endpoint for bidirectional streaming communication.

**How invocation works under the hood:**

1. An HTTP POST comes in to `/invocations`
2. The SDK extracts request context (session ID, request ID, access tokens, custom headers) from HTTP headers into a [`RequestContext`](../src/bedrock_agentcore/runtime/context.py) (Pydantic model) and stores them in Python `contextvars` via [`BedrockAgentCoreContext`](../src/bedrock_agentcore/runtime/context.py)
3. The JSON payload is parsed and passed to your `@app.entrypoint` handler
4. If your handler is sync, it's run in a thread executor to not block the event loop; if async, it's awaited directly
5. The return value is JSON-serialized (with progressive fallback for complex objects like Pydantic models, dataclasses, etc. — see [`convert_complex_objects`](../src/bedrock_agentcore/runtime/utils.py))
6. If the return is a generator or async generator, a `StreamingResponse` with Server-Sent Events (SSE) format is returned

**The `/ping` health check** implements a state machine:

```
Priority: Forced Status > Custom @app.ping handler > Automatic (based on @app.async_task tracking)

States:
  - HEALTHY      = "Healthy"      → Ready for new work
  - HEALTHY_BUSY = "HealthyBusy"  → Currently processing, avoid new work
```

This is how AgentCore Runtime knows whether to route new requests to your container. This is documented in the [Ping Status Contract](../tests_integ/async/README.md).

**AWS API Communication** uses two types of endpoints ([`src/bedrock_agentcore/_utils/endpoints.py`](../src/bedrock_agentcore/_utils/endpoints.py)):
- **Control Plane** (`bedrock-agentcore-control.{region}.amazonaws.com`): Resource management (create/delete memories, browsers, code interpreters)
- **Data Plane** (`bedrock-agentcore.{region}.amazonaws.com`): Runtime operations (invoke agents, read/write memory events, execute code)

---

## 2. Writing Agent Code — Multiple Approaches

The key idea is that **`BedrockAgentCoreApp` is framework-agnostic**. Your `@app.entrypoint` handler is just a Python function that accepts a JSON payload and returns a response. What you do inside that function — which LLM you call, which framework you use — is entirely up to you.

### 2.1 Minimal Agent (Framework-Free)

The simplest possible agent uses no framework at all:

```python
# Reference: tests_integ/agents/sample_agent.py
import asyncio
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload):
    app.logger.info("Received payload: %s", payload)
    user_message = payload.get("prompt", "No prompt provided")
    # Your custom logic here — call an LLM API directly, run rules, etc.
    return {"response": f"You said: {user_message}"}

app.run()
```

**Under the hood:** `app.run()` calls `uvicorn.run(self, host=..., port=8080)`, starting the Starlette ASGI app. The handler is registered in `self.handlers["main"]` by the `@app.entrypoint` decorator ([`app.py` line 116-127](../src/bedrock_agentcore/runtime/app.py#L116-L127)).

**Sync handlers are also supported.** If you don't use `async def`, the SDK detects this and runs your handler in a thread pool to prevent blocking the event loop:

```python
@app.entrypoint
def invoke(payload):  # Sync function — runs in executor
    return {"response": "Hello from sync handler"}
```

This is handled by `_invoke_handler` ([`app.py` lines 466-479](../src/bedrock_agentcore/runtime/app.py#L466-L479)):
```python
if asyncio.iscoroutinefunction(handler):
    return await handler(*args)
else:
    loop = asyncio.get_event_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(None, ctx.run, handler, *args)
```

Note the use of `contextvars.copy_context()` — this ensures that context variables (session ID, request ID, etc.) are available in the executor thread.

### 2.2 Strands Agent Framework

[Strands](https://github.com/strands-agents/sdk-python) is AWS's own open-source agent framework. The SDK has deep, first-class integration with Strands:

```python
# Reference: tests_integ/runtime/test_simple_agent.py (lines 14-25)
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp(debug=True)
agent = Agent()

@app.entrypoint
async def agent_invocation(payload):
    return agent(payload.get("message"))

app.run()
```

**Streaming with Strands** uses async generators for real-time token streaming:

```python
# Reference: tests_integ/agents/streaming_agent.py
from strands import Agent
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
agent = Agent()

@app.entrypoint
async def agent_invocation(payload):
    user_message = payload.get("prompt", "No prompt found")
    stream = agent.stream_async(user_message)
    async for event in stream:
        yield event  # Yielding makes this an async generator → SSE response

if __name__ == "__main__":
    app.run()
```

**How SSE streaming works:** When `_handle_invocation` detects that your handler returns a generator (sync or async), it wraps it in a `StreamingResponse` with `media_type="text/event-stream"`. Each yielded value is serialized to JSON and wrapped in SSE format (`data: {...}\n\n`) by `_convert_to_sse` ([`app.py` lines 569-580](../src/bedrock_agentcore/runtime/app.py#L569-L580)).

### 2.3 Streaming Agent

You can stream from any framework, not just Strands:

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
async def stream_handler(payload):
    """Async generator for streaming responses."""
    prompt = payload.get("prompt", "")
    # Call any streaming LLM API
    for i in range(5):
        yield {"chunk": i, "text": f"Processing chunk {i}..."}
    yield {"chunk": "final", "text": "Done!"}

app.run()
```

Sync generators work too:

```python
@app.entrypoint
def stream_handler(payload):
    """Sync generator — also works for streaming."""
    for i in range(5):
        yield {"chunk": i}
```

### 2.4 WebSocket Agent

For bidirectional, long-lived communication, use the `@app.websocket` decorator:

```python
# Reference: tests_integ/runtime/test_websocket_agent.py (lines 17-54)
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.websocket
async def websocket_handler(websocket, context):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "echo":
                await websocket.send_json({
                    "type": "echo_response",
                    "message": data.get("message"),
                    "session_id": context.session_id
                })
            elif data.get("action") == "stream":
                count = data.get("count", 3)
                for i in range(count):
                    await websocket.send_json({
                        "type": "stream_chunk",
                        "chunk_id": i,
                        "data": f"Chunk {i+1} of {count}"
                    })
                await websocket.send_json({"type": "stream_complete"})
            elif data.get("action") == "close":
                break
    finally:
        await websocket.close()

app.run()
```

**Under the hood:** The `@app.websocket` decorator stores the handler in `self._websocket_handler`. When a WebSocket connection comes in to `/ws`, Starlette's `WebSocketRoute` calls `_handle_websocket`, which builds a `RequestContext` (extracting session ID from the `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header) and passes it along with the raw Starlette `WebSocket` object to your handler.

### 2.5 LangGraph, CrewAI, Autogen, or Custom Frameworks

The SDK is **framework-agnostic by design**. Here's how you'd use other popular frameworks:

**LangGraph:**

> **Note:** LangGraph is not included in this repository, but the SDK's framework-agnostic design means it works with any Python agent framework. The following is an illustrative example based on LangGraph's public API.

```python
from bedrock_agentcore import BedrockAgentCoreApp
from langgraph.graph import StateGraph
# ... define your LangGraph workflow ...

app = BedrockAgentCoreApp()
graph = build_my_langgraph()  # Your compiled LangGraph

@app.entrypoint
async def invoke(payload):
    result = await graph.ainvoke({"input": payload.get("prompt")})
    return result

app.run()
```

**CrewAI:**

> **Note:** CrewAI is not included in this repository. The following is an illustrative example showing the same framework-agnostic pattern.

```python
from bedrock_agentcore import BedrockAgentCoreApp
from crewai import Crew, Agent, Task

app = BedrockAgentCoreApp()
crew = Crew(agents=[...], tasks=[...])

@app.entrypoint
def invoke(payload):
    result = crew.kickoff(inputs={"query": payload.get("prompt")})
    return {"result": str(result)}

app.run()
```

**Any framework works** because the entrypoint is just a function. As long as you can call your framework from Python and return a JSON-serializable result, it works with the SDK. The serialization layer ([`_safe_serialize_to_json_string`](../src/bedrock_agentcore/runtime/app.py#L541-L567)) has progressive fallbacks that handle Pydantic models (`.model_dump()`), dataclasses (`asdict()`), and arbitrary objects (`str()`).

---

## 3. Agent Powerfulness Dimensions

### 3.1 Simple Chat Agent

A basic chat agent that forwards prompts to an LLM:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()
agent = Agent(system_prompt="You are a helpful assistant.")

@app.entrypoint
async def chat(payload):
    result = agent(payload.get("prompt"))
    return {"response": result.message}

app.run()
```

### 3.2 Using Tools

Tools make agents vastly more powerful by allowing them to take actions in the world. The SDK provides cloud-managed tools plus the ability to define custom tools.

#### Custom Tools with Strands

```python
# Reference: tests_integ/async/interactive_async_strands.py (lines 229-296, 509)
from strands import Agent, tool
from bedrock_agentcore import BedrockAgentCoreApp

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # In production, use a safe evaluator

app = BedrockAgentCoreApp()
agent = Agent(tools=[get_weather, calculate])

@app.entrypoint
def invoke(payload):
    result = agent(payload.get("prompt"))
    return {"response": result.message}

app.run()
```

**How Strands tools work:** Based on the Strands SDK's public documentation, when you decorate a function with `@tool`, Strands converts it into a tool schema that the LLM can call. When the LLM decides to use a tool, Strands intercepts the tool call, executes your function, and passes the result back to the LLM for incorporation into the response. See the [Strands Agents documentation](https://github.com/strands-agents/sdk-python) for details.

#### Cloud Browser Tool

The [`BrowserClient`](../src/bedrock_agentcore/tools/browser_client.py) provides cloud-managed Playwright browser sessions for web automation:

```python
from bedrock_agentcore.tools.browser_client import BrowserClient, browser_session

# Context manager pattern (recommended)
with browser_session('us-west-2') as client:
    # Get WebSocket URL and headers for connecting Playwright
    ws_url, headers = client.generate_ws_headers()
    # Use ws_url with playwright.chromium.connect_over_cdp(ws_url)

    # Or get a live view URL to watch the browser in real-time
    live_view_url = client.generate_live_view_url()
```

**Underlying technology:** The browser service runs Chromium instances in AWS-managed sandboxes. Communication happens over WebSocket, authenticated via SigV4. The SDK generates signed URLs for both automation (CDP protocol) and live viewing. The browser supports Web Bot Auth (`browserSigning`) for cryptographic identity that reduces CAPTCHA friction.

#### Cloud Code Interpreter

The [`CodeInterpreter`](../src/bedrock_agentcore/tools/code_interpreter_client.py) provides sandboxed Python/shell execution:

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session

with code_session('us-west-2') as client:
    # Execute code in sandboxed environment
    result = client.execute_code("print('Hello from sandbox!')")

    # Install packages
    client.install_packages(['pandas', 'matplotlib'])

    # Upload files for processing
    client.upload_file('data.csv', csv_content, description='Sales data')

    # Execute code that uses the uploaded data
    result = client.execute_code("""
        import pandas as pd
        df = pd.read_csv('data.csv')
        print(df.describe())
    """)
```

**Underlying technology:** Each code interpreter session runs in an isolated AWS sandbox with its own filesystem, networking (configurable — PUBLIC or VPC), and Python runtime. Like the browser, it uses control plane APIs for lifecycle management and data plane APIs for execution.

### 3.3 Persistent Memory (Short-Term and Long-Term)

The memory system is one of the most sophisticated parts of the SDK. It provides **persistent knowledge across sessions** with both conversational (short-term) and semantic (long-term) memory.

#### Architecture

The memory system has a hierarchical structure:
- **Memory** → Top-level container (has an ID, strategies, and configuration)
- **Actor** → Represents a user or entity
- **Session** → A conversation context within an actor
- **Events** → Individual conversation turns within a session
- **Branches** → Alternative conversation paths (for A/B testing)

There are two client interfaces:
1. **`MemorySessionManager` + `MemorySession`** (recommended) — Session-scoped, clean API
2. **`MemoryClient`** (legacy) — Lower-level, requires passing IDs everywhere

#### Short-Term Memory (Conversation History)

```python
# Reference: src/bedrock_agentcore/memory/README.md (Quick Start)
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

manager = MemorySessionManager(
    memory_id="your-memory-id",
    region_name="us-east-1"
)

session = manager.create_memory_session(
    actor_id="user-123",
    session_id="session-456"
)

# Add conversation turns — stored as events in the Memory service
session.add_turns([
    ConversationalMessage("I love eating apples and cherries", MessageRole.USER),
    ConversationalMessage("Apples are very good for you!", MessageRole.ASSISTANT),
])

# Retrieve recent conversation
turns = session.get_last_k_turns(k=10)
```

**How it works under the hood:** `add_turns()` sends a `create_event` API call to the AgentCore data plane. Each event contains a list of messages (USER, ASSISTANT, TOOL roles), metadata, and timestamps. The `get_last_k_turns()` method uses `list_events` with pagination to retrieve recent conversation.

#### Long-Term Memory (Semantic Extraction)

Long-term memory uses **strategies** that automatically extract and store knowledge from conversations:

```python
# Reference: src/bedrock_agentcore/memory/integrations/strands/README.md
from bedrock_agentcore.memory import MemoryClient

client = MemoryClient(region_name="us-east-1")

# Create memory with strategies
memory = client.create_memory_and_wait(
    name="SmartAgentMemory",
    description="Memory with automatic knowledge extraction",
    strategies=[
        {
            "summaryMemoryStrategy": {
                "name": "SessionSummarizer",
                "namespaces": ["/summaries/{actorId}/{sessionId}/"]
            }
        },
        {
            "userPreferenceMemoryStrategy": {
                "name": "PreferenceLearner",
                "namespaces": ["/preferences/{actorId}/"]
            }
        },
        {
            "semanticMemoryStrategy": {
                "name": "FactExtractor",
                "namespaces": ["/facts/{actorId}/"]
            }
        }
    ]
)
```

**How strategies work:** Based on the API patterns observed in the SDK and the [official memory strategies documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-strategies.html), when events are created in the memory, AgentCore's backend asynchronously processes them through configured strategies. These strategies use LLMs to extract structured knowledge:
- **`summaryMemoryStrategy`** — Summarizes conversation sessions
- **`userPreferenceMemoryStrategy`** — Learns user preferences ("likes sushi", "prefers dark mode")
- **`semanticMemoryStrategy`** — Extracts factual information ("works at Company X", "has a dog named Max")

Extracted knowledge is stored as **memory records** in namespaces (like folders). You can then search them semantically:

```python
# Search long-term memories
memories = session.search_long_term_memories(
    query="what food does the user like",
    namespace_prefix="/preferences/user-123/",
    top_k=5
)
```

#### Strands Memory Integration

The deepest integration is with Strands via [`AgentCoreMemorySessionManager`](../src/bedrock_agentcore/memory/integrations/strands/session_manager.py), which acts as a Strands `SessionManager`:

```python
# Reference: src/bedrock_agentcore/memory/integrations/strands/README.md
from strands import Agent
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig, RetrievalConfig
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)

config = AgentCoreMemoryConfig(
    memory_id="your-memory-id",
    session_id="session-456",
    actor_id="user-123",
    retrieval_config={
        "/preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.7),
        "/facts/{actorId}/": RetrievalConfig(top_k=10, relevance_score=0.3),
    }
)

session_manager = AgentCoreMemorySessionManager(
    agentcore_memory_config=config,
    region_name="us-east-1"
)

# Agent automatically uses AgentCore Memory for persistence
agent = Agent(
    system_prompt="You are a helpful assistant with memory.",
    session_manager=session_manager,
)

# Conversations are automatically persisted and memories retrieved
agent("I like sushi with tuna")
agent("What should I eat for lunch?")  # Remembers preferences
```

**How the Strands integration works:** `AgentCoreMemorySessionManager` implements Strands' `RepositorySessionManager` and `SessionRepository` interfaces. When Strands saves a message, the session manager:
1. Converts Strands `Message` objects to AgentCore Memory format (via [`AgentCoreMemoryConverter`](../src/bedrock_agentcore/memory/integrations/strands/bedrock_converter.py))
2. Sends them as events to the AgentCore Memory data plane
3. On session load, retrieves events from memory and reconstructs the conversation
4. Retrieves long-term memories from configured namespaces and injects them into the agent's context

#### LLM-Integrated Memory Turns

For direct LLM integration without Strands:

```python
# Reference: src/bedrock_agentcore/memory/README.md (Enhanced LLM Integration)
def my_llm(user_input: str, memories: list) -> str:
    context = "\n".join([m.get('content', {}).get('text', '') for m in memories])
    # Call your LLM with context
    return f"Based on {context}, here's my response to: {user_input}"

retrieval_config = {
    "support/facts/{sessionId}/": RetrievalConfig(top_k=5, relevance_score=0.3),
    "user/preferences/{actorId}/": RetrievalConfig(top_k=3, relevance_score=0.5)
}

memories, response, event = session.process_turn_with_llm(
    user_input="What did we discuss about my preferences?",
    llm_callback=my_llm,
    retrieval_config=retrieval_config
)
```

### 3.4 Identity and Authentication (OAuth2 & API Keys)

The identity system ([`src/bedrock_agentcore/identity/auth.py`](../src/bedrock_agentcore/identity/auth.py)) provides decorators for securely accessing external services using OAuth2 or API keys — without embedding credentials in your code.

#### OAuth2 Access Tokens

```python
# Reference: src/bedrock_agentcore/identity/auth.py (lines 22-101)
from bedrock_agentcore.identity import requires_access_token

@requires_access_token(
    provider_name="my-github-provider",
    scopes=["repo", "user"],
    auth_flow="USER_FEDERATION",  # or "M2M" for machine-to-machine
)
def call_github_api(query: str, *, access_token: str) -> str:
    """The access_token is automatically injected by the decorator."""
    import requests
    response = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return response.json()
```

**How it works under the hood:**
1. The decorator creates an `IdentityClient` ([`src/bedrock_agentcore/services/identity.py`](../src/bedrock_agentcore/services/identity.py))
2. It retrieves the workload access token from `BedrockAgentCoreContext` (set by the runtime during invocation from the `WorkloadAccessToken` header)
3. Calls the AgentCore data plane `get_resource_oauth2_token` API
4. If the token exists in the token vault, it's returned directly
5. If user consent is needed (3-legged OAuth), an authorization URL is returned, and the SDK polls until the user completes authorization
6. The token is injected into your function as a keyword argument

#### API Keys

```python
from bedrock_agentcore.identity import requires_api_key

@requires_api_key(provider_name="my-openai-provider")
def call_openai(prompt: str, *, api_key: str) -> str:
    """API key injected automatically."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    return client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
```

#### AWS IAM JWT Tokens

For M2M auth with external services that accept AWS-signed JWTs:

```python
# Reference: src/bedrock_agentcore/identity/auth.py (lines 104-217)
from bedrock_agentcore.identity import requires_iam_access_token

@requires_iam_access_token(
    audience=["https://api.example.com"],
    signing_algorithm="ES384",
    duration_seconds=300,
)
def call_external_api(query: str, *, access_token: str) -> str:
    import requests
    response = requests.get(
        "https://api.example.com/data",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"q": query},
    )
    return response.text
```

**How it works:** This decorator calls AWS STS `GetWebIdentityToken` to get an AWS-signed JWT. No client secrets needed — the token is signed by AWS. The external service validates the JWT by trusting the AWS account's OIDC issuer URL.

### 3.5 Cloud-Based Tools: Browser and Code Interpreter

See [Section 3.2](#32-using-tools) for code examples. Key architectural points:

**Browser Tool** ([`src/bedrock_agentcore/tools/browser_client.py`](../src/bedrock_agentcore/tools/browser_client.py)):
- Uses control plane for CRUD operations on browsers
- Uses data plane for session management
- Provides `generate_ws_headers()` for SigV4-authenticated Playwright CDP connections
- Provides `generate_live_view_url()` for presigned viewing URLs
- Supports VPC networking, session recording to S3, and Web Bot Auth (browser signing)
- Has `take_control()` and `release_control()` for switching between automated and manual mode

**Code Interpreter** ([`src/bedrock_agentcore/tools/code_interpreter_client.py`](../src/bedrock_agentcore/tools/code_interpreter_client.py)):
- Similar control/data plane split
- Provides `execute_code()`, `install_packages()`, `upload_file()`, `download_file()`
- Each session has isolated filesystem and Python runtime
- Supports both PUBLIC and VPC network configurations

Both tools follow the same **configuration pattern** using dataclasses in [`src/bedrock_agentcore/tools/config.py`](../src/bedrock_agentcore/tools/config.py) with convenient factory methods:

```python
from bedrock_agentcore.tools.config import (
    create_browser_config,
    NetworkConfiguration,
    ViewportConfiguration
)

config = create_browser_config(
    name="my_browser",
    execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
    enable_web_bot_auth=True,
    enable_recording=True,
    recording_bucket="my-recordings",
)
```

### 3.6 Async Task Management and Health Tracking

For long-running background tasks, the SDK provides async task management that integrates with the health check system:

```python
# Reference: tests_integ/async/README.md and async_status_example.py

# Method 1: Decorator-based (automatic)
@app.async_task
async def background_work():
    await asyncio.sleep(600)  # Ping status → "HealthyBusy" while running
    return "done"             # Ping status → "Healthy" when complete

@app.entrypoint
async def handler(event):
    asyncio.create_task(background_work())
    return {"status": "started"}

# Method 2: Manual tracking (fine-grained control)
@app.entrypoint
async def handler(event):
    task_id = app.add_async_task("data_processing", {"batch": 100})
    # ... do work ...
    app.complete_async_task(task_id)
    return {"status": "completed"}

# Method 3: Custom ping handler
@app.ping
def custom_status():
    if system_busy():
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY
```

**How it works:** The `@app.async_task` decorator wraps your async function to call `add_async_task()` before execution and `complete_async_task()` after. The `_active_tasks` dictionary tracks all running tasks with thread-safe locking. When `/ping` is called, `get_current_ping_status()` checks if any tasks are active and returns `HEALTHY_BUSY` if so.

---

## 4. Testing Agents

### 4.1 Unit Testing

The repository uses pytest with `pytest-asyncio` for async support. Tests are in `tests/` and mirror the source structure:

```bash
# Run unit tests
uv run pytest tests/ --cov=src --cov-report=xml

# Run specific test file
uv run pytest tests/bedrock_agentcore/runtime/test_app.py -v
```

Example of testing an agent entrypoint:

```python
import pytest
from starlette.testclient import TestClient
from bedrock_agentcore import BedrockAgentCoreApp

@pytest.fixture
def app():
    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def handler(payload):
        return {"echo": payload.get("message")}

    return app

def test_invocation(app):
    client = TestClient(app)
    response = client.post("/invocations", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json()["echo"] == "hello"

def test_ping(app):
    client = TestClient(app)
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"
```

### 4.2 Integration Testing

Integration tests live in `tests_integ/` and test against running agent servers:

```python
# Reference: tests_integ/runtime/test_simple_agent.py
# These tests:
# 1. Write an agent script to a temp file
# 2. Start it as a subprocess
# 3. Make HTTP requests to test behavior
# 4. Assert responses

from tests_integ.runtime.base_test import BaseSDKRuntimeTest, start_agent_server
from tests_integ.runtime.http_client import HttpClient

class TestSDKSimpleAgent(BaseSDKRuntimeTest):
    def setup(self):
        # Write agent code to file
        self.agent_module = "agent"
        with open(self.agent_module + ".py", "w") as file:
            file.write("""...""")

    def run_test(self):
        with start_agent_server(self.agent_module):
            client = HttpClient("http://127.0.0.1:8080")
            response = client.invoke_endpoint("tell me a joke")
            assert "joke" in response.lower()
```

### 4.3 Agent Evaluation

The evaluation system ([`src/bedrock_agentcore/evaluation/`](../src/bedrock_agentcore/evaluation/)) integrates with the Strands Evals framework to evaluate agent quality:

```python
# Reference: src/bedrock_agentcore/evaluation/integrations/strands_agents_evals/README.md
from strands import Agent, tool
from strands_evals import Experiment, Case
from strands_evals.telemetry import StrandsEvalsTelemetry
from bedrock_agentcore.evaluation import create_strands_evaluator

# Setup
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
agent = Agent(tools=[calculator], system_prompt="You are a math assistant.")

# Task function — runs agent and captures telemetry
def task_fn(case):
    result = agent(case.input)
    raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
    return {"output": str(result), "trajectory": raw_spans}

# Evaluate
cases = [
    Case(input="What is 5 + 3?", expected_output="8"),
    Case(input="Calculate 10 + 7", expected_output="17"),
]
evaluator = create_strands_evaluator("Builtin.Helpfulness")
experiment = Experiment(cases=cases, evaluators=[evaluator])
reports = experiment.run_evaluations(task_fn)
print(f"Overall score: {reports[0].overall_score:.2f}")
```

**Built-in evaluators:**
- `Builtin.Helpfulness` — How helpful is the response?
- `Builtin.Accuracy` — Factual accuracy
- `Builtin.Harmfulness` — Detects harmful content
- `Builtin.Relevance` — Response relevance to query

**How it works under the hood:** The `StrandsEvalsAgentCoreEvaluator` ([`evaluator.py`](../src/bedrock_agentcore/evaluation/integrations/strands_agents_evals/evaluator.py)) takes raw OpenTelemetry spans from Strands, converts them to ADOT (AWS Distro for OpenTelemetry) format using [`convert_strands_to_adot`](../src/bedrock_agentcore/evaluation/span_to_adot_serializer/), and sends them to the AgentCore Evaluation API. The API analyzes the agent's trace (LLM calls, tool usage, reasoning) alongside the output to compute quality scores.

---

## 5. Deploying Agents

### 5.1 Local Development and Running

```bash
# Install the SDK
pip install bedrock-agentcore

# For Strands integration
pip install 'bedrock-agentcore[strands-agents]'

# For evaluation
pip install 'bedrock-agentcore[strands-agents-evals]'

# Run your agent locally
python my_agent.py
# Server starts on http://127.0.0.1:8080

# Test it
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello!"}'
```

**Auto-detection:** The SDK detects whether it's running in Docker and adjusts the bind address accordingly:
```python
# From src/bedrock_agentcore/runtime/app.py, lines 449-453
if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
    host = "0.0.0.0"  # Docker needs this to expose the port
else:
    host = "127.0.0.1"
```

### 5.2 Containerization

To deploy to AgentCore Runtime, your agent must be containerized:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# AgentCore expects port 8080
EXPOSE 8080

CMD ["python", "agent.py"]
```

`requirements.txt`:
```
bedrock-agentcore
strands-agents  # if using Strands
```

### 5.3 Deploying to Bedrock AgentCore Runtime

**Using the Starter Toolkit** (recommended for getting started):

The [Bedrock AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit) automates the deployment process.

**Using AWS CDK** (production):

The [AWS CDK for AgentCore](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_bedrockagentcore-readme.html) provides infrastructure-as-code for production deployments.

**What happens during deployment (based on SDK patterns and [official documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-toolkit.html)):**

1. Your container image is pushed to ECR
2. AgentCore provisions compute infrastructure
3. Your container is started — `app.run()` starts the Starlette/Uvicorn server on port 8080
4. AgentCore begins health-checking via `GET /ping`
5. Once healthy, the runtime is ready to receive invocations
6. AgentCore handles scaling, load balancing, and routing based on ping status

---

## 6. Invoking Deployed Agents

### 6.1 HTTP Invocation

Once deployed, the simplest invocation is via the AWS CLI or boto3:

```python
import boto3
import json

client = boto3.client("bedrock-agentcore", region_name="us-west-2")

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    payload=json.dumps({"prompt": "Tell me a joke"}).encode()
)

result = json.loads(response["payload"].read().decode("utf-8"))
print(result)
```

### 6.2 WebSocket Invocation (SigV4 Auth)

For streaming/bidirectional communication, use the [`AgentCoreRuntimeClient`](../src/bedrock_agentcore/runtime/agent_core_runtime_client.py):

```python
# Reference: docs/examples/agent_runtime_client_examples.md
import asyncio
import websockets
from bedrock_agentcore.runtime import AgentCoreRuntimeClient

async def main():
    client = AgentCoreRuntimeClient(region="us-west-2")

    # Generate SigV4-signed WebSocket URL and headers
    ws_url, headers = client.generate_ws_connection(
        runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
        session_id="my-session-id",
        endpoint_name="DEFAULT"
    )

    async with websockets.connect(ws_url, extra_headers=headers) as ws:
        await ws.send('{"prompt": "Hello!"}')
        response = await ws.recv()
        print(f"Received: {response}")

asyncio.run(main())
```

**How SigV4 signing works:** The client converts the `wss://` URL to `https://` for signing, creates an `AWSRequest`, signs it with `SigV4Auth` using your AWS credentials, and extracts the `Authorization` and `X-Amz-Date` headers. These are passed as WebSocket connection headers for authentication.

### 6.3 WebSocket with Presigned URLs (Frontend)

For browser/frontend clients that can't use AWS credentials:

```python
# Backend generates presigned URL
client = AgentCoreRuntimeClient(region="us-west-2")

presigned_url = client.generate_presigned_url(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    endpoint_name="DEFAULT",
    custom_headers={"user_type": "premium"},
    expires=300  # 5 minutes
)

# Frontend JavaScript:
# const ws = new WebSocket(presigned_url);
```

**How presigned URLs work:** Instead of `SigV4Auth` (header-based), this uses `SigV4QueryAuth` which embeds the signature into query parameters (`X-Amz-Algorithm`, `X-Amz-Credential`, `X-Amz-Signature`, etc.). The session ID is also embedded as a query parameter. The URL is valid for `expires` seconds (max 300).

### 6.4 OAuth WebSocket Invocation

For OAuth-based authentication:

```python
# Reference: docs/examples/agent_runtime_client_examples.md (OAuth section)
client = AgentCoreRuntimeClient(region="us-west-2")

ws_url, headers = client.generate_ws_connection_oauth(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    bearer_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    endpoint_name="DEFAULT"
)

async with websockets.connect(ws_url, extra_headers=headers) as ws:
    await ws.send('{"prompt": "Hello!"}')
```

---

## 7. Observability and Monitoring

AgentCore provides built-in observability through OpenTelemetry:

- **Structured logging** — The SDK uses a custom `RequestContextFormatter` ([`app.py` lines 42-73](../src/bedrock_agentcore/runtime/app.py#L42-L73)) that outputs JSON logs with request ID, session ID, timestamps, and error details — optimized for CloudWatch Logs
- **ADOT tracing** — OpenTelemetry spans are exported in AWS Distro for OpenTelemetry format. The [`span_to_adot_serializer`](../src/bedrock_agentcore/evaluation/span_to_adot_serializer/) module handles conversion
- **CloudWatch integration** — Spans flow to CloudWatch via ADOT, where they can be queried using [`fetch_spans_from_cloudwatch`](../src/bedrock_agentcore/evaluation/utils/cloudwatch_span_helper.py)

**ADOT integration:** Based on the SDK's `span_to_adot_serializer` module and CloudWatch integration code, when running in AgentCore Runtime, an ADOT (AWS Distro for OpenTelemetry) collector is configured alongside the container, automatically collecting OpenTelemetry spans from instrumented frameworks (like Strands) and exporting them to CloudWatch. The evaluation system then queries these spans via the `fetch_spans_from_cloudwatch` utility. For more details, see the [AgentCore Observability documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html).

---

## 8. Security Considerations

- **Context isolation** — Each request gets its own `contextvars` context, preventing data leakage between concurrent requests
- **Authentication** — SigV4 for AWS-native auth, OAuth2 for external services, presigned URLs for frontend clients
- **Network isolation** — Tools (browser, code interpreter) support VPC networking for private access
- **Credential management** — No secrets in code; credentials flow through the AgentCore Identity service and are injected at runtime via decorators
- **Docker awareness** — The SDK detects Docker environments and adjusts binding (`0.0.0.0` vs `127.0.0.1`)
- **Input validation** — The SDK validates ARN formats, expiry timeouts, network configurations, etc.

---

## Summary: End-to-End Agent Lifecycle

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│  1. WRITE    │    │  2. TEST     │    │  3. DEPLOY    │    │  4. INVOKE     │
│              │    │              │    │               │    │                │
│ - Choose     │    │ - Unit tests │    │ - Containerize│    │ - HTTP POST    │
│   framework  │───▶│ - Integration│───▶│ - Push to ECR │───▶│ - WebSocket    │
│ - Define     │    │   tests      │    │ - Deploy to   │    │ - Presigned URL│
│   entrypoint │    │ - Evaluation │    │   AgentCore   │    │ - OAuth        │
│ - Add tools  │    │   (quality)  │    │   Runtime     │    │                │
│ - Add memory │    │              │    │               │    │                │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │  AgentCore Handles │
                                    │  - Scaling         │
                                    │  - Health checks   │
                                    │  - Auth routing    │
                                    │  - Observability   │
                                    └───────────────────┘
```

**Key files to explore:**
- [Runtime app](../src/bedrock_agentcore/runtime/app.py) — Core application server
- [Memory README](../src/bedrock_agentcore/memory/README.md) — Memory system documentation
- [Strands memory integration](../src/bedrock_agentcore/memory/integrations/strands/README.md) — Strands + AgentCore Memory
- [Identity auth](../src/bedrock_agentcore/identity/auth.py) — OAuth2, API key, and IAM JWT decorators
- [Browser client](../src/bedrock_agentcore/tools/browser_client.py) — Cloud browser automation
- [Code interpreter](../src/bedrock_agentcore/tools/code_interpreter_client.py) — Sandboxed code execution
- [Evaluation README](../src/bedrock_agentcore/evaluation/integrations/strands_agents_evals/README.md) — Agent quality evaluation
- [Async examples](../tests_integ/async/) — Background task management
- [WebSocket client examples](../docs/examples/agent_runtime_client_examples.md) — Invocation patterns
