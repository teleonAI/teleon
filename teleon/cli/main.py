"""
Teleon CLI - Command Line Interface

Main entry point for the `teleon` command.
"""

import os
import typer
from rich.console import Console
from rich.panel import Panel
from typing import Optional

from teleon.__version__ import __version__
from teleon.cli.commands.dev import dev as dev_cmd
from teleon.cli.commands.exec import exec_app
from teleon.cli.commands.deploy import app as deploy_app
from teleon.cli.commands.push import app as push_app
from teleon.cli.commands.agents import app as agents_app
from teleon.cli.commands.helix import app as helix_app
from teleon.cli.commands.cortex import app as cortex_app
from teleon.cli.commands.sentinel import app as sentinel_app
from teleon.cli.commands.auth import app as auth_app

# Initialize Typer app
app = typer.Typer(
    name="teleon",
    help="The Platform for Intelligent Agents - Deploy production-ready AI agents in minutes",
    add_completion=False,
)

# Add subcommands
app.add_typer(auth_app, name="auth")
app.add_typer(dev_cmd, name="dev")
app.add_typer(exec_app, name="exec")
app.add_typer(deploy_app, name="deploy")
app.add_typer(push_app, name="push")
app.add_typer(agents_app, name="agents")
app.add_typer(helix_app, name="helix")
app.add_typer(cortex_app, name="cortex")
app.add_typer(sentinel_app, name="sentinel")

# Add login as a top-level command (alias to auth login)
@app.command()
def login(api_key: str = typer.Option(None, "--api-key", help="Your Teleon API key")):
    """Authenticate with Teleon Platform (alias for 'teleon auth login')"""
    from teleon.cli.commands.auth import login as auth_login_cmd
    auth_login_cmd(api_key=api_key)

console = Console()


@app.command()
def version():
    """Show Teleon version."""
    console.print(f"[bold green]Teleon[/bold green] version [cyan]{__version__}[/cyan]")


TEMPLATES = {
    "customer-support": {
        "name": "Customer Support Bot",
        "description": "AI support agent with conversation memory and semantic search",
        "features": ["Cortex memory", "Semantic search", "Multi-tenant isolation"],
        "default_project_name": "support-agent",
    },
    "data-pipeline": {
        "name": "Data Analysis Pipeline",
        "description": "Data processing agent with custom tools and parallel execution",
        "features": ["Custom @tool decorators", "Parallel processing", "CSV analysis"],
        "default_project_name": "data-pipeline",
    },
    "code-review": {
        "name": "Code Review Agent",
        "description": "Code analysis agent using LLM gateway with structured output",
        "features": ["LLM Gateway", "JSON structured output", "Multi-language"],
        "default_project_name": "code-review-agent",
    },
    "content-generator": {
        "name": "Content Generator",
        "description": "Content creation agent with tone control and auto-scaling",
        "features": ["LLM Gateway", "Tone/style config", "Auto-scaling"],
        "default_project_name": "content-generator",
    },
    "research-system": {
        "name": "Multi-Agent Research System",
        "description": "Coordinated multi-agent system with task delegation",
        "features": ["Multi-agent collaboration", "Task delegation", "Orchestration"],
        "default_project_name": "research-system",
    },
}


def _get_agent_content(template_key: str) -> str:
    """Return the agents/main.py content for the given template."""
    if template_key == "customer-support":
        return '''"""Customer Support Agent with Cortex Memory

AI support agent with conversation memory and semantic search.
Uses Cortex for persistent memory scoped per customer.
"""

import os
from dotenv import load_dotenv
from teleon import TeleonClient

load_dotenv()

client = TeleonClient(
    api_key=os.getenv("TELEON_API_KEY", "tlk_test_dev"),
    environment=os.getenv("TELEON_ENV", "dev"),
    verify_key=False,
)


@client.agent(
    name="support-agent",
    description="Customer support agent with persistent memory",
    model="gpt-4",
    temperature=0.7,
    cortex={
        "auto": True,
        "scope": ["customer_id"],
        "auto_context": {
            "enabled": True,
            "history_limit": 10,
            "relevant_limit": 5,
            "max_tokens": 2000,
        },
    },
)
async def support_agent(query: str, customer_id: str = "default", cortex=None) -> dict:
    """
    Handle customer support queries with memory.

    Args:
        query: Customer\'s question or message
        customer_id: Unique customer identifier
        cortex: Injected Memory instance (auto-provided by Teleon)

    Returns:
        dict with response and metadata
    """
    past_resolutions = []
    recent_history = []
    if cortex is not None:
        past_resolutions = await cortex.search(
            query=query,
            filter={"type": "resolution"},
            limit=3,
        )

        recent_history = await cortex.get(
            filter={"type": "conversation"},
            limit=5,
        )

    response_parts = []

    # Check if this is a known issue
    if past_resolutions:
        best_match = past_resolutions[0]
        if best_match.score and best_match.score > 0.85:
            response_parts.append(
                f"I found a similar issue we resolved before. "
                f"Here\'s what helped: {best_match.content}"
            )

    # Generate response based on query type
    query_lower = query.lower()

    if any(word in query_lower for word in ["refund", "money back", "return"]):
        response_parts.append(
            "For refunds, our policy allows returns within 30 days of purchase. "
            "I can help process your refund request. Could you provide your order number?"
        )
        if cortex is not None:
            await cortex.store(content=f"Customer inquired about refund: {query}", type="query", topic="refund")
    elif any(word in query_lower for word in ["password", "login", "access", "account"]):
        response_parts.append(
            "For account access issues, you can reset your password at "
            "Settings > Security > Reset Password."
        )
        if cortex is not None:
            await cortex.store(content=f"Customer had account issue: {query}", type="query", topic="account")
    else:
        response_parts.append(
            f"I\'d be happy to help you with that. Let me look into \'{query}\' for you."
        )
        if cortex is not None:
            await cortex.store(content=f"General inquiry: {query}", type="query", topic="general")

    response = " ".join(response_parts)

    # Store conversation turn
    total_interactions = 0
    if cortex is not None:
        await cortex.store(content=f"Q: {query}\\nA: {response}", type="conversation")
        total_interactions = await cortex.count()

    return {
        "response": response,
        "customer_id": customer_id,
        "total_interactions": total_interactions,
        "similar_issues_found": len(past_resolutions),
    }


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await support_agent(
            query="How do I get a refund for my order?",
            customer_id="cust_123",
        )
        print(f"Response: {result[\'response\']}")
        print(f"Interactions: {result[\'total_interactions\']}")

    asyncio.run(main())
'''

    elif template_key == "data-pipeline":
        return '''"""Data Analysis Pipeline

Data processing agent with custom tools and parallel execution.
Demonstrates @tool decorators, async gathering, and structured output.
"""

import asyncio
import csv
import io
from typing import Any

from teleon.decorators.agent import agent
from teleon.decorators.tool import tool


@tool(name="parse-csv", description="Parse CSV data into records", category="data")
async def parse_csv(csv_text: str) -> list[dict[str, str]]:
    """Parse raw CSV text into a list of dictionaries."""
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


@tool(name="compute-stats", description="Compute basic statistics for a numeric column", category="data")
async def compute_stats(values: list[float]) -> dict[str, float]:
    """Return min, max, mean, and count for a list of numbers."""
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


@tool(name="filter-records", description="Filter records by a field value", category="data")
async def filter_records(
    records: list[dict[str, Any]], field: str, value: str
) -> list[dict[str, Any]]:
    """Return records where field matches value."""
    return [r for r in records if r.get(field) == value]


@agent(
    name="data-pipeline",
    memory=True,
    tools=[parse_csv, compute_stats, filter_records],
)
async def data_pipeline(csv_text: str, numeric_column: str) -> dict:
    """
    Analyze CSV data: parse, compute stats, and return a summary.

    Args:
        csv_text: Raw CSV content
        numeric_column: Name of the column to compute statistics on

    Returns:
        dict with parsed records, statistics, and row count
    """
    # Parse CSV and compute stats in parallel
    records_task = parse_csv(csv_text)
    records = await records_task

    # Extract numeric values
    values = []
    for record in records:
        try:
            values.append(float(record[numeric_column]))
        except (KeyError, ValueError):
            continue

    stats = await compute_stats(values)

    return {
        "row_count": len(records),
        "columns": list(records[0].keys()) if records else [],
        "stats": stats,
        "sample": records[:3],
    }


if __name__ == "__main__":

    async def main():
        sample_csv = """name,department,salary
Alice,Engineering,95000
Bob,Marketing,72000
Carol,Engineering,105000
Dave,Marketing,68000
Eve,Engineering,110000"""

        result = await data_pipeline(csv_text=sample_csv, numeric_column="salary")
        print(f"Rows: {result[\'row_count\']}")
        print(f"Columns: {result[\'columns\']}")
        print(f"Stats: {result[\'stats\']}")

    asyncio.run(main())
'''

    elif template_key == "code-review":
        return '''"""Code Review Agent

Code analysis agent using the LLM Gateway with structured JSON output.
Demonstrates LLMGateway, LLMConfig, and LLMMessage usage.
"""

import os
import json
from typing import Dict, Any

from dotenv import load_dotenv

from teleon.client import TeleonClient
from teleon.llm.gateway import LLMGateway
from teleon.llm.types import LLMConfig, LLMMessage

load_dotenv()

client = TeleonClient(
    api_key=os.getenv("TELEON_API_KEY", "tlk_test_dev"),
    environment=os.getenv("TELEON_ENV", "dev"),
    verify_key=False,
)

# Initialize LLM Gateway
gateway = LLMGateway()


@client.agent(
    name="code-reviewer",
    description="Code review agent that provides structured feedback",
    model="gpt-4",
    temperature=0.3,
    max_tokens=1500,
)
async def code_reviewer(
    code: str, language: str = "python", focus_areas: str = "all"
) -> Dict[str, Any]:
    """
    Analyze code and return a structured review.

    Args:
        code: Source code to review
        language: Programming language (default: python)
        focus_areas: Comma-separated areas — security, performance, style, bugs, all

    Returns:
        dict with overall_score, summary, issues, suggestions, etc.
    """
    system_prompt = (
        "You are an expert code reviewer. Review the code and return a JSON object with: "
        "overall_score (0-100), summary, strengths (list), "
        "issues (list of {severity, line, issue, suggestion}), "
        "suggestions (list), security_concerns (list), performance_tips (list)."
    )

    focus_instruction = ""
    if focus_areas and focus_areas != "all":
        focus_instruction = f"\\nFocus on: {focus_areas}"

    user_prompt = (
        f"Review this {language} code:\\n\\n```{language}\\n{code}\\n```\\n"
        f"Provide a comprehensive code review.{focus_instruction}\\n"
        "Return your review as valid JSON only."
    )

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]

    config = LLMConfig(
        model=os.getenv("TELEON_LLM_MODEL", "gpt-4"),
        temperature=0.3,
        max_tokens=1500,
    )

    response = await gateway.complete(messages, config)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "overall_score": 0,
            "summary": "Failed to parse review — raw response attached.",
            "strengths": [],
            "issues": [],
            "suggestions": [],
            "security_concerns": [],
            "performance_tips": [],
            "raw_response": response.content,
        }


if __name__ == "__main__":
    import asyncio

    async def main():
        sample_code = """
def process_user(user_id, data):
    db.execute(f"INSERT INTO users VALUES ({user_id}, \'{data}\')")
    return True
"""
        review = await code_reviewer(code=sample_code, focus_areas="security,best_practices")
        print(f"Score: {review.get(\'overall_score\', \'N/A\')}/100")
        print(f"Summary: {review.get(\'summary\', \'\')}")
        for issue in review.get("issues", []):
            print(f"  [{issue.get(\'severity\', \'?\')}] {issue.get(\'issue\', \'\')}")

    asyncio.run(main())
'''

    elif template_key == "content-generator":
        return '''"""Content Generator Agent

Content creation agent with tone/style control and auto-scaling.
Demonstrates LLM Gateway usage with configurable system prompts.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv

from teleon.decorators.agent import agent
from teleon.llm.gateway import LLMGateway
from teleon.llm.types import LLMConfig, LLMMessage

load_dotenv()

gateway = LLMGateway()

TONE_PROMPTS = {
    "professional": "Write in a professional, authoritative tone suitable for business audiences.",
    "casual": "Write in a friendly, conversational tone suitable for blog posts and social media.",
    "technical": "Write in a precise, technical tone suitable for developer documentation.",
    "persuasive": "Write in a compelling, persuasive tone suitable for marketing copy.",
}


@agent(
    name="content-generator",
    memory=True,
    scale={"min": 1, "max": 20, "target_cpu": 70},
)
async def content_generator(
    topic: str,
    content_type: str = "blog_post",
    tone: str = "professional",
    max_words: int = 500,
) -> Dict[str, Any]:
    """
    Generate content on a given topic.

    Args:
        topic: Subject to write about
        content_type: One of blog_post, social_media, email, documentation
        tone: One of professional, casual, technical, persuasive
        max_words: Approximate word limit

    Returns:
        dict with title, content, word_count, and metadata
    """
    tone_instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["professional"])

    system_prompt = (
        f"You are an expert content writer. {tone_instruction} "
        f"Generate a {content_type.replace('_', ' ')} of roughly {max_words} words."
    )

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=f"Write about: {topic}"),
    ]

    config = LLMConfig(
        model=os.getenv("TELEON_LLM_MODEL", "gpt-4"),
        temperature=0.8,
        max_tokens=max_words * 2,
    )

    response = await gateway.complete(messages, config)

    content = response.content
    word_count = len(content.split())

    # Extract a title from the first line if present
    lines = content.strip().splitlines()
    title = lines[0].strip("# ").strip() if lines else topic

    return {
        "title": title,
        "content": content,
        "word_count": word_count,
        "tone": tone,
        "content_type": content_type,
        "model": response.model,
    }


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await content_generator(
            topic="The future of AI agents in enterprise software",
            content_type="blog_post",
            tone="professional",
            max_words=300,
        )
        print(f"Title: {result['title']}")
        print(f"Words: {result['word_count']}")
        print(f"\n{result['content'][:500]}...")

    asyncio.run(main())
'''

    elif template_key == "research-system":
        return '''"""Multi-Agent Research System

Coordinated multi-agent system with task delegation.
Demonstrates defining multiple agents and orchestrating them together.
"""

import os
import asyncio
from typing import Dict, Any, List

from dotenv import load_dotenv

from teleon import TeleonClient
from teleon.llm.gateway import LLMGateway
from teleon.llm.types import LLMConfig, LLMMessage

load_dotenv()

client = TeleonClient(
    api_key=os.getenv("TELEON_API_KEY", "tlk_test_dev"),
    environment=os.getenv("TELEON_ENV", "dev"),
    verify_key=False,
)

gateway = LLMGateway()


@client.agent(
    name="research-collector",
    description="Collects raw information on a topic",
    model="gpt-4",
    temperature=0.5,
)
async def collector(topic: str, num_points: int = 5) -> Dict[str, Any]:
    """Gather key facts and data points about a topic."""
    messages = [
        LLMMessage(
            role="system",
            content="You are a research assistant. Return a JSON object with a \'points\' list of factual findings.",
        ),
        LLMMessage(role="user", content=f"Collect {num_points} key facts about: {topic}"),
    ]
    config = LLMConfig(model=os.getenv("TELEON_LLM_MODEL", "gpt-4"), temperature=0.5, max_tokens=800)
    response = await gateway.complete(messages, config)

    import json
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"points": [response.content]}


@client.agent(
    name="research-analyzer",
    description="Analyzes and synthesizes collected research",
    model="gpt-4",
    temperature=0.4,
)
async def analyzer(points: List[str], question: str) -> Dict[str, Any]:
    """Analyze collected research points and produce insights."""
    points_text = "\\n".join(f"- {p}" for p in points)
    messages = [
        LLMMessage(
            role="system",
            content="You are a research analyst. Synthesize the facts into a JSON object with: summary, key_insights (list), and confidence (0-1).",
        ),
        LLMMessage(
            role="user",
            content=f"Question: {question}\\n\\nCollected facts:\\n{points_text}\\n\\nProvide your analysis as JSON.",
        ),
    ]
    config = LLMConfig(model=os.getenv("TELEON_LLM_MODEL", "gpt-4"), temperature=0.4, max_tokens=800)
    response = await gateway.complete(messages, config)

    import json
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"summary": response.content, "key_insights": [], "confidence": 0.0}


@client.agent(
    name="research-orchestrator",
    description="Orchestrates the research pipeline",
    model="gpt-4",
    cortex=True,
)
async def orchestrator(question: str) -> Dict[str, Any]:
    """
    Run a full research pipeline: collect facts then analyze them.

    Args:
        question: The research question to investigate

    Returns:
        dict with collected points, analysis, and final answer
    """
    # Step 1 — Collect facts (could fan out to multiple collectors)
    collected = await collector(topic=question, num_points=5)
    points = collected.get("points", [])

    # Step 2 — Analyze
    analysis = await analyzer(points=points, question=question)

    return {
        "question": question,
        "collected_points": points,
        "analysis": analysis.get("summary", ""),
        "key_insights": analysis.get("key_insights", []),
        "confidence": analysis.get("confidence", 0.0),
    }


if __name__ == "__main__":

    async def main():
        result = await orchestrator(
            question="What are the main benefits and risks of AI agents in production?"
        )
        print(f"Question: {result[\'question\']}")
        print(f"\\nAnalysis: {result[\'analysis\']}")
        print(f"\\nKey Insights:")
        for insight in result.get("key_insights", []):
            print(f"  - {insight}")
        print(f"\\nConfidence: {result[\'confidence\']}")

    asyncio.run(main())
'''


def _get_requirements(template_key: str) -> str:
    base = [
        "teleon>=0.1.0",
        "python-dotenv>=1.0.0",
    ]

    cortex_templates = {"customer-support", "research-system"}
    if template_key in cortex_templates:
        base.extend([
            "asyncpg>=0.29.0",
            "fastembed>=0.3.0",
        ])

    return "\n".join(base) + "\n"


def _get_yaml_content(project_name: str, template_key: str) -> str:
    """Return teleon.yaml content tailored to the chosen template."""
    base = f"""# Teleon Configuration
project: {project_name}
environment: development

llm:
  default_provider: openai
  default_model: gpt-4
  max_retries: 3
  timeout: 30.0
"""
    extras = {
        "customer-support": """
memory:
  enabled: true
  working_ttl: 3600

cortex:
  auto: true
  scope: ["customer_id"]

observability:
  log_level: INFO
  metrics_enabled: true
""",
        "data-pipeline": """
memory:
  enabled: true
  working_ttl: 3600

tools:
  enabled: true
  max_concurrent: 10

observability:
  log_level: INFO
  metrics_enabled: true
""",
        "code-review": """
memory:
  enabled: false

tools:
  enabled: true
  max_concurrent: 5

observability:
  log_level: INFO
  metrics_enabled: true
""",
        "content-generator": """
memory:
  enabled: true
  working_ttl: 3600

scaling:
  min_replicas: 1
  max_replicas: 20
  target_cpu: 70

observability:
  log_level: INFO
  metrics_enabled: true
""",
        "research-system": """
memory:
  enabled: true
  working_ttl: 3600

cortex:
  auto: true

observability:
  log_level: INFO
  metrics_enabled: true
""",
    }
    return base + extras.get(template_key, "")


def _get_env_example(template_key: str) -> str:
    """Return .env.example content for the chosen template."""
    base = "# Teleon\nTELEON_API_KEY=tlk_test_your_key_here\nTELEON_ENV=dev\n"
    if template_key in ("code-review", "content-generator", "research-system"):
        base += "\n# LLM Provider\nTELEON_LLM_MODEL=gpt-4\nOPENAI_API_KEY=sk-your-key-here\n"
    return base


def _get_readme(project_name: str, template_key: str) -> str:
    """Return README.md content for the chosen template."""
    info = TEMPLATES[template_key]
    features = ", ".join(info["features"])
    return f"""# {project_name}

{info['description']}

**Features:** {features}

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example env file and fill in your keys:
   ```bash
   cp .env.example .env
   ```

3. Start the development server:
   ```bash
   teleon dev start agents/
   ```

## Project Structure

- `agents/` - Your AI agents
- `tools/` - Custom tools (if applicable)
- `tests/` - Test files
- `teleon.yaml` - Configuration

## Documentation

Visit https://teleon.ai/docs for complete documentation.
"""


@app.command()
def init(
    name: Optional[str] = typer.Argument(None),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Template: customer-support, data-pipeline, code-review, content-generator, research-system"),
):
    """
    Initialize a new Teleon project.

    When run without flags an interactive menu lets you pick a template.

    Examples:
        teleon init
        teleon init my-agent
        teleon init --template customer-support my-project
    """
    from pathlib import Path
    from rich.prompt import Prompt
    from rich.table import Table

    template_keys = list(TEMPLATES.keys())

    # --- resolve template --------------------------------------------------
    if template is None:
        # Interactive menu
        console.print()
        console.print(Panel.fit(
            "[bold green]Teleon Init[/bold green] — Choose a starter template",
            title="teleon init",
        ))
        console.print()

        for idx, key in enumerate(template_keys, 1):
            info = TEMPLATES[key]
            features_str = " | ".join(info["features"])
            console.print(f"  [bold cyan][{idx}][/bold cyan] {info['name']}")
            console.print(f"      [dim]{info['description']}[/dim]")
            console.print(f"      [dim italic]{features_str}[/dim italic]")
            console.print()

        choice = Prompt.ask(
            "  Select a template",
            choices=[str(i) for i in range(1, len(template_keys) + 1)],
            default="1",
        )
        template = template_keys[int(choice) - 1]
    else:
        if template not in TEMPLATES:
            console.print(f"[red]Unknown template '{template}'. Available: {', '.join(template_keys)}[/red]")
            raise typer.Exit(1)

    # --- resolve project name ----------------------------------------------
    default_name = TEMPLATES[template]["default_project_name"]
    if name is None:
        name = Prompt.ask("  Project name", default=default_name)

    project_name: str = name  # type: ignore[assignment]

    console.print()
    console.print(Panel.fit(
        f"[bold green]Creating project:[/bold green] [cyan]{project_name}[/cyan]\n"
        f"Template: [yellow]{TEMPLATES[template]['name']}[/yellow]",
        title="teleon init",
    ))

    project_path = Path(project_name)

    if project_path.exists():
        console.print(f"[red]Directory '{project_name}' already exists[/red]")
        raise typer.Exit(1)

    try:
        # Create directory structure
        project_path.mkdir(parents=True)
        (project_path / "agents").mkdir()
        (project_path / "tests").mkdir()

        # Only create tools/ for templates that use custom tools
        if template in ("data-pipeline",):
            (project_path / "tools").mkdir()

        # Write files
        (project_path / "agents" / "main.py").write_text(_get_agent_content(template))
        (project_path / "teleon.yaml").write_text(_get_yaml_content(project_name, template))
        (project_path / ".env.example").write_text(_get_env_example(template))
        (project_path / "README.md").write_text(_get_readme(project_name, template))

        # .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Environment
.env

# Teleon
.teleon/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
        (project_path / ".gitignore").write_text(gitignore_content)

        # requirements.txt
        (project_path / "requirements.txt").write_text(_get_requirements(template))

        console.print(f"\n[green]Project '{project_name}' created successfully![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. cd {project_name}")
        console.print("  2. pip install -r requirements.txt")
        console.print("  3. cp .env.example .env  # fill in your keys")
        console.print("  4. teleon dev start")

    except Exception as e:
        console.print(f"[red]Failed to create project: {e}[/red]")
        raise typer.Exit(1)




@app.command()
def logs(
    agent: Optional[str] = None,
    deployment: Optional[str] = typer.Option(None, "--deployment", "-d", help="Deployment ID to view logs for"),
    follow: bool = False,
    lines: int = 100,
    level: str = "INFO"
):
    """
    View agent logs.
    
    Example:
        teleon logs my-agent
        teleon logs --deployment abc12345
        teleon logs my-agent --follow
        teleon logs --deployment abc12345 --lines 50 --level ERROR
    
    Args:
        agent: Agent name to view logs for (local logs)
        deployment: Deployment ID to view logs for (platform logs)
        follow: Follow log output (like tail -f)
        lines: Number of lines to show
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    """
    import time
    from pathlib import Path
    from datetime import datetime
    import httpx
    import json
    
    # If deployment ID is provided, fetch logs from platform
    if deployment:
        console.print(Panel.fit(
            f"[bold green]Deployment Logs[/bold green]\n"
            f"Deployment ID: [cyan]{deployment}[/cyan]\n"
            f"Lines: [yellow]{lines}[/yellow] | Level: [yellow]{level}[/yellow]",
            title="📋 Teleon Logs"
        ))
        
        # Get auth token and platform URL
        config_file = Path.home() / ".teleon" / "config.json"
        if not config_file.exists():
            console.print("\n[red]❌ Not authenticated. Run: teleon login[/red]")
            raise typer.Exit(1)
        
        config_data = json.loads(config_file.read_text())
        auth_token = config_data.get("auth_token")
        platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
        
        if not auth_token:
            console.print("\n[red]❌ No auth token found. Run: teleon login[/red]")
            raise typer.Exit(1)
        
        try:
            # If deployment ID looks like a short ID (8 chars), try to expand it
            full_deployment_id = deployment
            
            if len(deployment) == 8 and '-' not in deployment:
                console.print(f"\n[dim]Looking up full deployment ID for: {deployment}...[/dim]")
                
                # Get user's deployments to find the matching one
                deployments_response = httpx.get(
                    f"{platform_url}/api/v1/deployments",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30.0
                )
                
                if deployments_response.status_code == 200:
                    deployments = deployments_response.json().get("deployments", [])
                    
                    # Find deployment that starts with the short ID
                    matching = [d for d in deployments if d.get("id", "").startswith(deployment)]
                    
                    if len(matching) == 1:
                        full_deployment_id = matching[0]["id"]
                        console.print(f"[dim]Found: {full_deployment_id}[/dim]")
                    elif len(matching) > 1:
                        console.print(f"[yellow]⚠️  Multiple deployments match '{deployment}':[/yellow]")
                        for d in matching[:5]:
                            console.print(f"  • {d['id']}")
                        console.print("\n[dim]Please use a more specific deployment ID[/dim]")
                        return
                    else:
                        console.print(f"[yellow]⚠️  No deployment found starting with '{deployment}'[/yellow]")
                        return
            
            # Fetch logs from platform API
            console.print(f"\n[dim]Fetching logs from {platform_url}...[/dim]\n")
            
            response = httpx.get(
                f"{platform_url}/api/v1/deployments/{full_deployment_id}/logs",
                headers={"Authorization": f"Bearer {auth_token}"},
                params={"lines": lines, "level": level},
                timeout=30.0
            )
            
            if response.status_code == 404:
                console.print(f"[yellow]⚠️  Deployment '{full_deployment_id}' not found[/yellow]")
                console.print("[dim]Make sure the deployment ID is correct[/dim]")
                return
            elif response.status_code == 403:
                console.print("[red]❌ Access denied to this deployment[/red]")
                return
            elif response.status_code != 200:
                console.print(f"[red]❌ Failed to fetch logs: {response.text}[/red]")
                return
            
            logs_data = response.json()
            log_lines = logs_data.get("logs", [])
            
            if not log_lines:
                console.print("[yellow]⚠️  No logs available for this deployment yet[/yellow]")
                console.print("\n[dim]Logs may take a few moments to appear after deployment[/dim]")
                return
            
            # Display logs
            for log_entry in log_lines:
                timestamp = log_entry.get("timestamp", "")
                log_level = log_entry.get("level", "INFO")
                message = log_entry.get("message", "")
                
                # Color code by level
                level_colors = {
                    "DEBUG": "dim",
                    "INFO": "cyan",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold red"
                }
                color = level_colors.get(log_level, "white")
                
                console.print(f"[dim]{timestamp}[/dim] [{color}]{log_level}[/{color}] {message}")
            
            console.print(f"\n[dim]Showing {len(log_lines)} log entries[/dim]")
            
        except httpx.RequestError as e:
            console.print(f"[red]❌ Network error: {e}[/red]")
            console.print(f"\n[yellow]Make sure the Teleon platform is running at: {platform_url}[/yellow]")
            raise typer.Exit(1)
        
        return
    
    # Otherwise, show local logs
    console.print(Panel.fit(
        f"[bold green]Logs for:[/bold green] [cyan]{agent or 'all agents'}[/cyan]\n"
        f"Follow: [yellow]{follow}[/yellow] | Lines: [yellow]{lines}[/yellow] | Level: [yellow]{level}[/yellow]",
        title="📋 Teleon Logs"
    ))
    
    # Implement log streaming
    log_dir = Path(".teleon/logs")
    
    if not log_dir.exists():
        console.print("[yellow]⚠️  No logs found. Run an agent first to generate logs.[/yellow]")
        console.print("\n[dim]Logs will be stored in .teleon/logs/[/dim]")
        console.print("\n[bold]To view deployment logs:[/bold]")
        console.print("[cyan]  teleon logs --deployment <deployment-id>[/cyan]")
        return
    
    # Find log files
    if agent:
        log_files = list(log_dir.glob(f"{agent}*.log"))
    else:
        log_files = list(log_dir.glob("*.log"))
    
    if not log_files:
        console.print(f"[yellow]⚠️  No logs found for '{agent or 'any agent'}'[/yellow]")
        console.print("\n[bold]To view deployment logs:[/bold]")
        console.print("[cyan]  teleon logs --deployment <deployment-id>[/cyan]")
        return
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    console.print(f"\n[dim]Found {len(log_files)} log file(s)[/dim]\n")
    
    try:
        if follow:
            # Follow mode - tail -f like behavior
            console.print("[dim]Following logs (Ctrl+C to stop)...[/dim]\n")
            
            log_file = log_files[0]  # Most recent
            with open(log_file, 'r') as f:
                # Seek to end
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        # Parse and format log line
                        if level.upper() in line or level == "DEBUG":
                            console.print(line.strip())
                    else:
                        time.sleep(0.1)
        else:
            # Static mode - show last N lines
            all_lines = []
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    file_lines = f.readlines()
                    all_lines.extend([
                        (log_file.name, line.strip())
                        for line in file_lines
                        if level.upper() in line or level == "DEBUG"
                    ])
            
            # Show last N lines
            for filename, line in all_lines[-lines:]:
                console.print(f"[dim]{filename}:[/dim] {line}")
            
            console.print(f"\n[dim]Showing last {min(lines, len(all_lines))} lines[/dim]")
    
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following logs[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Error reading logs: {e}[/red]")


# CLI entry point for Poetry
cli = app


if __name__ == "__main__":
    app()

