"""Setup configuration for Teleon package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read version from __version__.py
version_file = Path(__file__).parent / "teleon" / "__version__.py"
version_info = {}
with open(version_file) as f:
    exec(f.read(), version_info)

# Read README
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="teleon",
    version=version_info["__version__"],
    description="The Platform for Intelligent Agents - Deploy production-ready AI agents in minutes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Teleon AI, Inc.",
    author_email="founders@teleon.ai",
    url="https://teleon.ai",
    project_urls={
        "Homepage": "https://teleon.ai",
        "Documentation": "https://docs.teleon.ai",
        "Source": "https://github.com/teleonAI/teleon",
        "Bug Tracker": "https://github.com/teleonAI/teleon/issues",
    },
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "teleon-platform",
            "teleon-platform-azure",
            "teleon-dashboard",
            "teleon-dashboard-test",
            "teleon-website",
            "teleon-infra",
            "demo_project",
            "demo_project_2",
            "examples",
            "docs",
        ]
    ),
    python_requires=">=3.11",
    install_requires=[
        # Web Framework
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.30.0,<1.0.0",
        # Data Validation
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        # CLI
        "typer>=0.12.0",
        "rich>=13.7.0",
        # Utilities
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "httpx>=0.25.2",
        "tenacity>=8.2.3",
        "structlog>=23.2.0",
        "watchdog>=3.0.0",  # File system monitoring for hot reload
        # Observability
        "prometheus-client>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.13.0",
        ],
        "openai": ["openai>=1.3.7"],
        "anthropic": ["anthropic>=0.7.7"],
        "redis": ["redis[asyncio]>=4.5.0"],
        "postgres": ["asyncpg>=0.28.0"],
        "chromadb": [
            "chromadb>=0.4.18",
            "fastembed>=0.2.0",
        ],
        "all-storage": [
            "redis[asyncio]>=4.5.0",
            "asyncpg>=0.28.0",
            "chromadb>=0.4.18",
            "fastembed>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "teleon=teleon.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: FastAPI",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "ai",
        "agents",
        "llm",
        "automation",
        "machine-learning",
        "agentic",
        "multi-agent",
        "ai-agents",
        "fastapi",
        "production",
    ],
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)

