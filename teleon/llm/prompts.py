"""Prompt management system with templates and versioning."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import re


class PromptTemplate(BaseModel):
    """A prompt template with variables."""
    
    name: str = Field(..., description="Template name")
    template: str = Field(..., description="Template string with {variables}")
    description: str = Field("", description="Template description")
    variables: List[str] = Field(default_factory=list, description="Required variables")
    version: str = Field("1.0.0", description="Template version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation time")
    
    def render(self, **kwargs) -> str:
        """
        Render template with variables.
        
        Args:
            **kwargs: Variable values
        
        Returns:
            Rendered prompt
        
        Raises:
            ValueError: If required variables missing
        """
        # Check required variables
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Render template
        return self.template.format(**kwargs)
    
    @classmethod
    def from_string(
        cls,
        name: str,
        template_str: str,
        description: str = "",
        version: str = "1.0.0"
    ) -> "PromptTemplate":
        """
        Create template from string.
        
        Args:
            name: Template name
            template_str: Template string
            description: Description
            version: Version
        
        Returns:
            PromptTemplate
        """
        # Extract variables
        variables = re.findall(r'\{(\w+)\}', template_str)
        variables = list(set(variables))  # Unique
        
        return cls(
            name=name,
            template=template_str,
            description=description,
            variables=variables,
            version=version
        )


class PromptManager:
    """
    Manage prompts with templates, versioning, and A/B testing.
    
    Features:
    - Template storage and retrieval
    - Versioning
    - A/B testing
    - Usage tracking
    - Best practice prompts
    """
    
    def __init__(self):
        """Initialize prompt manager."""
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}  # name -> version -> template
        self.usage_stats: Dict[str, int] = {}  # template_id -> usage count
        self.ab_tests: Dict[str, Dict[str, Any]] = {}  # test_id -> config
        
        # Load default templates
        self._load_defaults()
    
    def register_template(
        self,
        template: PromptTemplate,
        override: bool = False
    ) -> None:
        """
        Register a prompt template.
        
        Args:
            template: Template to register
            override: Whether to override existing version
        """
        if template.name not in self.templates:
            self.templates[template.name] = {}
        
        if template.version in self.templates[template.name] and not override:
            raise ValueError(
                f"Template '{template.name}' version '{template.version}' already exists"
            )
        
        self.templates[template.name][template.version] = template
    
    def get_template(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get a template.
        
        Args:
            name: Template name
            version: Template version (None = latest)
        
        Returns:
            Template or None
        """
        if name not in self.templates:
            return None
        
        versions = self.templates[name]
        
        if version:
            return versions.get(version)
        else:
            # Get latest version
            latest = max(versions.keys())
            return versions[latest]
    
    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a template."""
        if name not in self.templates:
            return []
        return list(self.templates[name].keys())
    
    def render(
        self,
        template_name: str,
        version: Optional[str] = None,
        track_usage: bool = True,
        **kwargs
    ) -> str:
        """
        Render a template.
        
        Args:
            template_name: Template name
            version: Template version
            track_usage: Whether to track usage
            **kwargs: Template variables
        
        Returns:
            Rendered prompt
        """
        template = self.get_template(template_name, version)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Track usage
        if track_usage:
            template_id = f"{template_name}@{template.version}"
            self.usage_stats[template_id] = self.usage_stats.get(template_id, 0) + 1
        
        return template.render(**kwargs)
    
    def create_ab_test(
        self,
        test_name: str,
        variant_a: str,
        variant_b: str,
        version_a: Optional[str] = None,
        version_b: Optional[str] = None,
        traffic_split: float = 0.5
    ) -> None:
        """
        Create an A/B test for prompts.
        
        Args:
            test_name: Test name
            variant_a: Template A name
            variant_b: Template B name
            version_a: Template A version
            version_b: Template B version
            traffic_split: % traffic to A (0-1)
        """
        self.ab_tests[test_name] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "results_a": 0,
            "results_b": 0,
            "created_at": datetime.now(timezone.utc)
        }
    
    def get_ab_variant(
        self,
        test_name: str,
        user_id: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get A/B test variant for user.
        
        Args:
            test_name: Test name
            user_id: User ID (for consistent assignment)
        
        Returns:
            Template variant
        """
        if test_name not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_name]
        
        # Determine variant
        if user_id:
            # Consistent assignment based on user_id
            import hashlib
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            use_a = (hash_val % 100) < (test["traffic_split"] * 100)
        else:
            # Random assignment
            import random
            use_a = random.random() < test["traffic_split"]
        
        if use_a:
            template = self.get_template(
                test["variant_a"],
                test["version_a"]
            )
            test["results_a"] += 1
        else:
            template = self.get_template(
                test["variant_b"],
                test["version_b"]
            )
            test["results_b"] += 1
        
        return template
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get template usage statistics."""
        return self.usage_stats.copy()
    
    def get_ab_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        return self.ab_tests.get(test_name)
    
    def _load_defaults(self):
        """Load default templates."""
        
        # System message templates
        system_assistant = PromptTemplate.from_string(
            name="system.assistant",
            template_str="You are a helpful AI assistant. {instructions}",
            description="Basic assistant system message",
            version="1.0.0"
        )
        self.register_template(system_assistant)
        
        # Code generation
        code_gen = PromptTemplate.from_string(
            name="code.generate",
            template_str="""Generate {language} code for the following task:

Task: {task}

Requirements:
{requirements}

Please provide clean, well-documented code with error handling.""",
            description="Code generation prompt",
            version="1.0.0"
        )
        self.register_template(code_gen)
        
        # Analysis
        analysis = PromptTemplate.from_string(
            name="analysis.data",
            template_str="""Analyze the following data:

{data}

Focus on: {focus_areas}

Provide insights and recommendations.""",
            description="Data analysis prompt",
            version="1.0.0"
        )
        self.register_template(analysis)
        
        # Summarization
        summary = PromptTemplate.from_string(
            name="summarize",
            template_str="""Summarize the following text in {max_words} words or less:

{text}

Focus on the main points and key takeaways.""",
            description="Text summarization prompt",
            version="1.0.0"
        )
        self.register_template(summary)
        
        # Question answering
        qa = PromptTemplate.from_string(
            name="qa.answer",
            template_str="""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer based only on the context provided.""",
            description="Question answering prompt",
            version="1.0.0"
        )
        self.register_template(qa)
        
        # Classification
        classify = PromptTemplate.from_string(
            name="classify",
            template_str="""Classify the following text into one of these categories:
{categories}

Text: {text}

Category:""",
            description="Text classification prompt",
            version="1.0.0"
        )
        self.register_template(classify)
        
        # Extraction
        extract = PromptTemplate.from_string(
            name="extract.entities",
            template_str="""Extract the following entities from the text:
Entity types: {entity_types}

Text: {text}

Entities:""",
            description="Entity extraction prompt",
            version="1.0.0"
        )
        self.register_template(extract)


# Global prompt manager instance
_global_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get global prompt manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = PromptManager()
    return _global_manager

