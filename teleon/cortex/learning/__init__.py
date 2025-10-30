"""
Cortex Learning System.

Automatic learning and optimization for Teleon agents.

Components:
- LearningEngine: Orchestrates learning across memory types
- PatternRecognizer: Advanced pattern extraction and analysis
- CostOptimizer: Model selection and cost optimization

Example:
    ```python
    from teleon.cortex.learning import LearningEngine, LearningConfig
    from teleon.cortex.memory import EpisodicMemory, SemanticMemory, ProceduralMemory
    
    # Create learning engine
    engine = LearningEngine(
        episodic=episodic_memory,
        semantic=semantic_memory,
        procedural=procedural_memory,
        config=LearningConfig(
            auto_learn=True,
            batch_size=10
        )
    )
    
    # Process interaction
    await engine.process_interaction(
        input_data={"query": "Hello"},
        output_data={"response": "Hi!"},
        success=True,
        cost=0.001,
        duration_ms=100
    )
    
    # Trigger learning
    results = await engine.learn()
    print(f"Patterns learned: {results['patterns_learned']}")
    
    # Get optimization recommendations
    from teleon.cortex.learning import CostOptimizer
    optimizer = CostOptimizer()
    report = await optimizer.analyze(episodes)
    ```
"""

from teleon.cortex.learning.engine import (
    LearningEngine,
    LearningConfig,
    LearningMetrics,
)
from teleon.cortex.learning.patterns import (
    PatternRecognizer,
    PatternSignature,
)
from teleon.cortex.learning.optimization import (
    CostOptimizer,
    OptimizationStrategy,
    ModelRecommendation,
    CachingRecommendation,
    OptimizationReport,
)

__all__ = [
    # Learning Engine
    "LearningEngine",
    "LearningConfig",
    "LearningMetrics",
    
    # Pattern Recognition
    "PatternRecognizer",
    "PatternSignature",
    
    # Optimization
    "CostOptimizer",
    "OptimizationStrategy",
    "ModelRecommendation",
    "CachingRecommendation",
    "OptimizationReport",
]

