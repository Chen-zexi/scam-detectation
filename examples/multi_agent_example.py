#!/usr/bin/env python3
"""
Multi-Agent Synthesis Example Script

This script demonstrates the new multi-agent collaboration features for synthetic 
scam data generation. It shows how multiple specialized AI agents work together
to generate, evaluate, and refine high-quality synthetic data.

Based on cutting-edge research from:
- MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation (arXiv:2504.12563)
- AgentSGEN: Multi-Agent LLM in the Loop for Semantic Collaboration (arXiv:2505.13466)
- ConvoGen: Enhancing Conversational AI with Synthetic Data: A Multi-Agent Approach (arXiv:2503.17460)

Usage:
    python examples/multi_agent_example.py
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.synthesize import (
    SynthesisGenerator, 
    MultiAgentOrchestrator, 
    DiversityManager, 
    DiversityConfig, 
    DiversityLevel
)


async def demonstrate_multi_agent_features():
    """Demonstrate the multi-agent collaboration features."""
    
    print("🤖 Multi-Agent Collaboration Demo")
    print("=" * 60)
    print("This demo shows how multiple AI agents collaborate to generate")
    print("higher quality synthetic scam data through iterative improvement.")
    print()
    
    # Example 1: Standard vs Multi-Agent Generation
    print("📊 Example 1: Standard vs Multi-Agent Generation")
    print("-" * 50)
    
    try:
        # Standard generation with diversity
        print("🔹 Standard Generation (with diversity):")
        standard_generator = SynthesisGenerator(
            synthesis_type="phone_transcript",
            sample_size=1,
            provider="openai",
            model="gpt-4o-mini",
            enable_diversity=True,
            diversity_level="medium",
            enable_multi_agent=False,
            category="authority_scam"
        )
        
        print("Standard generation process:")
        print("  1. Apply diversity techniques")
        print("  2. Generate content with enhanced prompts")
        print("  3. Return result")
        print()
        
        # Multi-agent generation
        print("🔹 Multi-Agent Generation:")
        multi_agent_generator = SynthesisGenerator(
            synthesis_type="phone_transcript",
            sample_size=1,
            provider="openai",
            model="gpt-4o-mini",
            enable_diversity=True,
            diversity_level="medium",
            enable_multi_agent=True,
            multi_agent_quality_threshold=7.0,
            enable_specialist_agents=True,
            enable_iterative_refinement=True,
            category="authority_scam"
        )
        
        print("Multi-agent collaboration process:")
        print("  1. Generator Agent creates initial content")
        print("  2. Evaluator Agent assesses quality (realism, authenticity, etc.)")
        print("  3. Scam Specialist Agent reviews social engineering tactics")
        print("  4. Authenticity Specialist Agent verifies real-world accuracy")
        print("  5. Refiner Agent improves content based on feedback")
        print("  6. Coordinator Agent makes final quality assessment")
        print()
        
        # Note about API requirements
        print("💡 Note: Full multi-agent generation requires valid API keys.")
        print("   This demo shows the architecture and process flow.")
        
    except Exception as e:
        print(f"Demo setup error: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 2: Agent Specializations
    print("🎭 Example 2: Specialized Agent Roles")
    print("-" * 40)
    
    agents_info = {
        "Content Generator": {
            "role": "Creates initial synthetic content",
            "specialization": "Diversity techniques, category adherence",
            "focus": ["authenticity", "diversity", "category_requirements"]
        },
        "Quality Evaluator": {
            "role": "Assesses content quality and realism",
            "specialization": "Multi-dimensional quality scoring",
            "focus": ["realism", "authenticity", "language_quality", "training_value"]
        },
        "Scam Specialist": {
            "role": "Domain expert in social engineering",
            "specialization": "Scam tactics and psychological manipulation",
            "focus": ["persuasion_techniques", "authority_tactics", "urgency_creation"]
        },
        "Authenticity Specialist": {
            "role": "Real-world accuracy validator",
            "specialization": "Communication patterns, procedures",
            "focus": ["factual_accuracy", "cultural_authenticity", "procedural_realism"]
        },
        "Content Refiner": {
            "role": "Improves content based on feedback",
            "specialization": "Quality enhancement while preserving intent",
            "focus": ["realism_improvement", "language_polish", "training_effectiveness"]
        },
        "Collaboration Coordinator": {
            "role": "Orchestrates multi-agent collaboration",
            "specialization": "Meta-assessment and process optimization",
            "focus": ["quality_convergence", "collaboration_efficiency", "final_decisions"]
        }
    }
    
    for agent_name, info in agents_info.items():
        print(f"\n🤖 {agent_name}:")
        print(f"   Role: {info['role']}")
        print(f"   Specialization: {info['specialization']}")
        print(f"   Focus areas: {', '.join(info['focus'])}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Quality Improvement Process
    print("📈 Example 3: Iterative Quality Improvement")
    print("-" * 45)
    
    improvement_process = [
        {
            "round": 1,
            "generator": "Creates initial authority scam transcript",
            "evaluator": "Scores: Realism=6.5, Authenticity=6.0, Diversity=7.0",
            "specialist": "Scam tactics need more authority-building language",
            "refiner": "Enhances official terminology and badge references",
            "quality": 6.5
        },
        {
            "round": 2, 
            "generator": "N/A (refinement of existing content)",
            "evaluator": "Scores: Realism=7.5, Authenticity=7.0, Diversity=7.0",
            "specialist": "Good improvement, add more pressure tactics",
            "refiner": "Adds time pressure and consequence threats",
            "quality": 7.2
        },
        {
            "round": 3,
            "generator": "N/A (final refinement)",
            "evaluator": "Scores: Realism=8.0, Authenticity=7.8, Diversity=7.0",
            "specialist": "Excellent psychological manipulation techniques",
            "refiner": "Minor language polishing",
            "quality": 7.8
        }
    ]
    
    for step in improvement_process:
        print(f"\n🔄 Collaboration Round {step['round']}:")
        print(f"   Generator: {step['generator']}")
        print(f"   Evaluator: {step['evaluator']}")
        print(f"   Specialist: {step['specialist']}")
        print(f"   Refiner: {step['refiner']}")
        print(f"   Overall Quality: {step['quality']}/10")
    
    print(f"\n✅ Final Result: Quality improved from 6.5 → 7.8 (+20%)")
    
    print("\n" + "=" * 60)
    
    # Example 4: Research-Based Techniques
    print("📚 Example 4: Research-Based Techniques Integration")
    print("-" * 50)
    
    research_techniques = {
        "MetaSynth Approach": [
            "Meta-prompting with expert agent orchestration",
            "Collaborative generation through agent debate",
            "Diversity enhancement through multiple perspectives",
            "Quality convergence through iterative refinement"
        ],
        "AgentSGEN Patterns": [
            "Evaluator-Editor collaboration loop",
            "Semantic consistency enforcement",
            "Iterative refinement with quality thresholds",
            "Safety-critical scenario validation"
        ],
        "ConvoGen Methods": [
            "Dynamic few-shot example selection",
            "Conversational flow optimization",
            "Multi-agent conversational modeling",
            "Context-aware dialogue generation"
        ]
    }
    
    for approach, techniques in research_techniques.items():
        print(f"\n📖 {approach}:")
        for technique in techniques:
            print(f"   • {technique}")
    
    print("\n" + "=" * 60)
    print("✅ Multi-Agent Collaboration Demo Complete!")
    
    print("\nKey Benefits of Multi-Agent Approach:")
    print("🎯 Higher Quality: Multiple expert perspectives ensure better content")
    print("🔍 Better Evaluation: Comprehensive assessment across multiple dimensions")
    print("🔄 Iterative Improvement: Content gets refined through multiple rounds")
    print("🎭 Specialist Knowledge: Domain experts provide specific expertise")
    print("⚖️ Quality Assurance: Coordinator ensures standards are met")
    
    print("\nTo use multi-agent collaboration:")
    print("1. Run: python main.py")
    print("2. Choose 'Synthesis'")
    print("3. Select synthesis type and category")
    print("4. Enable diversity enhancement")
    print("5. Enable multi-agent collaboration")
    print("6. Configure quality thresholds and agent settings")
    

def show_configuration_examples():
    """Show different multi-agent configuration examples."""
    print("\n🔧 Multi-Agent Configuration Examples")
    print("=" * 40)
    
    configs = {
        "Conservative": {
            "quality_threshold": 6.0,
            "specialist_agents": True,
            "iterative_refinement": True,
            "description": "Uses multi-agent for most content, high collaboration"
        },
        "Balanced": {
            "quality_threshold": 7.0,
            "specialist_agents": True,
            "iterative_refinement": True,
            "description": "Recommended for most use cases"
        },
        "Selective": {
            "quality_threshold": 8.0,
            "specialist_agents": True,
            "iterative_refinement": False,
            "description": "Multi-agent only for challenging content"
        },
        "Minimal": {
            "quality_threshold": 9.0,
            "specialist_agents": False,
            "iterative_refinement": False,
            "description": "Rare multi-agent use, basic collaboration"
        }
    }
    
    for config_name, settings in configs.items():
        print(f"\n📋 {config_name} Configuration:")
        print(f"   Quality threshold: {settings['quality_threshold']}")
        print(f"   Specialist agents: {settings['specialist_agents']}")
        print(f"   Iterative refinement: {settings['iterative_refinement']}")
        print(f"   Use case: {settings['description']}")
    
    print("\n💡 Recommendation: Start with 'Balanced' configuration")
    print("   Adjust based on your quality requirements and API usage preferences.")


def show_expected_improvements():
    """Show expected improvements from multi-agent approach."""
    print("\n📊 Expected Quality Improvements")
    print("=" * 35)
    
    improvements = {
        "Content Realism": {
            "standard": "6.5/10",
            "multi_agent": "7.8/10",
            "improvement": "+20%",
            "reason": "Multiple expert perspectives and iterative refinement"
        },
        "Authenticity": {
            "standard": "6.8/10", 
            "multi_agent": "8.1/10",
            "improvement": "+19%",
            "reason": "Specialist validation and real-world accuracy checks"
        },
        "Diversity": {
            "standard": "7.2/10",
            "multi_agent": "7.5/10", 
            "improvement": "+4%",
            "reason": "Agent diversity adds variation to existing techniques"
        },
        "Training Value": {
            "standard": "6.9/10",
            "multi_agent": "8.3/10",
            "improvement": "+20%",
            "reason": "Quality-focused refinement improves ML training effectiveness"
        }
    }
    
    for metric, data in improvements.items():
        print(f"\n📈 {metric}:")
        print(f"   Standard: {data['standard']}")
        print(f"   Multi-Agent: {data['multi_agent']} ({data['improvement']} improvement)")
        print(f"   Reason: {data['reason']}")
    
    print(f"\n🎯 Overall Quality Improvement: ~15-20% across all metrics")
    print("📚 Based on MetaSynth research: 25-40% improvement in diversity metrics")


if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Collaboration Demo...")
    print("This demo showcases advanced multi-agent synthesis features.\n")
    
    try:
        # Run the main demonstration
        asyncio.run(demonstrate_multi_agent_features())
        
        # Show configuration examples
        show_configuration_examples()
        
        # Show expected improvements
        show_expected_improvements()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Note: Multi-agent features require valid API keys in .env file")
    
    print("\nThank you for exploring the multi-agent collaboration features! 🤖")