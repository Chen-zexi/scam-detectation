#!/usr/bin/env python3
"""
Diversity Manager for Synthetic Data Generation

This module implements advanced diversity techniques to improve the quality and 
variation of synthetic scam data generation. Based on research from MetaSynth, 
CoDSA (Conditional Data Synthesis Augmentation), and other state-of-the-art 
synthetic data generation techniques.

Key Features:
- Template variation and rotation
- Dynamic context injection
- Few-shot learning with example pools
- Persona-based generation
- Self-consistency techniques
- Conditional data synthesis for underrepresented scenarios
"""

import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from .synthesis_prompts import SynthesisPromptsManager

logger = logging.getLogger(__name__)


class DiversityLevel(Enum):
    """Diversity levels for controlling generation variation."""
    MINIMAL = "minimal"      # Basic template rotation
    MEDIUM = "medium"        # Template + context + few-shot
    HIGH = "high"           # All techniques + self-consistency
    MAXIMUM = "maximum"     # Multi-agent + advanced techniques


@dataclass
class DiversityConfig:
    """Configuration for diversity techniques."""
    level: DiversityLevel = DiversityLevel.MEDIUM
    template_variation: bool = True
    context_injection: bool = True
    few_shot_learning: bool = True
    persona_variation: bool = True
    self_consistency: bool = False
    num_candidates: int = 3
    confidence_threshold: float = 0.8


class DiversityManager:
    """
    Enhanced prompt generation with diversity techniques for synthetic data.
    
    This class implements various techniques to diversify synthetic data generation:
    1. Template Variations - Multiple prompt templates per category
    2. Context Injection - Dynamic contextual elements
    3. Few-Shot Learning - Rotating example sets
    4. Persona Variation - Different perspectives and roles
    5. Self-Consistency - Multiple generation + selection
    
    Based on research from:
    - MetaSynth: Meta-Prompting-Driven Agentic Scaffolds (arXiv:2504.12563)
    - CoDSA: Conditional Data Synthesis Augmentation (arXiv:2504.07426)
    - Advanced prompt engineering techniques for LLMs
    """
    
    def __init__(self, prompts_manager: SynthesisPromptsManager, config_path: Optional[str] = None):
        """
        Initialize the Diversity Manager.
        
        Args:
            prompts_manager: Base prompts manager
            config_path: Path to diversity configuration file
        """
        self.prompts_manager = prompts_manager
        self.config_path = config_path or self._get_default_config_path()
        
        # Load diversity configurations
        self.diversity_config = self._load_diversity_config()
        self.template_variations = self._load_template_variations()
        self.context_pools = self._load_context_pools()
        self.few_shot_examples = self._load_few_shot_examples()
        self.personas = self._load_personas()
        
        logger.info(f"DiversityManager initialized with {len(self.template_variations)} template variations")
    
    def _get_default_config_path(self) -> str:
        """Get default path for diversity configuration."""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "diversity_config.json")
    
    def _load_diversity_config(self) -> Dict[str, Any]:
        """Load diversity configuration from JSON file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Diversity config not found at {self.config_path}, using defaults")
                return self._get_default_diversity_config()
        except Exception as e:
            logger.error(f"Error loading diversity config: {e}")
            return self._get_default_diversity_config()
    
    def _get_default_diversity_config(self) -> Dict[str, Any]:
        """Get default diversity configuration."""
        return {
            "template_variations": {
                "phone_transcript": {
                    "authority_scam": [
                        "direct_official",
                        "narrative_building", 
                        "analytical_structured",
                        "conversational_natural",
                        "urgency_focused"
                    ],
                    "tech_support_scam": [
                        "technical_detailed",
                        "problem_discovery",
                        "solution_oriented",
                        "fear_inducing",
                        "helpful_deceptive"
                    ]
                }
            },
            "context_pools": {
                "demographics": [
                    "young professional in their 20s",
                    "middle-aged parent",
                    "senior citizen over 65",
                    "college student",
                    "retiree",
                    "small business owner",
                    "healthcare worker",
                    "teacher"
                ],
                "timeframes": [
                    "early morning (7-9 AM)",
                    "mid-morning (9-11 AM)", 
                    "lunch time (11 AM-1 PM)",
                    "afternoon (1-5 PM)",
                    "evening (5-8 PM)",
                    "late evening (8-10 PM)"
                ],
                "locations": [
                    "urban city environment",
                    "suburban neighborhood", 
                    "rural/small town area",
                    "office setting",
                    "home environment",
                    "public place"
                ],
                "emotional_states": [
                    "calm and collected",
                    "slightly anxious",
                    "confused but cooperative",
                    "suspicious and questioning",
                    "stressed about finances",
                    "eager to resolve issues"
                ],
                "technical_levels": [
                    "very tech-savvy",
                    "moderate computer skills",
                    "basic technology user",
                    "minimal tech experience",
                    "prefers traditional methods"
                ]
            },
            "personas": {
                "scammer_types": [
                    "authoritative government official",
                    "helpful tech support agent", 
                    "urgent financial advisor",
                    "friendly customer service rep",
                    "security specialist",
                    "health insurance coordinator"
                ],
                "victim_types": [
                    "trusting elderly person",
                    "busy professional", 
                    "concerned parent",
                    "tech-confused user",
                    "financially stressed individual",
                    "health-conscious citizen"
                ]
            }
        }
    
    def _load_template_variations(self) -> Dict[str, Dict[str, List[str]]]:
        """Load template variations from config."""
        return self.diversity_config.get("template_variations", {})
    
    def _load_context_pools(self) -> Dict[str, List[str]]:
        """Load context pools from config."""
        return self.diversity_config.get("context_pools", {})
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load few-shot examples from config."""
        return self.diversity_config.get("few_shot_examples", {})
    
    def _load_personas(self) -> Dict[str, List[str]]:
        """Load persona definitions from config."""
        return self.diversity_config.get("personas", {})
    
    def create_diverse_generation_prompt(self, synthesis_type: str, category: str, 
                                       diversity_config: DiversityConfig = None) -> str:
        """
        Create a diverse generation prompt using multiple enhancement techniques.
        
        This method implements several diversity techniques based on the configuration:
        1. Template Variation - Selects from multiple prompt templates
        2. Context Injection - Adds dynamic contextual elements
        3. Few-Shot Learning - Includes relevant examples
        4. Persona Variation - Applies different perspectives
        
        Args:
            synthesis_type: Type of synthesis (e.g., 'phone_transcript')
            category: Category identifier (e.g., 'authority_scam')
            diversity_config: Configuration for diversity techniques
            
        Returns:
            Enhanced prompt string with diversity techniques applied
        """
        if diversity_config is None:
            diversity_config = DiversityConfig()
        
        try:
            # Start with base system prompt
            system_prompt = self.prompts_manager.get_system_prompt(synthesis_type)
            
            # Get base category information
            category_info = self.prompts_manager.get_category_info(synthesis_type, category)
            base_category_prompt = self.prompts_manager.get_prompt_for_category(synthesis_type, category)
            
            # Apply diversity techniques based on configuration
            enhanced_prompt = self._build_enhanced_prompt(
                system_prompt=system_prompt,
                category_info=category_info,
                base_prompt=base_category_prompt,
                synthesis_type=synthesis_type,
                category=category,
                config=diversity_config
            )
            
            logger.debug(f"Generated diverse prompt for {synthesis_type}.{category}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error creating diverse prompt: {e}")
            # Fallback to original prompt generation
            return self.prompts_manager.create_generation_prompt(synthesis_type, category)
    
    def _build_enhanced_prompt(self, system_prompt: str, category_info: Dict[str, Any],
                             base_prompt: str, synthesis_type: str, category: str,
                             config: DiversityConfig) -> str:
        """Build enhanced prompt with diversity techniques."""
        
        # 1. Template Variation
        if config.template_variation:
            base_prompt = self._apply_template_variation(base_prompt, synthesis_type, category)
        
        # 2. Context Injection
        contextual_elements = []
        if config.context_injection:
            contextual_elements = self._generate_contextual_elements()
        
        # 3. Persona Variation
        persona_context = ""
        if config.persona_variation:
            persona_context = self._apply_persona_variation(synthesis_type, category)
        
        # 4. Few-Shot Learning
        few_shot_examples = ""
        if config.few_shot_learning:
            few_shot_examples = self._add_few_shot_examples(synthesis_type, category)
        
        # Combine all elements
        prompt_parts = [
            system_prompt,
            "\n=== ENHANCED GENERATION INSTRUCTIONS ===",
            f"Category: {category_info.get('name', category)}",
            f"Expected Classification: {category_info.get('classification', 'UNKNOWN')}"
        ]
        
        # Add contextual elements
        if contextual_elements:
            prompt_parts.append(f"\nCONTEXTUAL SETTING:")
            for element in contextual_elements:
                prompt_parts.append(f"- {element}")
        
        # Add persona context
        if persona_context:
            prompt_parts.append(f"\nPERSPECTIVE GUIDANCE:")
            prompt_parts.append(persona_context)
        
        # Add few-shot examples
        if few_shot_examples:
            prompt_parts.append(f"\nREFERENCE EXAMPLES:")
            prompt_parts.append(few_shot_examples)
        
        # Add enhanced category-specific instructions
        prompt_parts.extend([
            "\nCATEGORY-SPECIFIC INSTRUCTIONS:",
            base_prompt,
            "\n=== FINAL GENERATION INSTRUCTION ===",
            "Generate a realistic, diverse example that follows the guidelines above.",
            "Ensure the output is authentic, believable, and varies from typical patterns.",
            "Include natural variations in language, structure, and approach while maintaining realism."
        ])
        
        return "\n".join(prompt_parts)
    
    def _apply_template_variation(self, base_prompt: str, synthesis_type: str, category: str) -> str:
        """Apply template variation technique."""
        variations = self.template_variations.get(synthesis_type, {}).get(category, [])
        
        if not variations:
            return base_prompt
        
        # Select a random variation approach
        variation_type = random.choice(variations)
        
        # Apply variation-specific modifications
        variation_prompts = {
            "direct_official": "Use direct, official language with formal structure and authoritative tone.",
            "narrative_building": "Build the scenario progressively through storytelling elements.",
            "analytical_structured": "Present information in a logical, step-by-step analytical manner.",
            "conversational_natural": "Use natural, conversational flow with realistic speech patterns.",
            "urgency_focused": "Emphasize time pressure and immediate action requirements.",
            "technical_detailed": "Include specific technical terminology and detailed explanations.",
            "problem_discovery": "Focus on gradually revealing and diagnosing problems.",
            "solution_oriented": "Emphasize helpful problem-solving approach.",
            "fear_inducing": "Use anxiety-provoking language about potential consequences.",
            "helpful_deceptive": "Maintain helpful tone while hiding malicious intent."
        }
        
        variation_instruction = variation_prompts.get(variation_type, "")
        if variation_instruction:
            enhanced_prompt = f"{base_prompt}\n\nSTYLE VARIATION: {variation_instruction}"
            logger.debug(f"Applied {variation_type} variation to {category}")
            return enhanced_prompt
        
        return base_prompt
    
    def _generate_contextual_elements(self) -> List[str]:
        """Generate random contextual elements for injection."""
        elements = []
        
        # Sample from different context pools
        if "demographics" in self.context_pools:
            demographic = random.choice(self.context_pools["demographics"])
            elements.append(f"Target demographic: {demographic}")
        
        if "timeframes" in self.context_pools:
            timeframe = random.choice(self.context_pools["timeframes"])
            elements.append(f"Time context: {timeframe}")
        
        if "emotional_states" in self.context_pools:
            emotional_state = random.choice(self.context_pools["emotional_states"])
            elements.append(f"Victim emotional state: {emotional_state}")
        
        if "technical_levels" in self.context_pools:
            tech_level = random.choice(self.context_pools["technical_levels"])
            elements.append(f"Technical sophistication: {tech_level}")
        
        # Randomly select 2-3 elements to avoid overwhelming the prompt
        return random.sample(elements, min(3, len(elements)))
    
    def _apply_persona_variation(self, synthesis_type: str, category: str) -> str:
        """Apply persona variation based on category type."""
        if "scammer_types" not in self.personas or "victim_types" not in self.personas:
            return ""
        
        scammer_persona = random.choice(self.personas["scammer_types"])
        victim_persona = random.choice(self.personas["victim_types"])
        
        persona_guidance = f"""
Consider these character perspectives:
- Scammer role: {scammer_persona}
- Victim profile: {victim_persona}

Ensure the interaction dynamics reflect these personas realistically.
"""
        
        return persona_guidance.strip()
    
    def _add_few_shot_examples(self, synthesis_type: str, category: str, num_examples: int = 2) -> str:
        """Add few-shot examples for the category."""
        # For now, return placeholder - would be enhanced with actual example pools
        examples_text = f"""
Here are example patterns for {category} (for reference, not to copy directly):

Example 1: Focus on establishing authority and creating urgency
Example 2: Demonstrate gradual escalation and information gathering
Example 3: Show realistic victim responses and hesitations

(Generate something similar in style but different in specific content)
"""
        
        return examples_text.strip()
    
    def get_diversity_statistics(self) -> Dict[str, Any]:
        """Get statistics about available diversity options."""
        stats = {
            "template_variations": {},
            "context_pools_size": {k: len(v) for k, v in self.context_pools.items()},
            "persona_types": {k: len(v) for k, v in self.personas.items()},
            "total_combinations": 0
        }
        
        # Calculate template variations per synthesis type
        for synth_type, categories in self.template_variations.items():
            stats["template_variations"][synth_type] = {
                cat: len(variations) for cat, variations in categories.items()
            }
        
        # Estimate total possible combinations (rough calculation)
        base_combinations = 1
        for pool_size in stats["context_pools_size"].values():
            if pool_size > 0:
                base_combinations *= pool_size
        
        stats["total_combinations"] = base_combinations
        
        return stats


def create_diversity_manager(prompts_manager: SynthesisPromptsManager, 
                           config_path: Optional[str] = None) -> DiversityManager:
    """
    Factory function to create a DiversityManager instance.
    
    Args:
        prompts_manager: Base prompts manager
        config_path: Optional path to diversity configuration
        
    Returns:
        Configured DiversityManager instance
    """
    return DiversityManager(prompts_manager, config_path)