#!/usr/bin/env python3
"""
Diversity Enhancement Example Script

This script demonstrates the new diversity enhancement features for synthetic 
scam data generation. It shows how to use the DiversityManager and enhanced 
SynthesisGenerator to create more varied and realistic synthetic data.

Based on research from:
- MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation
- CoDSA: Conditional Data Synthesis Augmentation  
- Advanced prompt engineering techniques for LLMs

Usage:
    python examples/diversity_example.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.synthesize import SynthesisGenerator, DiversityManager, DiversityConfig, DiversityLevel


async def demonstrate_diversity_features():
    """Demonstrate the diversity enhancement features."""
    
    print("🎯 Diversity Enhancement Demo")
    print("=" * 60)
    print("This demo shows how diversity techniques improve synthetic data generation.")
    print()
    
    # Example 1: Standard Generation vs Diversity Enhanced
    print("📊 Example 1: Standard vs Enhanced Generation")
    print("-" * 40)
    
    # Standard generation
    print("🔹 Standard Generation:")
    standard_generator = SynthesisGenerator(
        synthesis_type="phone_transcript",
        sample_size=2,
        provider="openai",
        model="gpt-4o-mini",
        enable_diversity=False,
        category="authority_scam"
    )
    
    try:
        # Generate standard prompts
        standard_prompt = standard_generator.create_generation_prompt("authority_scam")
        print("Standard prompt structure:")
        print(f"Length: {len(standard_prompt)} characters")
        print("First 200 chars:", standard_prompt[:200], "...\n")
        
    except Exception as e:
        print(f"Note: Standard generation requires API keys: {e}\n")
    
    # Enhanced generation
    print("🔹 Diversity Enhanced Generation:")
    enhanced_generator = SynthesisGenerator(
        synthesis_type="phone_transcript",
        sample_size=2,
        provider="openai", 
        model="gpt-4o-mini",
        enable_diversity=True,
        diversity_level="medium",
        category="authority_scam"
    )
    
    try:
        # Generate enhanced prompts (multiple variations)
        print("Generating 3 diverse prompt variations...")
        for i in range(3):
            enhanced_prompt = enhanced_generator.create_generation_prompt("authority_scam")
            print(f"\nVariation {i+1}:")
            print(f"Length: {len(enhanced_prompt)} characters")
            print("First 200 chars:", enhanced_prompt[:200], "...")
            
            # Show diversity indicators
            diversity_indicators = []
            if "CONTEXTUAL SETTING" in enhanced_prompt:
                diversity_indicators.append("Context Injection")
            if "PERSPECTIVE GUIDANCE" in enhanced_prompt:
                diversity_indicators.append("Persona Variation")
            if "REFERENCE EXAMPLES" in enhanced_prompt:
                diversity_indicators.append("Few-Shot Learning")
            if "STYLE VARIATION" in enhanced_prompt:
                diversity_indicators.append("Template Variation")
            
            print(f"Applied techniques: {', '.join(diversity_indicators)}")
        
    except Exception as e:
        print(f"Note: Enhanced generation requires API keys: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 2: Different Diversity Levels
    print("📈 Example 2: Diversity Levels Comparison")
    print("-" * 40)
    
    levels = ["minimal", "medium", "high", "maximum"]
    
    for level in levels:
        print(f"\n🔸 {level.upper()} Level:")
        
        try:
            level_generator = SynthesisGenerator(
                synthesis_type="phone_transcript",
                sample_size=1,
                provider="openai",
                model="gpt-4o-mini", 
                enable_diversity=True,
                diversity_level=level,
                category="tech_support_scam"
            )
            
            prompt = level_generator.create_generation_prompt("tech_support_scam")
            
            # Analyze prompt features
            features = {
                "Template Variation": "STYLE VARIATION" in prompt,
                "Context Injection": "CONTEXTUAL SETTING" in prompt,
                "Few-Shot Learning": "REFERENCE EXAMPLES" in prompt,
                "Persona Variation": "PERSPECTIVE GUIDANCE" in prompt,
                "Enhanced Instructions": "ENHANCED GENERATION INSTRUCTIONS" in prompt
            }
            
            enabled_features = [name for name, enabled in features.items() if enabled]
            print(f"Enabled features: {', '.join(enabled_features)}")
            print(f"Prompt length: {len(prompt)} characters")
            
        except Exception as e:
            print(f"Error generating prompt: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Direct Diversity Manager Usage
    print("🔧 Example 3: Direct Diversity Manager Usage")
    print("-" * 40)
    
    try:
        from src.synthesize.synthesis_prompts import SynthesisPromptsManager
        
        # Create prompts manager and diversity manager
        prompts_manager = SynthesisPromptsManager()
        diversity_manager = DiversityManager(prompts_manager)
        
        # Get diversity statistics
        stats = diversity_manager.get_diversity_statistics()
        print("📊 Diversity Configuration Statistics:")
        print(f"Template variations available: {stats['template_variations']}")
        print(f"Context pool sizes: {stats['context_pools_size']}")
        print(f"Persona types: {stats['persona_types']}")
        print(f"Estimated combinations: {stats['total_combinations']:,}")
        
        # Generate diverse prompts for different categories
        print(f"\n🎭 Prompt Variations for Different Categories:")
        
        categories = ["authority_scam", "tech_support_scam", "financial_fraud"]
        
        for category in categories:
            print(f"\n--- {category} ---")
            
            # Create diversity config
            config = DiversityConfig(
                level=DiversityLevel.MEDIUM,
                template_variation=True,
                context_injection=True,
                few_shot_learning=True,
                persona_variation=True
            )
            
            try:
                prompt = diversity_manager.create_diverse_generation_prompt(
                    "phone_transcript", category, config
                )
                
                # Extract and show applied techniques
                techniques_used = []
                if "STYLE VARIATION" in prompt:
                    # Extract the style variation
                    style_start = prompt.find("STYLE VARIATION:")
                    if style_start != -1:
                        style_end = prompt.find("\n", style_start)
                        style_info = prompt[style_start:style_end] if style_end != -1 else prompt[style_start:style_start+100]
                        techniques_used.append(style_info.replace("STYLE VARIATION: ", "Style: "))
                
                if "Target demographic:" in prompt:
                    # Extract demographic context
                    demo_start = prompt.find("Target demographic:")
                    if demo_start != -1:
                        demo_end = prompt.find("\n", demo_start)
                        demo_info = prompt[demo_start:demo_end] if demo_end != -1 else prompt[demo_start:demo_start+100]
                        techniques_used.append(demo_info)
                
                print(f"Applied techniques: {'; '.join(techniques_used[:2])}")  # Show first 2
                print(f"Total prompt length: {len(prompt)} characters")
                
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"Error initializing diversity manager: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Diversity Enhancement Demo Complete!")
    print("\nTo use diversity enhancement in your synthesis:")
    print("1. Run: python main.py")
    print("2. Choose 'Synthesis'")
    print("3. Select your synthesis type and category")
    print("4. When prompted, enable diversity enhancement")
    print("5. Choose your preferred diversity level")
    print("\nFor more details, see: config/diversity_config.json")


def show_configuration_options():
    """Show available configuration options."""
    print("\n🔧 Configuration Options")
    print("=" * 30)
    
    print("Diversity Levels:")
    print("- minimal: Basic template rotation")
    print("- medium: Templates + context + examples (recommended)")
    print("- high: All techniques + self-consistency")
    print("- maximum: Multi-agent + advanced techniques")
    
    print("\nTechniques Available:")
    print("✓ Template Variation - Multiple prompt styles per category")
    print("✓ Context Injection - Dynamic demographic/situational context")
    print("✓ Few-Shot Learning - Rotating example sets")
    print("✓ Persona Variation - Different character perspectives")
    print("✓ Self-Consistency - Multiple generation + selection (high/max levels)")
    
    print("\nConfiguration Files:")
    print("- config/diversity_config.json - Main diversity settings")
    print("- config/synthesis_config.json - Base synthesis configuration")
    

if __name__ == "__main__":
    print("🚀 Starting Diversity Enhancement Demo...")
    print("This demo showcases new synthetic data diversity features.\n")
    
    try:
        # Run the main demonstration
        asyncio.run(demonstrate_diversity_features())
        
        # Show configuration info
        show_configuration_options()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Note: Some features require valid API keys in .env file")
    
    print("\nThank you for trying the diversity enhancement features! 🎯")