"""
User interface helper functions.
"""

import sys
import logging
from typing import Dict, Any, List
from pathlib import Path
from ..database import get_knowledge_base_service
from ..synthesize import SynthesisPromptsManager

logger = logging.getLogger(__name__)


class UIHelper:
    """Provides UI helper functions for the CLI."""
    
    @staticmethod
    def print_header():
        """Prints the application header."""
        print("="*80)
        print("INTERACTIVE DATASET PROCESSOR WITH CHECKPOINTING")
        print("="*80)
        print("This tool helps you process datasets for scam detection using various LLM providers.")
        print("You'll be guided through each step of the configuration process.")
        print()
    
    @staticmethod
    def choose_task() -> str:
        """
        Interactive task type selection.
        
        Returns:
            Selected task type
        """
        print("\nSTEP 1: Choose Task Type")
        print("-" * 40)
        print("1. Annotation - Generate structured annotations for datasets")
        print("2. Evaluation - Evaluate model performance on datasets")
        print("3. Synthesis - Generate synthetic scam detection data")
        
        while True:
            try:
                choice = int(input("\nSelect task (1-3): ").strip())
                if choice == 1:
                    print("Selected: Annotation")
                    return "annotation"
                elif choice == 2:
                    print("Selected: Evaluation")
                    return "evaluation"
                elif choice == 3:
                    print("Selected: Synthesis")
                    return "synthesis"
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
    
    @staticmethod
    def choose_synthesis_type() -> str:
        """
        Interactive synthesis type selection.
        
        Returns:
            Selected synthesis type
        """
        print("\nSTEP 2: Select Synthesis Type")
        print("-" * 40)
        
        try:
            # Try to get from database first
            kb_service = get_knowledge_base_service()
            types = kb_service.get_all_types()
            
            if types:
                print("Available synthesis types (from database):")
                for i, type_key in enumerate(types, 1):
                    # Get description from first entry of this type
                    entries = kb_service.get_knowledge_by_type(type_key)
                    if entries:
                        desc = entries[0].description
                        print(f"{i}. {type_key}")
                        print(f"   {desc[:80]}..." if len(desc) > 80 else f"   {desc}")
                
                while True:
                    try:
                        choice = int(input(f"\nSelect type (1-{len(types)}): ").strip())
                        if 1 <= choice <= len(types):
                            selected = types[choice - 1]
                            print(f"Selected: {selected}")
                            return selected
                        else:
                            print(f"Please enter a number between 1 and {len(types)}")
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\nOperation cancelled")
                        sys.exit(0)
                        
        except Exception as e:
            logger.info(f"Using JSON configuration: {e}")
        
        # Fallback to JSON-based types
        try:
            prompts_manager = SynthesisPromptsManager(use_database=False)
            types = prompts_manager.get_synthesis_types()
            
            if not types:
                print("No synthesis types available!")
                sys.exit(1)
            
            print("Available synthesis types:")
            for i, type_key in enumerate(types, 1):
                type_info = prompts_manager.get_synthesis_type_info(type_key)
                print(f"{i}. {type_info.get('name', type_key)}")
                print(f"   {type_info.get('description', 'No description')}")
            
            while True:
                try:
                    choice = int(input(f"\nSelect type (1-{len(types)}): ").strip())
                    if 1 <= choice <= len(types):
                        selected = types[choice - 1]
                        print(f"Selected: {selected}")
                        return selected
                    else:
                        print(f"Please enter a number between 1 and {len(types)}")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nOperation cancelled")
                    sys.exit(0)
                    
        except Exception as e:
            logger.error(f"Failed to load synthesis types: {e}")
            print(f"Error loading synthesis types: {e}")
            sys.exit(1)
    
    @staticmethod
    def choose_synthesis_category(synthesis_type: str) -> str:
        """
        Interactive category selection for synthesis.
        
        Args:
            synthesis_type: Selected synthesis type
            
        Returns:
            Selected category or "ALL"
        """
        print(f"\nSelect Category for {synthesis_type}")
        print("-" * 40)
        
        try:
            # Try database first
            kb_service = get_knowledge_base_service()
            categories = kb_service.get_knowledge_by_type(synthesis_type)
            
            if categories:
                # Group by classification
                by_classification = {}
                for cat in categories:
                    classification = cat.classification
                    if classification not in by_classification:
                        by_classification[classification] = []
                    by_classification[classification].append(cat)
                
                # Display categories grouped by classification
                all_categories = []
                for classification in sorted(by_classification.keys()):
                    print(f"\n{classification}:")
                    for cat in by_classification[classification]:
                        all_categories.append(cat)
                        idx = len(all_categories)
                        print(f"{idx}. {cat.name}")
                        desc_preview = cat.description[:80] + "..." if len(cat.description) > 80 else cat.description
                        print(f"    {desc_preview}")
                
                print(f"\n0. Generate ALL categories (mixed dataset)")
                
                while True:
                    try:
                        choice = int(input(f"\nSelect category (0-{len(all_categories)}): ").strip())
                        if choice == 0:
                            print("Selected: ALL categories (will generate a mix)")
                            return "ALL"
                        elif 1 <= choice <= len(all_categories):
                            selected_cat = all_categories[choice - 1]
                            print(f"Selected: {selected_cat.name}")
                            return selected_cat.category
                        else:
                            print(f"Please enter a number between 0 and {len(all_categories)}")
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\nOperation cancelled")
                        sys.exit(0)
                        
        except Exception as e:
            logger.info(f"Using JSON configuration for categories: {e}")
        
        # Fallback to JSON
        try:
            prompts_manager = SynthesisPromptsManager(use_database=False)
            categories = prompts_manager.get_categories(synthesis_type)
            
            if categories:
                print(f"Available categories:")
                cat_list = list(categories.items())
                for i, (cat_key, cat_info) in enumerate(cat_list, 1):
                    print(f"{i}. {cat_info.get('name', cat_key)}")
                    print(f"    Classification: {cat_info.get('classification', 'Unknown')}")
                
                print(f"\n0. Generate ALL categories (mixed dataset)")
                
                while True:
                    try:
                        choice = int(input(f"\nSelect category (0-{len(cat_list)}): ").strip())
                        if choice == 0:
                            print("Selected: ALL categories")
                            return "ALL"
                        elif 1 <= choice <= len(cat_list):
                            selected_key, selected_info = cat_list[choice - 1]
                            print(f"Selected: {selected_info.get('name', selected_key)}")
                            return selected_key
                        else:
                            print(f"Please enter a number between 0 and {len(cat_list)}")
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\nOperation cancelled")
                        sys.exit(0)
            else:
                print("No categories available for this synthesis type.")
                return "ALL"
                
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            print(f"Error loading categories: {e}")
            return "ALL"
    
    @staticmethod
    def print_results_summary(results: Dict[str, Any], task: str):
        """
        Print processing results summary.
        
        Args:
            results: Results dictionary from processing
            task: Task type
        """
        print("\n" + "="*80)
        print("PROCESSING RESULTS")
        print("="*80)
        
        if task == "evaluation":
            # Handle both formats
            if 'metrics' in results:
                # Format from run_full_evaluation
                metrics = results['metrics']
                print(f"\nTotal evaluated: {metrics.get('successfully_processed', 0)}")
                print(f"Correct predictions: {metrics.get('correct_predictions', 0)}")
                print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
                
                if 'save_paths' in results:
                    paths = results['save_paths']
                    if 'detailed_results' in paths:
                        print(f"\nDetailed results: {paths['detailed_results']}")
                    if 'metrics' in paths:
                        print(f"Metrics summary: {paths['metrics']}")
                    if 'summary' in paths:
                        print(f"Summary report: {paths['summary']}")
            elif 'total_evaluated' in results:
                # Format from async checkpoint methods
                print(f"\nTotal evaluated: {results['total_evaluated']}")
                print(f"Correct predictions: {results['correct_predictions']}")
                print(f"Accuracy: {results['accuracy']:.2%}")
                
                if 'detailed_results_path' in results:
                    print(f"\nDetailed results: {results['detailed_results_path']}")
                if 'metrics_path' in results:
                    print(f"Metrics summary: {results['metrics_path']}")
                if 'summary_path' in results:
                    print(f"Summary report: {results['summary_path']}")
                    
        elif task == "annotation":
            # Handle both async and sync result formats
            if 'summary' in results:
                # Sync format from run_full_annotation
                summary = results['summary']
                print(f"\nTotal annotated: {summary.get('total_records', 0)}")
                print(f"Successful: {summary.get('successful_annotations', 0)}")
                print(f"Usable: {summary.get('usable_annotations', 0)}")
                print(f"Success rate: {summary.get('success_rate', 0):.2%}")
                
                if 'save_paths' in results:
                    paths = results['save_paths']
                    if 'annotated_dataset_path' in paths:
                        print(f"\nAnnotated dataset: {paths['annotated_dataset_path']}")
                    if 'summary_path' in paths:
                        print(f"Summary file: {paths['summary_path']}")
            elif 'annotated' in results:
                # Async format
                print(f"\nTotal annotated: {results['annotated']}")
                print(f"Errors: {results.get('errors', 0)}")
                
                if 'annotated_file' in results:
                    print(f"\nAnnotated dataset: {results['annotated_file']}")
                if 'summary_file' in results:
                    print(f"Summary file: {results['summary_file']}")
                    
        elif task == "synthesis":
            if 'results' in results:
                synthesis_results = results['results']
                print(f"\nTotal generated: {synthesis_results.get('total_count', 0)}")
                print(f"Successful: {synthesis_results.get('success_count', 0)}")
                print(f"Errors: {synthesis_results.get('error_count', 0)}")
                
                if 'detailed_results' in synthesis_results:
                    print(f"\nDetailed results: {synthesis_results['detailed_results']}")
                if 'summary_report' in synthesis_results:
                    print(f"Summary report: {synthesis_results['summary_report']}")
                
                if 'mongodb_result' in synthesis_results:
                    mongodb = synthesis_results['mongodb_result']
                    if mongodb.get('success'):
                        print(f"\nMongoDB: {mongodb.get('inserted_count', 0)} records saved")
        
        print("\n" + "="*80)
    
    @staticmethod
    def choose_diversity_settings() -> Dict[str, Any]:
        """
        Interactive diversity enhancement configuration.
        
        Returns:
            Dictionary with diversity settings
        """
        print("\n🎯 Diversity Enhancement Settings")
        print("=" * 40)
        print("Diversity enhancement improves the variety and quality of generated data")
        print("using advanced prompt engineering techniques from MetaSynth and CoDSA research.")
        print()
        
        # Ask if user wants diversity enhancement
        while True:
            try:
                enable_choice = input("Enable diversity enhancement? (y/n) [n]: ").strip().lower()
                if enable_choice in ['', 'n', 'no']:
                    return {'enable_diversity': False}
                elif enable_choice in ['y', 'yes']:
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        # Choose diversity level
        print("\nSelect diversity enhancement level:")
        print("1. Minimal   - Basic template rotation")
        print("2. Medium    - Templates + context + examples (recommended)")
        print("3. High      - All techniques + self-consistency")
        print("4. Maximum   - Multi-agent + advanced techniques (experimental)")
        print()
        
        level_map = {
            1: "minimal",
            2: "medium", 
            3: "high",
            4: "maximum"
        }
        
        while True:
            try:
                choice = input("Enter choice [2]: ").strip()
                if choice == "":
                    level = "medium"
                    break
                
                choice_num = int(choice)
                if choice_num in level_map:
                    level = level_map[choice_num]
                    break
                else:
                    print("Please enter a number between 1-4")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        # Show what techniques will be enabled
        techniques = {
            "minimal": ["Template Variation"],
            "medium": ["Template Variation", "Context Injection", "Few-Shot Learning", "Persona Variation"],
            "high": ["Template Variation", "Context Injection", "Few-Shot Learning", "Persona Variation", "Self-Consistency"],
            "maximum": ["All techniques", "Multi-Agent Generation", "Advanced Meta-Prompting"]
        }
        
        print(f"\n✅ Diversity level '{level}' selected")
        print("Enabled techniques:")
        for technique in techniques[level]:
            print(f"  • {technique}")
        
        return {
            'enable_diversity': True,
            'diversity_level': level
        }
    
    @staticmethod
    def choose_multi_agent_settings() -> Dict[str, Any]:
        """
        Interactive multi-agent collaboration configuration.
        
        Returns:
            Dictionary with multi-agent settings
        """
        print("\n🤖 Multi-Agent Collaboration Settings")
        print("=" * 45)
        print("Multi-agent collaboration uses specialized AI agents that work together")
        print("to generate, evaluate, and refine synthetic data for higher quality.")
        print()
        print("Based on research from:")
        print("• MetaSynth: Meta-Prompting-Driven Agentic Scaffolds")
        print("• AgentSGEN: Multi-Agent LLM Collaboration")
        print("• ConvoGen: Multi-Agent Conversational Data Generation")
        print()
        
        # Ask if user wants multi-agent collaboration
        while True:
            try:
                enable_choice = input("Enable multi-agent collaboration? (y/n) [n]: ").strip().lower()
                if enable_choice in ['', 'n', 'no']:
                    return {'enable_multi_agent': False}
                elif enable_choice in ['y', 'yes']:
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        print("\n🎯 Multi-Agent Configuration:")
        
        # Quality threshold setting
        print("\nQuality threshold (when to activate multi-agent vs standard generation):")
        print("1. Conservative (6.0) - Use multi-agent more often")
        print("2. Balanced (7.0) - Recommended balance")
        print("3. Selective (8.0) - Use multi-agent for challenging content")
        print("4. Minimal (9.0) - Use multi-agent rarely")
        
        threshold_map = {
            1: 6.0,
            2: 7.0,
            3: 8.0, 
            4: 9.0
        }
        
        while True:
            try:
                choice = input("Enter choice [2]: ").strip()
                if choice == "":
                    threshold = 7.0
                    break
                
                choice_num = int(choice)
                if choice_num in threshold_map:
                    threshold = threshold_map[choice_num]
                    break
                else:
                    print("Please enter a number between 1-4")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        # Specialist agents setting
        print("\nEnable specialist agents? (domain experts for scam tactics, authenticity, etc.)")
        while True:
            try:
                specialist_choice = input("Enable specialist agents? (y/n) [y]: ").strip().lower()
                if specialist_choice in ['', 'y', 'yes']:
                    enable_specialists = True
                    break
                elif specialist_choice in ['n', 'no']:
                    enable_specialists = False
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        # Iterative refinement setting
        print("\nEnable iterative refinement? (agents improve content through multiple rounds)")
        while True:
            try:
                refinement_choice = input("Enable iterative refinement? (y/n) [y]: ").strip().lower()
                if refinement_choice in ['', 'y', 'yes']:
                    enable_refinement = True
                    break
                elif refinement_choice in ['n', 'no']:
                    enable_refinement = False
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                sys.exit(0)
        
        # Show configuration summary
        print(f"\n✅ Multi-agent collaboration configured:")
        print(f"  • Quality threshold: {threshold}")
        print(f"  • Specialist agents: {'Enabled' if enable_specialists else 'Disabled'}")
        print(f"  • Iterative refinement: {'Enabled' if enable_refinement else 'Disabled'}")
        print()
        print("Multi-agent process:")
        print("  1. Generator Agent creates initial content")
        print("  2. Evaluator Agent assesses quality and realism")
        if enable_specialists:
            print("  3. Specialist Agents provide domain expertise")
        if enable_refinement:
            print(f"  {'4' if enable_specialists else '3'}. Refiner Agent improves content")
            print(f"  {'5' if enable_specialists else '4'}. Coordinator Agent makes final assessment")
        
        return {
            'enable_multi_agent': True,
            'multi_agent_quality_threshold': threshold,
            'enable_specialist_agents': enable_specialists,
            'enable_iterative_refinement': enable_refinement
        }