#!/usr/bin/env python3
"""
Example: Using the Scam Knowledge Base

This script demonstrates how to use the new MongoDB knowledge base
for managing prompts and scam information.
"""

from src.database import get_knowledge_base_service, ScamKnowledge
from src.synthesize import SynthesisPromptsManager
from datetime import datetime


def example_1_basic_queries():
    """Example 1: Basic CRUD operations with the knowledge base"""
    print("=== Example 1: Basic Knowledge Base Operations ===\n")
    
    kb_service = get_knowledge_base_service()
    
    # 1. Get all scam types
    print("1. Getting all scam types:")
    types = kb_service.get_all_types()
    print(f"   Found {len(types)} types: {', '.join(types)}\n")
    
    # 2. Get all categories for a specific type
    print("2. Getting categories for 'phone_transcript':")
    categories = kb_service.get_categories_for_type("phone_transcript")
    for cat in categories[:3]:  # Show first 3
        print(f"   - {cat['category']}: {cat['name']} ({cat['classification']})")
    print(f"   ... and {len(categories) - 3} more\n")
    
    # 3. Get a specific knowledge entry
    print("3. Getting specific knowledge entry:")
    knowledge = kb_service.get_knowledge("phone_transcript.tech_support_scam")
    if knowledge:
        print(f"   ID: {knowledge.id}")
        print(f"   Name: {knowledge.name}")
        print(f"   Classification: {knowledge.classification}")
        print(f"   Prompt preview: {knowledge.prompt[:100]}...\n")
    
    # 4. Search by classification
    print("4. Finding all legitimate phone call templates:")
    legitimate = kb_service.get_knowledge_by_type("phone_transcript", classification="LEGITIMATE")
    print(f"   Found {len(legitimate)} legitimate templates:")
    for item in legitimate:
        print(f"   - {item.name}")
    print()


def example_2_prompt_management():
    """Example 2: Using the enhanced SynthesisPromptsManager"""
    print("=== Example 2: Database-Backed Prompt Management ===\n")
    
    # Initialize with database support
    prompts_manager = SynthesisPromptsManager(use_database=True)
    
    # 1. Get prompt for a specific category
    print("1. Getting prompt from database:")
    prompt = prompts_manager.get_prompt_for_category("phone_transcript", "authority_scam")
    print(f"   Prompt length: {len(prompt)} characters")
    print(f"   Preview: {prompt[:150]}...\n")
    
    # 2. Create complete generation prompt
    print("2. Creating complete generation prompt:")
    full_prompt = prompts_manager.create_generation_prompt("phone_transcript", "tech_support_scam")
    print(f"   Full prompt includes system prompt + category instructions")
    print(f"   Total length: {len(full_prompt)} characters\n")
    
    # 3. Database fallback demonstration
    print("3. Database fallback (if MongoDB is down):")
    prompts_manager_json = SynthesisPromptsManager(use_database=False)
    types = prompts_manager_json.get_synthesis_types()
    print(f"   JSON fallback still works: {len(types)} types available\n")


def example_3_add_custom_knowledge():
    """Example 3: Adding custom scam knowledge"""
    print("=== Example 3: Adding Custom Knowledge ===\n")
    
    kb_service = get_knowledge_base_service()
    
    # Create a new custom scam type
    custom_knowledge = ScamKnowledge(
        id="phone_transcript.crypto_investment_scam",
        type="phone_transcript",
        category="crypto_investment_scam",
        name="Cryptocurrency Investment Scam",
        description="Scammers promoting fake cryptocurrency investment opportunities",
        classification="OBVIOUS_SCAM",
        prompt="""Generate a phone conversation where the caller promotes a fake cryptocurrency investment opportunity.

SCENARIOS TO INCLUDE:
- Claims of guaranteed high returns (200-500% in weeks)
- Pressure to invest immediately before 'opportunity closes'
- Name-dropping famous investors or celebrities
- Technical jargon to sound legitimate
- Request for immediate wire transfer or crypto payment

Make the conversation realistic with natural dialogue flow.""",
        tags=["crypto", "investment", "financial", "scam"],
        is_active=True
    )
    
    # Add to knowledge base
    result = kb_service.create_knowledge(custom_knowledge)
    if result:
        print(f"✅ Successfully added custom knowledge: {result}")
        
        # Verify it was added
        retrieved = kb_service.get_knowledge("phone_transcript.crypto_investment_scam")
        if retrieved:
            print(f"   Verified: {retrieved.name} is now in the knowledge base\n")
    else:
        print("❌ Failed to add custom knowledge (may already exist)\n")


def example_4_analytics_queries():
    """Example 4: Analytics and statistics"""
    print("=== Example 4: Knowledge Base Analytics ===\n")
    
    kb_service = get_knowledge_base_service()
    
    # Get statistics
    stats = kb_service.get_stats()
    
    print("Knowledge Base Statistics:")
    print(f"  Total entries: {stats.get('total_knowledge', 0)}")
    print(f"  Active entries: {stats.get('active_knowledge', 0)}")
    
    print("\nBy Type:")
    for type_name, type_info in stats.get('by_type', {}).items():
        print(f"  - {type_name}: {type_info.get('count', 0)} entries")
    
    print("\nBy Classification:")
    for classification, count in stats.get('by_classification', {}).items():
        print(f"  - {classification}: {count} entries")
    print()


def example_5_search_and_filter():
    """Example 5: Advanced search and filtering"""
    print("=== Example 5: Search and Filter ===\n")
    
    kb_service = get_knowledge_base_service()
    
    # 1. Search by tags
    print("1. Searching by tags ['authority', 'government']:")
    results = kb_service.search_by_tags(['authority', 'government'])
    print(f"   Found {len(results)} entries with these tags\n")
    
    # 2. Get all scam entries (exclude legitimate)
    print("2. Getting all phone scam entries (excluding legitimate):")
    scam_types = ["OBVIOUS_SCAM", "SUBTLE_SCAM", "BORDERLINE_SUSPICIOUS"]
    all_scams = []
    for scam_class in scam_types:
        entries = kb_service.get_knowledge_by_type("phone_transcript", classification=scam_class)
        all_scams.extend(entries)
    print(f"   Found {len(all_scams)} scam templates\n")
    
    # 3. Update knowledge entry
    print("3. Updating a knowledge entry:")
    success = kb_service.update_knowledge(
        "phone_transcript.tech_support_scam",
        {"tags": ["tech", "support", "computer", "virus", "remote_access"]}
    )
    if success:
        print(" Successfully updated tags for tech support scam\n")


if __name__ == "__main__":
    print("\nScam Knowledge Base Examples\n")
    print("This script demonstrates various ways to use the new MongoDB-backed")
    print("knowledge base for managing scam detection prompts and templates.\n")
    
    try:
        # Run examples
        example_1_basic_queries()
        example_2_prompt_management()
        example_3_add_custom_knowledge()
        example_4_analytics_queries()
        example_5_search_and_filter()
        
        print("\n All examples completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python migrate_to_knowledge_base.py' to populate the database")
        print("2. Use the knowledge base in your synthesis pipeline")
        print("3. Add new scam types and categories as needed")
        print("4. Query generated data by knowledge_id to analyze prompt effectiveness")
        
    except Exception as e:
        print(f"\n Error running examples: {e}")
        print("\nMake sure:")
        print("1. MongoDB is running (docker-compose up -d)")
        print("2. You've run the migration script")
        print("3. Your .env file has MongoDB configuration")