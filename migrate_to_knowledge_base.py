#!/usr/bin/env python3
"""
Migration Script: JSON Config to MongoDB Knowledge Base

This script migrates the existing synthesis configuration from JSON
to the MongoDB knowledge base for easier management and querying.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.database.knowledge_base_models import ScamKnowledge
from src.database.knowledge_base_service import get_knowledge_base_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_config(config_path: str = "config/synthesis_config.json") -> Dict[str, Any]:
    """
    Load the existing JSON configuration.
    
    Args:
        config_path: Path to the JSON config file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON config: {e}")
        raise


def migrate_synthesis_config():
    """
    Migrate synthesis configuration from JSON to MongoDB.
    """
    print("🚀 Starting migration from JSON to MongoDB Knowledge Base")
    print("=" * 60)
    
    # Load JSON configuration
    try:
        config = load_json_config()
        logger.info("✅ Loaded JSON configuration")
    except Exception as e:
        print(f"❌ Failed to load JSON config: {e}")
        return
    
    # Get knowledge base service
    kb_service = get_knowledge_base_service()
    
    # Track migration stats
    total_entries = 0
    successful_entries = 0
    failed_entries = 0
    
    # Process each synthesis type
    for type_key, type_config in config.get("synthesis_types", {}).items():
        type_name = type_config.get("name", type_key)
        type_description = type_config.get("description", "")
        system_prompt = type_config.get("system_prompt", "")
        
        print(f"\n📁 Processing type: {type_name}")
        print(f"   Description: {type_description}")
        
        # Process each category within the type
        categories = type_config.get("categories", {})
        for category_key, category_config in categories.items():
            total_entries += 1
            
            # Create ScamKnowledge instance
            knowledge = ScamKnowledge(
                id=f"{type_key}.{category_key}",
                type=type_key,
                category=category_key,
                name=category_config.get("name", category_key),
                description=category_config.get("description", 
                           f"{category_config.get('name', category_key)} for {type_name}"),
                classification=category_config.get("classification", "UNKNOWN"),
                prompt=category_config.get("prompt_template", ""),
                system_prompt=system_prompt if system_prompt else None,
                tags=[type_key, category_key, category_config.get("classification", "").lower()],
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={
                    "migrated_from": "synthesis_config.json",
                    "migration_date": datetime.utcnow().isoformat(),
                    "original_config": {
                        "llm_response_schema": type_config.get("llm_response_schema", {}),
                        "metadata_schema": type_config.get("metadata_schema", {})
                    }
                }
            )
            
            # Try to create the knowledge entry
            result = kb_service.create_knowledge(knowledge)
            if result:
                successful_entries += 1
                print(f"   ✅ Migrated: {knowledge.id} - {knowledge.name}")
            else:
                failed_entries += 1
                print(f"   ❌ Failed: {knowledge.id} - {knowledge.name}")
    
    # Print migration summary
    print("\n" + "=" * 60)
    print("📊 Migration Summary:")
    print(f"   Total entries: {total_entries}")
    print(f"   ✅ Successful: {successful_entries}")
    print(f"   ❌ Failed: {failed_entries}")
    
    # Get and display stats
    stats = kb_service.get_stats()
    if stats:
        print("\n📈 Knowledge Base Statistics:")
        print(f"   Total knowledge entries: {stats.get('total_knowledge', 0)}")
        print(f"   Active entries: {stats.get('active_knowledge', 0)}")
        
        print("\n   By Type:")
        for type_name, type_stats in stats.get('by_type', {}).items():
            print(f"     - {type_name}: {type_stats.get('count', 0)} total, "
                  f"{type_stats.get('active_count', 0)} active")
        
        print("\n   By Classification:")
        for classification, count in stats.get('by_classification', {}).items():
            print(f"     - {classification}: {count}")
    
    print("\n✨ Migration completed!")
    
    # Provide example queries
    print("\n📚 Example queries you can now run:")
    print("   - Get all phone scam prompts:")
    print("     kb_service.get_knowledge_by_type('phone_transcript')")
    print("   - Get a specific prompt:")
    print("     kb_service.get_knowledge('phone_transcript.tech_support_scam')")
    print("   - Search by tags:")
    print("     kb_service.search_by_tags(['scam', 'phishing'])")


def verify_migration():
    """
    Verify that the migration was successful by running some test queries.
    """
    print("\n🔍 Verifying migration...")
    kb_service = get_knowledge_base_service()
    
    # Test 1: Get all types
    types = kb_service.get_all_types()
    print(f"\n✅ Found {len(types)} scam types: {', '.join(types)}")
    
    # Test 2: Get a specific knowledge entry
    test_id = "phone_transcript.tech_support_scam"
    knowledge = kb_service.get_knowledge(test_id)
    if knowledge:
        print(f"\n✅ Successfully retrieved '{test_id}':")
        print(f"   Name: {knowledge.name}")
        print(f"   Classification: {knowledge.classification}")
        print(f"   Prompt length: {len(knowledge.prompt)} characters")
    else:
        print(f"\n❌ Could not retrieve '{test_id}'")
    
    # Test 3: Search by classification
    scam_entries = kb_service.get_knowledge_by_type("phone_transcript", classification="OBVIOUS_SCAM")
    print(f"\n✅ Found {len(scam_entries)} obvious scam entries for phone transcripts")
    
    print("\n✨ Verification complete!")


if __name__ == "__main__":
    try:
        # Run migration
        migrate_synthesis_config()
        
        # Verify the migration
        verify_migration()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"\n❌ Migration failed: {e}")