#!/usr/bin/env python3
"""
Test MongoDB Integration

This script tests the MongoDB integration for the scam detection project.
Run this to verify that your MongoDB setup is working correctly.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_mongodb_connection():
    """Test basic MongoDB connection."""
    print("🔍 Testing MongoDB Connection...")
    
    try:
        from src.database import test_connection
        if test_connection():
            print("✅ MongoDB connection successful!")
            return True
        else:
            print("❌ MongoDB connection failed!")
            print("💡 Make sure MongoDB is running and check your .env configuration")
            return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        print("💡 Make sure you have installed pymongo: pip install pymongo")
        return False

def test_scam_data_service():
    """Test the scam data service."""
    print("\n🔍 Testing Scam Data Service...")
    
    try:
        from src.database import get_scam_data_service
        
        service = get_scam_data_service()
        
        # Test with sample data
        sample_data = [
            {
                'id': 1,
                'synthesis_type': 'test',
                'classification': 'TEST',
                'category': 'test_category',
                'content': 'This is a test message',
                'generation_timestamp': datetime.now().isoformat()
            }
        ]
        
        result = service.store_synthesis_results('test', sample_data)
        
        if result.get('success'):
            print(f"✅ Successfully stored test data!")
            print(f"   Collection: {result['collection']}")
            print(f"   Documents: {result['inserted_count']}")
            
            # Test retrieval
            retrieved = service.get_scam_data('test', limit=1)
            if retrieved:
                print("✅ Successfully retrieved test data!")
            else:
                print("⚠️ Could not retrieve test data")
                
            return True
        else:
            print(f"❌ Failed to store test data: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing scam data service: {e}")
        return False

def test_collection_stats():
    """Test getting collection statistics."""
    print("\n🔍 Testing Collection Statistics...")
    
    try:
        from src.database import get_scam_data_service
        
        service = get_scam_data_service()
        stats = service.get_collection_stats()
        
        print("📊 Collection Statistics:")
        if stats:
            for collection, data in stats.items():
                print(f"   {collection}: {data.get('total_documents', 0)} documents")
        else:
            print("   No data found (this is normal for a fresh setup)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return False

def show_configuration():
    """Show current MongoDB configuration."""
    print("\n⚙️ Current MongoDB Configuration:")
    
    config_vars = [
        'MONGODB_HOST', 'MONGODB_PORT', 'MONGODB_DATABASE',
        'MONGODB_USERNAME', 'MONGODB_PASSWORD'
    ]
    
    for var in config_vars:
        value = os.getenv(var, 'Not set')
        if 'PASSWORD' in var and value != 'Not set':
            value = '*' * len(value)  # Hide password
        print(f"   {var}: {value}")

def main():
    """Run all tests."""
    print("🧪 MongoDB Integration Test")
    print("=" * 50)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment")
    except Exception as e:
        print(f"⚠️ Could not load .env file: {e}")
    
    show_configuration()
    
    # Run tests
    tests = [
        test_mongodb_connection,
        test_scam_data_service,
        test_collection_stats
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n🎉 MongoDB integration is working correctly!")
        print("You can now run synthesis and your data will be saved to MongoDB.")
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("\n💡 Troubleshooting:")
        print("1. Make sure MongoDB is running")
        print("2. Check your .env configuration")
        print("3. Verify network connectivity")
        print("4. See mongodb_setup.md for detailed setup instructions")

if __name__ == "__main__":
    main() 