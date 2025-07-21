# MongoDB Setup Guide

This guide helps you set up MongoDB integration for the scam detection project.

## Environment Configuration

Create a `.env` file in your project root with the following MongoDB configuration:

```bash
# MongoDB Configuration
# Copy these variables to your .env file and update the values

# MongoDB Host (default: localhost)
MONGODB_HOST=localhost

# MongoDB Port (default: 27017)
MONGODB_PORT=27017

# MongoDB Database Name
MONGODB_DATABASE=scam_detection

# MongoDB Authentication (leave empty for local development without auth)
# MONGODB_USERNAME=your_username
# MONGODB_PASSWORD=your_password

# MongoDB Authentication Source (default: admin)
MONGODB_AUTH_SOURCE=admin

# Connection timeout in milliseconds (default: 5000)
MONGODB_TIMEOUT=5000
```

## Quick Setup Options

### Option 1: Local MongoDB (Recommended for Development)

1. **Install MongoDB locally:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install mongodb
   
   # On macOS with Homebrew
   brew install mongodb-community
   
   # On Windows, download from: https://www.mongodb.com/try/download/community
   ```

2. **Start MongoDB service:**
   ```bash
   # On Ubuntu/Debian
   sudo systemctl start mongod
   
   # On macOS
   brew services start mongodb-community
   
   # On Windows, MongoDB should start automatically after installation
   ```

3. **Create your .env file with local settings:**
   ```bash
   MONGODB_HOST=localhost
   MONGODB_PORT=27017
   MONGODB_DATABASE=scam_detection
   ```

### Option 2: MongoDB Atlas (Cloud)

1. **Create a MongoDB Atlas account:** https://www.mongodb.com/cloud/atlas
2. **Create a new cluster and get connection details**
3. **Update your .env file:**
   ```bash
   MONGODB_HOST=cluster0.abcd123.mongodb.net
   MONGODB_USERNAME=your_atlas_username
   MONGODB_PASSWORD=your_atlas_password
   MONGODB_DATABASE=scam_detection
   ```

### Option 3: Docker MongoDB

```bash
# Run MongoDB in Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Your .env file:
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=scam_detection
```

## Testing Your Setup

After setting up MongoDB, test the connection by running:

```python
from src.database import test_connection
if test_connection():
    print("✅ MongoDB connection successful!")
else:
    print("❌ MongoDB connection failed!")
```

## Collections Created

The system will automatically create the following collections:

- `phone_scams` - Phone transcript scam data
- `email_scams` - Email phishing scam data  
- `sms_scams` - SMS scam data

## Data Schema

Each document will contain:
- `id` - Unique identifier
- `synthesis_type` - Type of scam data
- `classification` - Scam classification
- `category` - Specific category
- `generation_timestamp` - When generated
- `storage_timestamp` - When stored
- Plus all the synthesized content fields

The system automatically creates indexes for optimal performance. 