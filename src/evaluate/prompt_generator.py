from typing import List, Dict, Any

class PromptGenerator:
    """
    Generate prompts dynamically based on available dataset features.
    Adapts to different datasets for scam detection across various content types.
    """
    
    def __init__(self, features: List[str], content_columns: List[str] = None):
        self.features = features
        self.content_columns = content_columns or features  # Use all features if not specified
        
    def get_system_prompt(self) -> str:
        """Generate system prompt for scam detection"""
        return """
You are an expert analyst specializing in scam detection across various types of content.
Your task is to analyze text content and determine if it's a scam or legitimate.

Guidelines:
- Scam content typically contains deceptive material designed to steal credentials, personal information, or money
- Look for suspicious indicators like urgent language, suspicious requests, grammatical errors, impersonation attempts, false promises
- Legitimate content is from real sources and doesn't attempt to deceive recipients
- Consider the context and type of content (email, message, conversation, etc.)

Respond with:
- Phishing: true if the content is a scam, false if legitimate
- Reason: Brief explanation of your decision
"""
    
    def create_user_prompt(self, row: Dict[str, Any]) -> str:
        """
        Create user prompt dynamically based on available features in the dataset row.
        Adapts to different dataset structures and content types.
        """
        prompt_parts = ["Please analyze the following content and determine if it's a scam or legitimate:\n"]
        
        # Handle content features
        for feature in self.content_columns:
            if feature in row:
                value = row.get(feature, "")
                if value and str(value).strip():  # Only include non-empty values
                    prompt_parts.append(f"{feature}: {str(value).strip()}\n")
        
        prompt_parts.append("\nAnalyze this content and provide your assessment.")
        
        return "".join(prompt_parts)
    
    def get_features_summary(self) -> str:
        """Get a summary of features that will be used in prompts"""
        return f"Content features used for evaluation: {', '.join(self.content_columns)}" 