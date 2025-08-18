from typing import Any, Dict, Optional, Type, Union, Tuple
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import asyncio
import re
import logging

# Configure logging for token usage
logger = logging.getLogger(__name__)

# Default response schema for backward compatibility
class response_schema(BaseModel):
    Phishing: bool
    Reason: str


def create_prompt_template(system_prompt: str, user_prompt: str) -> ChatPromptTemplate:
    """Create a standardized prompt template."""
    template = ChatPromptTemplate([
        ("system", "{system_prompt}"),
        ("user", "{user_prompt}")
    ])
    return template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})

def extract_token_usage(response: Any) -> Dict[str, Any]:
    """Extract token usage information from a response object.
    
    Args:
        response: Response object from LangChain
        
    Returns:
        Dictionary containing comprehensive token usage information
    """
    token_info = {}
    
    # Try to get usage_metadata (primary source for Response API)
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        if isinstance(usage, dict):
            token_info['input_tokens'] = usage.get('input_tokens', 0)
            token_info['output_tokens'] = usage.get('output_tokens', 0)
            token_info['total_tokens'] = usage.get('total_tokens', 0)
            
            # Extract cached tokens from input_token_details (Response API format)
            if 'input_token_details' in usage:
                details = usage['input_token_details']
                if 'cache_read' in details:
                    token_info['cached_tokens'] = details.get('cache_read', 0)
                if 'audio' in details:
                    token_info['audio_input_tokens'] = details.get('audio', 0)
            
            # Extract output token details
            if 'output_token_details' in usage:
                details = usage['output_token_details']
                if 'reasoning' in details:
                    token_info['reasoning_tokens'] = details.get('reasoning', 0)
                if 'audio' in details:
                    token_info['audio_output_tokens'] = details.get('audio', 0)
    
    # Also check response_metadata for additional details (Standard API)
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if isinstance(metadata, dict) and 'token_usage' in metadata:
            token_usage = metadata['token_usage']
            
            # Fallback if usage_metadata was not available
            if 'input_tokens' not in token_info:
                token_info['input_tokens'] = token_usage.get('prompt_tokens', 0)
                token_info['output_tokens'] = token_usage.get('completion_tokens', 0)
                token_info['total_tokens'] = token_usage.get('total_tokens', 0)
            
            # Extract prompt token details (Standard API format)
            if 'prompt_tokens_details' in token_usage:
                details = token_usage['prompt_tokens_details']
                if 'cached_tokens' in details:
                    # Prefer this over cache_read if both exist
                    token_info['cached_tokens'] = details.get('cached_tokens', 0)
                if 'audio_tokens' in details:
                    token_info['audio_input_tokens'] = details.get('audio_tokens', 0)
            
            # Extract completion token details (Standard API format)
            if 'completion_tokens_details' in token_usage:
                details = token_usage['completion_tokens_details']
                if 'reasoning_tokens' in details:
                    token_info['reasoning_tokens'] = details.get('reasoning_tokens', 0)
                if 'accepted_prediction_tokens' in details:
                    token_info['accepted_prediction_tokens'] = details.get('accepted_prediction_tokens', 0)
                if 'rejected_prediction_tokens' in details:
                    token_info['rejected_prediction_tokens'] = details.get('rejected_prediction_tokens', 0)
                if 'audio_tokens' in details:
                    token_info['audio_output_tokens'] = details.get('audio_tokens', 0)
    
    return token_info

def log_token_usage(token_info: Dict[str, Any], model_name: str = None, operation: str = None, verbose: bool = False):
    """Log token usage information.
    
    Args:
        token_info: Dictionary containing token usage
        model_name: Name of the model used
        operation: Description of the operation
        verbose: If True, log to console. If False, silent tracking only.
    """
    if not token_info:
        return
    
    if verbose:
        log_msg = f"Token Usage"
        if model_name:
            log_msg += f" [{model_name}]"
        if operation:
            log_msg += f" - {operation}"
        
        log_msg += f": Input={token_info.get('input_tokens', 0)}, "
        log_msg += f"Output={token_info.get('output_tokens', 0)}, "
        log_msg += f"Total={token_info.get('total_tokens', 0)}"
        
        if 'reasoning_tokens' in token_info:
            log_msg += f", Reasoning={token_info['reasoning_tokens']}"
        
        logger.info(log_msg)


async def make_api_call(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        return_token_usage: bool = False,
    ) -> Union[Dict[str, Any], BaseModel, str, Tuple[Any, Dict[str, Any]]]:
        """
        Make an async API call to an LLM with flexible response handling.
        
        Args:
            llm: The LLM instance
            system_prompt: System prompt
            user_prompt: User prompt
            response_schema: Optional Pydantic schema for structured output
            return_token_usage: If True, returns tuple of (response, token_usage)
            
        Returns:
            Structured response, dict, or string depending on configuration
            If return_token_usage=True, returns tuple of (response, token_usage)
        
        Note:
            When a model doesn't support native structured output, the system automatically
            falls back to using gpt-4.1-nano for parsing the response into the desired schema.
        """
        # Enable stream_usage for OpenAI models to get token counts
        if hasattr(llm, 'model_name') and hasattr(llm, 'stream_usage'):
            llm.stream_usage = True
        
        messages = create_prompt_template(system_prompt, user_prompt)
        
        # Get model name for logging
        model_name = getattr(llm, 'model_name', 'unknown')
        
        # Case 1: No schema requested, return raw content
        if response_schema is None:
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage (silent by default)
            token_info = extract_token_usage(response)
            
            if return_token_usage:
                return content, token_info
            return content
        
        # Case 2: Native structured output
        try:
            # Use include_raw=True to get token usage with structured output
            client = llm.with_structured_output(response_schema, include_raw=True)
            response_with_raw = await client.ainvoke(messages)
            
            token_info = {}
            
            # Extract the parsed response and raw response
            if isinstance(response_with_raw, dict) and 'raw' in response_with_raw:
                response = response_with_raw.get('parsed', response_with_raw)
                raw_response = response_with_raw['raw']
                token_info = extract_token_usage(raw_response)
            else:
                # Fallback to direct response if include_raw didn't work
                client = llm.with_structured_output(response_schema)
                response = await client.ainvoke(messages)
            
            # Silent token tracking
            
            if return_token_usage:
                return response, token_info
            return response
            
        except Exception as e:
            # Fallback to structure model
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage
            token_info = extract_token_usage(response)
            
            
            # Fallback: try to parse the unstructured response  
            # If the model doesn't support structured output natively,
            # we'll attempt to parse the JSON response manually
            import json
            try:
                # Try to extract JSON from the content
                if isinstance(content, str):
                    # Look for JSON structure in the content
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        data = json.loads(json_str)
                        parsed = response_schema(**data)
                    else:
                        # If no JSON found, create a basic response
                        # This is a best-effort fallback
                        parsed = response_schema(Phishing=False, Reason="Failed to parse response")
                else:
                    parsed = response_schema(Phishing=False, Reason="Failed to parse response")
            except:
                # Final fallback
                parsed = response_schema(Phishing=False, Reason="Failed to parse response")
            
            # Silent token tracking
            
            if return_token_usage:
                return parsed, token_info
            return parsed
    

async def parse_structured_output(llm: object, text: str, schema_class: Optional[Type[BaseModel]] = None) -> Union[BaseModel, Dict[str, Any]]:
    """
    Asynchronously parse unstructured text into a structured format using an LLM.
    
    Args:
        llm: The LLM instance to use for parsing
        text: The text to parse
        schema_class: The Pydantic schema class to parse into
        
    Returns:
        Parsed structured output as a Pydantic model instance
    """
    if schema_class is None:
        schema_class = response_schema
    
    system_prompt = f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {schema_class.model_json_schema()}"
    user_prompt = f"{text}"
    messages = create_prompt_template(system_prompt, user_prompt)
    
    try:
        client = llm.with_structured_output(schema_class)
        response = await client.ainvoke(messages)
        return response
    except Exception as e:
        # If structured output fails, try to parse manually
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        # Try to parse JSON manually
        import json
        try:
            data = json.loads(content)
            return schema_class(**data)
        except:
            raise e
    
