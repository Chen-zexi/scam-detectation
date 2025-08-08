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

def remove_thinking_tokens(text: str) -> str:
    """Remove thinking tokens from the response text."""
    if '<think>' in text and '</think>' in text:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text

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
        Dictionary containing token usage information
    """
    token_info = {}
    
    # Try to get usage_metadata (primary source)
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        if isinstance(usage, dict):
            token_info['input_tokens'] = usage.get('input_tokens', 0)
            token_info['output_tokens'] = usage.get('output_tokens', 0)
            token_info['total_tokens'] = usage.get('total_tokens', 0)
            
            # Check for reasoning tokens in output details
            if 'output_token_details' in usage:
                details = usage['output_token_details']
                if 'reasoning' in details:
                    token_info['reasoning_tokens'] = details['reasoning']
    
    # Also check response_metadata for additional details
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if isinstance(metadata, dict) and 'token_usage' in metadata:
            token_usage = metadata['token_usage']
            
            # Fallback if usage_metadata was not available
            if 'input_tokens' not in token_info:
                token_info['input_tokens'] = token_usage.get('prompt_tokens', 0)
                token_info['output_tokens'] = token_usage.get('completion_tokens', 0)
                token_info['total_tokens'] = token_usage.get('total_tokens', 0)
            
            # Check for reasoning tokens in completion details
            if 'completion_tokens_details' in token_usage:
                details = token_usage['completion_tokens_details']
                if 'reasoning_tokens' in details:
                    token_info['reasoning_tokens'] = details['reasoning_tokens']
    
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

def make_api_call(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        enable_thinking: bool = False,
        use_structure_model: bool = False,
        structure_model: Optional[object] = None,
        force_structure_model: bool = False,
        return_token_usage: bool = False,
    ) -> Union[Dict[str, Any], BaseModel, str, Tuple[Any, Dict[str, Any]]]:
        """
        Make an API call to an LLM with flexible response handling.
        
        Args:
            llm: The LLM instance
            system_prompt: System prompt
            user_prompt: User prompt
            response_schema: Optional Pydantic schema for structured output
            enable_thinking: Whether to enable thinking tokens
            use_structure_model: Whether to use a separate model for parsing
            structure_model: Optional structure model instance for parsing (defaults to gpt-4.1-nano)
            force_structure_model: Force parsing with structure model even if native works
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
        if enable_thinking:
            user_prompt += "\n \\think"
        
        messages = create_prompt_template(system_prompt, user_prompt)
        
        # Get model name for logging
        model_name = getattr(llm, 'model_name', 'unknown')
        
        # Case 1: No schema requested, return raw content
        if response_schema is None:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Extract token usage (silent by default)
            token_info = extract_token_usage(response)
            
            if return_token_usage:
                return content, token_info
            return content
        
        # Case 2: Schema requested with structure model parsing
        if use_structure_model or (structure_model and not hasattr(llm, 'with_structured_output')):
            # Get unstructured response first
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage from initial call
            token_info = extract_token_usage(response)
            
            # Remove thinking tokens if present
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Parse with structure model
            if structure_model:
                parsed = parse_structured_output(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                parsed = parse_structured_output(auto_structure_model, content, response_schema)
            
            # Silent token tracking
            
            if return_token_usage:
                return parsed, token_info
            return parsed
        
        # Case 3: Native structured output
        try:
            # Use include_raw=True to get token usage with structured output
            client = llm.with_structured_output(response_schema, include_raw=True)
            response_with_raw = client.invoke(messages)
            
            token_info = {}
            
            # Extract the parsed response and raw response
            if isinstance(response_with_raw, dict) and 'raw' in response_with_raw:
                response = response_with_raw.get('parsed', response_with_raw)
                raw_response = response_with_raw['raw']
                token_info = extract_token_usage(raw_response)
            else:
                # Fallback to direct response if include_raw didn't work
                client = llm.with_structured_output(response_schema)
                response = client.invoke(messages)
            
            # If force_structure_model is True, also parse with structure model
            if force_structure_model and structure_model:
                # Convert response to string for re-parsing
                response_str = response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)
                response = parse_structured_output(structure_model, response_str, response_schema)
            
            # Silent token tracking
            
            if return_token_usage:
                return response, token_info
            return response
            
        except Exception as e:
            # Fallback to structure model
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage
            token_info = extract_token_usage(response)
            
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            if structure_model:
                parsed = parse_structured_output(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano as fallback
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                parsed = parse_structured_output(auto_structure_model, content, response_schema)
            
            # Silent token tracking
            
            if return_token_usage:
                return parsed, token_info
            return parsed

async def make_api_call_async(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        enable_thinking: bool = False,
        use_structure_model: bool = False,
        structure_model: Optional[object] = None,
        force_structure_model: bool = False,
        return_token_usage: bool = False,
    ) -> Union[Dict[str, Any], BaseModel, str, Tuple[Any, Dict[str, Any]]]:
        """
        Make an async API call to an LLM with flexible response handling.
        
        Args:
            llm: The LLM instance
            system_prompt: System prompt
            user_prompt: User prompt
            response_schema: Optional Pydantic schema for structured output
            enable_thinking: Whether to enable thinking tokens
            use_structure_model: Whether to use a separate model for parsing
            structure_model: Optional structure model instance for parsing (defaults to gpt-4.1-nano)
            force_structure_model: Force parsing with structure model even if native works
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
        if enable_thinking:
            user_prompt += "\n \\think"
        
        messages = create_prompt_template(system_prompt, user_prompt)
        
        # Get model name for logging
        model_name = getattr(llm, 'model_name', 'unknown')
        
        # Case 1: No schema requested, return raw content
        if response_schema is None:
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Extract token usage (silent by default)
            token_info = extract_token_usage(response)
            
            if return_token_usage:
                return content, token_info
            return content
        
        # Case 2: Schema requested with structure model parsing
        if use_structure_model or (structure_model and not hasattr(llm, 'with_structured_output')):
            # Get unstructured response first
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage from initial call
            token_info = extract_token_usage(response)
            
            # Remove thinking tokens if present
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Parse with structure model
            if structure_model:
                parsed = await parse_structured_output_async(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                parsed = await parse_structured_output_async(auto_structure_model, content, response_schema)
            
            # Silent token tracking
            
            if return_token_usage:
                return parsed, token_info
            return parsed
        
        # Case 3: Native structured output
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
            
            # If force_structure_model is True, also parse with structure model
            if force_structure_model and structure_model:
                # Convert response to string for re-parsing
                response_str = response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)
                response = await parse_structured_output_async(structure_model, response_str, response_schema)
            
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
            
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            if structure_model:
                parsed = await parse_structured_output_async(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano as fallback
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                parsed = await parse_structured_output_async(auto_structure_model, content, response_schema)
            
            # Silent token tracking
            
            if return_token_usage:
                return parsed, token_info
            return parsed
    
def parse_structured_output(llm: object, text: str, schema_class: Optional[Type[BaseModel]] = None) -> Union[BaseModel, Dict[str, Any]]:
    """
    Parse unstructured text into a structured format using an LLM.
    
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
        response = client.invoke(messages)
        return response
    except Exception as e:
        # If structured output fails, try to parse manually
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        # Try to parse JSON manually
        import json
        try:
            data = json.loads(content)
            return schema_class(**data)
        except:
            raise e

async def parse_structured_output_async(llm: object, text: str, schema_class: Optional[Type[BaseModel]] = None) -> Union[BaseModel, Dict[str, Any]]:
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
    
