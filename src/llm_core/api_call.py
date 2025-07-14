from typing import Any, Dict, Optional, Type, Union
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import asyncio
import re

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

def make_api_call(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        enable_thinking: bool = False,
        use_structure_model: bool = False,
        structure_model: Optional[object] = None,
        force_structure_model: bool = False,
    ) -> Union[Dict[str, Any], BaseModel, str]:
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
            
        Returns:
            Structured response, dict, or string depending on configuration
        
        Note:
            When a model doesn't support native structured output, the system automatically
            falls back to using gpt-4.1-nano for parsing the response into the desired schema.
        """
        if enable_thinking:
            user_prompt += "\n \\think"
        
        messages = create_prompt_template(system_prompt, user_prompt)
        
        # Case 1: No schema requested, return raw content
        if response_schema is None:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            if enable_thinking:
                content = remove_thinking_tokens(content)
            return content
        
        # Case 2: Schema requested with structure model parsing
        if use_structure_model or (structure_model and not hasattr(llm, 'with_structured_output')):
            # Get unstructured response first
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Remove thinking tokens if present
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Parse with structure model
            if structure_model:
                return parse_structured_output(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                return parse_structured_output(auto_structure_model, content, response_schema)
        
        # Case 3: Native structured output
        try:
            client = llm.with_structured_output(response_schema)
            response = client.invoke(messages)
            
            # If force_structure_model is True, also parse with structure model
            if force_structure_model and structure_model:
                # Convert response to string for re-parsing
                response_str = response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)
                return parse_structured_output(structure_model, response_str, response_schema)
            
            return response
            
        except Exception as e:
            # Fallback to structure model
            if structure_model:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if enable_thinking:
                    content = remove_thinking_tokens(content)
                return parse_structured_output(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano as fallback
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                response = llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if enable_thinking:
                    content = remove_thinking_tokens(content)
                return parse_structured_output(auto_structure_model, content, response_schema)

async def make_api_call_async(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        enable_thinking: bool = False,
        use_structure_model: bool = False,
        structure_model: Optional[object] = None,
        force_structure_model: bool = False,
    ) -> Union[Dict[str, Any], BaseModel, str]:
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
            
        Returns:
            Structured response, dict, or string depending on configuration
        
        Note:
            When a model doesn't support native structured output, the system automatically
            falls back to using gpt-4.1-nano for parsing the response into the desired schema.
        """
        if enable_thinking:
            user_prompt += "\n \\think"
        
        messages = create_prompt_template(system_prompt, user_prompt)
        
        # Case 1: No schema requested, return raw content
        if response_schema is None:
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            if enable_thinking:
                content = remove_thinking_tokens(content)
            return content
        
        # Case 2: Schema requested with structure model parsing
        if use_structure_model or (structure_model and not hasattr(llm, 'with_structured_output')):
            # Get unstructured response first
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Remove thinking tokens if present
            if enable_thinking:
                content = remove_thinking_tokens(content)
            
            # Parse with structure model
            if structure_model:
                return await parse_structured_output_async(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                return await parse_structured_output_async(auto_structure_model, content, response_schema)
        
        # Case 3: Native structured output
        try:
            client = llm.with_structured_output(response_schema)
            response = await client.ainvoke(messages)
            
            # If force_structure_model is True, also parse with structure model
            if force_structure_model and structure_model:
                # Convert response to string for re-parsing
                response_str = response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)
                return await parse_structured_output_async(structure_model, response_str, response_schema)
            
            return response
            
        except Exception as e:
            # Fallback to structure model
            if structure_model:
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if enable_thinking:
                    content = remove_thinking_tokens(content)
                return await parse_structured_output_async(structure_model, content, response_schema)
            else:
                # Auto-create structure model using gpt-4.1-nano as fallback
                from .api_provider import LLM
                temp_llm_instance = LLM(provider="openai", model="gpt-4.1-nano")
                auto_structure_model = temp_llm_instance.get_structure_model()
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if enable_thinking:
                    content = remove_thinking_tokens(content)
                return await parse_structured_output_async(auto_structure_model, content, response_schema)
    
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
    
