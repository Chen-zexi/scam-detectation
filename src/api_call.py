from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class response_schema(BaseModel):
    Phishing: bool
    Reason: str

def make_api_call(
        llm: object,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: bool = False,
        structure_model: bool = False,
    ) -> Dict[str, Any]:
        if enable_thinking:
            user_prompt += "\n \\think"
        prompt_template = ChatPromptTemplate(
            [
                ("system", "{system_prompt}"),
                ( "user", "{user_prompt}")
            ]
        )
        messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
        
        if structure_model:
            client = llm
            response = client.invoke(messages).content
        else:
            client = llm.with_structured_output(response_schema)
            response = client.invoke(messages)
            
        
        return response
    
def parse_structured_output(llm: object, text: str, schema_class=None) -> Dict[str, Any]:
    if schema_class is None:
        schema_class = response_schema
    
    system_prompt = f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {schema_class.model_json_schema()}"
    user_prompt = f"{text}"
    prompt_template = ChatPromptTemplate(
            [
                ("system", "{system_prompt}"),
                ( "user", "{user_prompt}")
            ]
        )
    messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
    client = llm.with_structured_output(schema_class)
    response = client.invoke(messages)
    return response
    
