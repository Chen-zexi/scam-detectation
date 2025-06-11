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
    ) -> Dict[str, Any]:

        prompt_template = ChatPromptTemplate(
            [
                ("system", "{system_prompt}"),
                ( "user", "{user_prompt}")
            ]
        )
        messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
            
        client = llm.with_structured_output(response_schema)
            
        response = client.invoke(messages)
        print(response)
                
        return response
