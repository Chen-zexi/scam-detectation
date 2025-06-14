import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class LLM:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def get_llm(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            print(api_key)
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            return ChatOpenAI(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            print(api_key)
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            if self.model.startswith("gemini-2.5-flash"):
                return ChatGoogleGenerativeAI(api_key=api_key, model=self.model, temperature=0)
            else:
                return ChatGoogleGenerativeAI(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "lm-studio":
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise ValueError("HOST_IP is not set")
            return ChatOpenAI(base_url=f"http://{host_ip}:1234/v1", api_key='lm-studio', model=self.model, temperature=0)
        elif self.provider == "vllm":
            # Assume using lm-studio
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise ValueError("HOST_IP is not set")
            print(f"http://{host_ip}:8000/v1")
            return ChatOpenAI(base_url=f"http://{host_ip}:8000/v1", api_key='EMPTY', model=self.model, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        