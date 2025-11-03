import os
import base64
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()


class GeminiResponse(BaseModel):
    """Structured response from Gemini."""
    text: str = Field(default="", description="Generated text output from Gemini")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Raw response metadata")


class GeminiAPIClient:
    """
    Unified Gemini API client via LangChain.
    Supports:
      • Text-only prompts
      • Multimodal (image + text) prompts
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables.")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            convert_system_message_to_human=True
        )

    def generate_content(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/png"
    ) -> GeminiResponse:
        """
        Generate Gemini response for text or image tasks.

        Args:
            prompt (str): User or system prompt.
            image_bytes (bytes, optional): Raw image bytes for visual input.
            mime_type (str): MIME type of image, e.g. "image/png" or "image/jpeg".
        """
        try:
            content = [{"type": "text", "text": prompt}]

            if image_bytes:
                image_b64 = base64.b64encode(image_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_b64}"
                })

            msg = HumanMessage(content=content)
            response = self.llm.invoke([msg])

            return GeminiResponse(
                text=response.content if hasattr(response, "content") else "",
                metadata=getattr(response, "response_metadata", {})
            )

        except Exception as e:
            print(f"[ERROR] LangChain Gemini failed: {e}")
            return GeminiResponse(text="", metadata={"error": str(e)})
