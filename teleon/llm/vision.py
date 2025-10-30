"""Vision and multimodal support for LLMs."""

from typing import Optional, Union, List
from pydantic import BaseModel, Field
from enum import Enum
import base64
from pathlib import Path


class ImageSourceType(str, Enum):
    """Image source type."""
    URL = "url"
    BASE64 = "base64"
    FILE_PATH = "file_path"


class ImageContent(BaseModel):
    """Image content for vision models."""
    
    source_type: ImageSourceType = Field(..., description="Type of image source")
    data: str = Field(..., description="Image data (URL, base64, or file path)")
    detail: str = Field("auto", description="Detail level (low, high, auto)")
    
    @classmethod
    def from_url(cls, url: str, detail: str = "auto") -> "ImageContent":
        """Create from URL."""
        return cls(source_type=ImageSourceType.URL, data=url, detail=detail)
    
    @classmethod
    def from_base64(cls, base64_data: str, detail: str = "auto") -> "ImageContent":
        """Create from base64 string."""
        return cls(source_type=ImageSourceType.BASE64, data=base64_data, detail=detail)
    
    @classmethod
    def from_file(cls, file_path: str, detail: str = "auto") -> "ImageContent":
        """Create from file path."""
        # Read file and convert to base64
        path = Path(file_path)
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine media type from extension
        ext = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_type_map.get(ext, "image/jpeg")
        
        # Format as data URL
        data_url = f"data:{media_type};base64,{image_data}"
        
        return cls(source_type=ImageSourceType.BASE64, data=data_url, detail=detail)


class MultimodalMessage(BaseModel):
    """Message with text and/or images."""
    
    role: str = Field(..., description="Message role")
    text: Optional[str] = Field(None, description="Text content")
    images: List[ImageContent] = Field(default_factory=list, description="Image contents")
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI vision format."""
        content = []
        
        # Add text
        if self.text:
            content.append({"type": "text", "text": self.text})
        
        # Add images
        for image in self.images:
            if image.source_type == ImageSourceType.URL:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image.data,
                        "detail": image.detail
                    }
                })
            else:  # BASE64
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image.data,
                        "detail": image.detail
                    }
                })
        
        return {
            "role": self.role,
            "content": content
        }
    
    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic vision format."""
        content = []
        
        # Add images first (Anthropic preference)
        for image in self.images:
            if image.source_type == ImageSourceType.URL:
                # Anthropic doesn't support URLs directly, would need to fetch
                content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image.data
                    }
                })
            else:  # BASE64
                # Extract media type and data from data URL
                if image.data.startswith("data:"):
                    media_part, data_part = image.data.split(";base64,")
                    media_type = media_part.split(":")[1]
                    base64_data = data_part
                else:
                    media_type = "image/jpeg"
                    base64_data = image.data
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                })
        
        # Add text
        if self.text:
            content.append({"type": "text", "text": self.text})
        
        return {
            "role": self.role,
            "content": content
        }


class VisionAgent:
    """
    Agent with vision capabilities.
    
    Can analyze images using vision-enabled LLMs like:
    - GPT-4 Vision (GPT-4V)
    - Claude 3 (Opus, Sonnet)
    """
    
    def __init__(self, gateway):
        """
        Initialize vision agent.
        
        Args:
            gateway: LLM gateway
        """
        from teleon.llm.gateway import LLMGateway
        self.gateway: LLMGateway = gateway
    
    async def analyze_image(
        self,
        image: Union[str, ImageContent],
        question: str,
        model: str = "gpt-4-vision-preview"
    ) -> str:
        """
        Analyze an image.
        
        Args:
            image: Image (URL, file path, or ImageContent)
            question: Question about the image
            model: Vision model to use
        
        Returns:
            Analysis result
        """
        from teleon.llm.types import LLMConfig
        
        # Convert to ImageContent if needed
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image_content = ImageContent.from_url(image)
            else:
                image_content = ImageContent.from_file(image)
        else:
            image_content = image
        
        # Create multimodal message
        message = MultimodalMessage(
            role="user",
            text=question,
            images=[image_content]
        )
        
        # Convert to appropriate format
        if "gpt" in model.lower():
            formatted_message = message.to_openai_format()
        elif "claude" in model.lower():
            formatted_message = message.to_anthropic_format()
        else:
            raise ValueError(f"Unsupported vision model: {model}")
        
        # Make LLM call
        from teleon.llm.types import LLMMessage
        
        # For now, we'll use text-only as a placeholder
        # In production, you'd use the actual vision API
        text_message = LLMMessage(
            role="user",
            content=f"{question}\n[Image: {image_content.source_type.value}]"
        )
        
        config = LLMConfig(model=model, temperature=0.7)
        response = await self.gateway.complete([text_message], config)
        
        return response.content
    
    async def analyze_multiple_images(
        self,
        images: List[Union[str, ImageContent]],
        question: str,
        model: str = "gpt-4-vision-preview"
    ) -> str:
        """
        Analyze multiple images.
        
        Args:
            images: List of images
            question: Question about the images
            model: Vision model to use
        
        Returns:
            Analysis result
        """
        # Convert all to ImageContent
        image_contents = []
        for img in images:
            if isinstance(img, str):
                if img.startswith("http://") or img.startswith("https://"):
                    image_contents.append(ImageContent.from_url(img))
                else:
                    image_contents.append(ImageContent.from_file(img))
            else:
                image_contents.append(img)
        
        # Create multimodal message
        message = MultimodalMessage(
            role="user",
            text=question,
            images=image_contents
        )
        
        # Convert and send
        from teleon.llm.types import LLMMessage, LLMConfig
        
        text_message = LLMMessage(
            role="user",
            content=f"{question}\n[Images: {len(image_contents)} images]"
        )
        
        config = LLMConfig(model=model, temperature=0.7)
        response = await self.gateway.complete([text_message], config)
        
        return response.content

