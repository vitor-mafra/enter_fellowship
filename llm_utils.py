"""LLM client utilities for document classification and data extraction."""

import json
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError(
        "Failed to import OpenAI SDK. Install with `pip install openai`."
    ) from e

load_dotenv()

logger = logging.getLogger(__name__)

_CLIENT: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client instance.
    
    The API key is read from the OPENAI_API_KEY environment variable,
    which can be set in a .env file.
    
    Returns:
        OpenAI client instance.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def classify_document_structure(
    document_label: str, max_retries: int = 2
) -> str:
    """Classify a document type as stricted or flexible using LLM.
    
    Stricted documents have fixed layouts (e.g., ID cards, utility bills, forms).
    Flexible documents have variable layouts (e.g., contracts, emails, legal opinions).
    
    Args:
        document_label: Document type label (e.g., "carteira_oab", "contrato").
        max_retries: Maximum number of retry attempts on failure.
        
    Returns:
        "stricted" or "flexible" classification result.
        
    Raises:
        ValueError: If the API call fails after all retries.
    """
    model = "gpt-5-mini"  # Always use ChatGPT 5 mini
    prompt = (
        "Classifique o tipo de documento informado como 'rígida' (layout fixo, "
        "campos em posições estáveis, como carteira de identidade, CNH, conta de água, fatura, "
        "carteira da oab, tela de sistema, formulário) ou 'flexível' (texto livre e altamente variável, como "
        "contrato, email, parecer jurídico).\n\n"
        "Responda APENAS com uma palavra: 'rígida' ou 'flexível'.\n\n"
        f"Tipo de documento: {document_label}"
    )
    
    client = get_client()
    
    for attempt in range(max_retries + 1):
        try:
            request_params = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um assistente especializado em classificação de documentos.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            
            if model.startswith("gpt-5"):
                request_params["reasoning_effort"] = "minimal"
            # Note: gpt-5-mini doesn't support temperature parameter
                
            response = client.chat.completions.create(**request_params)
            content = response.choices[0].message.content.strip().lower()
            
            # Normalize response: keep prompt in Portuguese but return English terms
            if "rígida" in content or "rigida" in content or "rigid" in content or "stricted" in content:
                return "stricted"
            elif "flexível" in content or "flexivel" in content or "flexible" in content:
                return "flexible"
            else:
                logger.warning(
                    f"Unexpected LLM response: '{content}'. Defaulting to 'flexible'."
                )
                return "flexible"
                
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                )
                continue
            else:
                logger.error(
                    f"LLM API call failed after {max_retries + 1} attempts: {str(e)}"
                )
                raise ValueError(
                    f"Failed to classify document type '{document_label}': {str(e)}"
                ) from e
    
    raise ValueError(f"Failed to classify document type '{document_label}'")


def extract_data_with_llm(
    pdf_text: str, extraction_schema: Dict[str, str]
) -> Dict[str, Any]:
    """Extract data from PDF text using LLM based on extraction schema.
    
    Args:
        pdf_text: Extracted text from the PDF.
        extraction_schema: Dictionary mapping field names to extraction descriptions.
        
    Returns:
        Dictionary with extracted field values (null for missing fields).
    """
    model = "gpt-5-mini"  # Always use ChatGPT 5 mini
    
    # Format schema as JSON
    schema_json = json.dumps(extraction_schema, ensure_ascii=False, indent=2)
    
    prompt = f"""Você deve extrair informações de um documento PDF baseado no schema fornecido.

Schema de extração:
{schema_json}

Texto do PDF:
{pdf_text}

Importante: Se um campo não puder ser encontrado no texto, retorne null para esse campo"""
    
    client = get_client()
    
    try:
        request_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente especializado em extração de dados estruturados de documentos. Sempre retorne apenas JSON válido, sem texto adicional.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        
        if model.startswith("gpt-5"):
            request_params["reasoning_effort"] = "minimal"
        # Note: gpt-5-mini doesn't support temperature parameter
        
        response = client.chat.completions.create(**request_params)
        content = response.choices[0].message.content.strip()
        
        try:
            extracted_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {str(e)}")
            logger.error(f"Response content: {content}")
            return {field: None for field in extraction_schema.keys()}
        
        # Ensure all schema fields are present, set to None (will be serialized as null in JSON) if missing
        result = {}
        for field in extraction_schema.keys():
            value = extracted_data.get(field, None)
            # Normalize: convert string "null" or "None" to Python None
            if isinstance(value, str) and value.lower() in ("null", "none", ""):
                value = None
            result[field] = value
        
        return result
        
    except Exception as e:
        logger.error(f"LLM extraction failed: {str(e)}")
        return {field: None for field in extraction_schema.keys()}

