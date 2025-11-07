"""Heuristic-based field matching for positional extraction from stricted templates."""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from box_parser import extract_fields_from_box
from template_manager import find_template_by_label_and_fields, infer_type

logger = logging.getLogger(__name__)

# Field labels that should be avoided (static labels, not variable values)
FIELD_LABELS = {
    "nome", "nome completo", "nome do titular", "nome da mãe", "nome do pai",
    "filiação", "nacionalidade", "naturalidade", "sexo", "gênero",
    "data de nascimento", "nascimento", "estado civil", "cor", "raça",
    "rg", "r.g.", "registro geral", "número do rg", "órgão emissor",
    "emissor", "uf emissor", "cpf", "c.p.f.", "cadastro de pessoa física",
    "número do cpf", "título de eleitor", "zona", "seção", "cnh",
    "carteira nacional de habilitação", "categoria", "validade",
    "data de emissão", "local de emissão", "assinatura", "assinatura do titular",
    "digital", "data de expedição", "identidade civil", "passaporte",
    "nº do passaporte", "país emissor", "data de validade",
    "número da inscrição", "inscrição", "inscrição profissional",
    "seccional", "subseção", "cargo", "função", "registro",
    "número de registro", "matrícula", "nº da matrícula", "conselho",
    "conselho regional", "conselho federal", "registro no conselho",
    "situação", "situação do profissional", "validade da carteira",
    "órgão de classe", "carteira profissional", "número da carteira",
    "uf de registro", "tipo de licença", "data de filiação",
    "data de renovação", "conta", "conta nº", "conta número",
    "código do cliente", "código de instalação", "código da unidade consumidora",
    "código de barras", "referência", "período de consumo",
    "leitura anterior", "leitura atual", "consumo", "medição",
    "nº do medidor", "data da leitura", "data de vencimento",
    "valor da fatura", "valor total", "total a pagar", "subtotal",
    "multa", "juros", "desconto", "vencimento anterior",
    "histórico de consumo", "número do contrato", "número da conta",
    "unidade consumidora", "classe de consumo", "endereço da instalação",
    "cep", "cidade", "estado", "uf", "cnpj da concessionária",
    "nome da concessionária", "autenticação", "autenticação eletrônica",
    "protocolo", "número do protocolo", "banco", "agência",
    "conta corrente", "conta poupança", "conta salário", "nº da conta",
    "número da conta", "nº da agência", "número da agência",
    "titular", "favorecido", "cpf/cnpj", "cnpj", "código do banco",
    "código da agência", "chave pix", "tipo de conta", "tipo de operação",
    "valor", "valor da operação", "data", "data de pagamento",
    "data de crédito", "data de débito", "hora", "horário", "saldo",
    "saldo anterior", "saldo atual", "lançamento", "histórico",
    "comprovante", "comprovante de transferência", "autenticação bancária",
    "descrição", "descrição da operação", "código da transação",
    "identificador", "nosso número", "linha digitável", "forma de pagamento",
    "modalidade", "nota fiscal", "nº da nota", "número da nota fiscal",
    "chave de acesso", "modelo", "série", "cfop", "cst", "código fiscal",
    "produto", "descrição do produto", "quantidade", "qtde", "unidade",
    "valor unitário", "base de cálculo", "imposto", "icms", "ipi", "iss",
    "cofins", "pis", "frete", "seguro", "valor líquido", "fornecedor",
    "cliente", "consumidor", "razão social", "nome fantasia",
    "endereço", "país", "inscrição estadual", "inscrição municipal",
    "data de saída", "data de entrada", "cond. de pagamento", "prazo",
    "transportadora", "placa", "uf do veículo", "motorista", "observações",
    "natureza da operação", "instituição", "universidade", "faculdade",
    "curso", "graduação", "pós-graduação", "aluno", "discente",
    "registro acadêmico", "ra", "período", "semestre", "ano",
    "carga horária", "disciplina", "código da disciplina", "professor",
    "docente", "nota", "média", "aprovado", "reprovado", "frequência",
    "coordenador", "diretor", "secretário", "paciente", "nome do paciente",
    "idade", "convênio", "plano", "carteirinha", "número da carteirinha",
    "médico", "crm", "especialidade", "hospital", "clínica", "procedimento",
    "diagnóstico", "cid", "cid-10", "exame", "tipo de exame", "material",
    "coleta", "data da coleta", "data do resultado", "resultado",
    "valor de referência", "carimbo", "processo", "número do processo",
    "nº do processo", "tribunal", "comarca", "vara", "juiz", "autor",
    "réu", "advogado", "oab", "parte", "requerente", "requerido",
    "petição", "tipo de ação", "assunto", "objeto", "data de protocolo",
    "data da audiência", "data de julgamento", "sentença", "despacho",
    "decisão", "assinatura digital", "certifico", "cópia autenticada",
    "documento anexo", "embarcador", "destinatário", "remetente",
    "endereço de entrega", "endereço de coleta", "cpf do motorista",
    "rntrc", "carga", "peso", "peso bruto", "peso líquido", "volume",
    "qtde volumes", "data de coleta", "data de entrega", "tipo de frete",
    "frete por conta", "valor do frete", "ct-e", "nº ct-e", "nfe relacionada",
    "empresa", "telefone", "email", "responsável", "contador", "crc",
    "período fiscal", "competência", "receita", "despesa", "lucro",
    "débito", "crédito", "tributo", "irpj", "csll", "declaro",
    "declaração", "declaramos", "recibo", "recebemos de", "referente a",
    "identificação", "identidade", "documento", "comprovante",
    "autenticação mecânica", "número", "emitente", "dia", "mês",
    "página", "nº", "código", "tipo", "status", "obs", "anexo",
    "informações", "informações complementares"
}

# Patterns that indicate labels (not values)
LABEL_PATTERNS = [
    re.compile(r'^[a-záàâãéêíóôõúç\s]+:\s*$', re.IGNORECASE),  # "Nome:"
    re.compile(r'^[a-záàâãéêíóôõúç\s]+:\s*$', re.IGNORECASE),  # "CPF:"
    re.compile(r'^\s*[a-záàâãéêíóôõúç]+[:\-]\s*$', re.IGNORECASE),  # "Nome-" or "Nome:"
]

# Weights for scoring function
W_SPATIAL = 0.30
W_TEXT_VARIABILITY = 0.15
W_PROXIMITY = 0.10
W_LENGTH_SIMILARITY = 0.25  # Increased weight for length matching
W_TYPE_MATCH = 0.20


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return text.strip().lower()


def _is_label_text(text: str) -> bool:
    """Check if text block is likely a label (not a value).
    
    Args:
        text: Text content to check.
        
    Returns:
        True if text appears to be a label, False otherwise.
    """
    normalized = _normalize_text(text)
    
    # Check against known field labels
    if normalized in FIELD_LABELS:
        return True
    
    # Check for label patterns (ends with colon, dash, etc.)
    for pattern in LABEL_PATTERNS:
        if pattern.match(text.strip()):
            return True
    
    # Check if text is very short and contains only label-like words
    words = normalized.split()
    if len(words) <= 3:
        if any(word in FIELD_LABELS for word in words):
            return True
    
    return False


def _compute_text_variability_score(text: str) -> float:
    """Compute a score indicating how variable/dynamic the text is.
    
    Higher score = more likely to be a variable value (not a static label).
    
    Args:
        text: Text content to analyze.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0
    
    normalized = _normalize_text(text)
    
    # Penalize if it's a known label
    if _is_label_text(text):
        return 0.0
    
    score = 1.0
    
    # Boost for numeric content (likely variable)
    if re.search(r'\d', text):
        score += 0.2
    
    # Boost for mixed alphanumeric (likely variable)
    if re.search(r'[a-záàâãéêíóôõúç]', text, re.IGNORECASE) and re.search(r'\d', text):
        score += 0.2
    
    # Boost for longer text (less likely to be a label)
    if len(text.strip()) > 10:
        score += 0.1
    
    # Penalize if it looks like a label pattern
    if re.match(r'^[a-záàâãéêíóôõúç\s]+:\s*$', text.strip(), re.IGNORECASE):
        score -= 0.5
    
    # Normalize to [0, 1]
    return min(1.0, max(0.0, score))


def _compute_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box with keys x0, y0, x1, y1.
        box2: Second bounding box with keys x0, y0, x1, y1.
        
    Returns:
        IoU value between 0.0 and 1.0.
    """
    x0_1, y0_1 = box1.get("x0", 0), box1.get("y0", 0)
    x1_1, y1_1 = box1.get("x1", 0), box1.get("y1", 0)
    x0_2, y0_2 = box2.get("x0", 0), box2.get("y0", 0)
    x1_2, y1_2 = box2.get("x1", 0), box2.get("y1", 0)
    
    # Calculate intersection
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)
    
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    
    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def _compute_center_distance(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Compute Euclidean distance between box centers.
    
    Args:
        box1: First bounding box.
        box2: Second bounding box.
        
    Returns:
        Distance in pixels.
    """
    center1_x = (box1.get("x0", 0) + box1.get("x1", 0)) / 2
    center1_y = (box1.get("y0", 0) + box1.get("y1", 0)) / 2
    center2_x = (box2.get("x0", 0) + box2.get("x1", 0)) / 2
    center2_y = (box2.get("y0", 0) + box2.get("y1", 0)) / 2
    
    dx = center1_x - center2_x
    dy = center1_y - center2_y
    
    return math.sqrt(dx * dx + dy * dy)


def _compute_spatial_similarity(
    ref_box: Dict[str, float],
    candidate_box: Dict[str, float],
    page_width: float = 800.0,
    page_height: float = 1000.0,
    tolerance_pixels: float = 10.0,
    tolerance_percent: float = 0.02,
) -> float:
    """Compute spatial similarity score between reference and candidate boxes.
    
    Args:
        ref_box: Reference bounding box from template.
        candidate_box: Candidate text block bounding box.
        page_width: Page width for percentage-based tolerance.
        page_height: Page height for percentage-based tolerance.
        tolerance_pixels: Absolute tolerance in pixels.
        tolerance_percent: Relative tolerance as percentage of page dimension.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    # Check if same page
    ref_page = ref_box.get("page", 0)
    cand_page = candidate_box.get("page", 0)
    if ref_page != cand_page:
        return 0.0
    
    # Compute IoU
    iou = _compute_iou(ref_box, candidate_box)
    
    # Compute center distance
    distance = _compute_center_distance(ref_box, candidate_box)
    
    # Calculate tolerance (max of absolute and relative)
    max_tolerance = max(
        tolerance_pixels,
        tolerance_percent * max(page_width, page_height)
    )
    
    # Convert distance to similarity (inverse, normalized)
    if distance <= max_tolerance:
        distance_score = 1.0 - (distance / max_tolerance)
    else:
        distance_score = max(0.0, 1.0 - (distance / (max_tolerance * 2)))
    
    # Combine IoU and distance
    spatial_score = (iou * 0.6 + distance_score * 0.4)
    
    return spatial_score


def _compute_proximity_bonus(
    ref_box: Dict[str, float],
    candidate_box: Dict[str, float],
    max_bonus: float = 0.15,
) -> float:
    """Compute a small bonus for blocks very close to reference position.
    
    Args:
        ref_box: Reference bounding box.
        candidate_box: Candidate bounding box.
        max_bonus: Maximum bonus value.
        
    Returns:
        Bonus score between 0.0 and max_bonus.
    """
    distance = _compute_center_distance(ref_box, candidate_box)
    
    # Bonus decreases linearly with distance
    # Full bonus at distance 0, zero bonus at distance 50
    if distance <= 10:
        return max_bonus
    elif distance <= 50:
        return max_bonus * (1.0 - (distance - 10) / 40)
    else:
        return 0.0


def _compute_length_similarity(
    candidate_text: str,
    expected_avg_length: Optional[float],
) -> float:
    """Compute similarity score based on text length matching expected average.
    
    More restrictive: heavily penalizes text that is much longer than expected.
    
    Args:
        candidate_text: Text content of candidate block.
        expected_avg_length: Expected average length from template (None if unknown).
        
    Returns:
        Score between 0.0 and 1.0. Returns 0.0 if text is much longer than expected.
    """
    if expected_avg_length is None or expected_avg_length == 0:
        return 0.5  # Neutral score if no template knowledge
    
    actual_length = len(candidate_text.strip())
    
    if actual_length == 0:
        return 0.0
    
    # Compute relative difference
    diff = abs(actual_length - expected_avg_length)
    relative_diff = diff / max(expected_avg_length, 1.0)
    
    # Heavily penalize text that is MUCH longer than expected (likely concatenated fields)
    # If text is 2x or more longer, reject it completely
    if actual_length > expected_avg_length * 2.0:
        return 0.0
    
    # If text is 1.5x longer, heavily penalize
    if actual_length > expected_avg_length * 1.5:
        excess_ratio = (actual_length - expected_avg_length * 1.5) / expected_avg_length
        return max(0.0, 0.2 - excess_ratio * 0.4)  # Score drops quickly from 0.2 to 0.0
    
    # Score decreases as difference increases
    # Perfect match (diff=0) -> score=1.0
    # 10% difference -> score=1.0
    # 30% difference -> score=0.7
    # 50% difference -> score=0.5
    if relative_diff <= 0.1:
        return 1.0
    elif relative_diff <= 0.3:
        return 1.0 - (relative_diff - 0.1) * 1.5  # Linear from 1.0 to 0.7
    elif relative_diff <= 0.5:
        return 0.7 - (relative_diff - 0.3) * 1.0  # Linear from 0.7 to 0.5
    else:
        return max(0.2, 0.5 - (relative_diff - 0.5) * 0.6)  # Linear from 0.5 to 0.2


def _compute_type_match_score(
    candidate_text: str,
    expected_type: Optional[str],
) -> float:
    """Compute score based on whether candidate text matches expected type.
    
    Args:
        candidate_text: Text content of candidate block.
        expected_type: Expected type from template (None if unknown).
        
    Returns:
        Score: 1.0 if type matches, 0.5 if unknown, 0.0 if mismatch.
    """
    if expected_type is None:
        return 0.5  # Neutral score if no template knowledge
    
    actual_type = infer_type(candidate_text)
    
    if actual_type == expected_type:
        return 1.0
    else:
        # Penalize type mismatch, but not too harshly
        # Some types are related (e.g., number vs money)
        if (expected_type == "money" and actual_type == "number") or \
           (expected_type == "number" and actual_type == "money"):
            return 0.7  # Partial match
        else:
            return 0.2  # Mismatch penalty


def _extract_field_from_box_text(
    box_text: str,
    field_name: str,
    expected_type: Optional[str],
    expected_avg_length: Optional[float],
) -> str:
    """Extract the relevant substring from a box text that may contain multiple fields.
    
    When a bounding box contains multiple fields (e.g., "101943 PR CONSELHO SECCIONAL - PARANÁ"),
    this function tries to extract the substring that best matches the expected field.
    
    Args:
        box_text: Full text content of the bounding box.
        field_name: Name of the field being extracted.
        expected_type: Expected type of the field (from template knowledge).
        expected_avg_length: Expected average length of the field.
        
    Returns:
        Extracted substring that best matches the field, or original text if no better match found.
    """
    if not box_text or not box_text.strip():
        return box_text
    
    box_text = box_text.strip()
    actual_length = len(box_text)
    
    # If text is within expected range, use it as-is
    if expected_avg_length:
        max_expected = expected_avg_length * 1.3
        if actual_length <= max_expected:
            return box_text
    
    # Try to extract substring based on expected type
    if expected_type:
        if expected_type == "number":
            # Extract number sequences and pick the best match
            number_matches = list(re.finditer(r'\d+', box_text))
            if number_matches:
                best_match = None
                best_score = float('inf')
                for match in number_matches:
                    extracted = match.group(0)
                    if expected_avg_length:
                        score = abs(len(extracted) - expected_avg_length)
                        if score < best_score:
                            best_score = score
                            best_match = extracted
                    else:
                        # If no expected length, use first number
                        return extracted
                if best_match:
                    return best_match
        
        elif expected_type == "text":
            # For text fields, try to extract based on position and length
            words = box_text.split()
            if expected_avg_length:
                target_length = int(expected_avg_length)
                
                # If expected length is very short (2-4 chars), look for siglas/codes
                if target_length <= 4:
                    # Look for short words (2-4 chars) that match
                    for word in words:
                        if 2 <= len(word) <= 4 and abs(len(word) - target_length) <= 1:
                            # Check if it's all uppercase (likely a sigla)
                            if word.isupper() or word.isalpha():
                                return word
                
                best_segment = None
                best_diff = float('inf')
                
                # Try all possible segments (1-6 words)
                for start in range(len(words)):
                    for end in range(start + 1, min(start + 7, len(words) + 1)):
                        segment = " ".join(words[start:end])
                        segment_len = len(segment)
                        diff = abs(segment_len - target_length)
                        if diff < best_diff:
                            best_diff = diff
                            best_segment = segment
                            # If we found a very close match, use it
                            if diff <= 2:
                                return best_segment
                
                if best_segment:
                    return best_segment
                
                # Fallback: try first word if it's reasonable
                if len(words) > 0:
                    first_word = words[0]
                    if len(first_word) <= expected_avg_length * 1.5:
                        return first_word
            else:
                # No expected length: try to extract meaningful text (skip numbers/short words)
                # Skip initial numbers and short words, take rest
                meaningful_words = []
                for word in words:
                    if not re.match(r'^\d+$', word) and len(word) > 2:
                        meaningful_words.append(word)
                if meaningful_words:
                    return " ".join(meaningful_words)
        
        elif expected_type in ["date", "time"]:
            # Extract date/time patterns
            date_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', box_text)
            if date_match:
                return date_match.group(0)
            time_match = re.search(r'\d{1,2}:\d{2}', box_text)
            if time_match:
                return time_match.group(0)
    
    # Fallback: try to extract based on position
    # If expected length is much smaller, try to extract from beginning
    if expected_avg_length and actual_length > expected_avg_length * 1.5:
        # Try extracting from start
        start_segment = box_text[:int(expected_avg_length * 1.3)]
        # Try to break at word boundary
        last_space = start_segment.rfind(' ')
        if last_space > expected_avg_length * 0.7:
            return start_segment[:last_space].strip()
        return start_segment.strip()
    
    # If all else fails, return original text
    return box_text


def _truncate_text_by_length(
    text: str,
    expected_avg_length: Optional[float],
    max_ratio: float = 1.3,
) -> str:
    """Truncate text if it's significantly longer than expected average length.
    
    Args:
        text: Text to potentially truncate.
        expected_avg_length: Expected average length from template.
        max_ratio: Maximum allowed ratio (text_length / avg_length). Default 1.3 (30% tolerance).
        
    Returns:
        Truncated text if needed, original text otherwise.
    """
    if expected_avg_length is None or expected_avg_length == 0:
        return text
    
    text = text.strip()
    actual_length = len(text)
    max_allowed_length = int(expected_avg_length * max_ratio)
    
    # If text is significantly longer, truncate it
    if actual_length > max_allowed_length:
        # Try to truncate at word boundary if possible
        truncated = text[:max_allowed_length]
        # Find last space before max length
        last_space = truncated.rfind(' ')
        if last_space > expected_avg_length * 0.8:  # Only use word boundary if not too short
            truncated = truncated[:last_space].strip()
        else:
            truncated = text[:max_allowed_length].strip()
        
        logger.debug(
            f"Truncated text from {actual_length} to {len(truncated)} chars "
            f"(expected avg: {expected_avg_length:.1f})"
        )
        return truncated
    
    return text


def _estimate_page_dimensions(text_blocks: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Estimate page dimensions from text blocks.
    
    Args:
        text_blocks: List of text blocks with coordinates.
        
    Returns:
        Tuple of (width, height).
    """
    if not text_blocks:
        return 800.0, 1000.0
    
    max_x = max((block.get("x1", 0) for block in text_blocks), default=800.0)
    max_y = max((block.get("y1", 0) for block in text_blocks), default=1000.0)
    
    return max_x, max_y


def find_best_matching_block(
    field_name: str,
    ref_position: Dict[str, Any],
    text_blocks: List[Dict[str, Any]],
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
    template_knowledge: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Find the best matching text block for a field using heuristic scoring.
    
    Args:
        field_name: Name of the field being matched.
        ref_position: Reference position from template (gabarito).
        text_blocks: List of candidate text blocks.
        page_width: Optional page width (estimated if not provided).
        page_height: Optional page height (estimated if not provided).
        template_knowledge: Optional template knowledge dict with field metadata.
        
    Returns:
        Best matching text block, or None if no good match found.
    """
    if not text_blocks:
        return None
    
    # Estimate page dimensions if not provided
    if page_width is None or page_height is None:
        page_width, page_height = _estimate_page_dimensions(text_blocks)
    
    # Filter blocks on same page
    ref_page = ref_position.get("page", 0)
    candidate_blocks = [
        block for block in text_blocks
        if block.get("page", 0) == ref_page
    ]
    
    if not candidate_blocks:
        logger.debug(f"No candidate blocks on page {ref_page} for field '{field_name}'")
        return None
    
    # Get template knowledge for this field
    field_metadata = None
    if template_knowledge:
        fields = template_knowledge.get("fields", {})
        field_metadata = fields.get(field_name)
    
    expected_avg_length = None
    expected_type = None
    if field_metadata:
        expected_avg_length = field_metadata.get("avg_length")
        expected_type = field_metadata.get("type")
    
    # Score each candidate
    scored_candidates: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
    
    for block in candidate_blocks:
        # Compute spatial similarity
        spatial_score = _compute_spatial_similarity(
            ref_position,
            block,
            page_width=page_width,
            page_height=page_height,
            tolerance_pixels=10.0,
            tolerance_percent=0.02,
        )
        
        # Compute text variability score
        text = block.get("text", "").strip()
        text_variability = _compute_text_variability_score(text)
        
        # Compute proximity bonus
        proximity_bonus = _compute_proximity_bonus(ref_position, block)
        
        # Compute length similarity (from template knowledge)
        length_similarity = _compute_length_similarity(text, expected_avg_length)
        
        # Compute type match score (from template knowledge)
        type_match = _compute_type_match_score(text, expected_type)
        
        # Composite score
        composite_score = (
            W_SPATIAL * spatial_score +
            W_TEXT_VARIABILITY * text_variability +
            W_PROXIMITY * proximity_bonus +
            W_LENGTH_SIMILARITY * length_similarity +
            W_TYPE_MATCH * type_match
        )
        
        scored_candidates.append((
            block,
            composite_score,
            {
                "spatial": spatial_score,
                "text_variability": text_variability,
                "proximity": proximity_bonus,
                "length_similarity": length_similarity,
                "type_match": type_match,
            }
        ))
    
    # Sort by score (descending)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Log debug information
    logger.debug(
        f"Field '{field_name}': Found {len(scored_candidates)} candidates on page {ref_page}"
    )
    
    if scored_candidates:
        best_block, best_score, score_breakdown = scored_candidates[0]
        best_text = best_block.get("text", "").strip()
        
        logger.debug(
            f"  Best match: '{best_text[:50]}...' (score: {best_score:.3f}, "
            f"spatial: {score_breakdown['spatial']:.3f}, "
            f"text_var: {score_breakdown['text_variability']:.3f}, "
            f"proximity: {score_breakdown['proximity']:.3f}, "
            f"length: {score_breakdown['length_similarity']:.3f}, "
            f"type: {score_breakdown['type_match']:.3f})"
        )
        
        # Log top 3 candidates if there are multiple
        if len(scored_candidates) > 1:
            logger.debug("  Top candidates:")
            for idx, (block, score, breakdown) in enumerate(scored_candidates[:3], 1):
                text = block.get("text", "").strip()[:50]
                logger.debug(
                    f"    {idx}. '{text}...' (score: {score:.3f})"
                )
        
        # Additional check: reject if text is way too long (even if score is OK)
        if expected_avg_length and expected_avg_length > 0:
            actual_length = len(best_text)
            if actual_length > expected_avg_length * 1.8:  # 80% longer than expected
                logger.debug(
                    f"  Rejecting match: text length ({actual_length}) is "
                    f"{actual_length/expected_avg_length:.1f}x longer than expected "
                    f"({expected_avg_length:.1f}), likely concatenated fields"
                )
                # Try to find a better candidate with appropriate length
                for block, score, breakdown in scored_candidates[1:]:
                    candidate_text = block.get("text", "").strip()
                    candidate_length = len(candidate_text)
                    # Accept if length is more reasonable and score is decent
                    if (candidate_length <= expected_avg_length * 1.5 and 
                        score >= 0.25 and
                        breakdown.get("length_similarity", 0) > 0.3):
                        logger.debug(
                            f"  Using alternative candidate with better length "
                            f"({candidate_length} vs {actual_length})"
                        )
                        return block
                # If no better candidate found, reject
                return None
        
        # Return best match if score is above threshold
        if best_score >= 0.3:  # Minimum threshold
            return best_block
        else:
            logger.debug(
                f"  Best match score ({best_score:.3f}) below threshold (0.3), "
                f"returning None"
            )
    
    return None


def extract_fields_with_heuristics(
    field_positions: Dict[str, Dict[str, Any]],
    text_blocks: List[Dict[str, Any]],
    extraction_schema: Dict[str, str],
    label: Optional[str] = None,
    extraction_modes: Optional[Dict[str, str]] = None,
    box_patterns: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Extract field values using heuristic-based positional matching.
    
    Args:
        field_positions: Dictionary mapping field names to their reference positions.
        text_blocks: List of text blocks with coordinates from PDF.
        extraction_schema: Dictionary mapping field names to extraction descriptions.
        label: Optional document label for loading template knowledge.
        extraction_modes: Optional dictionary mapping field names to "exact" or "partial".
        box_patterns: Optional dictionary mapping box text to learned pattern data.
        
    Returns:
        Dictionary with extracted field values.
    """
    extracted_data: Dict[str, Any] = {}
    
    # Load template knowledge if label is provided
    template_knowledge = None
    if label:
        try:
            field_names = list(extraction_schema.keys())
            template_knowledge = find_template_by_label_and_fields(label, field_names)
            if template_knowledge:
                logger.debug(f"Using template knowledge for label '{label}'")
        except Exception as e:
            logger.debug(f"Could not load template knowledge: {str(e)}")
    
    # Estimate page dimensions once
    page_width, page_height = _estimate_page_dimensions(text_blocks)
    
    for field_name in extraction_schema.keys():
        # Skip if already extracted (e.g., from a multi-field box)
        if field_name in extracted_data and extracted_data[field_name] is not None:
            logger.debug(f"Field '{field_name}' already extracted, skipping")
            continue
            
        if field_name in field_positions:
            ref_position = field_positions[field_name]
            
            # DEBUG: Print reference position (gabarito)
            ref_x0 = ref_position.get("x0", 0)
            ref_y0 = ref_position.get("y0", 0)
            ref_x1 = ref_position.get("x1", 0)
            ref_y1 = ref_position.get("y1", 0)
            ref_page = ref_position.get("page", 0)
            logger.debug(
                f"[DEBUG BOX] Field '{field_name}' reference (gabarito): "
                f"page={ref_page}, "
                f"bbox=({ref_x0:.1f}, {ref_y0:.1f}, {ref_x1:.1f}, {ref_y1:.1f})"
            )
            
            # Get expected avg_length and type for this field from template knowledge
            expected_avg_length = None
            expected_type = None
            field_extraction_mode = None
            if template_knowledge:
                fields = template_knowledge.get("fields", {})
                field_metadata = fields.get(field_name)
                if field_metadata:
                    expected_avg_length = field_metadata.get("avg_length")
                    expected_type = field_metadata.get("type")
                    field_extraction_mode = field_metadata.get("extraction_mode")
            
            # Use extraction_mode from template if available, otherwise from extraction_modes dict
            if not field_extraction_mode and extraction_modes:
                field_extraction_mode = extraction_modes.get(field_name)
            
            best_block = find_best_matching_block(
                field_name,
                ref_position,
                text_blocks,
                page_width=page_width,
                page_height=page_height,
                template_knowledge=template_knowledge,
            )
            
            if best_block:
                box_text = best_block.get("text", "").strip()
                
                # Check if this box has a learned pattern
                pattern_data = None
                if box_patterns and box_text in box_patterns:
                    pattern_data = box_patterns[box_text]
                    logger.debug(
                        f"Field '{field_name}' found in box with learned pattern: "
                        f"{pattern_data.get('pattern', [])}"
                    )
                
                # If extraction mode is "partial" or box has learned pattern, use pattern-based extraction
                if field_extraction_mode == "partial" or pattern_data:
                    if pattern_data and field_name in pattern_data.get("field_to_segment", {}):
                        # Use learned pattern: segment current box and extract specific segment
                        from box_parser import segment_box_text
                        seg_idx = pattern_data["field_to_segment"][field_name]
                        # Re-segment the current box text (it may differ slightly from original)
                        current_segments = segment_box_text(box_text)
                        if seg_idx < len(current_segments):
                            value = current_segments[seg_idx].strip()
                            logger.debug(
                                f"Extracted '{value}' from box using learned pattern "
                                f"(segment {seg_idx} of {len(current_segments)}) for field '{field_name}'"
                            )
                        else:
                            # Fallback to box_parser
                            logger.debug(
                                f"Segment index {seg_idx} out of range (box has {len(current_segments)} segments), using box_parser"
                            )
                            # Extract ALL fields from this box at once for better accuracy
                            # Find all fields that map to this same position (with tolerance)
                            tolerance = 5.0  # pixels
                            fields_in_same_box = [
                                fname for fname in extraction_schema.keys()
                                if fname in field_positions
                            ]
                            # Filter by position similarity
                            fields_in_same_box = [
                                fname for fname in fields_in_same_box
                                if abs(field_positions[fname].get("x0", 0) - ref_position.get("x0", 0)) < tolerance and
                                abs(field_positions[fname].get("y0", 0) - ref_position.get("y0", 0)) < tolerance and
                                field_positions[fname].get("page") == ref_position.get("page")
                            ]
                            box_schema = {f: extraction_schema[f] for f in fields_in_same_box}
                            box_assignments = extract_fields_from_box(
                                box_text,
                                box_schema,
                                template_knowledge,
                            )
                            value = box_assignments.get(field_name)
                    else:
                        # Use box_parser to extract the correct substring
                        # Extract ALL fields from this box at once for better accuracy
                        # Find all fields that map to this same position (with tolerance)
                        tolerance = 5.0  # pixels
                        fields_in_same_box = [
                            fname for fname in extraction_schema.keys()
                            if fname in field_positions
                        ]
                        # Filter by position similarity
                        fields_in_same_box = [
                            fname for fname in fields_in_same_box
                            if abs(field_positions[fname].get("x0", 0) - ref_position.get("x0", 0)) < tolerance and
                            abs(field_positions[fname].get("y0", 0) - ref_position.get("y0", 0)) < tolerance and
                            field_positions[fname].get("page") == ref_position.get("page")
                        ]
                        box_schema = {f: extraction_schema[f] for f in fields_in_same_box}
                        
                        logger.debug(
                            f"Field '{field_name}' uses PARTIAL extraction mode, "
                            f"extracting {len(fields_in_same_box)} fields from box '{box_text[:50]}...'"
                        )
                        box_assignments = extract_fields_from_box(
                            box_text,
                            box_schema,
                            template_knowledge,
                        )
                        value = box_assignments.get(field_name)
                        
                        # Store extracted values for other fields in the same box
                        for other_field in fields_in_same_box:
                            if other_field != field_name and other_field not in extracted_data:
                                extracted_data[other_field] = box_assignments.get(other_field)
                                logger.debug(
                                    f"Also extracted '{other_field}' = '{box_assignments.get(other_field)}' "
                                    f"from same box"
                                )
                        
                        if not value:
                            # Fallback to simple extraction if box_parser didn't find it
                            logger.debug(
                                f"box_parser didn't find field '{field_name}', using fallback extraction"
                            )
                            value = _extract_field_from_box_text(
                                box_text,
                                field_name,
                                expected_type,
                                expected_avg_length,
                            )
                else:
                    # Try to extract the relevant substring if box contains multiple fields
                    value = _extract_field_from_box_text(
                        box_text,
                        field_name,
                        expected_type,
                        expected_avg_length,
                    )
                
                # If extraction changed the value, log it
                if value != box_text:
                    logger.debug(
                        f"Field '{field_name}': Extracted substring '{value}' "
                        f"from box text '{box_text[:50]}...'"
                    )
                
                # DEBUG: Print bounding box information
                box_x0 = best_block.get("x0", 0)
                box_y0 = best_block.get("y0", 0)
                box_x1 = best_block.get("x1", 0)
                box_y1 = best_block.get("y1", 0)
                box_page = best_block.get("page", 0)
                box_width = box_x1 - box_x0 if box_x1 > box_x0 else 0
                box_height = box_y1 - box_y0 if box_y1 > box_y0 else 0
                
                # Calculate distance from reference position
                ref_center_x = (ref_position.get("x0", 0) + ref_position.get("x1", 0)) / 2
                ref_center_y = (ref_position.get("y0", 0) + ref_position.get("y1", 0)) / 2
                box_center_x = (box_x0 + box_x1) / 2
                box_center_y = (box_y0 + box_y1) / 2
                distance = math.sqrt(
                    (box_center_x - ref_center_x) ** 2 + 
                    (box_center_y - ref_center_y) ** 2
                )
                
                logger.debug(
                    f"[DEBUG BOX] Field '{field_name}' extracted: "
                    f"page={box_page}, "
                    f"bbox=({box_x0:.1f}, {box_y0:.1f}, {box_x1:.1f}, {box_y1:.1f}), "
                    f"size=({box_width:.1f}x{box_height:.1f}), "
                    f"center=({box_center_x:.1f}, {box_center_y:.1f}), "
                    f"distance_from_ref={distance:.1f}px, "
                    f"text_length={len(value)}, "
                    f"text='{value[:50]}{'...' if len(value) > 50 else ''}'"
                )
                # Truncate if value is much longer than expected
                if value and expected_avg_length:
                    original_length = len(value)
                    value = _truncate_text_by_length(value, expected_avg_length, max_ratio=1.3)
                    if len(value) < original_length:
                        logger.debug(
                            f"[DEBUG BOX] Field '{field_name}': "
                            f"Truncated from {original_length} to {len(value)} chars"
                        )
                
                # Normalize: convert string "null" or "None" to Python None
                if isinstance(value, str) and value.lower() in ("null", "none", ""):
                    value = None
                extracted_data[field_name] = value
            else:
                extracted_data[field_name] = None
                logger.debug(f"Field '{field_name}': No matching block found")
        else:
            extracted_data[field_name] = None
            logger.debug(f"Field '{field_name}': No reference position in template")
    
    return extracted_data

