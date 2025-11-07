## Sistema de Extração Estruturada de Documentos (OCR + LLM)

Solução em Python 3.13 para extrair dados estruturados de PDFs (com OCR) combinando três estratégias: templates posicionais para documentos rígidos, heurísticas inteligentes para caixas multi‑campos e fallback com LLM para casos flexíveis ou campos faltantes. Aprende continuamente padrões/posições e reutiliza conhecimento entre execuções.

### Arquitetura (alto nível)

```
dataset.json → Classificação (rígida/flexível) →
  - Rígida: OCR com coordenadas → Hashing → [Template conhecido?] → Heurísticas posicionais → (LLM só p/ faltantes)
  - Flexível: Texto simples → LLM
→ Resultados em results/<arquivo>.json (+ output.json)
→ Aprendizado: templates em memória (sessão) + conhecimento persistente por label (templates/templates.json)
```

## Instalação (uv + Python 3.13)

Pré‑requisitos:
- Python 3.13 instalado
- macOS/Linux com bash/zsh (comandos abaixo)

Passo a passo:

```bash
# 1) Instale o uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) No diretório do projeto, crie e ative o ambiente 3.13
uv venv -p 3.13
source .venv/bin/activate  # zsh/bash

# 3) Instale dependências com uv
uv pip install -r requirements.txt

# 4) Configure sua API key da OpenAI
export OPENAI_API_KEY="sua_chave"
# (opcional) ou use um .env:
echo "OPENAI_API_KEY=sua_chave" > .env
```

## Como usar

```bash
# Execução básica (serial)
python main.py dataset.json

# Saída verbosa
python main.py dataset.json -v

# Paralelizar por label
python main.py dataset.json --parallel

# Medir tempos/uso de LLM e imprimir resumo
python main.py dataset.json --benchmark

# Desativar cache de resultados
python main.py dataset.json --no-cache
```

Saídas:
- Arquivo por PDF em `results/<nome_arquivo>.json` (apenas os campos do schema)
- Consolidação incremental em `results/output.json`

### Formato de entrada (dataset.json)

```json
[
  {
    "label": "carteira_oab",
    "extraction_schema": {
      "nome": "Nome do profissional",
      "inscricao": "Número de inscrição",
      "situacao": "Situação do profissional"
    },
    "pdf_path": "files/oab_1.pdf"
  }
]
```

### Formato de saída (por PDF)

```json
{
  "nome": "MARIA SILVA",
  "inscricao": "123456",
  "situacao": "Ativo"
}
```

## Estrutura do projeto

```
enter_fellowship/
  box_parser.py           
  cache.py                
  classification.py       
  evaluator.py            
  extraction.py           
  format_validator.py     
  hashing.py              
  heuristics.py           
  llm_utils.py            
  main.py                 
  oracle.py               
  pipeline.py             
  requirements.txt        
  schemas/
    __init__.py           
  templates/
    templates.json        
  results/                
  oracle_results/         
  files/                  
  dataset_*.json          
```

### O que faz cada módulo (1 frase)
- `main.py`: ponto de entrada CLI (parsing de args, logging, delega ao pipeline).
- `pipeline.py`: orquestra o fluxo por documento (cache → classificação → extração → salvamento/benchmark).
- `extraction.py`: extrai texto do PDF com/sem coordenadas via PyMuPDF.
- `classification.py`: classifica o label como “rígida” ou “flexível” usando LLM e cache em memória.
- `hashing.py`: gera hashes perceptuais estáveis (página inteira/metades) e decide similaridade de template.
- `oracle.py`: faz extração via LLM e cria templates em memória com posições, modos e padrões de caixas.
- `heuristics.py`: extrai campos por matching posicional usando pontuação espacial/semântica e conhecimento prévio.
- `box_parser.py`: segmenta caixas multi‑campos por delimitadores/transições e casa segmentos com campos.
- `format_validator.py`: infere/valida formato esperado (data/hora/número/UF) e corrige substrings válidas.
- `template_manager.py`: persiste conhecimento por `label+campos` (tipo médio, comprimento, padrões e splits) em `templates.json`.
- `cache.py`: cacheia cada resultado em `results/<pdf>.json` e guarda code_version para invalidar quando o código muda.
- `evaluator.py`: compara `results/` com `oracle_results/` e imprime relatório de acurácia.
- `llm_utils.py`: cliente OpenAI (carrega `OPENAI_API_KEY`) e chamadas de classificação/extração com resposta JSON.
- `schemas/__init__.py`: espaço para definições futuras de schemas/validações.

## Boas práticas e notas
- O sistema tenta sempre heurísticas posicionais antes de recorrer ao LLM e chama o LLM apenas para campos faltantes/invalidáveis.
- Templates por hash vivem na memória durante a execução; o conhecimento agregado por `label` persiste em `templates/templates.json`.
- O cache invalida automaticamente quando PDFs mudam ou quando a versão do código de extração se altera.

## Avaliação de acurácia (opcional)

```bash
python evaluator.py
```
Compara `results/*.json` com os gabaritos em `oracle_results/*.json` e imprime um resumo de acurácia geral/por arquivo/campo.
