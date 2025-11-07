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

### TL;DR / Quickstart

```bash
uv venv -p 3.13 && source .venv/bin/activate
uv pip install -r requirements.txt
export OPENAI_API_KEY="sua_chave"
python main.py dataset*.json
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
├── box_parser.py                 # Segmenta caixas multi-campos e casa segmentos com campos
├── cache.py                      # Cache de resultados + versionamento de código
├── classification.py             # Classificação rígida/flexível via LLM (com cache por label)
├── evaluator.py                  # Avaliação de acurácia (results vs oracle_results)
├── extraction.py                 # Extração de texto com/sem coordenadas (PyMuPDF)
├── format_validator.py           # Inferência/validação de formatos e extração de substrings válidas
├── hashing.py                    # Hash perceptual estável (full/left/right) e matching de template
├── heuristics.py                 # Matching posicional com pontuação composta e truncamentos defensivos
├── llm_utils.py                  # Cliente OpenAI e utilitários de classificação/extração via LLM
├── main.py                       # CLI (parâmetros, logging, delega ao pipeline)
├── oracle.py                     # Extração via LLM + criação/uso de templates em memória
├── pipeline.py                   # Orquestração: cache → classificação → extração → salvamento/benchmark
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── templates.json           # Conhecimento persistente por label (gerado/atualizado em runtime)
├── results/                     # Saídas por PDF e output.json (gerados em runtime)
│   └── *.json
├── oracle_results/              # Gabaritos para avaliação (opcional)
│   └── *.json
├── files/                       # Exemplos pequenos de PDFs para teste rápido
│   └── *.pdf
├── dataset*.json               # Datasets de entrada (ex.: dataset.json, dataset_RG.json, ...)
├── dataset_CNH/                 # PDFs grandes (ignorados no Git)
├── dataset_RG/                  # PDFs grandes (ignorados no Git)
└── dataset_compras_BNDES/       # PDFs grandes (ignorados no Git)
```



## Custo x acurácia (resumo da estratégia)
- Classificação por label com cache (minimiza chamadas ao LLM).
- Hashing perceptual multi‑região + templates em memória para reuso imediato.
- Heurísticas posicionais e parsing de caixas multi‑campos antes do LLM.
- LLM apenas para campos faltantes/invalidáveis (incremental) e para documentos flexíveis.
- Conhecimento persistente por `label+campos` em `templates/templates.json` (tipos, comprimentos, padrões, delimitadores).
- Cache de resultados em `results/` com versionamento do código para invalidar automaticamente.

## Boas práticas e notas
- O sistema tenta sempre heurísticas posicionais antes de recorrer ao LLM e chama o LLM apenas para campos faltantes/invalidáveis.
- Templates por hash vivem na memória durante a execução; o conhecimento agregado por `label` persiste em `templates/templates.json`.
- O cache invalida automaticamente quando PDFs mudam ou quando a versão do código de extração se altera.

## Troubleshooting
- Erro de API: verifique `OPENAI_API_KEY` e conectividade; use `-v` para logs.
- Erro PyMuPDF: cheque instalação do `pymupdf` e permissões de leitura do PDF.
- Sem resultados em `results/`: confirme caminhos do `dataset.json` e existência dos arquivos.

## Avaliação de acurácia (opcional)

```bash
python evaluator.py
```
Compara `results/*.json` com os gabaritos em `oracle_results/*.json` e imprime um resumo de acurácia geral/por arquivo/campo.
