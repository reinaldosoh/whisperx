# Deploy WhisperX no EasyPanel (CPU-only)

## Visão Geral

Este guia explica como fazer deploy do WhisperX no EasyPanel sem GPU.

## Arquivos Criados

- `Dockerfile` - Imagem Docker otimizada para CPU
- `api.py` - API REST com FastAPI
- `docker-compose.yml` - Configuração Docker Compose

---

## Opção 1: Deploy via App Docker no EasyPanel

### Passo 1: Criar App no EasyPanel

1. Acesse seu painel EasyPanel
2. Clique em **"Create App"**
3. Selecione **"Docker"** ou **"App from Git"**

### Passo 2: Configurar o App

Se usando **Git Repository**:
- Repository URL: `https://github.com/m-bain/whisperX` (ou seu fork)
- Branch: `main`
- Dockerfile Path: `Dockerfile`

Se usando **Docker Image** (build local primeiro):
```bash
docker build -t whisperx-api .
docker tag whisperx-api seu-registry/whisperx-api:latest
docker push seu-registry/whisperx-api:latest
```

### Passo 3: Configurar Variáveis de Ambiente

No EasyPanel, adicione estas variáveis:

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `DEVICE` | `cpu` | Força uso de CPU |
| `COMPUTE_TYPE` | `int8` | Tipo de computação leve |
| `DEFAULT_MODEL` | `base` | Modelo padrão (base é recomendado para CPU) |

### Passo 4: Configurar Porta

- **Container Port**: `8000`
- **Domain**: Configure seu domínio ou use o domínio do EasyPanel

### Passo 5: Configurar Recursos

Recomendações mínimas para CPU:
- **Memory**: 4GB (mínimo), 8GB (recomendado)
- **CPU**: 2 cores (mínimo), 4 cores (recomendado)

Para modelos maiores:
- `small`: 4GB RAM
- `medium`: 8GB RAM
- `large-v2`: 16GB+ RAM

---

## Opção 2: Deploy via Docker Compose

Se seu EasyPanel suporta Docker Compose:

1. Faça upload do projeto
2. Use o arquivo `docker-compose.yml` incluído
3. Execute: `docker-compose up -d`

---

## Usando a API

### Endpoints Disponíveis

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Status do serviço |
| GET | `/health` | Health check |
| GET | `/models` | Lista modelos disponíveis |
| POST | `/transcribe` | Transcrever áudio |

### Exemplo: Transcrever Áudio

```bash
curl -X POST "https://seu-dominio.com/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=base" \
  -F "language=pt"
```

### Exemplo com Python

```python
import requests

url = "https://seu-dominio.com/transcribe"
files = {"file": open("audio.mp3", "rb")}
data = {
    "model": "base",
    "language": "pt",
    "align": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

### Parâmetros do /transcribe

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `file` | File | (obrigatório) | Arquivo de áudio |
| `model` | string | `base` | Modelo Whisper |
| `language` | string | auto | Código do idioma (pt, en, es...) |
| `align` | bool | `true` | Ativar alinhamento de palavras |
| `diarize` | bool | `false` | Ativar identificação de speakers |
| `hf_token` | string | - | Token HuggingFace (para diarização) |
| `min_speakers` | int | - | Mínimo de speakers |
| `max_speakers` | int | - | Máximo de speakers |

---

## Modelos Disponíveis

| Modelo | Velocidade | Precisão | RAM (CPU) |
|--------|------------|----------|-----------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐ | ~1GB |
| `base` | ⚡⚡⚡⚡ | ⭐⭐ | ~1GB |
| `small` | ⚡⚡⚡ | ⭐⭐⭐ | ~2GB |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐ | ~5GB |
| `large-v2` | ⚡ | ⭐⭐⭐⭐⭐ | ~10GB |

**Recomendação para CPU**: Use `base` ou `small`

---

## Diarização (Identificação de Speakers)

Para usar diarização, você precisa:

1. Criar conta no [HuggingFace](https://huggingface.co)
2. Gerar um token de acesso [aqui](https://huggingface.co/settings/tokens)
3. Aceitar os termos dos modelos:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Exemplo com diarização:
```bash
curl -X POST "https://seu-dominio.com/transcribe" \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "hf_token=hf_xxxxx" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

---

## Troubleshooting

### Erro de memória
- Reduza o modelo: use `tiny` ou `base`
- Aumente a RAM do container no EasyPanel

### Transcrição lenta
- Normal para CPU, especialmente com modelos grandes
- Use `tiny` ou `base` para velocidade
- Áudios longos demoram mais

### Idioma errado detectado
- Especifique o idioma manualmente: `language=pt`

### Container não inicia
- Verifique os logs no EasyPanel
- Certifique-se que a porta 8000 está configurada

---

## Idiomas Suportados

O WhisperX suporta múltiplos idiomas. Alguns com modelos de alinhamento nativos:
- `en` - Inglês
- `pt` - Português
- `es` - Espanhol
- `fr` - Francês
- `de` - Alemão
- `it` - Italiano
- E muitos outros via HuggingFace

---

## Suporte

- Repositório oficial: https://github.com/m-bain/whisperX
- Issues: https://github.com/m-bain/whisperX/issues
