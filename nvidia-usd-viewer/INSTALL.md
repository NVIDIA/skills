# Instalação local — USD Viewer RTX Streaming

Guia passo a passo para rodar o viewer na sua máquina local.

---

## Pré-requisitos

| Requisito | Versão mínima | Verificar |
|-----------|--------------|----------|
| NVIDIA GPU (RTX) | Turing (RTX 2xxx) ou mais recente | `nvidia-smi` |
| CUDA driver | 12.x | `nvidia-smi \| grep CUDA` |
| Python | 3.10–3.13 | `python3 --version` |
| Node.js | 18+ | `node --version` |
| npm | 10+ | `npm --version` |
| Linux x86_64 | Ubuntu 20.04+ recomendado | `uname -m` |

> **Windows**: Funciona em Windows x86_64. Substitua `./setup.sh` pelos
> passos manuais abaixo. A skill tem referências em
> `skills/omniverse-realtime-viewer/references/windows-native-setup/`.

---

## Passo 1 — Baixar o código

### Opção A: Clonar diretamente do branch
```bash
git clone --branch claude/nvidia-usd-viewer-tutorial-sRK8G \
  https://github.com/ItamarIliuk/skills.git
cd skills/nvidia-usd-viewer
```

### Opção B: Criar o repositório separado
```bash
# 1. Crie o repo em github.com/new → nvidia-usd-viewer-tutorial

# 2. Clone o branch e extraia a subpasta:
git clone --branch claude/nvidia-usd-viewer-tutorial-sRK8G \
  https://github.com/ItamarIliuk/skills.git skills-tmp

git init nvidia-usd-viewer-tutorial
cd nvidia-usd-viewer-tutorial

git remote add tmp ../skills-tmp
git fetch tmp
git checkout tmp/claude/nvidia-usd-viewer-tutorial-sRK8G -- nvidia-usd-viewer
mv nvidia-usd-viewer/* . && rmdir nvidia-usd-viewer
git remote rm tmp

git remote add origin https://github.com/itamariliuk/nvidia-usd-viewer-tutorial.git
git add . && git commit -m "Initial USD viewer tutorial"
git push -u origin main
```

---

## Passo 2 — Instalar a skill no Claude Code (local)

Isso ensina o Claude Code a usar as APIs de `ovrtx`, `ovstream`, e WebRTC:

```bash
bash skills-install.sh
# ou diretamente:
npx skills add nvidia/skills --skill omniverse-realtime-viewer --agent claude-code
```

---

## Passo 3 — Setup do ambiente

```bash
bash setup.sh
```

O que o script faz:
1. Verifica GPU e driver (`nvidia-smi`)
2. Cria virtualenv Python em `.venv/`
3. Instala `ovrtx` do **índice PyPI da NVIDIA** (`https://pypi.nvidia.com`)
4. Instala `ovstream`, `warp-lang`, `numpy` do PyPI padrão
5. Valida o renderer (`ovrtx.Renderer()`) — **primeira execução pode demorar vários minutos** (compilação de shaders)
6. Instala dependências npm do frontend

### Se o setup falhar

**`ovrtx` não encontrado:**
```bash
# Confirme Python e plataforma:
python3 -c "import platform, sys; print(platform.platform(), sys.version)"
# Instale manualmente:
pip install ovrtx --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple
```

**Materiais magenta / plugins não carregam:**
```bash
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))')"
export LD_LIBRARY_PATH="${OVRTX_BIN_PATH}/plugins${LD_LIBRARY_PATH:+:}"
```

**`@nvidia/ov-web-rtc` não encontrado:**
```bash
# Confirme o .npmrc na pasta frontend/
cat frontend/.npmrc
# Instale com o registro explícito:
cd frontend
npm install --registry https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/ @nvidia/ov-web-rtc
```

---

## Passo 4 — Iniciar o servidor

```bash
bash run.sh
```

O servidor sobe em:
- **WebRTC streaming**: porta `49100` (sinalização)
- **Health check**: `http://localhost:8888/healthz`
- **Frontend dev**: `http://localhost:5173` (rodar em outra aba)

Em outro terminal:
```bash
cd frontend
npm run dev
```

Abra **http://localhost:5173** em um browser baseado em Chromium (Chrome, Edge, Brave).

---

## Passo 5 — Verificar que está rodando

```bash
# Health check — deve retornar "ok" após o primeiro frame renderizado:
curl http://localhost:8888/healthz

# Se travar em "not ready" por mais de 60s:
tail -50 logs/server.log  # ou veja saída do terminal do servidor
```

Sinais de sucesso nos logs do servidor:
```
ovrtx.Renderer created (GPU 0)
Stage loaded  rp=/Render/OVServer  cam=/OVCamera
Warmup done (frame_index=8)
ovstream WebRTC started  signaling=127.0.0.1:49100
/healthz → 200 ready
```

---

## Controles de câmera

| Ação | Input |
|------|-------|
| Orbit | Arrastar com botão esquerdo |
| Pan | Arrastar com botão direito |
| Zoom | Scroll |

Os eventos de mouse são enviados via NVST (input nativo do ovstream) — não há JSON de mouse.

---

## Carregar uma cena USD

Digite o caminho no toolbar e pressione **Load** (ou Enter):

```
assets/samples/scene.usda           # cena de exemplo incluída
/caminho/absoluto/para/cena.usd     # arquivo local
omniverse://servidor/projeto.usd    # Omniverse Nucleus
```

---

## Deploy na nuvem (Brev)

```bash
brev open https://github.com/itamariliuk/nvidia-usd-viewer-tutorial
```

Isso cria uma instância GPU (AWS g5.xlarge por padrão) e executa
`setup.sh` + `run.sh` automaticamente via `brev.json`.

---

## Estrutura de portas

| Porta | Protocolo | Uso |
|-------|-----------|-----|
| 5173  | HTTP | Frontend React (dev server) |
| 8888  | HTTP | Health check (`/healthz`) |
| 49100 | WebSocket | WebRTC signaling (ovstream) |
| dinâmica | UDP/SRTP | WebRTC media (negociado via ICE) |

Para acesso LAN/remoto passe `--public-ip SEU_IP` para o servidor e
`?host=SEU_IP&port=49100` na URL do browser.
