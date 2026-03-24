# =============================================================================
# AutoVision — Análise de Danos Automotivos com Moondream AI
# =============================================================================
# Deploy: Streamlit Community Cloud
# API Key: configurada em st.secrets["MOONDREAM_API_KEY"]
#          (Settings > Secrets no painel do Streamlit Cloud)
# =============================================================================

import streamlit as st
import requests
import base64
import json
import re
import math
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoVision · Análise de Danos",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — IDENTIDADE VISUAL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fundo geral */
    .stApp {
        background: #0f1117;
        color: #e8eaf0;
    }

    /* Cabeçalho */
    .header-block {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border: 1px solid #2a3040;
        border-radius: 16px;
        padding: 32px 28px 24px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .header-block::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 6px;
        letter-spacing: -0.5px;
    }
    .header-sub {
        font-size: 0.92rem;
        color: #64748b;
        margin: 0;
    }

    /* Cards de resultado */
    .result-card {
        background: #151923;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 12px;
    }
    .result-card .label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #4b5563;
        margin-bottom: 6px;
    }
    .result-card .value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    /* Badge de severidade */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-ok       { background: #14532d; color: #4ade80; }
    .badge-baixa    { background: #1c3a20; color: #86efac; }
    .badge-media    { background: #451a03; color: #fb923c; }
    .badge-alta     { background: #450a0a; color: #f87171; }
    .badge-critico  { background: #3b0764; color: #e879f9; }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #1e2536;
        margin: 20px 0;
    }

    /* Botão customizado */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 0.95rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2a3040;
        border-radius: 12px;
        padding: 12px;
        background: #0d1018;
    }

    /* JSON block */
    .stJson {
        background: #0d1018 !important;
        border-radius: 10px;
    }

    /* Alertas */
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
MOONDREAM_QUERY_URL  = "https://api.moondream.ai/v1/query"
MOONDREAM_DETECT_URL = "https://api.moondream.ai/v1/detect"
MOONDREAM_POINT_URL  = "https://api.moondream.ai/v1/point"

PECAS_CONHECIDAS = [
    "para-brisa", "parabrisa", "windshield",
    "farol", "headlight",
    "para-choque", "parachoque", "bumper",
    "porta", "door",
    "retrovisor", "mirror",
    "vidro", "glass", "window",
    "capô", "capo", "hood",
    "teto", "roof",
    "paralama", "fender",
    "pneu", "tire", "roda", "wheel",
    "lanternas", "taillight",
]

DANOS_CONHECIDOS = [
    "trinca", "crack",
    "quebra", "quebrado", "broken",
    "arranhão", "arranhado", "scratch",
    "amassado", "amassamento", "dent",
    "deformação",
    "dano superficial",
    "dano severo",
]


# ──────────────────────────────────────────────────────────────────────────────
# FUNÇÕES UTILITÁRIAS
# ──────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Converte bytes de imagem para string base64 no formato esperado pela API."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def get_media_type(file_name: str) -> str:
    """Retorna o MIME type correto com base na extensão do arquivo."""
    ext = file_name.lower().split(".")[-1]
    mapping = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    return mapping.get(ext, "image/jpeg")


def call_moondream_query(image_url: str, question: str, api_key: str) -> str:
    """
    Chama o endpoint /query da API Moondream.

    Parâmetros:
        image_url: string base64 da imagem (data:image/jpeg;base64,...)
        question:  pergunta em linguagem natural
        api_key:   chave de autenticação da API

    Retorna:
        Texto da resposta ou mensagem de erro.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
    }
    payload = {
        "image_url": image_url,
        "question":  question,
        "stream":    False,
    }
    response = requests.post(MOONDREAM_QUERY_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("answer", "").strip()


def call_moondream_point(image_url: str, object_name: str, api_key: str) -> list:
    """
    Chama o endpoint /point da API Moondream para obter as coordenadas centrais
    exatas do objeto/dano na imagem.

    Retorna lista de pontos: [{"x": 0.65, "y": 0.42}, ...]
    Coordenadas normalizadas de 0 a 1.
    Retorna lista vazia em caso de falha (não bloqueia o fluxo principal).
    """
    headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
    }
    payload = {
        "image_url": image_url,
        "object":    object_name,
    }
    try:
        response = requests.post(MOONDREAM_POINT_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get("points", [])
    except Exception:
        # /point é opcional — falha silenciosa para não quebrar o MVP
        return []


def call_moondream_detect(image_url: str, object_name: str, api_key: str) -> list:
    """
    Fallback: chama /detect para bounding box caso /point não retorne resultado.
    Retorna lista de objetos com x_min, y_min, x_max, y_max (0 a 1).
    """
    headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
    }
    payload = {
        "image_url": image_url,
        "object":    object_name,
    }
    try:
        response = requests.post(MOONDREAM_DETECT_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get("objects", [])
    except Exception:
        return []


def inferir_peca(texto: str) -> str:
    """
    Extrai a peça automotiva da resposta do modelo de forma fuzzy.
    Se não reconhecer, retorna o primeiro segmento da resposta.
    """
    texto_lower = texto.lower()
    for peca in PECAS_CONHECIDAS:
        if peca in texto_lower:
            return peca
    # fallback: primeiras 5 palavras da resposta
    palavras = texto.split()
    return " ".join(palavras[:5]) if palavras else "não identificado"


def inferir_severidade(peca: str, tipo_dano: str, descricao: str) -> tuple[str, str]:
    """
    Aplica regras de negócio simples para classificar severidade.

    Retorna:
        (severidade_texto, badge_css_class)
    """
    peca_lower     = peca.lower()
    dano_lower     = tipo_dano.lower()
    desc_lower     = descricao.lower()

    sem_dano = any(t in desc_lower for t in [
        "no damage", "sem dano", "nenhum dano", "não há dano",
        "not visible", "não visível", "normal", "intact",
    ])
    if sem_dano:
        return "Sem dano aparente", "badge-ok"

    # Regras específicas de negócio
    if any(v in peca_lower for v in ["para-brisa", "parabrisa", "windshield", "vidro", "glass"]):
        if any(t in dano_lower for t in ["trinca", "crack", "quebra", "broken"]):
            return "🔴 CRÍTICO — Vidro comprometido", "badge-critico"

    if any(f in peca_lower for f in ["farol", "headlight", "lanterna", "taillight"]):
        if any(t in dano_lower for t in ["quebra", "broken", "trinca", "crack"]):
            return "Alta — Farol danificado", "badge-alta"

    if any(t in dano_lower for t in ["amassado", "dent", "deformação"]):
        return "Média — Dano estrutural", "badge-media"

    if any(t in dano_lower for t in ["arranhão", "scratch", "superficial"]):
        return "Baixa — Dano superficial", "badge-baixa"

    if any(t in dano_lower for t in ["severo", "severe", "grande", "extenso"]):
        return "Alta — Dano severo", "badge-alta"

    # Fallback com base em palavras-chave na descrição
    if any(t in desc_lower for t in ["severe", "severo", "crítico", "critical"]):
        return "Alta", "badge-alta"
    if any(t in desc_lower for t in ["moderate", "moderado", "média", "medium"]):
        return "Média", "badge-media"

    return "Média", "badge-media"


def desenhar_marcador_ponto(pil_image: Image.Image, pontos: list) -> Image.Image:
    """
    Desenha marcadores de precisão (crosshair + círculos pulsantes) nos pontos
    exatos do dano retornados pelo endpoint /point do Moondream.

    Coordenadas normalizadas 0–1 → convertidas para pixels.
    """
    img = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = pil_image.size
    # Tamanho do marcador proporcional à imagem
    raio_base = max(18, min(w, h) // 30)

    for ponto in pontos:
        cx = int(ponto["x"] * w)
        cy = int(ponto["y"] * h)

        # ── Círculo externo (halo) ──────────────────────────────────────────
        r_ext = raio_base * 3
        draw.ellipse(
            [cx - r_ext, cy - r_ext, cx + r_ext, cy + r_ext],
            outline=(239, 68, 68, 60),
            width=2,
        )

        # ── Círculo médio ───────────────────────────────────────────────────
        r_mid = raio_base * 2
        draw.ellipse(
            [cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid],
            outline=(239, 68, 68, 140),
            width=2,
        )

        # ── Círculo interno preenchido ──────────────────────────────────────
        r_inn = raio_base
        draw.ellipse(
            [cx - r_inn, cy - r_inn, cx + r_inn, cy + r_inn],
            fill=(239, 68, 68, 200),
            outline=(255, 255, 255, 230),
            width=2,
        )

        # ── Crosshair (cruz) ────────────────────────────────────────────────
        arm = raio_base * 4
        gap = raio_base + 4  # gap entre centro e início do traço

        # Horizontal
        draw.line([cx - arm, cy, cx - gap, cy], fill=(239, 68, 68, 220), width=2)
        draw.line([cx + gap, cy, cx + arm, cy], fill=(239, 68, 68, 220), width=2)
        # Vertical
        draw.line([cx, cy - arm, cx, cy - gap], fill=(239, 68, 68, 220), width=2)
        draw.line([cx, cy + gap, cx, cy + arm], fill=(239, 68, 68, 220), width=2)

        # ── Ponto central branco ────────────────────────────────────────────
        dot = 3
        draw.ellipse(
            [cx - dot, cy - dot, cx + dot, cy + dot],
            fill=(255, 255, 255, 255),
        )

    resultado = Image.alpha_composite(img, overlay)
    return resultado.convert("RGB")


def desenhar_bounding_boxes(pil_image: Image.Image, boxes: list) -> Image.Image:
    """
    Fallback: desenha bounding boxes quando /point não retorna coordenadas.
    Também desenha o ponto central dentro de cada box.
    """
    img = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = pil_image.size
    for box in boxes:
        x0 = int(box["x_min"] * w)
        y0 = int(box["y_min"] * h)
        x1 = int(box["x_max"] * w)
        y1 = int(box["y_max"] * h)

        # Fundo semitransparente
        draw.rectangle([x0, y0, x1, y1], fill=(239, 68, 68, 25))
        # Borda
        draw.rectangle([x0, y0, x1, y1], outline=(239, 68, 68, 220), width=3)

        # Marcador no centro da box como fallback
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r = 8
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(239, 68, 68, 200), outline=(255, 255, 255, 220), width=2)

    resultado = Image.alpha_composite(img, overlay)
    return resultado.convert("RGB")


def analisar_imagem(image_bytes: bytes, file_name: str, api_key: str) -> dict:
    """
    Orquestra as chamadas à API Moondream e retorna um dicionário estruturado.

    Fluxo:
        1. Identificar a peça automotiva principal
        2. Pergunta binária YES/NO — ancora confiável para ha_dano
        3. Se há dano: detalhar tipo, localização e severidade
        4. Aplicar regras de negócio de severidade
        5. Localizar o dano na imagem via /point (fallback /detect)
    """
    media_type = get_media_type(file_name)
    image_url  = image_to_base64(image_bytes, media_type)

    # ── PASSO 1: Identificar a peça ──────────────────────────────────────────
    resposta_peca = call_moondream_query(
        image_url,
        (
            "What is the main automotive part or component visible in this image? "
            "Be specific and concise. Answer with just the part name in one or two words, "
            "for example: windshield, headlight, bumper, door, mirror, hood, fender, tire."
        ),
        api_key,
    )
    peca_identificada = inferir_peca(resposta_peca)

    # ── PASSO 2: Pergunta binária — sinal principal para ha_dano ─────────────
    # Esta é a âncora mais confiável. Resposta esperada: "YES" ou "NO".
    # Não depende de parsing de frases longas, que é onde o erro ocorria antes.
    resposta_binaria = call_moondream_query(
        image_url,
        (
            f"Look very carefully at every part of this {peca_identificada} in the image. "
            "Check for any scratch, scuff, mark, stain, dent, crack, paint chip, "
            "discoloration, or any other imperfection — even subtle ones. "
            "Is there ANY visible damage or imperfection present? "
            "Answer with a single word: YES or NO."
        ),
        api_key,
    )
    # Determina ha_dano de forma binária e robusta — imune a variações de frase
    ha_dano = resposta_binaria.strip().upper().startswith("YES")

    # ── PASSO 3: Detalhar o dano (só se confirmado no passo 2) ──────────────
    resposta_dano = ""
    tipo_dano_raw = "none"
    tipo_dano     = "sem dano"

    if ha_dano:
        resposta_dano = call_moondream_query(
            image_url,
            (
                f"There is visible damage on the {peca_identificada}. "
                "Describe it precisely: "
                "(1) What type of damage is it? (scratch, dent, crack, break, stain, etc.) "
                "(2) Where exactly on the part is it located? "
                "(3) How severe does it appear? "
                "Be direct and specific."
            ),
            api_key,
        )

        tipo_dano_raw = call_moondream_query(
            image_url,
            (
                "Look at the damage visible in this image. "
                "Reply with ONE word only — the damage type. "
                "Choose exactly one: crack / scratch / dent / break / stain / superficial. "
                "Do NOT explain. Just the one word."
            ),
            api_key,
        )

        tipo_dano_map = {
            "crack":       "trinca",
            "scratch":     "arranhão",
            "dent":        "amassado",
            "break":       "quebra",
            "stain":       "mancha",
            "deformation": "deformação",
            "superficial": "dano superficial",
            "none":        "sem dano",
        }
        # Busca correspondência parcial para tolerar respostas como "scratches"
        tipo_dano = "dano visível"  # fallback se nenhum match
        for key, valor in tipo_dano_map.items():
            if key in tipo_dano_raw.lower():
                tipo_dano = valor
                break
    else:
        # Pede descrição sucinta mesmo quando sem dano
        resposta_dano = call_moondream_query(
            image_url,
            f"Briefly confirm the condition of the {peca_identificada} — no damage visible.",
            api_key,
        )

    # ── PASSO 4: Regras de negócio — severidade ──────────────────────────────
    severidade_texto, badge_class = inferir_severidade(
        peca_identificada, tipo_dano, resposta_dano
    )

    # ── PASSO 5: Localização precisa do dano (/point → /detect como fallback) ─
    pontos = []
    bboxes = []

    if ha_dano:
        # Descrição específica aumenta a precisão do /point
        objeto_busca = tipo_dano_raw.split()[0] if tipo_dano_raw != "none" else "damage"

        # Tenta /point — retorna coordenada central exata do dano
        pontos = call_moondream_point(image_url, objeto_busca, api_key)

        if not pontos:
            pontos = call_moondream_point(image_url, "visible damage", api_key)

        # Se /point falhar, fallback para /detect (bounding box)
        if not pontos:
            bboxes = call_moondream_detect(image_url, objeto_busca, api_key)

        if not pontos and not bboxes:
            bboxes = call_moondream_detect(image_url, "damage mark", api_key)

    return {
        "peca":             peca_identificada,
        "resposta_peca":    resposta_peca,
        "resposta_binaria": resposta_binaria,
        "ha_dano":          ha_dano,
        "tipo_dano":        tipo_dano,
        "severidade":       severidade_texto,
        "badge_class":      badge_class,
        "descricao":        resposta_dano,
        "pontos":           pontos,
        "bboxes":           bboxes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE — CABEÇALHO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <p class="header-title">🔍 AutoVision</p>
    <p class="header-sub">
        Análise inteligente de peças e danos automotivos com Moondream AI.<br>
        Faça upload de uma imagem para iniciar o diagnóstico.
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LEITURA DA API KEY (via st.secrets — configurado no Streamlit Cloud)
# ──────────────────────────────────────────────────────────────────────────────
# No Streamlit Cloud: vá em Settings > Secrets e adicione:
#   MOONDREAM_API_KEY = "seu_token_aqui"
#
# Localmente: crie o arquivo .streamlit/secrets.toml com:
#   MOONDREAM_API_KEY = "seu_token_aqui"
# ─────────────────────────────────────────────────────────────────────────────

api_key = None
try:
    api_key = st.secrets["MOONDREAM_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "🔑 **API Key não encontrada.**\n\n"
        "Configure `MOONDREAM_API_KEY` em **Settings > Secrets** no Streamlit Cloud, "
        "ou em `.streamlit/secrets.toml` no ambiente local."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# UPLOAD DA IMAGEM
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Selecione uma imagem",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceitos: JPG, JPEG, PNG",
)

if uploaded_file is None:
    st.markdown("""
    <div style="
        background: #0d1018;
        border: 1px dashed #2a3040;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        color: #374151;
        margin-top: 12px;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">🚗</div>
        <div style="font-size: 0.9rem;">
            Faça upload de uma foto de para-brisa, farol, para-choque, porta ou qualquer peça automotiva.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PREVIEW DA IMAGEM
# ──────────────────────────────────────────────────────────────────────────────
image_bytes = uploaded_file.read()

try:
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
except Exception:
    st.error("❌ Não foi possível abrir a imagem. Certifique-se de que é um arquivo válido (JPG/PNG).")
    st.stop()

st.image(pil_image, caption="Imagem enviada", use_container_width=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# BOTÃO DE ANÁLISE
# ──────────────────────────────────────────────────────────────────────────────
if not st.button("🔍  Iniciar Análise de Danos"):
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DA ANÁLISE
# ──────────────────────────────────────────────────────────────────────────────
resultado = None

with st.spinner("Analisando imagem com Moondream AI..."):
    try:
        resultado = analisar_imagem(image_bytes, uploaded_file.name, api_key)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 401:
            st.error("🔑 **API Key inválida ou expirada.** Verifique o valor em Secrets.")
        elif status == 429:
            st.warning("⏳ **Limite de requisições atingido.** Aguarde alguns segundos e tente novamente.")
        else:
            st.error(f"❌ **Erro na API Moondream (HTTP {status}):** {e}")
        st.stop()
    except requests.exceptions.ConnectionError:
        st.error("🌐 **Falha de conexão.** Verifique sua internet e tente novamente.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⏱️ **Tempo de resposta excedido.** A API demorou demais para responder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Erro inesperado:** {e}")
        st.stop()

if not resultado:
    st.error("A análise não retornou resultado. Tente novamente.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# RESULTADO — IMAGEM COM MARCAÇÃO PRECISA DO DANO
# ──────────────────────────────────────────────────────────────────────────────
if resultado["pontos"] or resultado["bboxes"]:
    st.subheader("📍 Localização do Dano")

    if resultado["pontos"]:
        # Marcador de precisão via /point (crosshair + círculos)
        imagem_anotada = desenhar_marcador_ponto(pil_image, resultado["pontos"])
        st.image(imagem_anotada, caption="⊕ Ponto exato do dano identificado", use_container_width=True)
    else:
        # Fallback: bounding box via /detect
        imagem_anotada = desenhar_bounding_boxes(pil_image, resultado["bboxes"])
        st.image(imagem_anotada, caption="Região do dano identificada", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTADO — CARDS VISUAIS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.subheader("📋 Diagnóstico")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="result-card">
        <div class="label">Peça Identificada</div>
        <div class="value">🔧 {resultado['peca'].title()}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    dano_icon  = "⚠️" if resultado["ha_dano"] else "✅"
    dano_texto = "Dano detectado" if resultado["ha_dano"] else "Sem dano aparente"
    st.markdown(f"""
    <div class="result-card">
        <div class="label">Status do Dano</div>
        <div class="value">{dano_icon} {dano_texto}</div>
    </div>
    """, unsafe_allow_html=True)

if resultado["ha_dano"]:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Tipo de Dano</div>
            <div class="value">💥 {resultado['tipo_dano'].title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        badge = resultado["badge_class"]
        sev   = resultado["severidade"]
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Severidade</div>
            <div class="value">
                <span class="badge {badge}">{sev}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Descrição completa
st.markdown(f"""
<div class="result-card" style="margin-top: 4px;">
    <div class="label">Descrição da Análise</div>
    <div class="value" style="font-size: 0.95rem; font-weight: 400; line-height: 1.6; color: #cbd5e1;">
        {resultado['descricao']}
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTADO — JSON ESTRUTURADO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

json_saida = {
    "peca":       resultado["peca"],
    "ha_dano":    resultado["ha_dano"],
    "tipo_dano":  resultado["tipo_dano"],
    "severidade": resultado["severidade"],
    "descricao":  resultado["descricao"],
}

with st.expander("🗂️  Ver saída JSON estruturada"):
    st.json(json_saida)

# ──────────────────────────────────────────────────────────────────────────────
# RODAPÉ
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #374151; font-size: 0.78rem; margin-top: 40px; padding-bottom: 20px;">
    Powered by <strong style="color: #4b5563;">Moondream AI</strong> · AutoVision MVP
</div>
""", unsafe_allow_html=True)
