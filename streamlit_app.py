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


# Paleta de cores para múltiplos marcadores — uma cor por peça danificada
CORES_MARCADORES = [
    (239, 68,  68),   # vermelho
    (59,  130, 246),  # azul
    (34,  197, 94),   # verde
    (251, 191, 36),   # amarelo
    (168, 85,  247),  # roxo
    (20,  184, 166),  # teal
]


def desenhar_marcadores_multi(
    pil_image: Image.Image,
    pecas_danificadas: list,          # lista de dicts com "label", "pontos", "bboxes", "cor"
) -> Image.Image:
    """
    Desenha marcadores de precisão para MÚLTIPLAS peças danificadas.
    Cada peça recebe uma cor diferente com label identificando qual é.

    pecas_danificadas: [
        {"label": "bumper", "pontos": [{"x":0.5,"y":0.7}], "bboxes": [], "cor": (R,G,B)},
        ...
    ]
    """
    img     = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    w, h      = pil_image.size
    raio_base = max(16, min(w, h) // 32)

    for peca in pecas_danificadas:
        cor   = peca.get("cor", (239, 68, 68))
        label = peca.get("label", "dano")
        r, g, b = cor

        # ── Desenha por pontos (/point) ──────────────────────────────────────
        for ponto in peca.get("pontos", []):
            cx = int(ponto["x"] * w)
            cy = int(ponto["y"] * h)
            _desenhar_crosshair(draw, cx, cy, raio_base, r, g, b, label)

        # ── Fallback: bounding boxes (/detect) ───────────────────────────────
        for box in peca.get("bboxes", []):
            x0 = int(box["x_min"] * w)
            y0 = int(box["y_min"] * h)
            x1 = int(box["x_max"] * w)
            y1 = int(box["y_max"] * h)
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 20))
            draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 200), width=3)
            # Ponto central na box
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            _desenhar_crosshair(draw, cx, cy, raio_base, r, g, b, label)

    resultado = Image.alpha_composite(img, overlay)
    return resultado.convert("RGB")


def _desenhar_crosshair(draw, cx, cy, raio_base, r, g, b, label=""):
    """Desenha crosshair + halo + label para um único ponto."""
    # Halo externo
    r_ext = raio_base * 3
    draw.ellipse([cx-r_ext, cy-r_ext, cx+r_ext, cy+r_ext],
                 outline=(r, g, b, 55), width=2)
    # Círculo médio
    r_mid = raio_base * 2
    draw.ellipse([cx-r_mid, cy-r_mid, cx+r_mid, cy+r_mid],
                 outline=(r, g, b, 130), width=2)
    # Círculo interno preenchido
    draw.ellipse([cx-raio_base, cy-raio_base, cx+raio_base, cy+raio_base],
                 fill=(r, g, b, 190), outline=(255, 255, 255, 230), width=2)
    # Crosshair
    arm = raio_base * 4
    gap = raio_base + 4
    draw.line([cx-arm, cy, cx-gap, cy], fill=(r, g, b, 210), width=2)
    draw.line([cx+gap, cy, cx+arm, cy], fill=(r, g, b, 210), width=2)
    draw.line([cx, cy-arm, cx, cy-gap], fill=(r, g, b, 210), width=2)
    draw.line([cx, cy+gap, cx, cy+arm], fill=(r, g, b, 210), width=2)
    # Ponto central branco
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255, 255, 255, 255))
    # Label da peça acima do marcador
    if label:
        texto = label.upper()
        pad   = 5
        tx    = cx - len(texto) * 4
        ty    = cy - raio_base * 3 - 22
        # Fundo do label
        draw.rectangle(
            [tx - pad, ty - pad, tx + len(texto) * 8 + pad, ty + 16 + pad],
            fill=(r, g, b, 200),
        )
        draw.text((tx, ty), texto, fill=(255, 255, 255, 255))


# Mantém assinaturas antigas para compatibilidade interna
def desenhar_marcador_ponto(pil_image, pontos):
    pecas = [{"label": "dano", "pontos": pontos, "bboxes": [], "cor": CORES_MARCADORES[0]}]
    return desenhar_marcadores_multi(pil_image, pecas)


def desenhar_bounding_boxes(pil_image, boxes):
    pecas = [{"label": "dano", "pontos": [], "bboxes": boxes, "cor": CORES_MARCADORES[0]}]
    return desenhar_marcadores_multi(pil_image, pecas)


def analisar_imagem(image_bytes: bytes, file_name: str, api_key: str) -> dict:
    """
    Analisa a imagem identificando TODAS as peças danificadas presentes,
    não apenas a principal.

    Fluxo:
        1. Perguntar quais peças com dano estão visíveis → lista
        2. Para cada peça: confirmar dano (YES/NO) + tipo + localizar (/point)
        3. Montar resultado consolidado com uma entrada por peça danificada
    """
    media_type = get_media_type(file_name)
    image_url  = image_to_base64(image_bytes, media_type)

    # ── PASSO 1: Listar TODAS as peças danificadas na imagem ─────────────────
    # Pergunta aberta que força o modelo a varrer toda a imagem
    resposta_lista = call_moondream_query(
        image_url,
        (
            "Look carefully at the entire image. "
            "List ALL automotive parts that show ANY visible damage, scratch, dent, "
            "crack, break, stain, or imperfection. "
            "Reply ONLY with a comma-separated list of part names. "
            "Example: bumper, taillight, door. "
            "If nothing is damaged, reply: none."
        ),
        api_key,
    )

    # Parseia a lista de peças
    pecas_raw = [p.strip().lower() for p in resposta_lista.split(",") if p.strip()]
    pecas_raw = [p for p in pecas_raw if p not in ("none", "no damage", "nothing")]

    # Fallback: se não retornou lista, identifica peça principal
    if not pecas_raw:
        peca_principal = call_moondream_query(
            image_url,
            (
                "What is the main automotive part visible in this image? "
                "Answer with just the part name in 1-2 words."
            ),
            api_key,
        ).strip().lower()
        pecas_raw = [peca_principal] if peca_principal else ["car part"]

    # ── PASSO 2: Analisar cada peça individualmente ──────────────────────────
    pecas_analisadas = []   # lista de dicts com info completa por peça

    tipo_dano_map = {
        "crack":       "trinca",
        "scratch":     "arranhão",
        "dent":        "amassado",
        "break":       "quebra",
        "broken":      "quebra",
        "stain":       "mancha",
        "deformation": "deformação",
        "superficial": "dano superficial",
        "none":        "sem dano",
    }

    for idx, peca in enumerate(pecas_raw[:5]):   # limita a 5 peças para não estourar rate limit
        cor = CORES_MARCADORES[idx % len(CORES_MARCADORES)]

        # 2a. Confirmar dano com pergunta binária
        confirmacao = call_moondream_query(
            image_url,
            (
                f"Focus only on the {peca}. "
                "Is there visible damage, scratch, dent, crack, break or any imperfection on it? "
                "Answer YES or NO only."
            ),
            api_key,
        )
        ha_dano_peca = confirmacao.strip().upper().startswith("YES")

        if not ha_dano_peca:
            continue   # pula peça sem dano confirmado

        # 2b. Tipo de dano
        tipo_raw = call_moondream_query(
            image_url,
            (
                f"What is the damage type on the {peca}? "
                "Reply with ONE word only: crack / scratch / dent / break / stain / superficial."
            ),
            api_key,
        ).strip().lower()

        tipo_dano = "dano visível"
        for key, valor in tipo_dano_map.items():
            if key in tipo_raw:
                tipo_dano = valor
                break

        # 2c. Descrição detalhada
        descricao = call_moondream_query(
            image_url,
            (
                f"Describe the damage on the {peca} in one sentence: "
                "type, location on the part, and severity."
            ),
            api_key,
        )

        # 2d. Severidade via regras de negócio
        severidade_texto, badge_class = inferir_severidade(peca, tipo_dano, descricao)

        # 2e. Localizar o dano via /point → fallback /detect
        pontos = call_moondream_point(image_url, f"damage on {peca}", api_key)
        if not pontos:
            pontos = call_moondream_point(image_url, peca, api_key)

        bboxes = []
        if not pontos:
            bboxes = call_moondream_detect(image_url, f"damaged {peca}", api_key)
        if not pontos and not bboxes:
            bboxes = call_moondream_detect(image_url, peca, api_key)

        pecas_analisadas.append({
            "peca":       peca,
            "tipo_dano":  tipo_dano,
            "severidade": severidade_texto,
            "badge_class": badge_class,
            "descricao":  descricao,
            "pontos":     pontos,
            "bboxes":     bboxes,
            "cor":        cor,
            "label":      peca,
        })

    # Se nenhuma peça passou pela confirmação, monta resultado "sem dano"
    ha_dano_geral = len(pecas_analisadas) > 0

    # Descrição consolidada
    if ha_dano_geral:
        desc_geral = "; ".join(
            f"{p['peca']}: {p['tipo_dano']}" for p in pecas_analisadas
        )
    else:
        desc_geral = "Nenhum dano visível foi identificado na imagem."

    return {
        "ha_dano":          ha_dano_geral,
        "pecas_analisadas": pecas_analisadas,        # lista completa por peça
        "descricao_geral":  desc_geral,
        "lista_raw":        resposta_lista,
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
# RESULTADO — IMAGEM COM MARCAÇÃO MULTI-PEÇA
# ──────────────────────────────────────────────────────────────────────────────
pecas = resultado.get("pecas_analisadas", [])
tem_marcacoes = any(p["pontos"] or p["bboxes"] for p in pecas)

if tem_marcacoes:
    st.subheader("📍 Localização dos Danos")
    imagem_anotada = desenhar_marcadores_multi(pil_image, pecas)
    st.image(imagem_anotada, caption="⊕ Cada cor representa uma peça danificada", use_container_width=True)

    # Legenda de cores
    if len(pecas) > 1:
        cols_leg = st.columns(len(pecas))
        for i, peca_info in enumerate(pecas):
            r, g, b = peca_info["cor"]
            with cols_leg[i]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<div style='width:14px;height:14px;border-radius:50%;"
                    f"background:rgb({r},{g},{b});flex-shrink:0;'></div>"
                    f"<span style='font-size:0.78rem;color:#94a3b8;'>{peca_info['peca'].title()}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ──────────────────────────────────────────────────────────────────────────────
# RESULTADO — CARDS POR PEÇA DANIFICADA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.subheader("📋 Diagnóstico")

if not resultado["ha_dano"]:
    st.success("✅ **Nenhum dano visível identificado** na imagem analisada.")
else:
    st.markdown(
        f"<p style='color:#64748b;font-size:0.88rem;margin-bottom:16px;'>"
        f"🔍 {len(pecas)} peça(s) com dano identificada(s)</p>",
        unsafe_allow_html=True,
    )

    for peca_info in pecas:
        r, g, b  = peca_info["cor"]
        badge    = peca_info["badge_class"]
        sev      = peca_info["severidade"]

        st.markdown(
            f"<div class='result-card' style='border-left:4px solid rgb({r},{g},{b});'>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.markdown(
                f"<div class='label'>Peça</div>"
                f"<div class='value'>🔧 {peca_info['peca'].title()}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='label'>Tipo de Dano</div>"
                f"<div class='value'>💥 {peca_info['tipo_dano'].title()}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='label'>Severidade</div>"
                f"<div class='value'><span class='badge {badge}'>{sev}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;'>"
            f"{peca_info['descricao']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTADO — JSON ESTRUTURADO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

json_saida = {
    "ha_dano": resultado["ha_dano"],
    "total_pecas_danificadas": len(pecas),
    "pecas": [
        {
            "peca":       p["peca"],
            "tipo_dano":  p["tipo_dano"],
            "severidade": p["severidade"],
            "descricao":  p["descricao"],
        }
        for p in pecas
    ],
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
