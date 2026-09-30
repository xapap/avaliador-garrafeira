import os
import json
import re
import streamlit as st
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ----------------------------------------------------
# Configurações da Página Streamlit
# ----------------------------------------------------
st.set_page_config(
    page_title="Avaliador e Precificador de Vinhos",
    page_icon="🍷",
    layout="wide"
)

# Modelo ativo recomendado pela Google API
MODELO_GEMINI = "gemini-3.8-flash"

# ----------------------------------------------------
# Obter Chave dos Segredos do Streamlit de Forma Segura
# ----------------------------------------------------
def obter_cliente_gemini():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return None

# ----------------------------------------------------
# Esquemas Pydantic
# ----------------------------------------------------
class InspecaoVisual(BaseModel):
    nome_vinho: str = Field(description="Nome do vinho, colheita/ano e produtor exatos identificados a partir das imagens.")
    produtor: str = Field(description="Nome da casa ou produtor.")
    ano_colheita: str = Field(description="Ano ou colheita visível (ou 'S/A' se não tiver ano).")
    nivel_enchimento: str = Field(description="Classificação do nível: In Neck, Base Neck, Top Shoulder, Mid Shoulder ou Low Shoulder.")
    estado_capsula: str = Field(description="Condição da cápsula e selo fiscal.")
    estado_rotulo: str = Field(description="Estado físico do rótulo e contrarrótulo.")
    resumo_conservacao: str = Field(description="Breve análise do estado geral para leilão/venda cruzando os vários ângulos.")

# ----------------------------------------------------
# Funções Auxiliares de Extração e Limpeza
# ----------------------------------------------------
def extrair_json_de_texto(texto: str) -> dict:
    """Extrai bloco JSON limpo mesmo se contiver marcações de markdown."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", texto)
    conteudo = match.group(1) if match else texto
    return json.loads(conteudo.strip())

def calcular_metricas(precos_lista):
    """Calcula Mínimo, Médio e Máximo de uma lista de garrafeiras."""
    valores = [
        item["preco"] 
        for item in precos_lista 
        if isinstance(item.get("preco"), (int, float)) and item["preco"] > 0
    ]
    if not valores:
        return 0.0, 0.0, 0.0
    return min(valores), sum(valores) / len(valores), max(valores)

# ----------------------------------------------------
# Lógica com Gemini
# ----------------------------------------------------
def inspecionar_multiplas_imagens(client: genai.Client, imagens: list[Image.Image]) -> InspecaoVisual:
    """Passo 1: Leitura visual cruzando todas as fotos carregadas."""
    prompt = """
    És um perito internacional em vinhos raros, garrafas antigas e leilões de vinho.
    Recebeste várias fotografias da mesma garrafa (podem incluir frente, contrarrótulo, topo da cápsula, selo fiscal e nível contra a luz).
    Cruza as informações visuais de todas as imagens para extrair com o máximo rigor:
    1. Nome exato do vinho, produtor e ano/colheita (confirma contrarrótulo e selos se visíveis).
    2. Nível de enchimento (In Neck, Base Neck, Top Shoulder, Mid Shoulder, Low Shoulder).
    3. Estado da cápsula, presença de cortes/corrosão e integridade do selo fiscal (IVDP, IVBAM ou garantia).
    4. Estado de conservação do papel do rótulo e contrarrótulo.
    """
    
    conteudos = list(imagens) + [prompt]

    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=conteudos,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InspecaoVisual,
            temperature=0.1
        )
    )
    return InspecaoVisual.model_validate_json(response.text)

def pesquisar_precos_mercado(client: genai.Client, nome_vinho: str, estado_geral: str) -> dict:
    """Passo 2: Pesquisa ativa na web com Google Search Grounding em Portugal e Europa."""
    prompt = f"""
    Pesquisa preços reais de mercado atualmente praticados em garrafeiras e lojas especializadas para o seguinte vinho:
    Vinho: "{nome_vinho}"
    Estado de conservação da garrafa avaliada: "{estado_geral}"

    O teu objetivo primordial é encontrar:
    - Até 10 preços em garrafeiras e lojas em PORTUGAL (ex: Garrafeira Nacional, Garrafeira Soares, Estado d'Alma, Gourmet Experience, etc.).
    - Até 10 preços em garrafeiras e lojas na EUROPA (ex: Wine-Searcher Europa, Millesima, Lavinia, Ideawine, garrafeiras de Espanha, França, Reino Unido, Itália ou Alemanha).
    - Se não conseguires encontrar 10 preços de cada, inclui TODOS os preços reais que encontrares e mantém a mesma estrutura.

    Com base nos preços encontrados e no estado de conservação, define uma recomendação de Preço Sugerido de Venda para Portugal e para a Europa.

    Responde ESTRITAMENTE em formato JSON com o seguinte formato:
    {{
      "precos_portugal": [
         {{"loja": "Nome da Garrafeira", "preco": 150.0, "ano_ou_nota": "Colheita 1980 / 75cl", "url": "link ou domínio"}}
      ],
      "precos_europa": [
         {{"loja": "Nome da Garrafeira", "preco": 175.0, "ano_ou_nota": "Vintage 1980", "url": "link ou domínio"}}
      ],
      "preco_sugerido_pt": 140.0,
      "justificacao_pt": "Explicação para o mercado português...",
      "preco_sugerido_europa": 165.0,
      "justificacao_europa": "Explicação para o mercado europeu..."
    }}
    """

    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2
        )
    )
    return extrair_json_de_texto(response.text)

# ----------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------
st.title("🍷 Avaliador de Garrafas & Pesquisa de Preços de Mercado")
st.write("Carrega várias fotografias da garrafa (frente, cápsula, selo, nível, contrarrótulo) para inspeção visual e pesquisa comparativa de mercado.")

# Verificação inicial da chave nos Secrets
client = obter_cliente_gemini()
if client is None:
    st.error("""
    **Configuração necessária:** A chave `GEMINI_API_KEY` não foi encontrada nos Segredos do Streamlit.
    
    * **No teu computador:** Cria o ficheiro `.streamlit/secrets.toml` com a linha:  
      `GEMINI_API_KEY = "A_TUA_CHAVE"`
    * **No Streamlit Cloud:** Vai a **Settings > Secrets** da tua app e adiciona a mesma linha.
    """)
    st.stop()

col_esq, col_dir = st.columns([1, 1], gap="large")

with col_esq:
    ficheiros = st.file_uploader(
        "Carrega as fotografias da garrafa (podes selecionar várias em simultâneo):", 
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    
    lista_imagens = []
    if ficheiros:
        st.write(f"📸 **{len(ficheiros)} fotografia(s) selecionada(s)**")
        colunas_grid = st.columns(2)
        for idx, f in enumerate(ficheiros):
            img = Image.open(f)
            lista_imagens.append(img)
            with colunas_grid[idx % 2]:
                st.image(img, caption=f"Foto {idx + 1}", use_container_width=True)

with col_dir:
    if lista_imagens:
        if st.button("🚀 Iniciar Avaliação Completa", type="primary", use_container_width=True):
            # Etapa 1: Análise Visual Cruzada
            with st.spinner(f"1/2: A cruzar detalhes de {len(lista_imagens)} foto(s)..."):
                try:
                    inspecao = inspecionar_multiplas_imagens(client, lista_imagens)
                except Exception as e:
                    st.error(f"Erro na inspeção visual: {e}")
                    st.stop()
            
            st.subheader(f"🏷️ {inspecao.nome_vinho}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Nível (Ullage)", inspecao.nivel_enchimento)
            c2.metric("Cápsula / Selo", inspecao.estado_capsula)
            c3.metric("Rótulo", inspecao.estado_rotulo)
            st.caption(f"**Diagnóstico de Conservação:** {inspecao.resumo_conservacao}")

            st.markdown("---")

            # Etapa 2: Pesquisa Web em PT e Europa
            with st.spinner(f"2/2: A pesquisar até 10 preços em Portugal e 10 na Europa para '{inspecao.nome_vinho}'..."):
                try:
                    mercado = pesquisar_precos_mercado(
                        client, 
                        inspecao.nome_vinho, 
                        f"Nível: {inspecao.nivel_enchimento}; Rótulo: {inspecao.estado_rotulo}; Cápsula: {inspecao.estado_capsula}"
                    )
                except Exception as e:
                    st.error(f"Erro na pesquisa de preços: {e}")
                    st.stop()

            # Processamento das listas
            pt_precos = mercado.get("precos_portugal", [])
            eu_precos = mercado.get("precos_europa", [])

            min_pt, med_pt, max_pt = calcular_metricas(pt_precos)
            min_eu, med_eu, max_eu = calcular_metricas(eu_precos)

            st.subheader("💰 Recomendações de Venda")
            rec1, rec2 = st.columns(2)
            with rec1:
                sug_pt = mercado.get("preco_sugerido_pt", 0.0)
                st.success(f"### Preço Sugerido (Portugal): {sug_pt:.2f} €")
                st.write(mercado.get("justificacao_pt", ""))
            with rec2:
                sug_eu = mercado.get("preco_sugerido_europa", 0.0)
                st.info(f"### Preço Sugerido (Europa): {sug_eu:.2f} €")
                st.write(mercado.get("justificacao_europa", ""))

            st.markdown("---")

            # Tabelas e Métricas Comparativas
            tab_pt, tab_eu = st.tabs(["🇵🇹 Mercado Portugal", "🇪🇺 Mercado Europa"])

            with tab_pt:
                st.markdown(f"**Amostras encontradas:** {len(pt_precos)}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Preço Mínimo", f"{min_pt:.2f} €" if min_pt > 0 else "N/D")
                m2.metric("Preço Médio", f"{med_pt:.2f} €" if med_pt > 0 else "N/D")
                m3.metric("Preço Máximo", f"{max_pt:.2f} €" if max_pt > 0 else "N/D")
                
                if pt_precos:
                    df_pt = pd.DataFrame(pt_precos)
                    st.dataframe(df_pt, use_container_width=True, hide_index=True)
                else:
                    st.warning("Não foram encontrados preços públicos suficientes para lojas em Portugal.")

            with tab_eu:
                st.markdown(f"**Amostras encontradas:** {len(eu_precos)}")
                e1, e2, e3 = st.columns(3)
                e1.metric("Preço Mínimo", f"{min_eu:.2f} €" if min_eu > 0 else "N/D")
                e2.metric("Preço Médio", f"{med_eu:.2f} €" if med_eu > 0 else "N/D")
                e3.metric("Preço Máximo", f"{max_eu:.2f} €" if max_eu > 0 else "N/D")
                
                if eu_precos:
                    df_eu = pd.DataFrame(eu_precos)
                    st.dataframe(df_eu, use_container_width=True, hide_index=True)
                else:
                    st.warning("Não foram encontrados preços públicos suficientes para lojas europeias.")
    else:
        st.info("👈 Carrega uma ou mais fotografias ao lado para iniciar a análise.")