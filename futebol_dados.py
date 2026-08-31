import os
from datetime import datetime
import pandas as pd
import requests
import numpy as np

# ==========================================================
# CONFIGURAÇÃO DA API
# ==========================================================
API_URL = "https://api.football-data.org/v4"

# Para o Windows:
#   PowerShell:
#       $env:FOOTBALL_DATA_API_TOKEN="SEU_TOKEN"
#
# Para o APK, o token não deve ser considerado secreto se ficar
# dentro da aplicação. Para produção, o ideal é colocar a API
# atrás de um pequeno servidor próprio.
API_TOKEN = os.environ.get("FOOTBALL_DATA_API_TOKEN", "")

if not API_TOKEN:
    try:
        from config_api import API_TOKEN as TOKEN_CONFIG
        API_TOKEN = TOKEN_CONFIG
    except Exception:
        API_TOKEN = ""

TIMEZONE = "Africa/Luanda"

# Códigos oficiais das competições usados pelo football-data.org
LEAGUES = {
    "Premier League": "PL",
    "Serie A": "SA",
    "La Liga": "PD",
    "Ligue 1": "FL1",
    "Bundesliga": "BL1"
}


# ==========================================================
# PEDIDO À API
# ==========================================================
def pedir_api(endpoint, params=None):
    if not API_TOKEN:
        raise ValueError(
            "Token da football-data.org não configurado. "
            "Defina a variável FOOTBALL_DATA_API_TOKEN."
        )

    url = f"{API_URL}{endpoint}"
    headers = {"X-Auth-Token": API_TOKEN, "Accept": "application/json"}

    resposta = requests.get(url, headers=headers, params=params, timeout=30)

    if resposta.status_code == 429:
        raise RuntimeError(
            "Limite de pedidos da API atingido. "
            "Aguarde alguns segundos e tente novamente."
        )

    if resposta.status_code == 403:
        raise RuntimeError(
            "A API recusou o pedido (403). "
            "Verifique o plano e o token da football-data.org."
        )

    resposta.raise_for_status()
    return resposta.json()

# ==========================================================
# CONVERTE OS JOGOS DA API PARA O FORMATO DO PROJECTO
# ==========================================================
def converter_jogos(matches):
    dados = []

    for jogo in matches:
        data = pd.to_datetime(
            jogo.get("utcDate"),
            errors="coerce",
            utc=True
        )

        if pd.notna(data):
            data = data.tz_convert(TIMEZONE).tz_localize(None)

        score = jogo.get("score", {}).get("fullTime", {})

        golos_casa = score.get("home")
        golos_fora = score.get("away")

        if golos_casa is not None and golos_fora is not None:
            score_text = f"{int(golos_casa)}–{int(golos_fora)}"
        else:
            score_text = np.nan

        dados.append({
            "Date": data,
            "Time": data.strftime("%H:%M") if pd.notna(data) else "",
            "Home": jogo.get("homeTeam", {}).get("name"),
            "Away": jogo.get("awayTeam", {}).get("name"),
            "Score": score_text,
            "Status": jogo.get("status"),
            "Matchday": jogo.get("matchday")
        })
    return pd.DataFrame(dados)


# ==========================================================
# DADOS DOS JOGOS DE UMA COMPETIÇÃO
# ==========================================================
def buscar_jogos_competicao(competition_code, season_year):
    print(
        f"A obter jogos da competição {competition_code} "
        f"da temporada {season_year}-{season_year + 1}..."
    )

    dados = pedir_api(f"/competitions/{competition_code}/matches", params={"season": season_year, "limit": 500})
    matches = dados.get("matches", [])

    if not matches:
        print(
            f"AVISO: Nenhum jogo encontrado para "
            f"{competition_code} / {season_year}."
        )
        return pd.DataFrame(columns=["Date", "Time", "Home", "Away", "Score", "Status", "Matchday"])

    df = converter_jogos(matches)
    df = df.sort_values("Date", na_position="last").reset_index(drop=True)
    print(f"Jogos encontrados: {len(df)}")
    return df

# ==========================================================
# CONSTRÓI AS ESTATÍSTICAS DAS EQUIPAS
# ==========================================================
def construir_estatisticas(jogos_df):
    jogos = jogos_df.copy()
    jogos = jogos[jogos["Home"].notna() & jogos["Away"].notna() & jogos["Score"].notna()].copy()
    if jogos.empty:
        return pd.DataFrame(columns=["Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP"])

    estatisticas = {}

    for _, row in jogos.iterrows():
        casa = row["Home"]
        fora = row["Away"]

        try:
            golos_casa, golos_fora = map(int, str(row["Score"]).replace("–", "-").split("-"))
        except (ValueError, TypeError):
            continue

        if casa not in estatisticas:
            estatisticas[casa] = {"MP": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}

        if fora not in estatisticas:
            estatisticas[fora] = {"MP": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}

        estatisticas[casa]["MP"] += 1
        estatisticas[fora]["MP"] += 1
        estatisticas[casa]["GF"] += golos_casa
        estatisticas[casa]["GA"] += golos_fora
        estatisticas[fora]["GF"] += golos_fora
        estatisticas[fora]["GA"] += golos_casa

        if golos_casa > golos_fora:
            estatisticas[casa]["W"] += 1
            estatisticas[fora]["L"] += 1
            estatisticas[casa]["Pts"] += 3

        elif golos_casa < golos_fora:
            estatisticas[casa]["L"] += 1
            estatisticas[fora]["W"] += 1
            estatisticas[fora]["Pts"] += 3

        else:
            estatisticas[casa]["D"] += 1
            estatisticas[fora]["D"] += 1
            estatisticas[casa]["Pts"] += 1
            estatisticas[fora]["Pts"] += 1

    lista = []

    for equipa, valores in estatisticas.items():
        valores["GD"] = valores["GF"] - valores["GA"]

        if valores["MP"] > 0:
            valores["Pts/MP"] = valores["Pts"] / valores["MP"]
        else:
            valores["Pts/MP"] = 0

        valores["Squad"] = equipa
        lista.append(valores)

    df = pd.DataFrame(lista)

    colunas = ["Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP"]
    df = df[colunas]
    df = df.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    return df


# ==========================================================
# OBTÉM OS DADOS COMPLETOS DE UMA LIGA
# ==========================================================
def buscar_dados_liga(league_name, season_year):
    if league_name not in LEAGUES:
        raise ValueError(f"Liga '{league_name}' não está configurada.")

    competition_code = LEAGUES[league_name]
    jogos_df = buscar_jogos_competicao(competition_code, season_year)
    estatisticas_df = construir_estatisticas(jogos_df)
    return estatisticas_df, jogos_df

# ==========================================================
# MANTÉM O NOME DA FUNÇÃO DO PROJECTO ORIGINAL
# ==========================================================
# Não estamos mais usando Selenium.
# O nome é mantido para que as chamadas existentes do projecto
# não precisem ser alteradas neste primeiro passo.

def busca_dados_com_selenium(url, match_text):
    print(
        "AVISO: busca_dados_com_selenium() agora usa a API "
        "estruturada em vez de Selenium."
    )

    liga = None

    for nome in LEAGUES:
        if nome.lower() in str(match_text).lower():
            liga = nome
            break

    if liga is None:
        texto_url = str(url).lower()

        if "comps/9/" in texto_url:
            liga = "Premier League"
        elif "comps/11/" in texto_url:
            liga = "Serie A"
        elif "comps/12/" in texto_url:
            liga = "La Liga"
        elif "comps/13/" in texto_url:
            liga = "Ligue 1"
        elif "comps/20/" in texto_url:
            liga = "Bundesliga"

    if liga is None:
        raise ValueError(
            f"Não foi possível identificar a liga a partir de "
            f"'{url}' / '{match_text}'."
        )

    ano = datetime.now().year
    # Se a URL antiga tiver uma temporada, tentamos preservá-la.
    partes = str(url).replace("/", " ").split()

    for parte in partes:
        if len(parte) == 9 and parte[:4].isdigit() and parte[4] == "-":
            try:
                ano = int(parte[:4])
                break
            except ValueError:
                pass

    estatisticas, jogos = buscar_dados_liga(liga, ano)

    if "Scores & Fixtures" in str(match_text):
        return jogos

    return estatisticas


# ==========================================================
# COMPATIBILIDADE COM O TREINO ANTIGO
# ==========================================================
def setup_driver():
    print(
        "O Selenium não é mais necessário. "
        "A fonte de dados agora é a API football-data.org."
    )
    return True

def fetch_data_selenium(driver, url, match_text):
    return busca_dados_com_selenium(url, match_text)
