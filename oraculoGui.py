import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from kivy.factory import Factory
from kivy.core.window import Window
from kivymd.app import MDApp
from kaki.app import App
from futebol_dados import buscar_dados_liga
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

# ==========================================================
# CONFIGURAÇÃO DOS FICHEIROS DO MODELO
# ==========================================================

def localizar_ficheiro(nome):
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome)
    if os.path.exists(caminho):
        return caminho
    return nome

caminho_modelo = localizar_ficheiro('best_model.keras')
caminho_scaler = localizar_ficheiro('scaler.joblib')
caminho_features = localizar_ficheiro('model_features.json')

# ==========================================================
# CARREGAR MODELO
# ==========================================================

try:
    modelo = load_model(caminho_modelo)
    scaler = load(caminho_scaler)
    with open(caminho_features, 'r', encoding='utf-8') as f:
        model_features_info = json.load(f)
    print("Modelo, Scaler e Info de Features carregados com sucesso.")
except Exception as erro:
    modelo = None
    scaler = None
    model_features_info = None
    print(f"ERRO CRÍTICO: Não foi possível carregar os ficheiros do modelo. {erro}")

# ==========================================================
# APURAR A ESTATÍSTICA DOS ÚLTIMOS JOGOS
# ==========================================================

def get_team_form(team_name, all_matches_df, current_match_date, window=5):
    team_matches = all_matches_df[((all_matches_df['Home'] == team_name) | (all_matches_df['Away'] == team_name)) & (all_matches_df['Date'] < current_match_date)].sort_values('Date').tail(window)
    if len(team_matches) < window:
        return {'avg_goals_scored': 1.0, 'avg_goals_conceded': 1.0, 'points': window * 1.0}
    goals_scored = 0
    goals_conceded = 0
    points = 0
    for _, row in team_matches.iterrows():
        try:
            home_goals, away_goals = map(int, str(row['Score']).replace('–', '-').split('-'))
        except (ValueError, TypeError):
            continue
        if row['Home'] == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            if home_goals > away_goals:
                points += 3
            elif home_goals == away_goals:
                points += 1
        else:
            goals_scored += away_goals
            goals_conceded += home_goals
            if away_goals > home_goals:
                points += 3
            elif away_goals == home_goals:
                points += 1
    num_games = len(team_matches)
    return {'avg_goals_scored': goals_scored / num_games, 'avg_goals_conceded': goals_conceded / num_games, 'points': points}

# ==========================================================
# CAIXA DE INFORMAÇÃO
# ==========================================================

def info(titulo, mensagem):

    dialog = MDDialog(
        title=titulo,
        text=mensagem,
        buttons=[
            MDFlatButton(
                text="OK",
                on_release=lambda x: dialog.dismiss()
            )
        ]
    )

    dialog.open()

# ==========================================================
# CLASSE DAS LIGAS
# ==========================================================

class MinhasLigas:
    def __init__(self, slider_value, league_name):
        print(f"\n--- A iniciar a busca de dados para: {league_name} ---")
        self.slider_value = slider_value
        ano_atual = datetime.now().year
        self.rank_df, self.caract_df = buscar_dados_liga(league_name, ano_atual)
        if self.rank_df is None or self.caract_df is None:
            raise ValueError(f"Falha ao obter dados para {league_name}.")
        self.caract_df['Date'] = pd.to_datetime(self.caract_df['Date'], errors='coerce')
        hoje = pd.Timestamp.now().normalize()
        jogos_futuros = self.caract_df[self.caract_df['Date'].dt.normalize() >= hoje].copy()
        jogos_futuros = jogos_futuros.sort_values('Date')
        self.caract_df = jogos_futuros.reset_index(drop=True)
        if self.caract_df.empty:
            self.inicio = 0
            self.fim = 0
        else:
            self.inicio = 0
            self.fim = min(self.inicio + self.slider_value, len(self.caract_df))

# ==========================================================
# CLASSE PRINCIPAL DE PREVISÃO
# ==========================================================

class Primaria:
    def __init__(self, tabela_df, feature_df, comeco, termino):
        self.tabela_df = tabela_df
        self.feature_df = feature_df
        self.comeco = comeco
        self.termino = termino

    def principal(self):
        if modelo is None or scaler is None or model_features_info is None:
            print("Ficheiros essenciais do modelo não foram carregados.")
            return None, None, None, None, None

        stats = self.tabela_df.copy()
        if isinstance(stats.columns, pd.MultiIndex):
            stats.columns = stats.columns.get_level_values(1)
        stats = stats.set_index('Squad')

        jogos_futuros = self.feature_df.iloc[self.comeco:self.termino].dropna(subset=['Home', 'Away', 'Date'])
        if jogos_futuros.empty:
            print("Não foram encontrados jogos futuros.")
            return None, None, None, None, None

        valid_mask = jogos_futuros['Home'].isin(stats.index) & jogos_futuros['Away'].isin(stats.index)
        jogos_validos = jogos_futuros[valid_mask].copy()
        if jogos_validos.empty:
            print("Nenhum jogo válido encontrado.")
            return None, None, None, None, None

        all_features_list = []

        for index, row in jogos_validos.iterrows():
            home_team = row['Home']
            away_team = row['Away']
            match_date = row['Date']
            home_stats = stats.loc[home_team]
            away_stats = stats.loc[away_team]
            home_form = get_team_form(home_team, self.feature_df, match_date)
            away_form = get_team_form(away_team, self.feature_df, match_date)
            combined_features = {}

            for col, val in home_stats.items():
                combined_features[f"{col}_home"] = val

            for col, val in away_stats.items():
                combined_features[f"{col}_away"] = val

            combined_features['form_points_diff'] = home_form['points'] - away_form['points']
            combined_features['form_avg_gs_diff'] = home_form['avg_goals_scored'] - away_form['avg_goals_scored']
            combined_features['form_avg_gc_diff'] = home_form['avg_goals_conceded'] - away_form['avg_goals_conceded']
            all_features_list.append(combined_features)

        X_df = pd.DataFrame(all_features_list)
        nomes_features = model_features_info['feature_names']
        X_final = pd.DataFrame(0.0, index=X_df.index, columns=nomes_features)

        for coluna in X_df.columns:
            if coluna in X_final.columns:
                X_final[coluna] = X_df[coluna]

        X_final = X_final[nomes_features]
        X_final = X_final.apply(pd.to_numeric, errors='coerce')
        X_final.fillna(0, inplace=True)

        jogo_real_normalizado = scaler.transform(X_final)
        previsao_neural = modelo.predict(jogo_real_normalizado, verbose=0)

        home_real_list = list(jogos_validos['Home'])
        away_real_list = list(jogos_validos['Away'])
        data = np.datetime_as_string(jogos_validos['Date'].values, unit='D')
        tempo = np.array(jogos_validos['Time'])

        return data, tempo, home_real_list, previsao_neural, away_real_list

# ==========================================================
# MAIN APP CLASS
# ==========================================================

class LiveApp(MDApp, App):
    DEBUG = 0

    # ======================================================
    # *.kv files to watch
    # ======================================================

    KV_FILES = {
        os.path.join(os.getcwd(), "telas", "screenmanager.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaPrincipal.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaEpl.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaLaLiga.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaSerieA.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaBundesliga.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaLigue1.kv"),
        os.path.join(os.getcwd(), "telas", "tela_principal", "telaOutrasLigas.kv")
    }

    # ======================================================
    # Classes
    # ======================================================

    CLASSES = {
        "MainScreenManager": "telas.screenmanager",
        "TelaPrincipal": "telas.tela_principal.telaPrincipal",
        "PremierLeague": "telas.tela_principal.telaEpl",
        "LaLiga": "telas.tela_principal.telaLaLiga",
        "SerieA": "telas.tela_principal.telaSerieA",
        "Bundesliga": "telas.tela_principal.telaBundesliga",
        "Ligue1": "telas.tela_principal.telaLigue1",
        "OutrasLigas": "telas.tela_principal.telaOutrasLigas",
        "MinhasLigas": "telas.oraculoGui",
        "Primaria": "telas.oraculoGui"
    }

    # ======================================================
    # Auto reload
    # ======================================================

    AUTORELOADER_PATHS = [(".", {"recursive": True})]

    # ======================================================
    # BUILD APP
    # ======================================================

    def build_app(self, **kwargs):
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Cyan'
        self.theme_cls.primary_hue = 'A700'
        return Factory.MainScreenManager()

# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    LiveApp().run()