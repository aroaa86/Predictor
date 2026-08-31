import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
from futebol_dados import buscar_dados_liga


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================
caminho_base = (
    'C:/Users/Admin/OneDrive/Documentos/'
    'Backups projecto python/teste/'
)

if not os.path.exists(caminho_base):
    os.makedirs(caminho_base)

# Número de temporadas históricas utilizadas
NUM_SEASONS = 5

# Última temporada COMPLETA.
# Em 2026, a última temporada completa é 2025-2026.
ULTIMA_TEMPORADA = datetime.now().year - 1

# A quantidade de temporadas de treino será NUM_SEASONS - 1.
# A última temporada será reservada exclusivamente para teste.
TEMPORADAS = [
    ULTIMA_TEMPORADA - i
    for i in range(NUM_SEASONS - 1, -1, -1)
]

TEMPORADAS_TREINO = TEMPORADAS[:-1]
TEMPORADA_TESTE = TEMPORADAS[-1]

print("\n==============================================")
print(" CONFIGURAÇÃO DO TREINO")
print("==============================================")
print(f"Temporadas utilizadas: {NUM_SEASONS}")
print(
    f"Temporadas de treino: "
    f"{TEMPORADAS_TREINO[0]}-{TEMPORADAS_TREINO[0] + 1} "
    f"até "
    f"{TEMPORADAS_TREINO[-1]}-{TEMPORADAS_TREINO[-1] + 1}"
)
print(
    f"Temporada de teste: "
    f"{TEMPORADA_TESTE}-{TEMPORADA_TESTE + 1}"
)


# ==========================================================
# LIGAS
# ==========================================================
leagues = {
    'Premier League': 'PL',
    'Serie A': 'SA',
    'La Liga': 'PD',
    'Ligue 1': 'FL1',
    'Bundesliga': 'BL1'
}


# ==========================================================
# PROCESSAR RESULTADO
# ==========================================================
def process_score(score):

    try:
        partes = (str(score).replace('–', '-').split('-'))
        if len(partes) != 2:
            return None

        home_goals, away_goals = map(int, partes)

    except (ValueError, TypeError):
        return None

    if home_goals > away_goals:
        return [1, 0, 0]

    if home_goals < away_goals:
        return [0, 1, 0]

    return [0, 0, 1]

# ==========================================================
# APURAR A ESTATÍSTICA DOS ÚLTIMOS JOGOS
# ==========================================================
def get_team_form(team_name, all_matches_df, current_match_date, window=5):
    team_matches = all_matches_df[((all_matches_df['Home'] == team_name) | (all_matches_df['Away'] == team_name))
                                  & (all_matches_df['Date'] < current_match_date)].sort_values('Date').tail(window)

    if team_matches.empty:
        return {'avg_goals_scored': 1.0, 'avg_goals_conceded': 1.0, 'points': window * 1.0}

    goals_scored = 0
    goals_conceded = 0
    points = 0
    jogos_validos = 0

    for _, row in team_matches.iterrows():
        try:
            home_goals, away_goals = map(int, str(row['Score']).replace('–', '-').split('-'))
        except (ValueError, TypeError):
            continue

        jogos_validos += 1

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

    if jogos_validos == 0:
        return {'avg_goals_scored': 1.0, 'avg_goals_conceded': 1.0, 'points': window * 1.0}

    return {'avg_goals_scored': (goals_scored / jogos_validos), 'avg_goals_conceded': (goals_conceded / jogos_validos), 'points': points}


# ==========================================================
# ESTATÍSTICAS ATÉ AO JOGO ANTERIOR
# ==========================================================
def get_team_stats(team_name, matches_before):
    team_matches = matches_before[((matches_before['Home'] == team_name) | (matches_before['Away'] == team_name))]

    if team_matches.empty:
        return {'MP': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0, 'Pts/MP': 0}

    mp = 0
    wins = 0
    draws = 0
    losses = 0
    gf = 0
    ga = 0
    pts = 0

    for _, row in team_matches.iterrows():
        try:
            home_goals, away_goals = map(int, str(row['Score']).replace('–', '-').split('-'))
        except (ValueError, TypeError):
            continue

        if row['Home'] == team_name:
            mp += 1
            gf += home_goals
            ga += away_goals

            if home_goals > away_goals:
                wins += 1
                pts += 3

            elif home_goals == away_goals:
                draws += 1
                pts += 1

            else:
                losses += 1

        elif row['Away'] == team_name:

            mp += 1
            gf += away_goals
            ga += home_goals

            if away_goals > home_goals:
                wins += 1
                pts += 3

            elif away_goals == home_goals:
                draws += 1
                pts += 1

            else:
                losses += 1

    gd = gf - ga
    pts_mp = (pts / mp if mp > 0 else 0)
    return {'MP': mp, 'W': wins, 'D': draws, 'L': losses, 'GF': gf, 'GA': ga, 'GD': gd, 'Pts': pts, 'Pts/MP': pts_mp}


# ==========================================================
# PREPARAR DADOS DE UMA TEMPORADA
# ==========================================================
def preparar_temporada(league_name, season_year, stats_df, fixtures_df):

    print(f"\nProcessando {league_name} " f"{season_year}-{season_year + 1}...")
    jogos = fixtures_df.copy()

    if jogos.empty:
        return [], []

    jogos['Date'] = pd.to_datetime(jogos['Date'], errors='coerce')
    jogos = jogos.dropna(subset=['Date', 'Home', 'Away', 'Score']).copy()
    jogos = jogos.sort_values('Date').reset_index(drop=True)

    if jogos.empty:
        return [], []

    features = []
    resultados = []

    # ------------------------------------------------------
    # IMPORTANTE:
    # Cada jogo só pode utilizar informação anterior
    # àquele jogo.
    # ------------------------------------------------------

    for index, row in jogos.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        match_date = row['Date']
        resultado = process_score(row['Score'])

        if resultado is None:
            continue

        jogos_anteriores = jogos[jogos['Date'] < match_date]
        home_stats = get_team_stats(home_team, jogos_anteriores)
        away_stats = get_team_stats(away_team, jogos_anteriores)
        home_form = get_team_form(home_team, jogos, match_date)
        away_form = get_team_form(away_team, jogos, match_date)
        combined_features = {}

        for col, val in home_stats.items():
            combined_features[f"{col}_home"] = val

        for col, val in away_stats.items():
            combined_features[f"{col}_away"] = val

        combined_features['form_points_diff'] = (home_form['points'] - away_form['points'])
        combined_features['form_avg_gs_diff'] = (home_form['avg_goals_scored'] - away_form['avg_goals_scored'])
        combined_features['form_avg_gc_diff'] = (home_form['avg_goals_conceded'] - away_form['avg_goals_conceded'])
        features.append(combined_features)
        resultados.append(resultado)

    print(
        f"Jogos válidos processados: "
        f"{len(features)}"
    )
    return features, resultados


# ==========================================================
# RECOLHER DADOS
# ==========================================================
all_train_features = []
all_train_results = []
all_test_features = []
all_test_results = []
total_jogos_treino = 0
total_jogos_teste = 0

for league in leagues:
    print(f"\n==============================================")
    print(f" LIGA: {league}")
    print(f"==============================================")

    for season_year in TEMPORADAS:
        print(
            f"\n--- {league} "
            f"{season_year}-{season_year + 1} ---"
        )
        try:
            stats_df, fixtures_df = buscar_dados_liga(league, season_year)
            if (stats_df is None or fixtures_df is None or fixtures_df.empty):
                print(
                    f"AVISO: Não foram obtidos dados "
                    f"para {league} "
                    f"{season_year}-{season_year + 1}."
                )
                continue

            features, resultados = preparar_temporada(league, season_year, stats_df, fixtures_df)

            if not features:
                print(
                    "AVISO: Nenhuma feature válida "
                    "foi criada."
                )
                continue

            if season_year in TEMPORADAS_TREINO:
                all_train_features.extend(features)
                all_train_results.extend(resultados)
                total_jogos_treino += len(features)

            elif season_year == TEMPORADA_TESTE:
                all_test_features.extend(features)
                all_test_results.extend(resultados)
                total_jogos_teste += len(features)

            print(
                f"Dados adicionados com sucesso: "
                f"{len(features)} jogos."
            )

        except Exception as erro:
            print(
                f"ERRO ao obter/processar "
                f"{league} "
                f"{season_year}-{season_year + 1}: "
                f"{erro}"
            )


# ==========================================================
# VERIFICAÇÃO DOS DADOS
# ==========================================================
if not all_train_features:
    raise ValueError("Nenhum dado de treino foi recolhido.")

if not all_test_features:
    raise ValueError("Nenhum dado de teste foi recolhido.")


print("\n==============================================")
print(" RESUMO DOS DADOS")
print("==============================================")
print(f"Jogos para treino: " f"{total_jogos_treino}")
print(f"Jogos para teste: " f"{total_jogos_teste}")
print(f"Total de jogos: " f"{total_jogos_treino + total_jogos_teste}")


# ==========================================================
# DATAFRAMES
# ==========================================================
X_train_df = pd.DataFrame(all_train_features)
X_test_df = pd.DataFrame(all_test_features)
y_train = np.array(all_train_results, dtype=np.float32)
y_test = np.array(all_test_results, dtype=np.float32)


# ==========================================================
# GARANTIR AS MESMAS FEATURES
# ==========================================================
feature_names = list(X_train_df.columns)
X_test_df = X_test_df.reindex(columns=feature_names, fill_value=0)


# ==========================================================
# CONVERTER FEATURES PARA NUMÉRICO
# ==========================================================
X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce')
X_train_df.fillna(0, inplace=True)
X_test_df.fillna(0, inplace=True)

print(f"\nNúmero de features: " f"{len(feature_names)}")


# ==========================================================
# DISTRIBUIÇÃO DAS CLASSES
# ==========================================================
print("\nDistribuição dos resultados:")
train_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)

nomes_classes = {0: 'CASA', 1: 'FORA', 2: 'EMPATE'}

for classe in range(3):
    treino_quantidade = np.sum(train_labels == classe)
    teste_quantidade = np.sum(test_labels == classe)
    print(f"{nomes_classes[classe]}: " f"treino={treino_quantidade}, "f"teste={teste_quantidade}")


# ==========================================================
# GUARDAR LISTA DE FEATURES
# ==========================================================
model_features_info = {'feature_names': feature_names}

with open(
    os.path.join(caminho_base, 'model_features.json'), 'w', encoding='utf-8') as f:
    json.dump(model_features_info, f, indent=4, ensure_ascii=False)

print("\nLista de features guardada com sucesso.")


# ==========================================================
# SCALER
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)
dump(scaler, os.path.join(caminho_base, 'scaler.joblib'))

print("Scaler guardado com sucesso.")


# ==========================================================
# PESOS DAS CLASSES
# ==========================================================
classes = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
class_weight_dict = dict(zip(classes, class_weights))

print(f"\nPesos das classes calculados: " f"{class_weight_dict}")


# ==========================================================
# MODELO
# ==========================================================
input_shape = X_train_scaled.shape[1]
model = tf.keras.models.Sequential([tf.keras.Input(shape=(input_shape,)),
                                    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                                    tf.keras.layers.Dropout(0.25),

                                    tf.keras.layers.Dense( 64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                                    tf.keras.layers.Dropout(0.25),

                                    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                                    tf.keras.layers.Dropout(0.25),

                                    tf.keras.layers.Dense(3, activation='softmax')])


# ==========================================================
# COMPILAR MODELO
# ==========================================================
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# ==========================================================
# TREINO
# ==========================================================
print("\n==============================================")
print(" A iniciar o treino do modelo...")
print("==============================================")

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=150, batch_size=16, validation_data=(X_test_scaled, y_test), callbacks=[early_stop], class_weight=class_weight_dict, verbose=2)


# ==========================================================
# GUARDAR MODELO
# ==========================================================
caminho_modelo = os.path.join(caminho_base, 'best_model.keras')
model.save(caminho_modelo)
print("\nModelo guardado com sucesso.")


# ==========================================================
# AVALIAÇÃO
# ==========================================================
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print("\n==============================================")
print(f"Acurácia no conjunto de teste: " f"{accuracy:.4f}")
print(f"Loss no conjunto de teste: " f"{loss:.4f}")
print("==============================================")


# ==========================================================
# PREVISÕES
# ==========================================================
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)


# ==========================================================
# RELATÓRIO
# ==========================================================
class_names = ['Vitória Casa', 'Vitória Fora', 'Empate']
print("\n==============================================")
print(" RELATÓRIO DE CLASSIFICAÇÃO")
print("==============================================")
print(classification_report(y_test_labels, y_pred, target_names=class_names, zero_division=0))


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================
cm = confusion_matrix(y_test_labels, y_pred)
print("\nMatriz de Confusão:")
print(cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.colorbar()
plt.xticks(range(3), class_names, rotation=20)
plt.yticks(range(3), class_names)
plt.tight_layout()
plt.show()


# ==========================================================
# GRÁFICOS DO TREINO
# ==========================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Histórico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Histórico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# FIM
# ==========================================================
print("\n==============================================")
print("--- Processo de treino concluído. ---")
print("==============================================")