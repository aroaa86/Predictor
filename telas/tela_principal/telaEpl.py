from datetime import datetime
from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel
from oraculoGui import MinhasLigas, Primaria, info
import numpy as np
import threading
from kivy.clock import Clock


class PremierLeague(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

    def adicionar_widget(self):
        """
        Inicia o processo de loading e dispara o thread.
        """
        # Mostrar o spinner e desativar o botão para evitar múltiplos cliques
        self.ids.spinner.active = True
        self.ids.prever_button.disabled = True
        self.ids.box.clear_widgets()

        # Iniciar a tarefa demorada num thread separado
        threading.Thread(target=self.buscar_e_prever).start()

    def buscar_e_prever(self):
        """
        Contém a lógica demorada e corre num thread de segundo plano.
        """
        try:
            # 1ª fase: coleta de dados
            minhas = MinhasLigas(slider_value=int(self.ids.slider.value), league_name='Premier League')

            # 2ª fase: processamento e previsão
            previsao_engine = Primaria(minhas.rank_df, minhas.caract_df, minhas.inicio,minhas.fim)
            data, hora, casa, previsao, fora = previsao_engine.principal()
            Clock.schedule_once(lambda dt: self.mostrar_resultados(data, hora, casa, previsao, fora))

        except Exception as e:
            error_message = f"Ocorreu um erro: {e}"
            Clock.schedule_once(lambda dt: self.mostrar_erro(error_message))

    def mostrar_resultados(self, data, hora, casa, previsao, fora):
        """
        Corre no thread principal e atualiza a UI do utilizador.
        """
        self.ids.spinner.active = False
        self.ids.prever_button.disabled = False

        if data is None:
            print("Nenhum dado de previsão para exibir.")
            return

        print('\n--- PREVISÕES PREMIER LEAGUE ---')

        class_map = {0: 'CASA', 1: 'FORA', 2: 'EMPATE'}

        # 3ª fase: mostrar os resultados
        for k in range(len(data)):
            resultado_idx = np.argmax(previsao[k])
            confianca = previsao[k][resultado_idx] * 100
            resultado_texto = class_map.get(resultado_idx, 'N/A')

            print(f"\t{data[k]} {hora[k]} | " f"{casa[k]} vs {fora[k]} | " f"Previsão: {resultado_texto} ({confianca:.1f}%)")
            texto_label = (f"[b]{casa[k]}[/b] vs [b]{fora[k]}[/b]\n" f"{data[k]} - {hora[k]}\n"
                           f"Previsão: [color=00BFA5]{resultado_texto}[/color] " f"({confianca:.1f}%)")

            label = MDLabel(text=texto_label, halign='left', size_hint_y=None, height='40dp', theme_text_color="Custom", text_color=(0, 0, 0, 1), markup=True)
            self.ids.box.add_widget(label)

    def mostrar_erro(self, error_message):
        """
        Mostra um erro que ocorreu no thread de background.
        """
        self.ids.spinner.active = False
        self.ids.prever_button.disabled = False

        print(error_message)
        info("Erro Inesperado","Ocorreu um erro. Verifique a sua ligação ou tente mais tarde.")

    def remover_widget(self):
        """Limpa todos os widgets de previsão da caixa."""
        if not self.ids.box.children:
            return

        self.ids.box.clear_widgets()

    def set_screen(self):
        self.manager.current = "ligas"
        self.manager.transition.direction = 'right'