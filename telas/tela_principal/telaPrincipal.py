from kivy.app import App
from kivymd.uix.button import MDFillRoundFlatIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivy.uix.popup import Popup
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.animation import Animation


class TelaPrincipal(MDScreen):
    def info(self):
        cx = MDBoxLayout(orientation='vertical')

        pop = Popup(title='Outras ligas', content=cx, size_hint=(None, None), size=(100, 75))

        sair = MDFillRoundFlatIconButton(text='SAIR', icon='door', on_release=pop.dismiss, pos_hint={"center_x": .5})
        texto = MDLabel(text='Ainda não existem outras ligas')

        cx.add_widget(texto)
        cx.add_widget(sair)
        anim = Animation(size=(300, 200), duration=0.2, t='out_back')
        anim.start(pop)

        pop.open()

    def sair(self):
        App.get_running_app().stop()

