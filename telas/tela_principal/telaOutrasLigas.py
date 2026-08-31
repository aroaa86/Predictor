from kivymd.uix.screen import MDScreen


class OutrasLigas(MDScreen):
    def set_screen(self):
        self.manager.current = "ligas"
        self.manager.transition.direction = 'right'
