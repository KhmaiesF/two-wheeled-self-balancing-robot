# Placeholder: l’implémentation dépend de la lib GPIO (interrupts).
# Structure recommandée.

class EncoderReader:
    def __init__(self):
        self.count_l = 0
        self.count_r = 0

    def reset(self):
        self.count_l = 0
        self.count_r = 0

    def read_counts(self):
        return self.count_l, self.count_r
