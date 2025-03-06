


import numpy as np


class TrainingDataCollector:
    def __init__(self, ui):
        self.ui = ui
        self.run = False
        self.requested_location = None

    def trigger_detected(self, trigger):
        if not self.run:
            return
        self.ui.project.save_training_example(trigger['data'], self.requested_location)
        self.request_next()

    def start(self):
        self.run = True
        self.locations = self.generate_locations()
        self.request_next()

    def stop(self):
        self.run = False

    def request_next(self):
        self.requested_location = next(self.locations)
        self.ui.projected_view.set_target(self.requested_location)

    def generate_locations(self):
        while True:
            for i,row in enumerate(np.linspace(0, 1, 40)):
                for col in np.linspace(0, 1, 60):
                    if i % 2 == 1:
                        col = 1 - col
                    yield (
                        np.clip(col, 0, 1),
                        np.clip(row, 0, 1), 
                    )
