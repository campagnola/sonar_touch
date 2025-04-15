import time
import numpy as np
import teleprox
import pyqtgraph as pg


tap_actions = [
    {'action': 'focus_constellation', 'location': (.055, .309), 'args': ['The Beast']},
    {'action': 'focus_constellation', 'location': (.729, .158), 'args': ['Lovers on the Water']},
    {'action': 'go_home', 'location': (.407, .764), 'args': []},
]

locations = np.array([action['location'] for action in tap_actions])

class SonarAstrolabe:
    def __init__(self):
        self.last_tap_time = time.perf_counter()
        self.recent_taps = []
        self.max_tap_distance = 0.1

        self.proc = teleprox.start_process(qt=True)
        mainwin = self.proc.client._import("sonar_touch.astrolabe.mainwindow")
        self.win, self.view = mainwin.main(_timeout=20)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.timed_update)
        self.timer.start(1000)

    def on_tap(self, location):
        now = time.perf_counter()
        self.recent_taps.append((now, location))
        # remove old taps > 10 seconds
        self.recent_taps = [(t, loc) for t, loc in self.recent_taps if now - t < 10]

        # get 10-sec and 3-sec average location of taps
        locations_10sec = np.array([loc for t, loc in self.recent_taps])
        locations_3sec = np.array([loc for t, loc in self.recent_taps if now - t < 3])
        avg_loc_10sec = np.mean(locations_10sec, axis=0)
        avg_loc_3sec = np.mean(locations_3sec, axis=0)

        if locations_3sec.shape[0] > 1 and locations_3sec.std(axis=0).max() > 0.1:
            loc = location
            print(f"Tap: *{location}   3s avg: {avg_loc_3sec}   10s avg: {avg_loc_10sec}")
        else:
            loc = avg_loc_3sec
            print(f"Tap: {location}   3s avg: *{avg_loc_3sec}   10s avg: {avg_loc_10sec}")

        # distance to all locations
        distances = np.linalg.norm(locations - loc[np.newaxis, :], axis=1)
        # find the closest location
        closest_index = np.argmin(distances)
        # check if the distance is less than the max tap distance
        if distances[closest_index] < self.max_tap_distance:
            action = tap_actions[closest_index]
            print("action:", action)
            # call the action
            getattr(self.view, action['action'])(*action['args'])
            self.last_tap_time = time.perf_counter()
        else:
            print("tap too far away")

    def timed_update(self):
        if time.perf_counter() - self.last_tap_time > 30:
            self.view.go_home()
