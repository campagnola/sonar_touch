import time
import numpy as np
import teleprox
import pyqtgraph as pg


tap_actions = [
    {'action': 'focus_constellation', 'location': [0.26535508, 0.50626415], 'args': ['The Hourglass']},
    {'action': 'focus_constellation', 'location': [0.72001714, 0.81999177], 'args': ['Lovers on the Water']},
    {'action': 'focus_constellation', 'location': [0.8255281,  0.41119882], 'args': ['The Hunter']},
    {'action': 'focus_constellation', 'location': [0.6951457, 0.614518 ], 'args': ['The Longneck']},
    {'action': 'focus_constellation', 'location': [0.7799518, 0.60777  ], 'args': ['Shurap enters the Crack']},
    {'action': 'focus_constellation', 'location': [0.3902337,  0.58700186], 'args': ['The Beast']},
    {'action': 'focus_constellation', 'location': [0.31247833, 0.34628168], 'args': ['The Kite']},
    {'action': 'focus_constellation', 'location': [0.57072484, 0.3306935 ], 'args': ['The Destroyer']},
]

locations = np.array([action['location'] for action in tap_actions])

class SonarAstrolabe:
    def __init__(self):
        self.last_tap_time = None
        self.recent_taps = []
        self.max_tap_distance = 0.2
        self.off = True

        self.proc = teleprox.start_process(qt=True)
        mainwin = self.proc.client._import("sonar_touch.astrolabe.mainwindow")
        self.win, self.view = mainwin.main(_timeout=20)

        self.view.zoom_off()

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
        locations_3sec = np.array([loc for t, loc in self.recent_taps if now - t < 1.5])
        avg_loc_10sec = np.mean(locations_10sec, axis=0)
        avg_loc_3sec = np.mean(locations_3sec, axis=0)

        # if locations_3sec.shape[0] > 1 and locations_3sec.std(axis=0).max() > 0.1:
        #     loc = location
        #     print(f"Tap: *{location}   3s avg: {avg_loc_3sec}   10s avg: {avg_loc_10sec}")
        # else:
        #     loc = avg_loc_3sec
        #     print(f"Tap: {location}   3s avg: *{avg_loc_3sec}   10s avg: {avg_loc_10sec}")
        loc = location

        # distance to all locations
        distances = np.linalg.norm(locations - loc[np.newaxis, :], axis=1)
        # find the closest location
        closest_index = np.argmin(distances)
        # check if the distance is less than the max tap distance
        action = None
        if self.off:
            action = {'action': 'go_home', 'args': []}
        elif distances[closest_index] < self.max_tap_distance:
            action = tap_actions[closest_index]
        else:
            if loc[1] > 0.9:
                t = np.clip(loc[0] * 400000 - 200000, -200000, 200000)
                if np.abs(t) < 40000:
                    t = 0
                action = {'action': 'slew_to_time', 'args': [t]}
            elif loc[0] < 0.1 or loc[0] > 0.9:
                direction = direction = loc * 2 - 1
                direction /= np.linalg.norm(direction)
                action = {'action': 'slew_in_direction', 'args': [(direction[0], -direction[1])]}
            elif loc[1] < 0.1:
                action = {'action': 'go_home', 'args': []}
            else:
                action = None
                print("tap too far away")
        
        if action is not None:
            self.off = False
            print("action:", action)
            getattr(self.view, action['action'])(*action['args'])
            self.last_tap_time = time.perf_counter()

    def timed_update(self):
        if self.last_tap_time is not None and time.perf_counter() - self.last_tap_time > 30 and not self.off:
            self.view.zoom_off()
            self.off = True
            self.last_tap_time = None

