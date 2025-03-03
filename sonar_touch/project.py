import json
import os

import numpy as np


class SonarTouchProject:
    """Stores saved training data and projection ROI state
    """
    def __init__(self, project_path):
        self.project_path = project_path
        self.state_file = os.path.join(project_path, 'project.json')
        self.training_data_path = os.path.join(project_path, 'training_data')
        self.training_index_file = os.path.join(project_path, 'training_index.jsonl')
        self.load()

    def load(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {}
        self.load_training_index()

    def save(self, **kwargs):
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            os.makedirs(self.training_data_path)

        self.state.update(kwargs)
        state_json = json.dumps(self.state)
        with open(self.project_path + '/project.json', 'w') as f:
            f.write(state_json)

    def load_training_index(self):
        index = []
        if os.path.exists(self.training_index_file):
            for line in open(self.training_index_file, 'r'):
                index.append(json.loads(line))
        self.training_index = index

    def save_training_example(self, data, location):
        """Save a training example to disk, where data is a numpy array (N, n_channels) and location is a tuple (x, y)
        """
        next_index = len(self.training_index)
        filename = os.path.join(self.training_data_path, f'{next_index:06d}.npy')        
        np.save(filename, data)
        self.training_index.append({'filename': filename, 'location': location})
        record = json.dumps(self.training_index[-1])
        with open(self.training_index_file, 'a') as f:
            f.write(record + '\n')
