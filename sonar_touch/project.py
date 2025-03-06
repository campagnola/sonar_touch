import json
import os
import torch
import numpy as np
import sonar_touch.models


class SonarTouchProject:
    """Stores saved training data and projection ROI state
    """
    def __init__(self, project_path):
        self.project_path = project_path
        self.state_file = os.path.join(project_path, 'project.json')
        self.training_data_path = 'training_data'
        self.training_index_file = os.path.join(project_path, 'training_index.jsonl')
        self.next_example_id = 0
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
        self.next_example_id = index[-1]['id'] + 1 if len(index) > 0 else 0

    def save_training_example(self, data, location):
        """Save a training example to disk, where data is a numpy array (N, n_channels) and location is a tuple (x, y)
        """
        full_training_data_path = os.path.join(self.project_path, self.training_data_path)
        if not os.path.exists(full_training_data_path):
            os.makedirs(full_training_data_path)
        next_id = self.next_example_id
        self.next_example_id += 1
        filename = os.path.join(self.training_data_path, f'{next_id:06d}.npy')
        np.save(os.path.join(self.project_path, filename), data)
        self.training_index.append({'filename': filename, 'location': location, 'id': next_id})
        record = json.dumps(self.training_index[-1])
        with open(self.training_index_file, 'a') as f:
            f.write(record + '\n')

    def load_training_data(self):
        for example in self.training_index:
            if 'data' in example:
                continue
            filename = os.path.join(self.project_path, example['filename'])
            example['data'] = np.load(filename)
        return self.training_index

    def list_models(self):
        model_fies = [f for f in os.listdir(self.project_path) if f.endswith('.pth')]
        return sorted(model_fies)
    
    def load_model(self, model_name):
        print(f"Loading model {model_name}")
        class_name = os.path.splitext(model_name.split('_')[1])[0]
        model_class = getattr(sonar_touch.models, class_name)
        model = model_class()
        model.load(os.path.join(self.project_path, model_name))
        # send model to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.device = device
        return model
    