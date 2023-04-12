import torch
import torch.nn as nn
import numpy as np
import pkg_resources

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model_pt = pkg_resources.resource_stream(__name__, 'model_pt/model.pt')

class WaveformSelectionNet(nn.Module):

    def __init__(self, w=82): # w: window length
        super(WaveformSelectionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2),
        )
        w = (w - 3 + 1) // 2
        w = (w - 3 + 1) // 2
        w = (w - 3 + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(64 * w, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def load_model(self, PATH = model_pt):
        self.load_state_dict(torch.load(PATH))
    
    def predict(self, x, device=device):
        self.to(device)
        self.eval()
        scores = []
        with torch.no_grad():
            input_x = torch.tensor(x, dtype=torch.float).to(device)
            score = self(input_x)
            scores.append(score.cpu().data.numpy())

        scores = np.concatenate(scores)
        preds = np.argmax(scores, axis=1)
        return preds

    