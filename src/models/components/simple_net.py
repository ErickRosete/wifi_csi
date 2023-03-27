from torch import nn
import torch


def get_width(width, kernel_size, padding=0, stride=1):
    return 1 + (width - kernel_size + 2 * padding) // stride


class SimpleNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 180,
        conv_h1_channels: int = 90,
        conv_h2_channels: int = 30,
        conv_h3_channels: int = 1,
        linear_h1_size: int = 128,
        output_size: int = 7,
    ):
        super().__init__()
        w = 12000
        for _ in range(3):
            w = get_width(w, 7, 0, 2)
        linear_in = w * conv_h3_channels
        print(linear_in)

        self.conv_amp_model = nn.Sequential(
            nn.Conv1d(in_channels, conv_h1_channels, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(conv_h1_channels, conv_h2_channels, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(conv_h2_channels, conv_h3_channels, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # self.conv_phase_model = nn.Sequential(
        #     nn.Conv1d(in_channels, conv_h1_channels, kernel_size=7, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(conv_h1_channels, conv_h2_channels, kernel_size=7, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(conv_h2_channels, conv_h3_channels, kernel_size=7, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.linear_model = nn.Sequential(
            nn.Linear(linear_in * 3, linear_h1_size),
            nn.ReLU(),
            nn.Linear(linear_h1_size, output_size),
        )

    def forward(self, x):
        features = []
        start_idx = 0
        for _ in range(3):
            features.append(self.conv_amp_model(x[:, start_idx : start_idx + 30]))
            start_idx += 30
        # for _ in range(3):
        #     features.append(self.conv_phase_model(x[:, start_idx : start_idx + 30]))
        #     start_idx += 30
        all_ant = torch.concatenate(features, dim=-1)
        return self.linear_model(all_ant)


if __name__ == "__main__":
    _ = SimpleNet()
