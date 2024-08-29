import torch.nn as nn
import torch.optim as optim


class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_channels=12, input_length=500, conv_filters=[256, 128, 64], lstm_hidden_size=256, num_lstm_layers=2, num_classes=75):
        super(SimpleCNNLSTM, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, conv_filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # LSTM layer 
        self.lstm = nn.LSTM(input_size=conv_filters[2], hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)

        self.conv_out = nn.Conv1d(in_channels=lstm_hidden_size * 2, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        
        # Permute the input to match with Conv1D architecture
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x) 

        x = x.permute(0, 2, 1)
        x = self.conv_out(x)

        # Permute back match output shape
        x = x.permute(0, 2, 1)

        return x
