import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class CRNN(nn.Module):
    def __init__(self, num_classes, lexicon_path, tokens):
        super(CRNN, self).__init__()
        self.lexicon_path = lexicon_path
        self.tokens = tokens

        # conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 1))
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        
        # rnn layers
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        
        # map to character size
        self.linear = nn.Linear(512, num_classes)
        
        # CTC Decoder
        if lexicon_path:
            self.decoder = torchaudio.models.decoder.CTCBPEBeamSearchDecoder(
            lexicon=lexicon_path,
            tokens=tokens,
            lm=None,
            beam_size=10,
            blank_id=0
            )
        else:
            self.decoder = None
            
    
    def forward(self, x):
        # conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = self.bn1(x)
        x = F.relu(self.conv6(x))
        x = self.bn2(x)
        x = self.pool4(x)
        x = F.relu(self.conv7(x))
        
        # map to sequence
        x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.reshape(N, W, C * H)
        
        # lstm layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # map to character size
        logits = self.linear(x)
        
        if self.training:
            return logits
        else:
            # do the CTC only during inference time
            if self.decoder:
                # lexicon based method
                lengths = torch.full((N,), W, dtype=torch.long) # create a tensor of shape (N,) filled with value W.
                logits = logits.permute(1, 0, 2)  # (T, N, C)
                results = self.decoder(logits, lengths)
                decoded = [result[0][0] for result in results] # get the top prediction
            else:
                # lexicon free method
                probs = F.softmax(logits, dim=-1) # apply softmax
                best_paths = torch.argmax(probs, dim=-1)
                decoded = []
                for path in best_paths:
                    seq = path.tolist()
                    collapsed = []
                    prev = None
                    for char_idx in seq:
                        # skip blank sapce and reptition
                        if char_idx != 0 and char_idx != prev:
                            collapsed.append(self.tokens[char_idx])
                        prev = char_idx
                    decoded.append("".join(collapsed))
            return decoded