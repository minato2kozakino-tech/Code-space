import torch
import torch.nn as nn

class OrigonGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.2):
        super(OrigonGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        out, hidden = self.gru(embeds, hidden) # out: (batch_size, seq_len, hidden_size)
        
        # We only take the last output for prediction in many-to-one cases,
        # but for language modeling we might take all.
        out = self.fc(out) # (batch_size, seq_len, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(device)
