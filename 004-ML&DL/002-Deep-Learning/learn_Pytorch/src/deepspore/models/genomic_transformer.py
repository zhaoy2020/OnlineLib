
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#@save
class Genomicformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_heads=8, num_layers=6, num_classes=2, dropout=0.1):
        super(Genomicformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.model_dim = model_dim

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x = x.permute(1, 0, 2)  # Transformer expects input of shape (seq_length, batch_size, model_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling over the sequence length
        x = self.dropout(x)
        x = self.fc_out(x)
        return x