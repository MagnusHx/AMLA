import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=24,
                 c1=8, c2=16, c3=32,
                 fc1=64,
                 dropout=0.2,
                 # training defaults
                 batch_size=32,
                 learning_rate=1e-3,
                 optimizer_cls=None,
                 criterion=None):
        super().__init__()

        # Use Adaptive pooling so classifier is independent of input image size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            # No adaptive pooling: keep spatial size to hard-code flatten length
            nn.Flatten()
        )

        # Hard-coded flatten size for input images of size 105x105 with three 2x pools:
        # 105 -> 52 -> 26 -> 13, so flatten = c3 * 13 * 13 = c3 * 169
        self.classifier = nn.Sequential(
            nn.Linear(5408, fc1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc1, num_classes)
        )

        # Training defaults stored on the model instance
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # store optimizer class (default to Adam)
        self.optimizer_cls = optimizer_cls if optimizer_cls is not None else torch.optim.Adam
        # store criterion (default to CrossEntropyLoss)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def evaluate(self, dataloader, device='cpu', criterion=None):
        if criterion is None:
            criterion = self.criterion
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = self(xb)
                loss = criterion(out, yb)
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return running_loss / total, correct / total

    def fit(self, train_loader, val_loader=None, num_epochs=10, learning_rate=None, device='cpu', optimizer_cls=None, criterion=None):
        device = torch.device(device)
        self.to(device)

        # Resolve hyperparameters: use provided arguments, else fall back to model defaults
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        optimizer_cls = optimizer_cls if optimizer_cls is not None else self.optimizer_cls
        criterion = criterion if criterion is not None else self.criterion

        optimizer = optimizer_cls(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = self(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, device=device, criterion=criterion)
            else:
                val_loss, val_acc = None, None

            print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            if val_loss is not None:
                print(f"               Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save best weights in memory
                    best_state = {k: v.cpu() for k, v in self.state_dict().items()}

        # restore best weights if available
        try:
            self.load_state_dict(best_state)
        except Exception:
            pass

        return self
