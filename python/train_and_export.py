import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

# Define CNN architecture
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 8,
            kernel_size=3,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            8, 16,
            kernel_size=3,
            bias=False
        )

        self.fc = nn.Linear(
            16 * 5 * 5,
            10,
            bias=False
        )

        self.pool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))

        x = self.pool(x)

        x = self.relu(self.conv2(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)


# Load MNIST dataset
loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '.',
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=64,
    shuffle=True
)

# Create model
model = MnistCNN()

opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.CrossEntropyLoss()

# Train model
for epoch in range(5):

    for imgs, labels in loader:

        opt.zero_grad()

        loss_fn(
            model(imgs),
            labels
        ).backward()

        opt.step()

    print(f"Epoch {epoch + 1} done")


# Q4.3 settings
FRAC_BITS = 3

SCALE = 2 ** FRAC_BITS


# Convert float to Q4.3
def quantise_q43(tensor):

    scaled = tensor * SCALE

    rounded = torch.round(scaled)

    clamped = torch.clamp(
        rounded,
        -128,
        127
    )

    return clamped.to(torch.int8)


# Save weights as hex
def write_hex(tensor_int8, filename):

    flat = tensor_int8.numpy() \
                      .flatten() \
                      .astype(np.uint8)

    with open(filename, 'w') as f:

        for byte in flat:
            f.write(f"{byte:02x}\n")


# Export Conv1 weights
write_hex(
    quantise_q43(model.conv1.weight.data),
    "conv1_weights.hex"
)

# Export Conv2 weights
write_hex(
    quantise_q43(model.conv2.weight.data),
    "conv2_weights.hex"
)

# Export FC weights
write_hex(
    quantise_q43(model.fc.weight.data),
    "fc_weights.hex"
)

print("Hex files written.")


# Test quantised accuracy
def test_accuracy():

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '.',
            train=False,
            transform=transforms.ToTensor()
        ),
        batch_size=1000
    )

    # Fake quantisation
    def fake_quantise(w):

        return (
            quantise_q43(w)
            .float() / SCALE
        )

    with torch.no_grad():

        model.conv1.weight.data = fake_quantise(
            model.conv1.weight.data
        )

        model.conv2.weight.data = fake_quantise(
            model.conv2.weight.data
        )

        model.fc.weight.data = fake_quantise(
            model.fc.weight.data
        )

        correct = sum(
            (
                model(imgs).argmax(1) == labels
            ).sum()
            for imgs, labels in test_loader
        )

    print(
        f"Quantised accuracy: {correct / 100:.1f}%"
    )


test_accuracy()
