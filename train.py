import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from alexnet_cifar10 import AlexNetCIFAR10
from utils import evaluate
from tqdm import tqdm
import time
import pandas as pd

# Efficient reduced grid
CONFIGS = []
LEARNING_RATES = {
    'adam': [0.001, 0.0001],
    'sgd': [0.01, 0.001],
    'rmsprop': [0.001, 0.0001]
}
BATCH_SIZES = [32, 64]
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
SPLITS = [(0.7, 0.3), (0.9, 0.1)]

# Efficient toggle: use Dropout if BN is off, else keep both off
for opt in OPTIMIZERS:
    for lr in LEARNING_RATES[opt]:
        for bs in BATCH_SIZES:
            for split in SPLITS:
                CONFIGS.append((lr, bs, opt, split, True, False))   # BN=True, DO=False
                CONFIGS.append((lr, bs, opt, split, False, True))   # BN=False, DO=True

# Constants
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

full_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
results = []
start_all = time.time()

for idx, (lr, bs, opt, (train_frac, test_frac), use_bn, use_dropout) in enumerate(CONFIGS):
    train_len = int(len(full_dataset) * train_frac)
    test_len = len(full_dataset) - train_len
    train_set, test_set = random_split(full_dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    model = AlexNetCIFAR10(use_bn=use_bn, use_dropout=use_dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    print(f"\n[{idx+1}/{len(CONFIGS)}] LR={lr}, BS={bs}, OPT={opt}, Split={train_frac}, BN={use_bn}, DO={use_dropout}")
    total_time = time.time()
    best_acc, best_epoch, epochs_since_improvement = 0, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=epoch_loss/len(train_loader), acc=100.*correct/total)

        acc = 100. * correct / total
        if acc > best_acc:
            best_acc, best_epoch, epochs_since_improvement = acc, epoch, 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            break

    elapsed = time.time() - total_time
    test_acc = evaluate(model, test_loader, DEVICE)

    results.append({
        'lr': lr,
        'batch_size': bs,
        'optimizer': opt,
        'train_pct': train_frac,
        'test_pct': test_frac,
        'batch_norm': use_bn,
        'dropout': use_dropout,
        'best_train_acc': best_acc,
        'epochs_ran': best_epoch,
        'test_accuracy': test_acc,
        'train_time': round(elapsed, 2)
    })

print("\nAll experiments complete. Total elapsed time: {:.2f}s".format(time.time() - start_all))

# Save results
pd.DataFrame(results).to_csv("results_reduced.csv", index=False)
print("Results saved to results_reduced.csv")
