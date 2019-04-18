import collections
from tqdm import tqdm
import torch.nn as nn
import torch


def validate(model, dl, products):
    model.eval()
    metrics = collections.defaultdict(list)
    criterion = nn.NLLLoss()
    with tqdm(total=len(dl)) as pbar:
        correct = []
        for j, (batch, targets) in enumerate(dl):
            metrics['positives'].append(targets.sum().item())
            hypothesis = model(batch)
            losses = []
            for i, (prediction, target) in enumerate(zip(hypothesis, targets)):
                losses.append(criterion(prediction, target))
                prediction = torch.argmax(prediction, dim=-1)
                correct.extend([int(p == t) for p, t in zip(prediction, target)])
            loss = torch.sum(torch.stack(losses))
            metrics['loss'].append(loss.item())
            pbar.update(1)
            desc = f'loss: {loss.item():.6f}'
            pbar.set_description(desc)
        else:
            accuracies = []
            for j, product in enumerate(products):
                accuracy = correct[j::len(products)]
                accuracy = 100 * sum(accuracy) / len(accuracy)
                metrics[f'{product}_accuracy'] = accuracy
                accuracies.append(accuracy)
                print(f'Product: {product}: {accuracy}%')
            accuracy = sum(accuracies) / len(accuracies)
            metrics[f'mean_accuracy'] = accuracy
            desc = f'loss: {loss.item():.6f} | acc: {accuracy:.2f}%'
            pbar.set_description(desc)
    return metrics


