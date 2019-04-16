import collections
from tqdm import tqdm
import torch.nn as nn
import torch


def validate(model, dl):
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
                correct.extend([(1 if p == t else 0) for p, t in zip(prediction, target)])
            loss = torch.sum(torch.stack(losses))
            metrics['loss'].append(loss.item())
            pbar.update(1)
            desc = f'loss: {loss.item():.6f}'
            pbar.set_description(desc)
        else:
            accuracies = []
            print(model.products)
            for j, product in enumerate(model.products):
                accuracy = correct[j::model.n_products]
                accuracy = 100 * sum(accuracy) / len(accuracy)
                metrics[f'{product}_accuracy'] = accuracy
                accuracies.append(accuracy)
            accuracy = sum(accuracies) / len(accuracies)
            metrics[f'mean_accuracy'] = accuracy
            desc = f'loss: {loss.item():.6f} | acc: {accuracy:.2f}%'
            pbar.set_description(desc)
    return metrics


