import collections
from tqdm import tqdm
import torch.nn as nn
import torch


def validate(model, dl):
    metrics = collections.defaultdict(list)
    criterion = nn.NLLLoss()
    with tqdm(total=len(dl)) as pbar:
        correct, total = 0, 0
        for j, (batch, targets) in enumerate(dl):
            metrics['positives'].append(targets.sum().item())
            hypothesis = model(batch)
            losses = []
            for i, (prediction, target) in enumerate(zip(hypothesis, targets)):
                losses.append(criterion(prediction, target))
                prediction = torch.argmax(prediction, dim=-1)
                correct += sum([(1 if p == t else 0) for p, t in zip(prediction, target)])
                total += len(prediction)
            loss = torch.sum(torch.stack(losses))
            metrics['loss'].append(loss.item())
            metrics['accuracy'].append(correct / total)
            pbar.update(1)
            accuracy = 100.0 * correct / total
            pbar.set_description('loss: {} | acc: {}%'.format(loss.item(), accuracy))
    return metrics


