import os
import numpy as np
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor


def save_checkpoint(savepath, data_name, model, optimizer, is_best, epoch):
    filename = 'checkpoint' + data_name + '.pth.tar'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if is_best:
        state = {'model': model, 'optimizer': optimizer}
        torch.save(state, os.path.join(savepath, 'BEST_' + filename))

    # state = {'model': model, 'optimizer': optimizer}
    # torch.save(state, os.path.join(savepath, f"{epoch}_" + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses, save_scores=False):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    all_scores = {} 

    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))

        if type(method_i) == list:
            all_scores['Bleu_4'] = scores_i[-1]
        else:
            all_scores[method_i] = scores_i

    score_dict = dict(zip(method, score))
    
    # if save_scores:
    #     json.dump(all_scores, open('all_scores_aenet_dubai.json', 'w'))

    return score_dict


def sentence_decode(sequences, rev_word_map, end_idx):
    for i, seq in enumerate(sequences):
        caption = []
        for w in seq:
            caption.append(rev_word_map[w])
            if w == end_idx:
                break   
        # print(i, "  ||  ", " ".join(caption))


def batch_sentence_decode(sequences, rev_word_map, end_idx):
    """
    sequences: [batch_size, beam_size, seq_len]
    """
    for seqs in sequences:
        sentence_decode(seqs, rev_word_map, end_idx)


def print_component_params(model, component_names=None):
    def count_parameters(model_component):
        total_params = sum(p.numel() for p in model_component.parameters())
        trainable_params = sum(p.numel() for p in model_component.parameters() if p.requires_grad)
        return total_params, trainable_params

    print("=" * 60)
    print("Params")
    print("=" * 60)

    if component_names is not None:
        total_model_params = 0
        total_trainable_params = 0        
        for name in component_names:
            if hasattr(model, name):
                component = getattr(model, name)
                total_params, trainable_params = count_parameters(component)
                total_model_params += total_params
                total_trainable_params += trainable_params
                
                print(f"{name.upper()}:")
                print(f"  total: {total_params/1e6:.4f}")
                print(f"  trainable: {trainable_params/1e6:.4f}")
                print(f"  ratio: {total_params/sum(p.numel() for p in model.parameters())*100:.2f}%")
                print("-" * 40)

        print("Total:")
        print(f"  total: {total_model_params/1e6:.4f}")
        print(f"  trainable: {total_trainable_params/1e6:.4f}")
        print(f"  frozen: {(total_model_params - total_trainable_params)/1e6:.4f}")
            
    else:
        total_params, trainable_params = count_parameters(model)
        print(f"  total: {total_params/1e6:.4f}")
        print(f"  trainable: {trainable_params/1e6:.4f}")   
        print(f"  frozen: {(total_params - trainable_params)/1e6:.4f}")



def expand_tensor(tensor, size, dim):
    assert size > 1, "size should be larger than 1"
    return torch.repeat_interleave(tensor, size, dim=dim)


def expand_numpy(x, size):
    assert size > 1, "size should be larger than 1"
    x = x.reshape((-1, 1))
    x = np.repeat(x, size, axis=1)
    x = x.reshape((-1))
    return x


def expand_list(x, size):
    return [item for item in x for _ in range(size)]


def compute_precision_recall(y_true, y_pred):
    classes = sorted(set(y_true))
    
    results = {}
    
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results[cls] = {
            'precision': precision,
            'recall': recall
        }
    
    return results


def weight_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
        nn.init.zeros_(module.bias)
