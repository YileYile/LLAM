"""Some helper functions for Python, including:
    - adjust_learning_rate: adjust the learning rate of optimizer
    - clip_gradient: clip gradient
"""
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools


def adjust_learning_rate(optimizer, epoch,
                         lr_base=0.01,
                         lr_decay_start=60,
                         lr_decay_every=8,
                         lr_decay_rate=0.9):

    if epoch > lr_decay_start:
        frac = (epoch - lr_decay_start) / lr_decay_every
        decay_factor = lr_decay_rate ** frac
        current_lr = lr_base * decay_factor
        for group in optimizer.param_groups:
            group['lr'] = current_lr
    else:
        current_lr = lr_base
    return current_lr


def adjust_learning_rate2(optimizer, epoch, lr_base,
                          lr_decay_every=40,):

    if epoch > 0:
        frac = epoch // lr_decay_every
        decay_factor = 10 ** frac
        current_lr = lr_base / decay_factor
        for group in optimizer.param_groups:
            group['lr'] = current_lr
    else:
        current_lr = lr_base
    return current_lr

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
#print(term_width)
term_width = 82

TOTAL_BAR_LENGTH = 30
last_time = time.time()
begin_time = time.time()


def progress_bar(current, total, msg=None):

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total -1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.around(cm, decimals=2)
    sum = 0
    for idx in range(7):
        sum += cm[idx][idx]
    print('mACC:%0.2f' % (sum / 7 * 100.))
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
