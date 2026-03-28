from model import LogReg
import torch.nn as nn
import torch as th
import numpy as np
from sklearn.model_selection import train_test_split


def evaluation(embeddings, y, name, device, data, learning_rate2, weight_decay2):
    print("=== Evaluation ===")
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()

    # 内置数据集类别数
    num_class = {
        'Cora': 7,
        'CiteSeer': 6,
        'PubMed': 3,
        'DBLP': 10,
        'computers': 10,
        'photo': 8,
        'CS': 15,
        'Physics': 5
    }

    train_embs, val_embs, test_embs = None, None, None
    train_labels, val_labels, test_labels = None, None, None

    # --- 如果 data 有 mask，就用 mask ---
    if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
        train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze().cpu().numpy()
        val_idx = th.nonzero(data.val_mask, as_tuple=False).squeeze().cpu().numpy()
        test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze().cpu().numpy()

        train_embs = X[train_idx]
        val_embs = X[val_idx]
        test_embs = X[test_idx]

        train_labels = Y[train_idx]
        val_labels = Y[val_idx]
        test_labels = Y[test_idx]
    else:
        # --- 如果没有 mask，就随机划分 ---
        train_embs, test_embs, train_labels, test_labels = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        train_embs, val_embs, train_labels, val_labels = train_test_split(
            train_embs, train_labels, test_size=0.25, random_state=42
        )
        # 0.6 train / 0.2 val / 0.2 test

    # 转为 tensor
    train_embs = th.tensor(train_embs, dtype=th.float32).to(device)
    val_embs = th.tensor(val_embs, dtype=th.float32).to(device)
    test_embs = th.tensor(test_embs, dtype=th.float32).to(device)

    train_labels = th.tensor(train_labels, dtype=th.long).to(device)
    val_labels = th.tensor(val_labels, dtype=th.long).to(device)
    test_labels = th.tensor(test_labels, dtype=th.long).to(device)

    # --- Linear Evaluation ---
    num_cls = num_class.get(name, len(np.unique(Y)))  # 若未知数据集，则自动统计类别数
    logreg = LogReg(train_embs.shape[1], num_cls).to(device)
    opt = th.optim.Adam(logreg.parameters(), lr=learning_rate2, weight_decay=weight_decay2)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = (val_preds == val_labels).float().mean()
            test_acc = (test_preds == test_labels).float().mean()

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc

    print('Linear evaluation accuracy: {:.4f}'.format(eval_acc))
    return eval_acc
