import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETP
from utils.metrics import credit_rating_metrics

from models.CGDPM import CGDPM


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {'true', '1', 'yes', 'y', 't'}:
        return True
    if v in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    cluster_dispersion_weight=0.01,
    latent_sep_weight=0.01,
):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_cluster = 0.0
    total_sep = 0.0

    for batch in dataloader:
        x = batch['series'].to(device)
        s = batch['static'].to(device)
        y = batch['label'].to(device)
        x_mark = batch['time_marks'].to(device)

        logits, cluster_probs, mod_info = model(x, s, x_mark)

        cls_loss = criterion(logits, y)
        cluster_dispersion_loss = model.compute_cluster_entropy_loss(cluster_probs)

        if latent_sep_weight > 0 and 'coarse_delta' in mod_info and 'fine_delta' in mod_info:
            latent_sep_loss = model.compute_latent_orthogonality_loss(
                mod_info['coarse_delta'],
                mod_info['fine_delta'],
            )
        else:
            latent_sep_loss = torch.zeros((), device=logits.device)

        loss = (
            cls_loss
            + cluster_dispersion_weight * cluster_dispersion_loss
            + latent_sep_weight * latent_sep_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_cluster += cluster_dispersion_loss.item()
        total_sep += latent_sep_loss.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'cls_loss': total_cls / n_batches,
        'cluster_dispersion_loss': total_cluster / n_batches,
        'latent_sep_loss': total_sep / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    for batch in dataloader:
        x = batch['series'].to(device)
        s = batch['static'].to(device)
        y = batch['label'].to(device)
        x_mark = batch['time_marks'].to(device)

        logits, _, _ = model(x, s, x_mark)
        preds = logits.argmax(dim=1)

        all_logits.extend(logits.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_logits = np.array(all_logits)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_logits, all_preds, all_labels


def compute_metrics(logits, preds, labels, num_classes):
    core = credit_rating_metrics(
        logits=logits,
        preds=preds,
        labels=labels,
        num_classes=num_classes,
    )
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    core['cm'] = cm
    return core


def plot_confusion_matrix(cm, num_classes, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_setting_string(args, ii=0):
    return '{}_{}_{}_{}_{}_clt{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_revin{}_factor{}_embed{}_distil{}_norm{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.time_model,
        args.data,
        args.features,
        args.num_clusters,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        int(args.revin),
        args.factor,
        args.embed,
        int(args.distil),
        int(args.norm),
        args.des,
        ii,
    )


def main(args):
    # Normalize output path.
    args.output_dir = str(Path(args.output_dir).resolve())
    
    setting = generate_setting_string(args, ii=0)
    exp_output_dir = str(Path(args.output_dir) / setting)
    os.makedirs(exp_output_dir, exist_ok=True)
    args.output_dir = exp_output_dir

    print(f"\n{'=' * 80}")
    print(f'Experiment: {setting}')
    print(f'Output Dir: {exp_output_dir}')
    print(f"{'=' * 80}\n")
    for k, v in sorted(vars(args).items()):
        print(f'  {k}: {v}')
    print()

    root_path = args.data_path
    data_name = str(args.data).upper()
    train_dataset = Dataset_ETP(
        args=args,
        root_path=root_path,
        flag='train',
        features='S',
        data_path=data_name,
        target='level',
        scale=True,
        norm=args.norm,
    )
    val_dataset = Dataset_ETP(
        args=args,
        root_path=root_path,
        flag='test',
        features='S',
        data_path=data_name,
        target='level',
        scale=True,
        norm=args.norm,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = args.num_classes

    train_labels = train_dataset.labels
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    class_weights = np.ones(num_classes)
    for label, count in zip(unique_labels, label_counts):
        class_weights[label] = len(train_labels) / (num_classes * count)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f'Class Weights: {class_weights.cpu().numpy()}')

    model = CGDPM(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    print('=' * 80)
    print('TRAINING START (Dual Phase)')
    print('=' * 80)

    for epoch in range(args.num_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cluster_dispersion_weight=getattr(args, 'cluster_dispersion_weight', 0.01),
            latent_sep_weight=getattr(args, 'latent_sep_weight', 0.01),
        )

        val_logits, val_preds, val_labels = evaluate(model, val_loader)
        metrics = compute_metrics(
            val_logits,
            val_preds,
            val_labels,
            num_classes,
        )
        val_acc = metrics['eda_acc']

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"Cls: {train_stats['cls_loss']:.4f} | "
            f"ClusterDisp: {train_stats['cluster_dispersion_loss']:.4f} | "
            f"LatSep: {train_stats['latent_sep_loss']:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"F1-macro: {metrics['f1_macro']:.4f}"
        )

        if epoch == 0 or val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            plot_confusion_matrix(
                metrics['cm'],
                num_classes,
                os.path.join(args.output_dir, 'confusion_matrix_best.png'),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stop at epoch {epoch}, best accuracy: {best_val_acc:.4f}')
                break

        scheduler.step()

    print('\n' + '=' * 80)
    print('FINAL VALIDATION SUMMARY')
    print('=' * 80)

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    final_logits, final_preds, final_labels = evaluate(model, val_loader)
    final_metrics = compute_metrics(
        final_logits,
        final_preds,
        final_labels,
        num_classes,
    )

    print('\nValidation Set Results:')
    print(f"  Accuracy:      {final_metrics['acc']:.4f}")
    print(f"  F1-Macro:      {final_metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted:   {final_metrics['f1_weighted']:.4f}")
    print(f"  AUC:           {final_metrics['auc']:.4f}")
    print(f"  EDA-Acc:       {final_metrics['eda_acc']:.4f}")
    print('\nConfusion Matrix:')
    print(final_metrics['cm'])
    print('\nClassification Report:')
    print(classification_report(final_labels, final_preds))

    metrics_file = os.path.join(args.output_dir, 'best_metrics.txt')
    with open(metrics_file, 'a', encoding='utf-8') as f:
        try:
            if os.path.getsize(metrics_file) > 0:
                f.write('\n' + '=' * 80 + '\n')
        except OSError:
            pass
        f.write('Experiment Parameters:\n')
        f.write(f"{'-' * 80}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')
        f.write('\n')
        f.write('Final Validation Results\n')
        f.write(f"{'-' * 80}\n")
        f.write(f"Accuracy: {final_metrics['acc']:.4f}\n")
        f.write(f"F1-Macro: {final_metrics['f1_macro']:.4f}\n")
        f.write(f"F1-Weighted: {final_metrics['f1_weighted']:.4f}\n")
        f.write(f"AUC: {final_metrics['auc']:.4f}\n")
        f.write(f"EDA-Acc: {final_metrics['eda_acc']:.4f}\n")
        f.write('\nLoss Weights:\n')
        f.write(f"cluster_dispersion_weight: {args.cluster_dispersion_weight}\n")
        f.write(f"latent_sep_weight: {args.latent_sep_weight}\n")
        f.write(f"\nConfusion Matrix:\n{final_metrics['cm']}\n")
        f.write('\nClassification Report:\n')
        f.write(classification_report(final_labels, final_preds))

    print(f'\n[OK] Best metrics saved to: {metrics_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Phase Training')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--series_file', type=str, default='ETP_series.csv')
    parser.add_argument('--static_file', type=str, default='ETP_static_nostd.csv')
    parser.add_argument('--data', type=str, default='ETP')
    parser.add_argument('--uea_file', type=str, default='sample_data.ts')
    parser.add_argument('--uea_static_conv_kernel', type=int, default=5)
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--seq_len', type=int, default=251)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--random_split', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--norm', type=str2bool, default=True)

    parser.add_argument('--model', type=str, default='SCGTSC')
    parser.add_argument('--model_id', type=str, default='SGCTSC_DualPhase')
    parser.add_argument('--time_model', type=str, default='Transformer')
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--static_dim', type=int, default=36)
    parser.add_argument('--static_emb_dim', type=int, default=32)
    parser.add_argument('--cluster_emb_dim', type=int, default=16)
    parser.add_argument('--ts_hidden_dim', type=int, default=64)
    parser.add_argument('--coarse_rank', type=int, default=8)

    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--revin', action='store_true', default=False)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=96)
    
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--pos', type=str2bool, default=True)

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cluster_dispersion_weight', type=float, default=0.01)
    parser.add_argument('--latent_sep_weight', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--des', type=str, default='dual_phase')
    parser.add_argument('--task_name', default='classification')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
