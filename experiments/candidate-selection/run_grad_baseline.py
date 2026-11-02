"""Validation-only adapter for the authors' GIANT -> GraDBERT SAGN/SLE code.

Preserves source files and stage settings; records compatibility-only changes.
Official test labels are masked before upstream preprocessing or training.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch


def masked_labels(labels, test_ids):
    result = labels.clone()
    result[test_ids] = -1
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--giant', required=True)
    p.add_argument('--gradbert', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, nargs=4, default=[100, 100, 100, 10])
    p.add_argument('--torch-threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.torch_threads)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = Path(args.upstream).resolve() / 'GradBERT/ogbn-products/SAGN_with_SLE/src'
    runtime = out / 'source'
    shutil.copytree(original, runtime)
    replacements = {}
    # CPU aggregation leaves features in host memory, while upstream batches
    # follow the CUDA label device. Preserve the operation, fix index device.
    for name in ('sagn.py', 'train_process.py'):
        path = runtime / name
        source = path.read_text()
        fixed = source.replace('[x[batch].to(device) for x in feats]', '[x[batch.cpu()].to(device) for x in feats]')
        fixed = fixed.replace('label_emb[batch].to(device)', 'label_emb[batch.cpu()].to(device)')
        if name == 'sagn.py':
            fixed = fixed.replace('best_model.eval()',
                'best_model.eval()\n        torch.save(best_model.state_dict(), os.path.join(args.probs_dir, f"model_seed_{args.seed}_stage_{stage}.pt"))')
        path.write_text(fixed)
        replacements[name] = {'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                              'adapted_sha256': hashlib.sha256(fixed.encode()).hexdigest()}
    path = runtime / 'pre_process.py'
    source = path.read_text()
    # Upstream caches GIANT and GraDBERT under the same filename. Distinguish
    # them so stage 3 cannot silently reuse the early-stage GIANT cache.
    fixed = source.replace('f"smoothed_embs_K_{args.K}.pt"',
                           'f"smoothed_embs_K_{args.K}_{\'gradbert\' if stage == args.K else \'giant\'}.pt"')
    assert fixed != source
    path.write_text(fixed)
    replacements[path.name] = {'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                               'adapted_sha256': hashlib.sha256(fixed.encode()).hexdigest()}
    config = {**vars(args), 'scope': 'author-source SAGN/SLE with released GIANT/GraDBERT-KD features; reproduction pending',
              'test_labels_masked_before_training': True, 'test_labels_scored': False,
              'checkpoint_selection': 'full official validation set; controller comparisons must disclose this',
              'compatibility_changes': replacements, 'torch': torch.__version__,
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'driver.py').write_bytes(Path(__file__).read_bytes())
    sys.path.insert(0, str(runtime))
    import dgl.function as fn
    if not hasattr(fn, 'copy_src'):
        fn.copy_src = lambda src, out: fn.copy_u(src, out)
    if not hasattr(fn, 'copy_edge'):
        fn.copy_edge = lambda edge, out: fn.copy_e(edge, out)
    import sagn
    import dataset
    import pre_process
    original_load = dataset.load_dataset

    def load_without_test(device, settings):
        graph, labels, classes, train, valid, test, evaluator = original_load(device, settings)
        labels = masked_labels(labels, test)
        assert (labels[test] == -1).all()
        return graph, labels, classes, train, valid, test, evaluator

    dataset.load_dataset = pre_process.load_dataset = sagn.load_dataset = load_without_test

    @torch.no_grad()
    def validation_only(model, feats, label_emb, teacher_probs, labels, loss_fcn, val_loader,
                        test_loader, evaluator, train_nid, val_nid, test_nid, settings):
        del test_loader, train_nid, test_nid, teacher_probs
        model.eval()
        device = labels.device
        predictions, losses, sizes = [], [], []
        started = time.perf_counter()
        for batch in val_loader:
            features = [f[batch.cpu()].to(device) for f in feats]
            label_features = label_emb[batch.cpu()].to(device) if label_emb is not None else None
            logits, _ = model(features, label_features)
            assert (labels[batch] >= 0).all()
            losses.append(float(loss_fcn(logits, labels[batch])))
            sizes.append(len(batch))
            predictions.append(logits.argmax(-1))
        accuracy = evaluator(torch.cat(predictions), labels[val_nid])
        return float('nan'), accuracy, float('nan'), float(np.average(losses, weights=sizes)), time.perf_counter() - started

    sagn.test = validation_only
    settings = sagn.define_parser().parse_args([
        '--dataset', 'ogbn-products', '--gpu', '0', '--aggr-gpu', '-1', '--model', 'sagn',
        '--seed', str(args.seed), '--num-runs', '1', '--threshold', '.95', '.95', '.999',
        '--epoch-setting', *map(str, args.epochs), '--lr', '.001', '--zero-inits',
        '--batch-size', '50000', '--num-hidden', '256', '--num-heads', '1',
        '--dropout', '.5', '--attn-drop', '.4', '--input-drop', '.2', '--label-drop', '.5',
        '--K', '3', '--label-K', '9', '--use-labels', '--weight-decay', '0',
        '--warmup-stage', '-1', '--mlp-layer', '1', '--load-embs',
        '--data_root_dir', str(Path(args.data_root).resolve()),
        '--pretrain_path', str(Path(args.giant).resolve()), '--node_emb_path', str(Path(args.gradbert).resolve()),
        '--probs_dir', str(out / 'predictions')])
    settings.data_dir = settings.data_root_dir
    Path(settings.probs_dir).mkdir(parents=True)
    (out / 'upstream-arguments.json').write_text(json.dumps(vars(settings), indent=2) + '\n')
    device = torch.device('cuda:0')
    sagn.seed(args.seed)
    stages = []
    previous = ''
    for stage in range(4):
        print(f'Starting stage {stage}: {"GraDBERT-KD" if stage == 3 else "GIANT"}', flush=True)
        data = sagn.prepare_data(device, settings, previous, stage, load_embs=True, load_label_emb=False, subset_list=None)
        valid, _, probabilities, train_times, inference_times, val_accs, val_losses, attention = sagn.run(settings, data, device, stage)
        destination = out / f'seed-{args.seed}-stage-{stage}.pt'
        torch.save(probabilities, destination)
        record = {'stage': stage, 'feature': 'gradbert_kd' if stage == 3 else 'giant',
                  'best_validation_accuracy': valid, 'epochs': args.epochs[stage],
                  'preprocessing_seconds': data[-1], 'train_seconds': train_times,
                  'validation_seconds': inference_times, 'validation_accuracy_curve': val_accs,
                  'validation_loss_curve': val_losses, 'test_scored': False}
        stages.append(record)
        (out / 'stages.json').write_text(json.dumps(stages, indent=2) + '\n')
        previous = str(destination)
        del data, probabilities, attention
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Completed stage {stage}: validation={valid:.8f}', flush=True)
    (out / 'BASE_TRAINING_COMPLETE').write_text('Pre-C&S probabilities ready; no official test score computed.\n')


if __name__ == '__main__':
    main()
