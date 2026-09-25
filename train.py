import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss_stage1
from core.losses import MultimodalLoss_stage2
from core.scheduler import get_scheduler
from core.utils import setup_seed, save_model
from models.recap import build_model
from core.metric import MetricsTop, classification_metrics
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
parser.add_argument('--stage', type=str, choices=['completion', 'fusion_prediction'], default='fusion_prediction')
parser.add_argument('--time', type=str, default='') 
parser.add_argument('--stage1_ckpt', type=str, default='')
parser.add_argument('--missing_rate_eval_test', type=float, default=None) 
parser.add_argument('--batch_size', type=int, default=64) 
opt = parser.parse_args()
print(opt)
from datetime import datetime


def resolve_stage1_checkpoint(ckpt_root):
    if opt.stage1_ckpt:
        return opt.stage1_ckpt
    if opt.time:
        if opt.time.endswith('.pth'):
            return os.path.join(ckpt_root, opt.time)
        return os.path.join(ckpt_root, f'stage1_modules_{opt.time}.pth')
    raise ValueError("Please provide --stage1_ckpt or --time for stage 2 training.")


def compute_class_weights(train_loader):
    labels = torch.as_tensor(train_loader.dataset.labels['C']).view(-1).long()
    counts = torch.bincount(labels, minlength=3).float()
    if (counts == 0).any():
        raise ValueError(f"All three polarity classes are required; counts={counts.tolist()}")
    weights = labels.numel() / (3.0 * counts)
    print(f"Polarity class counts: {counts.int().tolist()}")
    print(f"Polarity class weights: {weights.tolist()}")
    return weights.to(device)


def build_stage2_optimizer(model, args):
    bert_lr = args['base'].get('bert_lr', args['base']['lr'] * 0.1)
    bert_parameters = [p for p in model.bertmodel.parameters() if p.requires_grad]
    bert_parameter_ids = {id(p) for p in bert_parameters}
    other_parameters = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in bert_parameter_ids
    ]
    print(f"Stage 2 learning rates: BERT={bert_lr}, other={args['base']['lr']}")
    return torch.optim.AdamW(
        [
            {'params': bert_parameters, 'lr': bert_lr},
            {'params': other_parameters, 'lr': args['base']['lr']},
        ],
        weight_decay=args['base']['weight_decay'],
    )


def validation_selection_score(results, args):
    """One validation-only score balancing both competition tasks."""
    corr_weight = args['base'].get('selection_corr_weight', 0.25)
    mae_weight = args['base'].get('selection_mae_weight', 0.25)
    return (
        results['Polarity_Macro_F1']
        + corr_weight * results['Corr']
        - mae_weight * results['MAE']
    )


def main():
    best_valid_results, best_test_results = {}, {}
    loss_history = {}
    config_file = 'configs/train_mosi.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    if opt.missing_rate_eval_test is not None:
        args['base']['missing_rate_eval_test'] = opt.missing_rate_eval_test
        print("train: ", args['base']['missing_rate_eval_test'])

    stage = opt.stage
    print(f"Training stage: {stage}")

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)


    loss_fn_stage1 = MultimodalLoss_stage1(args)
    if stage == 'fusion_prediction':
        class_weights = compute_class_weights(dataLoader['train'])
    else:
        class_weights = None
    loss_fn_stage2 = MultimodalLoss_stage2(
        args, class_weights=class_weights
    ).to(device)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'stage1_modules_seed{seed}_{timestamp}.pth'
    
    if opt.stage == 'completion':
        print("===> Stage 1: Completion training")
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args['base']['lr'],
                                    weight_decay=args['base']['weight_decay'])
        scheduler_warmup = get_scheduler(optimizer, args)
        best_model_state = None
        best_loss = float('inf')
        for epoch in range(1, args['base']['n_epochs_stage1'] + 1):
            train_loss_dict = train(model, dataLoader['train'], optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode='completion')
            for key, value in train_loss_dict.items():
                if key not in loss_history:
                    loss_history[key] = []
                loss_history[key].append(value)            
            scheduler_warmup.step()

            current_loss = train_loss_dict["loss"]
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {'generator': model.generator.state_dict()}
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(ckpt_root, filename))
            print(f"Best model saved with loss_total: {best_loss:.4f}")
        print("===> Saved completion & discriminator after Stage 1")
     
    elif opt.stage == 'fusion_prediction':
        print("===> Stage 2: Fusion training")
        
        # Load the Stage 1 module
        stage1_ckpt = resolve_stage1_checkpoint(ckpt_root)
        checkpoint = torch.load(stage1_ckpt, map_location=device)
        print(f"Loaded stage 1 checkpoint: {stage1_ckpt}")
        model.generator.load_state_dict(checkpoint['generator'])

        for p in model.generator.parameters():
            p.requires_grad = False

        optimizer = build_stage2_optimizer(model, args)
    
        # Recreate the learning rate scheduler
        scheduler_warmup = get_scheduler(optimizer, args)
        best_valid_f1 = float('-inf')
        best_valid_mae = float('inf')
        best_valid_joint = float('-inf')
        best_valid_epoch = None
        best_classification_path = os.path.join(
            ckpt_root, f'best_valid_polarity_f1_seed{seed}.pth'
        )
        best_regression_path = os.path.join(
            ckpt_root, f'best_valid_mae_seed{seed}.pth'
        )
        best_joint_path = os.path.join(
            ckpt_root, f'best_valid_joint_seed{seed}.pth'
        )
        epochs_without_improvement = 0
        early_stopping_patience = args['base'].get('early_stopping_patience', 25)

        for epoch in range(1, args['base']['n_epochs_stage2']+1):
            train_loss_dict = train(model, dataLoader['train'], optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode=stage) 
            for key, value in train_loss_dict.items():
                if key not in loss_history:
                    loss_history[key] = []
                loss_history[key].append(value)
            # train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

            if args['base']['do_validation']:
                valid_results = evaluate(model, dataLoader['valid'], loss_fn_stage2, epoch, metrics)
                current_valid_f1 = valid_results['Polarity_Macro_F1']
                current_valid_mae = valid_results['MAE']
                current_valid_joint = validation_selection_score(valid_results, args)
                print(f'Valid Results Epoch {epoch}: {valid_results}')
                print(f'Validation Joint Score Epoch {epoch}: {current_valid_joint:.4f}')

                if current_valid_f1 > best_valid_f1:
                    best_valid_f1 = current_valid_f1
                    save_model(best_classification_path, epoch, model, optimizer)
                    print(f'New best classification checkpoint: {best_classification_path}')

                if current_valid_mae < best_valid_mae:
                    best_valid_mae = current_valid_mae
                    save_model(best_regression_path, epoch, model, optimizer)
                    print(f'New best regression checkpoint: {best_regression_path}')

                if current_valid_joint > best_valid_joint:
                    best_valid_joint = current_valid_joint
                    best_valid_epoch = epoch
                    best_valid_results = dict(valid_results)
                    save_model(best_joint_path, epoch, model, optimizer)
                    epochs_without_improvement = 0
                    print(f'New best joint checkpoint: {best_joint_path}')
                else:
                    epochs_without_improvement += 1

                print(
                    f'Best Valid Epoch: {best_valid_epoch}; '
                    f'Best Valid Results: {best_valid_results}\n'
                )

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f'Early stopping after {early_stopping_patience} epochs '
                        'without joint-score improvement.'
                    )
                    break

            scheduler_warmup.step()

        if best_valid_epoch is not None:
            checkpoint = torch.load(best_joint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            best_test_results = evaluate(
                model, dataLoader['test'], loss_fn_stage2, best_valid_epoch, metrics
            )
            print(f'Final Selected Checkpoint: {best_joint_path}')
            print(f'Test Results at Selected Epoch {best_valid_epoch}: {best_test_results}')


def train(model, train_loader, optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode='completion'):
    y_pred, y_true = [], []
    polarity_logits, polarity_true = [], []
    loss_dict = {}
    results = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        classification_labels = data['labels'].get('C')
        label = {'sentiment_labels': sentiment_labels}
        if classification_labels is not None:
            label['classification_labels'] = classification_labels.to(device)

        if mode == 'completion':
            out = model(complete_input, incomplete_input, sentiment_labels, mode='completion')
            loss_stage1 = loss_fn_stage1(out, label)
            loss_stage1['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

            if cur_iter == 0:
                for key, value in loss_stage1.items():
                    loss_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
            else:
                for key, value in loss_stage1.items():
                    # loss_dict[key] += value.item()
                    loss_dict[key] += value.item() if isinstance(value, torch.Tensor) else value

        else:
            out = model(complete_input, incomplete_input, sentiment_labels, mode='fusion_prediction')
            loss_stage2 = loss_fn_stage2(out, label)
            optimizer.zero_grad()
            loss_stage2['loss'].backward()
            optimizer.step()
            
            y_pred.append(out['sentiment_preds'].cpu())
            y_true.append(label['sentiment_labels'].cpu())
            polarity_logits.append(out['polarity_logits'].cpu())
            polarity_true.append(label['classification_labels'].cpu())

            if cur_iter == 0:
                for key, value in loss_stage2.items():
                    loss_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
            else:
                for key, value in loss_stage2.items():
                    # loss_dict[key] += value.item()
                    loss_dict[key] += value.item() if isinstance(value, torch.Tensor) else value

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            results = metrics(pred, true)
            results.update(classification_metrics(
                torch.cat(polarity_logits), torch.cat(polarity_true)
            ))


    num_batches = cur_iter + 1 if 'cur_iter' in locals() else 1
    loss_dict = {key: value / num_batches for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')

    return loss_dict

def evaluate(model, eval_loader, loss_fn_stage2, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []
    polarity_logits, polarity_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        classification_labels = data['labels']['C'].to(device)
        label = {
            'sentiment_labels': sentiment_labels,
            'classification_labels': classification_labels,
        }
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input, sentiment_labels, mode='fusion_prediction')

        loss = loss_fn_stage2(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())
        polarity_logits.append(out['polarity_logits'].cpu())
        polarity_true.append(label['classification_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item() 
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    results.update(classification_metrics(
        torch.cat(polarity_logits), torch.cat(polarity_true)
    ))

    return results


if __name__ == '__main__':
    main()
