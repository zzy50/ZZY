import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):

    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")


    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad] # requires_grad=True로 설정되어있는 파라미터만 리스트에 담음 (requires_grad=False면 freeze시킨 파라미터이므로 가져올 필요가 없음)

    optimizer = optim.Adam(
        params,
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
    )

    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train() 

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i # 100부터 시작해서 batch를 뽑을 때마다 1씩 커짐. 1 epoch 막판에 199가 되고 2 epoch로 넘어가면 200이 됨

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # if문을 사용해 lr_scheduler를 직접 정의
                # 학습 초기에 lr을 매우 낮게 준 후 점점 키워감 
                # 학습이 끝날 때 쯤 되면 다시 lr를 대폭 낮춤 

                lr = model.hyperparams['learning_rate'] 
                if batches_done < model.hyperparams['burn_in']: # batches_done이 1000보다 작을수록 lr을 많이 감소시킴 (yolov3.cfg에 burn_in=1000으로 지정되어있음)
                    lr *= (batches_done / model.hyperparams['burn_in']) 
                else:

                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                            # batches_done이 max_batches인 500200의 80%(400000)에 도달했을 때 lr을 10배 줄임(0.1) 
                            # 90%(450000)에 도달하면 lr을 100배 줄임(for문으로 대입연산자가 두 번 실행되어 0.1*0.1 = 0.01)
            
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                
                # 위에서 정해진 lr을 optimizer에 적용
                for g in optimizer.param_groups:
                    g['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", loss.detach().cpu().item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", loss.detach().cpu().item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    run()
