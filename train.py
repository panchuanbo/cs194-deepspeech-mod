import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex.fp16_utils import FP16_Optimizer
from apex.parallel import DistributedDataParallel
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model_custom_dataset import DeepSpeech, supported_rnns
from utils import convert_model_to_half, reduce_tensor, check_loss

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--mixed-precision', action='store_true',
                    help='Uses mixed precision to train a model (suggested with volta and above)')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale for mixed precision, ' +
                         'positive power of 2 values can improve FP16 convergence,' +
                         'however dynamic loss scaling is preferred.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling for mixed precision. If supplied, this argument supersedes ' +
                         '--static_loss_scale. Suggested to turn on for mixed precision')
parser.add_argument('--custom-model', default=False, help='enable custom model')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


if __name__ == '__main__':
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mixed_precision and not args.cuda:
        raise ValueError('If using mixed precision training, CUDA must be enabled!')
    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    loss_results, age_results, gender_results, accent_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs), torch.Tensor(args.epochs)
    best_avg_acc = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = model.labels 
        audio_conf = model.audio_conf
        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            def update_tensor(t):
                if t.shape[0] <= args.epochs:
                    print("updated tensor from size %d to %d" % (t.shape[0], args.epochs))
                    new_tensor = torch.Tensor(args.epochs)
                    for i in range(t.shape[0]):
                        new_tensor[i] = t[i]
                    return new_tensor
                return t
            loss_results   = update_tensor(package['loss_results'])
            age_results    = update_tensor(package['age_results'])
            gender_results = update_tensor(package['gender_results'])
            accent_results = update_tensor(package['accent_results'])
            
            if main_proc and args.visdom:  # Add previous scores to visdom graph
                visdom_logger.load_previous_values(start_epoch, package)
            if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values(start_epoch, package)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional,
                           mixed_precision=args.mixed_precision)

    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    if args.mixed_precision:
        model = convert_model_to_half(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)
    if args.distributed:
        model = DistributedDataParallel(model)
    if args.mixed_precision:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    celoss = CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
#         print(list(enumerate(train_loader, start=start_iter)))
        for i, (data) in tqdm(enumerate(train_loader, start=start_iter), total=len(train_loader)):
            #enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            targets = targets.view(-1, 26).long().to(device)
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            print(inputs.shape)
            print(input_sizes.shape)
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
   
            out = model(inputs, input_sizes)
#             out = out.transpose(0, 1)  # TxNxH
            
            float_out = out.float()  # ensure float32 for loss
            #first category
            loss = celoss(float_out[:, :8], targets[:, :8].max(1)[1]).to(device)
            loss += celoss(float_out[:, 8:10], targets[:, 8:10].max(1)[1]).to(device)
            loss += celoss(float_out[:, 10:], targets[:, 10:].max(1)[1]).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            if args.distributed:
                loss = loss.to(device)
                loss_value = reduce_tensor(loss, args.world_size).item()
            else:
                loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()
                # compute gradient
                if args.mixed_precision:
                    optimizer.backward(loss)
                    optimizer.clip_master_grads(args.max_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                age_results=age_results, gender_results=gender_results, 
                                                accent_results=accent_results, avg_loss=avg_loss),
                           file_path)
            del loss, out, float_out, inputs, targets
        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_val, ages_correct, genders_correct, accents_correct = 0., 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                inputs = inputs.to(device)

                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                out = model(inputs, input_sizes)
                
                # Assume target sizes are all equal
                target_size = target_sizes[0]
                targets = targets.reshape((-1, target_size)).to(device)
                
                # print(torch.cat((torch.arange(10).to(device), out[:,:8].argmax(1))))
                results = torch.zeros(targets.shape)
                
                # results[torch.arange(10), out[:,:8].argmax(1)] = 1
                # results[torch.arange(10), out[:,8:10].argmax(1) + 8] = 1
                # results[torch.arange(10), out[:,10:].argmax(1) + 10] = 1
                
                ages_correct    += torch.sum(out[:,:8].argmax(1) == targets[:,:8].argmax(1))
                genders_correct += torch.sum(out[:,8:10].argmax(1) == targets[:,8:10].argmax(1))
                accents_correct += torch.sum(out[:,10:].argmax(1) == targets[:,10:].argmax(1))
                total_val += targets.shape[0]
                
                del out, inputs, targets
            
            ages_correct = ages_correct.type(torch.FloatTensor)
            genders_correct = genders_correct.type(torch.FloatTensor)
            accents_correct = accents_correct.type(torch.FloatTensor)
            
            loss_results[epoch] = avg_loss
            age_results[epoch] = ages_correct / total_val
            gender_results[epoch] = genders_correct / total_val
            accent_results[epoch] = accents_correct / total_val
            
            avg_acc = (age_results[epoch] + gender_results[epoch] + accent_results[epoch])/3
            print('Validation Summary Epoch: [{0}]\t'
                  'Age {age:.3f}\t'
                  'Gender {gender:.3f}\t'
                  'Accent {accent:.3f}\t'
                  'Average {average:.3f}\t'
                  'Loss {loss:.3f}\t'.format(
                epoch + 1, age=ages_correct/total_val, gender=genders_correct/total_val, 
                      accent=accents_correct/total_val, average=avg_acc, loss=avg_loss))

        values = {
            'loss_results': loss_results,
            'age_results': age_results,
            'gender_results': gender_results,
            'accent_results': accent_results,
        }
        if args.visdom and main_proc:
            visdom_logger.update(epoch, values)
        if args.tensorboard and main_proc:
            tensorboard_logger.update(epoch, values, model.named_parameters())
            values = {
                'loss_results': loss_results,
                'age_results': age_results,
                'gender_results': gender_results,
                'accent_results': accent_results,
            }

        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            age_results=age_results, gender_results=gender_results, accent_results=accent_results),
                       file_path)
        # anneal lr
        param_groups = optimizer.optimizer.param_groups if args.mixed_precision else optimizer.param_groups
        for g in param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if main_proc and (best_avg_acc is None or best_avg_acc < avg_acc):
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            age_results=age_results, gender_results=gender_results, accent_results=accent_results)
                       , args.model_path)
            best_avg_acc = avg_acc

            avg_loss = 0
            if not args.no_shuffle:
                print("Shuffling batches...")
                train_sampler.shuffle(epoch)
