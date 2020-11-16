"""Trainining script for VQVAE based on Mel. Quantizer + WaveLSTM

usage: train.py [options]

options:
    --conf=<json>             Path of configuration file (json).
    --gpu-id=<N>               ID of the GPU to use [default: 0]
    --exp-dir=<dir>           Experiment directory
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<dir>    Log Path [default: exp/log_tacotronOne]
    -h, --help                Show this help message and exit
"""
import os, sys
from docopt import docopt
args = docopt(__doc__)
print("Command line args:\n", args)
gpu_id = args['--gpu-id']
print("Using GPU ", gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id


from collections import defaultdict

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
##############################################
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from util import *
from model_vqvae import SILA
from model import *
from judith.experiment_tracking import RemoteTracker
from utils.plot import save_alignment
import json

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from os.path import join, expanduser

import tensorboard_logger
from tensorboard_logger import *
from hyperparameters import hyperparameters

vox_dir ='vox'

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None
finetune = None

hparams = hyperparameters()

fs = hparams.sample_rate
use_assistant = 1
use_reconstruction = 0
use_arff = 1

# Utility to return predictions
def return_classes(logits, dim=-1):
   _, predicted = torch.max(logits,dim)    
   return predicted.view(-1).cpu().numpy()



def get_metrics(predicteds, targets):
   print(confusion_matrix(targets, predicteds))
   print(classification_report(targets, predicteds))
   accuracy =  accuracy_score(targets, predicteds)
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   print("EER is ", EER)
   return recall_score(targets,predicteds,average='macro'), accuracy


def validate(model, val_loader, assistant=None):

 
    model.eval() 
    criterion = nn.CrossEntropyLoss() 
    y_true = [] 
    y_pred = [] 
    running_loss = 0.
    running_entropy = 0.
    for step, inputs in tqdm(enumerate(val_loader)): 
     with torch.no_grad(): 
        mels = Variable(inputs['mels']).cuda()
        lid = Variable(inputs['lid']).cuda()
        embedding = Variable(inputs['embedding']).cuda()
        if use_arff:
           lid_logits, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses = model.forward_getlid(mels, embedding) 
        else:
           lid_logits, entropy = model.forward_getlid(mels, embedding) 
        loss = criterion(lid_logits.contiguous().view(-1, 2), lid.contiguous().view(-1)) 
        targets = lid.cpu().view(-1).numpy() 
        y_true += targets.tolist() 
        predictions = return_classes(lid_logits) 
        y_pred += predictions.tolist() 

        running_loss += loss.item()
 
    ff = open(exp_dir + '/eval_epoch_' + str(global_epoch).zfill(4) ,'a') 
    for step_, (yp, yt) in enumerate(list(zip(y_pred, y_true))): 
      if yp == yt: 
        continue 
      ff.write( str(step_) + ' ' + str(yp) + ' ' + str(yt) + '\n') 
    ff.close() 
     
 
    recall,accuracy = get_metrics(y_pred, y_true) 
    if assistant:
         assistant.log_scalar("Validation Recall", recall)
         assistant.log_scalar("Validation Accuracy", accuracy)
         assistant.log_scalar("Validation LID loss", running_loss/step)
         if use_arff == 0:
            assistant.log_scalar("Validation Entropy", running_entropy/step)
         else:
            assistant.log_scalar("Validation 2 Class Entropy", entropy_2classes)
            assistant.log_scalar("Validation 3 Class Entropy", entropy_3classes)
            assistant.log_scalar("Validation 4 Class Entropy", entropy_4classes)
            assistant.log_scalar("Validation N Class Entropy", entropy_nclasses)


    return


def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0,  assistant=None):
    validate(model.eval(), val_loader, assistant)
    model.train()
    if use_cuda:
        model = model.cuda()

    criterion = nn.L1Loss()
    criterion_lid = nn.CrossEntropyLoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        model.train()
        h = open(logfile_name, 'a')
        running_loss = 0.
        running_loss_reconstruction = 0.
        running_loss_vq = 0.
        running_loss_encoder = 0.
        running_entropy = 0.
        running_loss_linear = 0.
        mel_loss = 0
        linear_loss = 0

        for step, inputs in tqdm(enumerate(train_loader)):

            max_learning_rate = 0.

            #if global_step == 6000:
            #   optimizer = optimizer_sgd 

            # learning rate
            if global_step > 12000:
               current_lr = learning_rate_decay(init_lr, global_step)
               max_learning_rate = current_lr
            else:
               for param_group in optimizer.param_groups:
                param_lr = param_group['lr'] 
                if param_lr > max_learning_rate:
                    max_learning_rate = param_lr  
                #param_group['lr'] = current_lr

            optimizer.zero_grad()

            mels = Variable(inputs['mels']).cuda()
            linears = Variable(inputs['linears']).cuda()
            embedding = Variable(inputs['embedding']).cuda()
            lid = Variable(inputs['lid']).cuda()

            assert mels.shape[1] == linears.shape[1]

            if use_reconstruction:
               mel_outputs, linear_outputs, attn, lid_logits, vq_penalty, encoder_penalty, entropy = model(mels, embedding)
            else:
               if use_arff:
                 lid_logits, vq_penalty, encoder_penalty, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses = model.forward_noreconstruction(mels, embedding)
               else:
                 lid_logits, vq_penalty, encoder_penalty, entropy = model.forward_noreconstruction(mels, embedding)
            #print("The shape of lid logits and targets: ", lid_logits.shape, lid.shape)

            # Loss
            if use_reconstruction:
              linear_dim = 1025
              mel_loss = criterion(mel_outputs, mels)
              n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
              linear_loss = 0.5 * criterion(linear_outputs, linears) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  linears[:, :, :n_priority_freq])
              reconstruction_loss = mel_loss + linear_loss
            else:
              reconstruction_loss = 0
 
            reconstruction_weight = 0.5
            lid_loss = criterion_lid(lid_logits.contiguous().view(-1,2), lid.view(-1))
            encoder_weight = 0.25 * min(1, max(0.1, global_step / 1000 - 1)) # https://github.com/mkotha/WaveRNN/blob/74b839b57a7e128b3f8f0b4eb224156c1e5e175d/models/vqvae.py#L209
            loss = reconstruction_loss * reconstruction_weight + vq_penalty + encoder_penalty * encoder_weight + lid_loss


            if global_step > 0 and global_step % hparams.save_states_interval == 0:
                if use_reconstruction:
                  alignment = attn[0].cpu().data.numpy()
                  path = checkpoint_dir + '/alignment_step' + str(global_step).zfill(7) + '.png'
                  save_alignment( path, alignment, global_step,  assistant)
                if use_assistant:
                   model.quantizer.plot_histogram = 1

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)


            # Update
            loss.backward(retain_graph=False)
            #grad_norm = torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), clip_thresh)
            #grad_norm_name = 'None'
            grad_norm, grad_norm_name = clip_gradients_custom(model, clip_thresh)
            optimizer.step()
            model.quantizer.after_update()

            # Logs
            #log_value("reconstruction loss", float(reconstruction_loss.item()), global_step)
            #log_value("loss", float(loss.item()), global_step)
            #log_value("gradient norm", grad_norm, global_step)
            #log_value("VQ Penalty", vq_penalty, global_step)
            #log_value("Encoder Penalty", encoder_penalty, global_step)
            #log_value("Entropy", entropy, global_step)
            #log_value("learning rate", max_learning_rate, global_step)

            if use_assistant and assistant is not None:
              assistant.log_scalar("loss", loss.item())
              assistant.log_scalar("VQ Penalty", vq_penalty)
              assistant.log_scalar("Gradient Norm", float(grad_norm))
              assistant.log_scalar("Learning Rate", float(max_learning_rate))
              assistant.log_scalar("Encoder Penalty", encoder_penalty)
              assistant.log_scalar("Encoder Weight", encoder_weight)
              assistant.log_scalar("Input length", mels.shape[1])
              assistant.log_text("Max Grad Norm", grad_norm_name)
              if use_reconstruction:
                assistant.log_scalar("Mel Loss", mel_loss.item())
                assistant.log_scalar("Linear Loss", linear_loss.item())
              assistant.log_scalar("LID Loss", lid_loss.item())
              if use_arff:
                 assistant.log_scalar("2 Class Entropy", entropy_2classes)
                 assistant.log_scalar("3 Class Entropy", entropy_3classes)
                 assistant.log_scalar("4 Class Entropy", entropy_4classes)
                 assistant.log_scalar("N Class Entropy", entropy_nclasses)

              else:
                 assistant.log_scalar("Entropy", entropy)



            global_step += 1
            running_loss += loss.item()
            #running_loss_reconstruction += reconstruction_loss.item()
            running_loss_vq += vq_penalty.item()
            running_loss_encoder += encoder_penalty.item()
            #running_entropy += entropy

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) 
                + '\n')
        h.close() 

        global_epoch += 1
        validate(model, val_loader, assistant) 

if __name__ == "__main__":

    exp_dir = args["--exp-dir"]
    checkpoint_dir = args["--exp-dir"] + '/checkpoints'
    checkpoint_path = args["--checkpoint-path"]
    log_path = args["--exp-dir"] + '/tracking'
    conf = args["--conf"]

    # Override hyper parameters
    if conf is not None:
        with open(conf) as f:
            hparams.update_params(f)

    print(hparams)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logfile_name = log_path + '/logfile'
    h = open(logfile_name, 'w')
    h.close()

    # Vocab size


    feats_name = 'mspec'
    Mel_train = float_datasource(vox_dir + '/' + 'fnames.train.dur', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Mel_val = float_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    feats_name = 'lspec'
    linear_train = float_datasource(vox_dir + '/' + 'fnames.train.dur', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    linear_val = float_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    feats_name = 'utteranceembeddings'
    embed_train = float_datasource(vox_dir + '/' + 'fnames.train.dur', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    embed_val = float_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    feats_name = 'lid'
    lid_train = categorical_datasource(vox_dir + '/' + 'fnames.train.dur', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    lid_val = categorical_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    # Dataset and Dataloader setup
    trainset = SILADataset(linear_train, Mel_train, embed_train, lid_train, hparams.outputs_per_step)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=4, shuffle=True,
        collate_fn=collate_fn_sila, pin_memory=hparams.pin_memory)

    ## Ok champion, tell me where you are using this  
    valset = SILADataset(linear_val, Mel_val, embed_val, lid_val, hparams.outputs_per_step)
    val_loader = data_utils.DataLoader(
        valset, batch_size=1,
        num_workers=hparams.num_workers, shuffle=False,
        collate_fn=collate_fn_sila, pin_memory=hparams.pin_memory)

    exp_name = os.path.basename(exp_dir)
    if use_assistant:
      assistant = RemoteTracker(exp_name, projects_dir='SILA', 
                                upload_source_files=[os.getcwd() + '/' + 'conf/vocoder.conf', 
                                                     os.getcwd() + '/local/model.py',
                                                     os.getcwd() + '/local/model_vqvae.py',
                                                     os.getcwd() + '/local/train_sila.py',
                                                     os.getcwd() + '/local/util.py'])
    else:
      assistant = None


    # Model
    model = SILA(embedding_dim=256,
                     input_dim=hparams.num_mels,
                     mel_dim = hparams.num_mels,
                     assistant = assistant,
                     r = hparams.outputs_per_step,
                     use_arff = use_arff
                     )
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)
    #optimizer = optim.SGD(model.parameters(), 
    #                      lr=hparams.initial_learning_rate,
    #                      momentum = 0.9)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = int(checkpoint["global_step"])
            global_epoch = int(checkpoint["global_epoch"])
        except:
            print("Houston! We have got problems")
            sys.exit()


    if finetune:
       assert os.path.exists(pretrained_checkpoint_path)
       model8 = WaveLSTM8(n_vocab=257,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     logits_dim=60,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
       model8 = model8.cuda()

       checkpoint = torch.load(pretrained_checkpoint_path)
       model8.load_state_dict(checkpoint["state_dict"])
       model.upsample_network = model8.upsample_network
       #model.joint_encoder = model8.joint_encoder
       #model.hidden2linear = model8.hidden2linear
       #model.linear2logits = model8.linear2logits1

    # Setup tensorboard logger
    tensorboard_logger.configure(log_path)
    exp_name = os.path.basename(exp_dir)

    # Train!
    try:
        train(model, train_loader, val_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh,
              assistant=assistant)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)
 
    print("Finished")
    sys.exit(0)
 
 



