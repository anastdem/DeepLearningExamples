import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
import copy

# from train import load_checkpoint
import models
from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from fastpitch.transformer import FFTransformer


class StyleEmbeddingEstimator(nn.Module):
    """
    Estimator takes encoder outputs and returns style embeddings
    You can use some aggregation layers here: GRU or FFT + FC and some dropout for regularization
    Output dimension should be hparams.symbols_embedding_dim
    """

    def __init__(self, model_config):
        super(StyleEmbeddingEstimator, self).__init__()

        self.fft = FFTransformer(
            n_layer=model_config['gst_n_layers'],
            n_head=model_config['gst_n_heads'],
            d_model=model_config['symbols_embedding_dim'],
            d_head=model_config['gst_d_head'],
            d_inner=4 * model_config['n_mel_channels'],
            kernel_size=model_config['gst_conv1d_kernel_size'],
            dropout=model_config['p_gst_dropout'],
            dropatt=model_config['p_gst_dropatt'],
            dropemb=model_config['p_gst_dropemb'],
            embed_input=False
        )

        self.fc = nn.Linear(model_config['symbols_embedding_dim'], model_config['symbols_embedding_dim'])
        self.dropout = nn.Dropout(model_config['p_gst_dropout'])

    def forward(self, x, lengths):
        emb, mask = self.fft(x, lengths)
        emb = emb.mean(dim=1)
        out = self.fc(emb)
        return self.dropout(out)


def parse_args(parser):
    parser.add_argument('-l', '--logdir', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-t', '--fastpitch_checkpoint_path', type=str, default=None,
                        required=False, help='fastpitch_checkpoint path')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--gst_estimator_epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--gst_estimator_lr', default=1e-3, type=float,
                        help='Learing rate')
    parser.add_argument('--gst_estimator_eval_interval', default=100, type=int,
                        help='Number of evaluation interval')
    parser.add_argument('--cudnn-enabled', action='store_true',
                        help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Run cudnn benchmark')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed for PyTorch random number generators')
    parser.add_argument('--cuda', action='store_true',
                        help='Run on GPU using CUDA')
    parser.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size per GPU')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--training-files', type=str, nargs='*', required=True,
                      help='Paths to training filelists.')
    data.add_argument('--validation-files', type=str, nargs='*',
                      required=True, help='Paths to validation filelists')
    data.add_argument('--text-cleaners', nargs='*',
                      default=['basic_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='russian_basic',
                      help='Define symbol set for input text')
    data.add_argument('--load-mel-from-disk', action='store_true',
                      help='Use mel-spectrograms cache on the disk')
    data.add_argument('--load-pitch-from-disk', action='store_true',
                      help='Use pitch cached on disk with prepare_dataset.py')

    return parser

def load_gst_estimator_checkpoint(predictor, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    predictor.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint['step'], checkpoint['min_eval_loss']


def save_gst_estimator_checkpoint(filepath, step, min_eval_loss, model, optimizer):
    checkpoint = {
        "step": step,
        'min_eval_loss': min_eval_loss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)


def train_gst_estimator():
    parser = argparse.ArgumentParser(description='PyTorch GST Estimator Training', allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    device = torch.device('cuda' if args.cuda else 'cpu')
    model, model_config, model_train_setup = models.load_and_setup_model('FastPitch', parser,
                                                                         args.fastpitch_checkpoint_path, False, device)

    wandb.init(project="FastPitch GST Estimator", sync_tensorboard=True, config=args)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val"))

    gst_estimator = StyleEmbeddingEstimator(model_config).to(device)

    optimizer = torch.optim.Adam(gst_estimator.parameters(), lr=args.gst_estimator_lr)

    collate_fn = TTSCollate()
    trainset = TTSDataset(audiopaths_and_text=args.training_files, n_mel_channels=model_config['n_mel_channels'], **vars(args))
    valset = TTSDataset(audiopaths_and_text=args.validation_files, n_mel_channels=model_config['n_mel_channels'], **vars(args))

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=None, batch_size=args.batch_size,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True, collate_fn=collate_fn)

    eval_loader = DataLoader(valset, sampler=None, num_workers=1,
                             shuffle=False, batch_size=args.batch_size,
                             pin_memory=False, collate_fn=collate_fn)

    criterion = nn.L1Loss()

    step = 0
    min_eval_loss = np.inf

    checkpoint_path = os.path.join(args.logdir, f"GST_estimator_best_checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        print("Resume training from checkpoint: ", checkpoint_path)
        step, min_eval_loss = load_gst_estimator_checkpoint(gst_estimator, optimizer, checkpoint_path)

    losses = []
    gst_estimator.train()

    for epoch in range(args.gst_estimator_epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()
            x, y, num_frames = batch_to_gpu(batch)

            with torch.no_grad():
                (text, text_lengths, mel_tgt, mel_lens, pitch_dense, energy_dense,
                speaker, attn_prior, audiopaths) = x
                gst_true, _ = model.gst(mel_tgt.transpose(1, 2), mel_lens)
                enc_out, _ = model.encoder(text, text_lengths)

            gst_pred = gst_estimator(enc_out, text_lengths)
            loss = criterion(gst_pred, gst_true.mean(dim=1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.gst_estimator_eval_interval == 0:
                train_writer.add_scalar('loss', np.mean(losses), step)
                print(f"train: {step:<3d} loss: {np.mean(losses):<5.4f}")

                losses = []
                gst_estimator.eval()
                for batch in eval_loader:
                    x, y, num_frames = batch_to_gpu(batch)

                    """
                    The same, but for validation:
                    """

                    with torch.no_grad():
                        (text, text_lengths, mel_tgt, mel_lens, pitch_dense, energy_dense,
                         speaker, attn_prior, audiopaths) = x
                        gst_true, _ = model.gst(mel_tgt.transpose(1, 2), mel_lens)
                        enc_out, _ = model.encoder(text, text_lengths)
                        gst_pred = gst_estimator(enc_out, text_lengths)
                        loss = criterion(gst_pred, gst_true.mean(dim=1))
                        losses.append(loss.item())

                val_writer.add_scalar('loss', np.mean(losses), step)
                print(f"val: {step:<3d} loss: {np.mean(losses):<5.4f}")

                """
                Fallback to the prev model if the new one is not better:
                """
                if np.mean(losses) < min_eval_loss:
                    min_eval_loss = np.mean(losses)
                    checkpoint_path = os.path.join(args.logdir, f"GST_estimator_best_checkpoint.pt")
                    save_gst_estimator_checkpoint(checkpoint_path, step, min_eval_loss, gst_estimator, optimizer)

                for w in train_writer.all_writers.values():
                    w.flush()
                for w in val_writer.all_writers.values():
                    w.flush()

                losses = []
                gst_estimator.train()


if __name__ == '__main__':
    train_gst_estimator()
