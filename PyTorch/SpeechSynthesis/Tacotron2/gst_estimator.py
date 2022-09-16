import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

from train import load_checkpoint
import models
import data_functions
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class StyleEmbeddingEstimator(nn.Module):
    """
    Estimator takes encoder outputs and returns style embeddings
    You can use some aggregation layers here: GRU + FC and some dropout for regularization
    Output dimension should be hparams.symbols_embedding_dim
    """

    def __init__(self, args):
        super(StyleEmbeddingEstimator, self).__init__()

        self.gru = nn.GRU(input_size=args.symbols_embedding_dim,
                          hidden_size=args.symbols_embedding_dim,
                          batch_first=True)

        self.fc = nn.Linear(args.symbols_embedding_dim, args.symbols_embedding_dim)
        self.dropout = nn.Dropout(args.p_gst_dropout)

    def forward(self, inputs, lengths=None):

        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        memory, emb = self.gru(inputs)
        out = self.fc(emb.squeeze(0))
        return self.dropout(out).unsqueeze(1)


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


def train_gst_estimator(parser):

    args, _ = parser.parse_known_args()

    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = args.rank
        world_size = args.world_size

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    parser = models.model_parser("Tacotron2", parser)
    args, _ = parser.parse_known_args()

    model_config = models.get_model_config("Tacotron2", args)
    tacotron = models.get_model("Tacotron2", model_config,
                                cpu_run=False,
                                uniform_initialize_bn_weight=not args.disable_uniform_initialize_bn_weight)

    tac_optimizer = torch.optim.Adam(tacotron.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = [0]
    model_config = load_checkpoint(tacotron, tac_optimizer, scaler, start_epoch,
                                   args.checkpoint_path, local_rank)
    tacotron.eval()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val"))

    gst_estimator = StyleEmbeddingEstimator(args).cuda()

    optimizer = torch.optim.Adam(gst_estimator.parameters(), lr=args.gst_estimator_lr)

    try:
        n_frames_per_step = args.n_frames_per_step
    except AttributeError:
        n_frames_per_step = None

    collate_fn = data_functions.get_collate_function("Tacotron2", n_frames_per_step)
    trainset = data_functions.get_data_loader("Tacotron2", args.dataset_path, args.training_files, args)
    valset = data_functions.get_data_loader("Tacotron2", args.dataset_path, args.validation_files, args)

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=None,
                              batch_size=args.batch_size, pin_memory=False,
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
            start = time.perf_counter()

            optimizer.zero_grad()
            x, y = tacotron.parse_batch(batch)

            with torch.no_grad():
                text, text_lengths, mel_tgt, gate_padded, mel_lengths, len_x, speakers_id = x
                gst_true, _ = tacotron.gst(mel_tgt, mel_lengths)
                embedded_inputs = tacotron.embedding(text).transpose(1, 2)
                enc_out = tacotron.encoder(embedded_inputs, text_lengths)

            gst_pred = gst_estimator(enc_out, text_lengths)
            loss = criterion(gst_pred, gst_true)
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
                    x, y = tacotron.parse_batch(batch)

                    """
                    The same, but for validation:
                    """

                    with torch.no_grad():
                        text, text_lengths, mel_tgt, gate_padded, mel_lengths, len_x, speakers_id = x
                        gst_true, _ = tacotron.gst(mel_tgt, mel_lengths)
                        embedded_inputs = tacotron.embedding(text).transpose(1, 2)
                        enc_out = tacotron.encoder(embedded_inputs, text_lengths)
                        gst_pred = gst_estimator(enc_out, text_lengths)
                        loss = criterion(gst_pred, gst_true)
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
    parser = argparse.ArgumentParser(description='GST Estimator Training')
    parser.add_argument('-l', '--logdir', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-t', '--tacotron_checkpoint_path', type=str, default=None,
                        required=False, help='tacotron_checkpoint path')
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

    train_gst_estimator(parser)

