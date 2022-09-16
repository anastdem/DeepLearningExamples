import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

from train import load_model, load_checkpoint, prepare_dataloaders
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from hparams import create_hparams

class StyleEmbeddingEstimator(nn.Module):
    """
    Estimator takes encoder outputs and returns style embeddings
    You can use some aggregation layers here: GRU + FC and some dropout for regularization
    Output dimension should be hparams.symbols_embedding_dim
    """

    def __init__(self, hparams):
        super(StyleEmbeddingEstimator, self).__init__()

        self.gru = nn.GRU(input_size=hparams.symbols_embedding_dim,
                          hidden_size=hparams.symbols_embedding_dim,
                          batch_first=True)

        self.fc = nn.Linear(hparams.symbols_embedding_dim, hparams.symbols_embedding_dim)
        self.dropout = nn.Dropout(hparams.p_gst_dropout)

    def forward(self, inputs, lengths=None):

        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # inputs = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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


def train_gst_estimator(tacotron_checkpoint_path, logdir, hparams):

    tacotron = load_model(hparams)
    learning_rate = hparams.learning_rate
    tac_optimizer = torch.optim.Adam(tacotron.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)
    tacotron, _, _, _ = load_checkpoint(tacotron_checkpoint_path, tacotron, tac_optimizer)
    tacotron.eval()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    train_writer = SummaryWriter(os.path.join(logdir, "train"))
    val_writer = SummaryWriter(os.path.join(logdir, "val"))

    gst_estimator = StyleEmbeddingEstimator(hparams).cuda()

    optimizer = torch.optim.Adam(gst_estimator.parameters(), lr=hparams.gst_estimator_lr)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    eval_loader = DataLoader(valset, sampler=None, num_workers=1,
                            shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=False, collate_fn=collate_fn)
    criterion = nn.L1Loss()

    step = 0
    min_eval_loss = np.inf

    checkpoint_path = os.path.join(logdir, f"GST_estimator_best_checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        print("Resume training from checkpoint: ", checkpoint_path)
        step, min_eval_loss = load_gst_estimator_checkpoint(gst_estimator, optimizer, checkpoint_path)

    losses = []
    gst_estimator.train()

    for epoch in range(hparams.gst_estimator_epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            optimizer.zero_grad()
            x, y = tacotron.parse_batch(batch)

            with torch.no_grad():
                text, text_lengths, mel_tgt, max_len, mel_lengths = x
                gst_true, _ = tacotron.gst(mel_tgt, mel_lengths)
                embedded_inputs = tacotron.embedding(text).transpose(1, 2)
                enc_out = tacotron.encoder(embedded_inputs, text_lengths)

            gst_pred = gst_estimator(enc_out, text_lengths)

            loss = criterion(gst_pred, gst_true)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step += 1

            if step % hparams.gst_estimator_eval_interval == 0:
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
                        text, text_lengths, mel_tgt, max_len, mel_lengths = x
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
                    checkpoint_path = os.path.join(logdir, f"GST_estimator_best_checkpoint.pt")
                    save_gst_estimator_checkpoint(checkpoint_path, step, min_eval_loss, gst_estimator, optimizer)

                for w in train_writer.all_writers.values():
                    w.flush()
                for w in val_writer.all_writers.values():
                    w.flush()


                losses = []
                gst_estimator.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-t', '--tacotron_checkpoint_path', type=str, default=None,
                        required=False, help='tacotron_checkpoint path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    train_gst_estimator(args.tacotron_checkpoint_path, args.log_directory, hparams)

