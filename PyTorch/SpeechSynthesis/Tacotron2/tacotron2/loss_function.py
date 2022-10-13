# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import lpips
from torch import nn
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)+'/../'))
from tacotron2_common.utils import to_gpu, get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, gate_positive_weight, use_guided_attention_loss, loss_attention_weight, diagonal_factor, use_lpips_loss):
        super(Tacotron2Loss, self).__init__()
        self.gate_positive_weight = gate_positive_weight

        self.use_guided_attention_loss = use_guided_attention_loss
        self.loss_attention_weight = loss_attention_weight
        self.diagonal_factor = diagonal_factor

        self.use_lpips_loss = use_lpips_loss

        if self.use_lpips_loss:
            self.lpips_loss = lpips.LPIPS(net='alex')  # best forward scores
            # self.lpips_loss = lpips.LPIPS(net='vgg').cuda()


    # @staticmethod
    def batch_diagonal_guide(self, input_lengths, output_lengths, g=0.2):
        dtype, device = torch.float32, input_lengths.device

        grid_text = torch.arange(input_lengths.max(), dtype=dtype, device=device)
        grid_text = grid_text.view(1, -1) / input_lengths.view(-1, 1)  # (B, T)

        grid_mel = torch.arange(output_lengths.max(), dtype=dtype, device=device)
        grid_mel = grid_mel.view(1, -1) / output_lengths.view(-1, 1)  # (B, M)

        grid = grid_text.unsqueeze(1) - grid_mel.unsqueeze(2)  # (B, M, T)

        # apply text and mel length masks
        grid.transpose(2, 1)[get_mask_from_lengths(input_lengths)] = 0.
        grid[get_mask_from_lengths(output_lengths)] = 0.

        W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
        return W

    def forward(self, model_output, targets, model_inputs):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        inputs, input_lengths, targets, max_len, output_lengths, speaker_id = model_inputs
        attention_loss = 0
        if self.use_guided_attention_loss:
            diagonal_guides = self.batch_diagonal_guide(input_lengths, output_lengths, g=self.diagonal_factor)
            attention_loss = torch.sum(alignments * diagonal_guides)
            active_elements = torch.sum(input_lengths * output_lengths)
            attention_loss = attention_loss / active_elements

        lpips_loss = 0
        if self.use_lpips_loss:
            lpips_loss = torch.tensor([
                self.lpips_loss(mel_out_postnet[i, :, :output_lengths[i]], mel_target[i, :, :output_lengths[i]])
                for i in range(mel_out_postnet.shape[0])
            ]).mean()

            lpips_loss += torch.tensor([
                self.lpips_loss(mel_out[i, :, :output_lengths[i]], mel_target[i, :, :output_lengths[i]])
                for i in range(mel_out_postnet.shape[0])
            ]).mean()

        return (mel_loss
                + self.gate_positive_weight * gate_loss
                + self.loss_attention_weight * attention_loss
                + lpips_loss
                ), {"mel_loss": mel_loss,
                    "gate_loss": self.gate_positive_weight * gate_loss,
                    "attention_loss": self.loss_attention_weight * attention_loss,
                    "lpips_loss": lpips_loss}
