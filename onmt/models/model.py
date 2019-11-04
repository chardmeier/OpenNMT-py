""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.alignment_weights = nn.Parameter(torch.zeros(2 * encoder.d_model, 2 * encoder.d_model))

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            TODO: Update return value docs
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns, emb = self.decoder(tgt, memory_bank,
                                           memory_lengths=lengths)

        src_vec = torch.cat([enc_state, memory_bank], dim=-1).transpose(0, 1)
        # do we need to shift the target embeddings?
        tgt_vec = torch.cat([emb, dec_out], dim=-1).permute(1, 2, 0)
        alignment = torch.nn.functional.softmax(src_vec @ self.alignment_weights @ tgt_vec, dim=-1)

        return {
            'dec_out': dec_out,
            'attns': attns,
            'src_emb': enc_state,
            'enc_out': memory_bank,
            'lengths': lengths,
            'alignment': alignment.permute(2, 0, 1)
        }
