# -*- coding: utf-8 -*-

import contextlib
import itertools
import torch
import random
import inspect


def make_shards(src_path, tgt_path, shard_size, docid_path=None):
    with contextlib.ExitStack() as stack:
        f_src = stack.enter_context(open(src_path, 'rb'))

        if tgt_path is not None:
            f_tgt = stack.enter_context(open(tgt_path, 'rb'))
        else:
            f_tgt = itertools.repeat(None)

        if docid_path is not None:
            f_docid = stack.enter_context(open(docid_path, 'r'))
        else:
            f_docid = itertools.repeat(None)

        if shard_size <= 0:
            yield zip(f_src.readlines(), f_tgt.readlines())
        else:
            src_shard = []
            tgt_shard = []
            docid_prefix = ''
            finish_doc = None
            for l_src, l_tgt, l_docid in zip(f_src, f_tgt, f_docid):
                if docid_path is not None:
                    docid_prefix = (l_docid.rstrip('\n') + '\t').encode('utf-8')

                if finish_doc is not None:
                    if docid_prefix == finish_doc:
                        src_shard.append(docid_prefix + l_src)
                        tgt_shard.append(l_tgt)
                        continue
                    else:
                        yield src_shard, tgt_shard
                        finish_doc = None
                        src_shard = []
                        tgt_shard = []

                if len(src_shard) < shard_size:
                    src_shard.append(docid_prefix + l_src)
                    tgt_shard.append(l_tgt)
                else:
                    if docid_path is not None:
                        finish_doc = docid_prefix
                    else:
                        yield src_shard, tgt_shard
                        src_shard = []
                        tgt_shard = []


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args
