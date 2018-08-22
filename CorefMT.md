Running coref-mt experiments
============================

Installation
------------

1. Clone the patched versions of _OpenNMT_ and _AllenNLP_ from Github:

   ```
   git clone -b coref-mt https://github.com/chardmeier/OpenNMT-py
   git clone -b coref-mt https://github.com/chardmeier/allennlp
   ```

2. Install all required prerequisites for the two libraries according to
   `OpenNMT-py/requirements.txt` and `allennlp/requirements.txt`.
  
3. Download required preprocessing and coreference models:

   ```
   wget https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz
   python -m spacy download en
   python -m spacy download fr
   ```

4. Add the `allennlp` directory to your `PYTHONPATH`.

Preprocessing
-------------

To train the system, you need a training and validation set. Each corpus 
should be stored in a set of three files with the same number of lines.
The first two files contain the sentence-aligned source and target text.
The third file contains a document identifier for each sentence in the corpus.
Documents define the scope of coreference resolution. Note that sentences
belonging to a document must be grouped together and ordered in the input
files. If the corpus files have a `.gz` extension, they are assumed to be
compressed.

Preprocessing of the input corpora consists of tokenisation (using _Spacy_),
coreference resolution (using _AllenNLP_) and conversion into the data structures
required by _OpenNMT_. It is done as follows:

```
python OpenNMT-py/preprocess_coref.py \
    -train training.en training.fr training.docids \
    -valid valid.en valid.fr valid.docids \
    -run_coref coref-model-2018.02.05.tar.gz \
    -save data_prefix
```

This will run for a while and generate a number of files whose names start
with `data_prefix`. The coreference resolver is run on the CPU. I haven't
tested running it on a GPU.

NMT Training
------------

The NMT training is launched with the standard _OpenNMT_ `train.py` script,
using an encoder type of `coref_transformer`. According to the [_OpenNMT_
FAQ](http://opennmt.net/OpenNMT-py/FAQ.html), the transformer model is
very sensitive to hyperparameters. The hyperparameter values below are taken
from there and should correspond to those used by Vaswani et al.

```
python OpenNMT-py/train.py -gpuid 0 \
        -data data_prefix -save_model model_path \
        -layers 6 -rnn_size 512 -word_vec_size 512 \
        -encoder_type coref_transformer -decoder_type transformer -position_encoding \
        -train_steps 100000 -max_generator_batches 32 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens \
        -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam \
        -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 \
        -param_init 0 -param_init_glorot -label_smoothing 0.1
```

Here, `data_prefix` is the pointer to where the preprocessed data files are
stored, and model_path is the prefix for saving the model.

TODO
----

- Fix GPU memory problems.
- Implement positional embeddings for members of coreference chains.