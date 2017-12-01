#!/bin/sh

set -e

name=$1

additional_args=${@:2}

# PTB-specific defs
#name="ptb"
data_dir="$DATA_DIR/wsj-parse-3.5.0"
training_corpus="$data_dir/wsj02-21-trn.sdep.spos.gold.conllu"
dev_corpus="$data_dir/wsj22-dev.sdep.spos.gold.conllu"
embeddings="$DATA_DIR/embeddings/glove/glove.6B.100d.floats.txt"
vocab="$DATA_DIR/embeddings/glove/glove.6B.100d.vocab.txt"

output_parent="trained"
output_dir="$output_parent/$name"
morph_to_pos=True
batch_size=64

# bazel build -c opt //dragnn/tools:trainer //dragnn/conll2017:make_parser_spec

mkdir -p $output_dir
bazel-bin/dragnn/transformer/make_parser_spec \
  --spec_file="$output_dir/parser_spec.textproto" \
  --embeddings_file=$embeddings \
  --embeddings_vocab=$vocab

bazel-bin/dragnn/tools/trainer \
  --logtostderr \
  --compute_lexicon \
  --dragnn_spec="$output_dir/parser_spec.textproto" \
  --resource_path="$output_dir/resources" \
  --training_corpus_path="$training_corpus" \
  --tune_corpus_path="$dev_corpus" \
  --tensorboard_dir="$output_dir/tensorboard" \
  --checkpoint_filename="$output_dir/checkpoint.model" \
  --morph_to_pos=$morph_to_pos \
  --batch_size=$batch_size \
  $additional_args
