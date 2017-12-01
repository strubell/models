#!/bin/sh

set -e

name=$1

additional_args=${@:2}

# global defs
output_parent="trained"
embeddings_parent="data/embeddings"

mkdir -p $embeddings_parent

# PTB-specific defs
#name="ptb"
data_dir="$DATA_DIR/wsj-parse-3.5.0"
training_corpus="$data_dir/wsj02-21-trn.sdep.spos.gold.conllu"
dev_corpus="$data_dir/wsj22-dev.sdep.spos.gold.conllu"
embeddings="$DATA_DIR/embeddings/glove/glove.6B.100d.txt"
vocab="$embeddings_parent/${embeddings##*/}.vocab"
embeddings_tfrecord_proto="$embeddings_parent/${embeddings##*/}.tfrecord"


output_dir="$output_parent/$name"
morph_to_pos=True
batch_size=64

# To build, run:
# bazel build -c opt //dragnn/tools:trainer //dragnn/transformer:convet_embeddings_for_syntaxnet \
#   //dragnn/transformer:make_parser_spec

mkdir -p $output_dir

# if embeddings proto doesn't exist, make it
if ! [[ -e "$embeddings_tfrecord_proto" ]]; then
    bazel-bin/dragnn/transformer/convet_embeddings_for_syntaxnet \
      --proto_embeddings_file=$embeddings_tfrecord_proto \
      --text_embeddings_file=$embeddings \
      --vocab_file=$vocab
fi

mkdir -p $output_dir
bazel-bin/dragnn/transformer/make_parser_spec \
  --spec_file="$output_dir/parser_spec.textproto" \
  --embeddings_file=$embeddings_tfrecord_proto \
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
