# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Construct the spec for a Transformer-based parser."""

import tensorflow as tf

from tensorflow.python.platform import gfile

from dragnn.protos import spec_pb2
from dragnn.python import spec_builder

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('spec_file', 'parser_spec.textproto',
                    'Filename to save the spec to.')
flags.DEFINE_string('embeddings_file', '',
                    'File containing pre-trained word embeddings')
flags.DEFINE_string('embeddings_vocab', '',
                    'File containing pre-trained word embedding vocab')


class BulkComponentSpecBuilder(spec_builder.ComponentSpecBuilder):
  def __init__(self, name, backend='SyntaxNetComponent'):
    builder = 'bulk_component.BulkFeatureExtractorComponentBuilder'
    super(BulkComponentSpecBuilder, self).__init__(name, builder, backend)


class BulkFeatureIdComponentSpecBuilder(spec_builder.ComponentSpecBuilder):
  def __init__(self, name, backend='SyntaxNetComponent'):
    builder = 'bulk_component.BulkFeatureIdExtractorComponentBuilder'
    super(BulkFeatureIdComponentSpecBuilder, self).__init__(name, builder, backend)


class BulkAnnotatorComponentSpecBuilder(spec_builder.ComponentSpecBuilder):
  def __init__(self, name, backend='SyntaxNetComponent'):
    builder = 'bulk_component.BulkAnnotatorComponentBuilder'
    super(BulkAnnotatorComponentSpecBuilder, self).__init__(name, builder, backend)


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def main(unused_argv):

  num_transformer_layers=4
  transformer_hidden_size=256
  num_heads=8
  head_size=64
  cnn_dim=1024
  num_cnn_layers=2
  heads_ff_size = 500
  deps_ff_size = 100
  transformer_total_dim = num_heads * head_size
  num_classes = 44

  # todo:
  # - set dropouts properly
  # - leaky relu
  # - batching
  # - lazyadam
  # - initilaize with glove embeddings
  # - add learned and fixed embeddings, concat with pos

  # Extract input features
  input_feats = BulkComponentSpecBuilder('input_feats')
  input_feats.set_network_unit('IdentityNetwork')
  input_feats.set_transition_system('shift-only')
  input_feats.add_fixed_feature(name='learned_embedding', embedding_dim=100, fml='input.token.word')
  input_feats.add_fixed_feature(name='pos_tag', embedding_dim=100, fml='input.tag')
  input_feats.add_fixed_feature(name='char_bigram', embedding_dim=16, fml='input.char-bigram')
  if FLAGS.embeddings_file != '':
    # todo assert that there is also a vocab file
    vocab_resource = spec_pb2.Resource()
    vocab_part = vocab_resource.part.add()
    vocab_part.file_pattern = FLAGS.embeddings_vocab
    vocab_part.file_format = 'text'

    embeddings_resource = spec_pb2.Resource()
    embedding_part = embeddings_resource.part.add()
    # todo pass this in
    embedding_part.file_pattern = FLAGS.embeddings_file

    input_feats.add_fixed_feature(name='fixed_embedding', embedding_dim=100,
                                  fml='input.token.known-word(outside=false)',
                                  pretrained_embedding_matrix=embeddings_resource,
                                  is_constant=True,
                                  vocab=vocab_resource,
                                  vocabulary_size=400000) # todo does this need tobe hard coded?
                                  # fml='input.token.known-word(outside=false)'),
                                  # pretrained_embedding_matrix=FLAGS.embeddings_file,
                                  # is_constant=True)

  lengths = BulkFeatureIdComponentSpecBuilder('lengths')
  lengths.set_network_unit('ExportFixedFeaturesNetwork')
  lengths.set_transition_system('once')
  lengths.add_fixed_feature(name='lengths', fml='sentence.length', embedding_dim=-1)

  # Embed tokens with CNN before passing representations to transformer
  convnet = BulkComponentSpecBuilder('convnet', backend='StatelessComponent')
  convnet.set_transition_system('shift-only')
  convnet.set_network_unit(name='ConvNetwork', depths='1024,1024', widths='3,3')
  convnet.add_link(source=input_feats, source_layer='input_embeddings', fml='input.focus')

  ff1 = BulkComponentSpecBuilder('ff1', backend='StatelessComponent')
  ff1.set_transition_system('shift-only')
  ff1.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes=str(transformer_total_dim), omit_logits='true')
  ff1.add_link(source=convnet, source_layer='conv_output', fml='input.focus')

  # Transformer layers
  transformer = BulkComponentSpecBuilder('transformer', backend='StatelessComponent')
  transformer.set_transition_system('shift-only')
  transformer.set_network_unit(name='transformer_units.TransformerEncoderNetwork',
                               num_layers=str(num_transformer_layers),
                               hidden_size=str(transformer_hidden_size),
                               num_heads=str(num_heads))
  transformer.add_link(source=ff1, source_layer='last_layer', name='features', fml='input.focus')
  transformer.add_link(source=lengths, source_layer='lengths', fml='input.focus')

  # ff heads representation
  heads_ff = BulkComponentSpecBuilder('heads_ff', backend='StatelessComponent')
  heads_ff.set_transition_system('shift-only')
  heads_ff.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes=str(heads_ff_size), omit_logits='true')
  heads_ff.add_link(source=transformer, source_layer='transformer_output')

  deps_ff = BulkComponentSpecBuilder('deps_ff', backend='StatelessComponent')
  deps_ff.set_transition_system('shift-only')
  deps_ff.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes=str(deps_ff_size), omit_logits='true')
  deps_ff.add_link(source=transformer, source_layer='transformer_output')

  bilinear = BulkComponentSpecBuilder('bilinear', backend='StatelessComponent')
  bilinear.set_transition_system('shift-only')
  bilinear.set_network_unit(name='transformer_units.PairwiseBilinearLabelNetwork', num_labels=str(num_classes))
  bilinear.add_link(source=heads_ff, name='sources', fml='input.focus')
  bilinear.add_link(source=deps_ff, name='targets', fml='input.focus')


  parser = BulkAnnotatorComponentSpecBuilder('parser')
  parser.set_network_unit(name='IdentityNetwork')
  parser.set_transition_system('heads_labels')
  parser.add_link(source=bilinear, source_layer='bilinear_scores', fml='input.focus')

  master_spec = spec_pb2.MasterSpec()
  master_spec.component.extend(
      [input_feats.spec, lengths.spec, convnet.spec, ff1.spec, transformer.spec, heads_ff.spec,
       deps_ff.spec, bilinear.spec, parser.spec])

  with gfile.FastGFile(FLAGS.spec_file, 'w') as f:
    f.write(str(master_spec).encode('utf-8'))

if __name__ == '__main__':
  tf.app.run()
