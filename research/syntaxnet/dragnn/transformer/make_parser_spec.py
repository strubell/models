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

class BulkComponentSpecBuilder(spec_builder.ComponentSpecBuilder):
    def __init__(self,
                 name,
                 builder='bulk_component.BulkFeatureExtractorComponentBuilder',
                 backend='SyntaxNetComponent'):
        """Initializes the ComponentSpec with some defaults for SyntaxNet.

        Args:
          name: The name of this Component in the pipeline.
          builder: The component builder type.
          backend: The component backend type.
        """
        self.spec = spec_pb2.ComponentSpec(
            name=name,
            backend=self.make_module(backend),
            component_builder=self.make_module(builder))


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

  lengths = BulkComponentSpecBuilder('input_feats')
  lengths.set_network_unit('IdentityNetwork')
  lengths.set_transition_system('shift-only')
  lengths.add_fixed_feature(name='lengths', fml='input.lengths')

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
  transformer.set_network_unit(name='transformer_units.TransformerEncoderNetwork', num_layers=str(num_transformer_layers),
                               hidden_size=str(transformer_hidden_size), num_heads=str(num_heads))
  transformer.add_link(source=ff1, source_layer='last_layer', fml='input.focus')
  transformer.add_link(source=lengths, source_layer='lengths', fml='input.focus')


  heads_ff = BulkComponentSpecBuilder('heads_ff', backend='StatelessComponent')
  heads_ff.set_transition_system('shift-only')
  heads_ff.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes=str(heads_ff_size))
  heads_ff.add_link(source=transformer, source_layer='transformer_output')

  deps_ff = BulkComponentSpecBuilder('deps_ff', backend='StatelessComponent')
  deps_ff.set_transition_system('shift-only')
  deps_ff.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes=str(deps_ff_size))
  deps_ff.add_link(source=transformer, source_layer='transformer_output')

  bilinear = BulkComponentSpecBuilder('bilinear', backend='StatelessComponent')
  bilinear.set_transition_system('shift-only')
  bilinear.set_network_unit(name='transformer_units.PairwiseBilinearLabelNetwork', num_labels=str(num_classes))
  bilinear.add_link(source=heads_ff, name='sources', fml='input.focus')
  bilinear.add_link(source=deps_ff, name='targets', fml='input.focus')


  parser = BulkComponentSpecBuilder('parser')
  parser.set_network_unit('IdentityNetwork')
  parser.set_transition_system('heads_labels')
  parser.add_link(source=bilinear, source_layer='bilinear_scores', fml='input.focus')

  master_spec = spec_pb2.MasterSpec()
  master_spec.component.extend(
      [input_feats.spec, lengths.spec, convnet.spec, ff1.spec, transformer.spec, heads_ff.spec, deps_ff.spec, bilinear.spec, parser.spec])


  # token_embeddings = BulkComponentSpecBuilder('token_embeddings', backend='StatelessComponent')
  # token_embeddings.set_transition_system('shift-only')
  # token_embeddings.set_network_unit('IdentityNetwork')
  # token_embeddings.add_link(source=input_feats, source_layer='input_embeddings', fml='input.focus', embedding_dim='-1', size='1')
  # token_embeddings.add_link(source=tagger_convnet, source_layer='conv_output', fml='input.focus', embedding_dim='-1', size='1')

  # parser_convnet = BulkComponentSpecBuilder('tagger_convnet', backend='StatelessComponent')
  # parser_convnet.set_transition_system('shift-only')
  # parser_convnet.set_network_unit(name='ConvNetwork', depths='128,128,128,128',
  #                                 output_embedding_dim='0', widths='3,3,3,3')


  # # Left-to-right, character-based LSTM.
  # char2word = spec_builder.ComponentSpecBuilder('char_lstm')
  # char2word.set_network_unit(
  #     name='wrapped_units.LayerNormBasicLSTMNetwork',
  #     hidden_layer_sizes='256')
  # char2word.set_transition_system(name='char-shift-only', left_to_right='true')
  # char2word.add_fixed_feature(name='chars', fml='char-input.text-char',
  #                             embedding_dim=16)
  #
  # # Lookahead LSTM reads right-to-left to represent the rightmost context of the
  # # words. It gets word embeddings from the char model.
  # lookahead = spec_builder.ComponentSpecBuilder('lookahead')
  # lookahead.set_network_unit(
  #     name='wrapped_units.LayerNormBasicLSTMNetwork',
  #     hidden_layer_sizes='256')
  # lookahead.set_transition_system(name='shift-only', left_to_right='false')
  # lookahead.add_link(source=char2word, fml='input.last-char-focus',
  #                    embedding_dim=64)
  #
  # # Construct the tagger. This is a simple left-to-right LSTM sequence tagger.
  # tagger = spec_builder.ComponentSpecBuilder('tagger')
  # tagger.set_network_unit(
  #     name='wrapped_units.LayerNormBasicLSTMNetwork',
  #     hidden_layer_sizes='256')
  # tagger.set_transition_system(name='tagger')
  # tagger.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)
  #
  # # Construct the parser.
  # parser = spec_builder.ComponentSpecBuilder('parser')
  # parser.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256',
  #                         layer_norm_hidden='true')
  # parser.set_transition_system(name='arc-standard')
  # parser.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)
  # parser.add_token_link(
  #     source=tagger, fml='input.focus stack.focus stack(1).focus',
  #     embedding_dim=64)
  #
  # # Add discrete features of the predicted parse tree so far, like in Parsey
  # # McParseface.
  # parser.add_fixed_feature(name='labels', embedding_dim=16,
  #                          fml=' '.join([
  #                              'stack.child(1).label',
  #                              'stack.child(1).sibling(-1).label',
  #                              'stack.child(-1).label',
  #                              'stack.child(-1).sibling(1).label',
  #                              'stack(1).child(1).label',
  #                              'stack(1).child(1).sibling(-1).label',
  #                              'stack(1).child(-1).label',
  #                              'stack(1).child(-1).sibling(1).label',
  #                              'stack.child(2).label',
  #                              'stack.child(-2).label',
  #                              'stack(1).child(2).label',
  #                              'stack(1).child(-2).label']))
  #
  # # Recurrent connection for the arc-standard parser. For both tokens on the
  # # stack, we connect to the last time step to either SHIFT or REDUCE that
  # # token. This allows the parser to build up compositional representations of
  # # phrases.
  # parser.add_link(
  #     source=parser,  # recurrent connection
  #     name='rnn-stack',  # unique identifier
  #     fml='stack.focus stack(1).focus',  # look for both stack tokens
  #     source_translator='shift-reduce-step',  # maps token indices -> step
  #     embedding_dim=64)  # project down to 64 dims
  #
  # master_spec = spec_pb2.MasterSpec()
  # master_spec.component.extend(
  #     [char2word.spec, lookahead.spec, tagger.spec, parser.spec])

  with gfile.FastGFile(FLAGS.spec_file, 'w') as f:
    f.write(str(master_spec).encode('utf-8'))

if __name__ == '__main__':
  tf.app.run()
