# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Convert plaintext pre-trained embeddings to the format expected by syntaxnet."""

from __future__ import print_function
import tensorflow as tf

from syntaxnet import dictionary_pb2

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('text_embeddings_file', '',
                    'File containing pre-trained word embeddings')
flags.DEFINE_string('proto_embeddings_file', '',
                    'File to write converted embeddings to')
flags.DEFINE_string('vocab_file', '',
                    'File to write vocab to. No vocab written if this is the empty string.')

def main(unused_argv):

  write_vocab_to_file = FLAGS.vocab_file != ''
  vocab = []

  max_to_read = 5

  tf.logging.info("Loading pretrained embeddings from: %s" % FLAGS.text_embeddings_file)
  tf.logging.info("Writing converted embeddings to: %s" % FLAGS.proto_embeddings_file)
  with open(FLAGS.text_embeddings_file, 'r') as f, \
          tf.python_io.TFRecordWriter(FLAGS.proto_embeddings_file) as w:
    for i, line in enumerate(f):
      split_line = line.split(' ')
      token = split_line[0]
      embedding = map(float, split_line[1:])
      assert token != ''
      token_embedding = dictionary_pb2.TokenEmbedding()
      token_embedding.token = token
      token_embedding.vector.values.extend(embedding)
      w.write(str(token_embedding))
      if write_vocab_to_file:
        vocab.append(token)
      if i == max_to_read:
        break

  if write_vocab_to_file:
    tf.logging.info("Writing pretrained embedding vocabulary to: %s" % FLAGS.vocab_file)
    with open(FLAGS.vocab_file, 'w') as f:
      for word in vocab:
        print(word, file=f)

  file_queue = tf.train.string_input_producer([FLAGS.proto_embeddings_file])
  reader = tf.TFRecordReader()
  key, record = reader.read(file_queue)
  print(key, record)
  print(tf.parse_tensor(record))
  proto = dictionary_pb2.TokenEmbedding()
  print(proto.ParseFromString(record))

if __name__ == '__main__':
  tf.app.run()
