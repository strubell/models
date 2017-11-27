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

"""Utilities for reading and writing sentences in dragnn."""
import tensorflow as tf
from syntaxnet.ops import gen_parser_ops


class FormatSentenceReader(object):
  """A reader for formatted files, with optional projectivizing."""

  def __init__(self,
               filepath,
               record_format,
               batch_size=32,
               check_well_formed=False,
               projectivize=False,
               morph_to_pos=False):
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)
    task_context_str = """
          input {
            name: 'documents'
            record_format: '%s'
            Part {
             file_pattern: '%s'
            }
          }""" % (record_format, filepath)
    if morph_to_pos:
      task_context_str += """
          Parameter {
            name: "join_category_to_pos"
            value: "true"
          }
          Parameter {
            name: "add_pos_as_attribute"
            value: "true"
          }
          Parameter {
            name: "serialize_morph_to_pos"
            value: "true"
          }
          """
    with self._graph.as_default():

      self._source, self._is_last = gen_parser_ops.document_source(
          task_context_str=task_context_str, batch_size=batch_size)
      self._is_last = tf.Print(self._is_last, [self._is_last], "self._is_last")

      if check_well_formed:
        self._is_last = tf.Print(self._is_last, [self._is_last], "check well formed")
        self._source = gen_parser_ops.well_formed_filter(self._source)
      if projectivize:
        self._is_last = tf.Print(self._is_last, [self._is_last], "projectivize")
        self._source = gen_parser_ops.projectivize_filter(self._source)
      self._is_last = tf.Print(self._is_last, [self._is_last], "returning")


  def read(self):
    """Reads a single batch of sentences."""
    if self._session:
      tf.logging.info("calling session.run")
      sentences, is_last = self._session.run([self._source, self._is_last])
      if is_last:
        self._session.close()
        self._session = None
    else:
      sentences, is_last = [], True
    tf.logging.info("returning")
    return sentences, is_last

  def corpus(self):
    """Reads the entire corpus, and returns in a list."""
    tf.logging.info('Reading corpus...')
    corpus = []
    while True:
      tf.logging.info("calling read")
      sentences, is_last = self.read()
      tf.logging.info(sentences)
      corpus.extend(sentences)
      if is_last:
        break
    tf.logging.info('Read %d sentences.' % len(corpus))
    return corpus


class ConllSentenceReader(FormatSentenceReader):
  """A sentence reader that uses an underlying 'conll-sentence' reader."""

  def __init__(self,
               filepath,
               batch_size=32,
               projectivize=False,
               morph_to_pos=False):
    super(ConllSentenceReader, self).__init__(
        filepath,
        'conll-sentence',
        check_well_formed=True,
        batch_size=batch_size,
        projectivize=projectivize,
        morph_to_pos=morph_to_pos)
