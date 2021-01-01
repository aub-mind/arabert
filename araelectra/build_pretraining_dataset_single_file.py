# coding=utf-8

import argparse
import os
import tensorflow as tf

import build_pretraining_dataset
from model import tokenization

class ExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, input_fname, vocab_file, output_dir, max_seq_length,
               blanks_separate_docs, do_lower_case):
    self._blanks_separate_docs = blanks_separate_docs
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    self._example_builder = build_pretraining_dataset.ExampleBuilder(tokenizer, max_seq_length)
    output_fname = os.path.join(output_dir, "{}.tfrecord".format(input_fname.split("/")[-1]))
    self._writer = tf.io.TFRecordWriter(output_fname)
    self.n_written = 0

  def write_examples(self, input_file):
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
        line = line.strip()
        if line or self._blanks_separate_docs:
          example = self._example_builder.add_line(line)
          if example:
            self._writer.write(example.SerializeToString())
            self.n_written += 1
      example = self._example_builder.add_line("")
      if example:
        self._writer.write(example.SerializeToString())
        self.n_written += 1

  def finish(self):
    self._writer.close()

def write_examples(args):
  """A single process creating and writing out pre-processed examples."""

  def log(*args):
    msg = " ".join(map(str, args))
    print(msg)

  log("Creating example writer")
  example_writer = ExampleWriter(
      input_fname=args.input_file,
      vocab_file=args.vocab_file,
      output_dir=args.output_dir,
      max_seq_length=args.max_seq_length,
      blanks_separate_docs=args.blanks_separate_docs,
      do_lower_case=args.do_lower_case
  )
  log("Writing tf example")

  example_writer.write_examples(args.input_file)
  example_writer.finish()
  log("Done!")
  return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True,
                        help="Location of pre-training text files.")
    parser.add_argument("--vocab-file", required=True,
                        help="Location of vocabulary file.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write out the tfrecords.")
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Number of tokens per example.")
    parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                        help="Whether blank lines indicate document boundaries.")
    parser.add_argument("--do-lower-case", dest='do_lower_case',
                        action='store_true', help="Lower case input text.")
    parser.add_argument("--no-lower-case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()

    write_examples(args)



if __name__ == "__main__":
  main()