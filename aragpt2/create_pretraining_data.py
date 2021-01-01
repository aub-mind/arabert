import collections
from transformers import GPT2TokenizerFast
import tensorflow as tf

import sys
sys.path.append("..")
from arabert.preprocess import preprocess

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None, "Input raw text file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "output_file", None, "Output TF example file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "tokenizer_dir", None, "The directory of a pretrained GPT2TokenizerFast"
)

flags.DEFINE_integer(
    "max_len", 1024, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_integer(
    "num_examples_print", 0, "Number of examples to print"
)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = tf.get_logger()
    logger.propagate = False

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    gpt2_tok = GPT2TokenizerFast.from_pretrained(FLAGS.tokenizer_dir)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_file + ".tfrecord")

    eos_id = gpt2_tok.eos_token_id
    all_examples = []
    for input_file in input_files:
        queue = []
        example = []
        with tf.gfile.GFile(input_file, "r") as reader:
            for line in reader.readlines():
                if line == "\n":
                    queue.append(eos_id)
                else:
                    line = line.replace("\n", " ")
                    line = preprocess(line,model='gpt2-base-arabic')
                    line = line.strip()
                    enc_line = gpt2_tok.encode(line)
                    queue.extend(enc_line)
                if len(queue) > FLAGS.max_len +1:
                    example = [queue.pop(0) for _ in range(FLAGS.max_len +1)]
                    assert len(example) == FLAGS.max_len +1
                    all_examples.append(example)


    for i, ex in enumerate(all_examples):
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(ex)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        if i < FLAGS.num_examples_print:
            tf.logging.info("*** Example ***")
            tf.logging.info("Length: %d" % len(ex))
            tf.logging.info("Tokens: %s" % gpt2_tok.decode(ex))
            tf.logging.info("ids: %s" % " ".join([str(x) for x in ex]))

    tf.logging.info("Wrote %d total instances", len(all_examples))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("tokenizer_dir")
    tf.app.run()
