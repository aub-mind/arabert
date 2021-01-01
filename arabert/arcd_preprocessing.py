# coding=utf-8
#
# This script applies AraBERT's cleaning process and segmentation to ARCD or
# any SQUAD-like structured files and "naively" re-alligns the answers start positions

import tensorflow as tf
import sys
sys.path.append('..')
from arabert.preprocess import ArabertPreprocessor
from tokenization import BasicTokenizer



import json

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input_file", None, "The input json file with a SQUAD like structure."
)

flags.DEFINE_string(
    "output_file", None, "The ouput json file with AraBERT preprocessing applied."
)

flags.DEFINE_bool(
    "model_name", None, "Check the accepted models list"
)



bt = BasicTokenizer()


def clean_preprocess(text, arabert_prep):
    text = " ".join(
        bt._run_split_on_punc(
            arabert_prep.preprocess(
                text
            )
        )
    )
    text = " ".join(text.split())  # removes extra whitespaces
    return text


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = tf.get_logger()
    logger.propagate = False

    arabert_prep = ArabertPreprocessor(model_name=FLAGS.model_name, keep_emojis=False)

    with tf.gfile.Open(FLAGS.input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph["context"] = clean_preprocess(
                paragraph["context"],
                arabert_prep
            )
            for qas in paragraph["qas"]:
                qas["question"] = clean_preprocess(
                    qas["question"],
                    arabert_prep
                )
                qas["answers"][0]["text"] = clean_preprocess(
                    qas["answers"][0]["text"],
                    arabert_prep
                )
                qas["answers"][0]["answer_start"] = paragraph["context"].find(
                    qas["answers"][0]["text"]
                )
                if qas["answers"][0]["answer_start"] == -1:
                    tf.logging.warning(
                        "Could not find answer for question '%s' : '%s' vs. '%s'",
                        qas["id"],
                        paragraph["context"],
                        qas["answers"][0]["text"],
                    )

    input_data = {
        "data": input_data,
        "version": "1.1",
        "preprocess": "True",
    }
    with tf.gfile.Open(FLAGS.output_file, "w") as writer:
        json.dump(input_data, writer)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("model_name")
    tf.app.run()
