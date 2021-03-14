# coding=utf-8
#
# This script applies AraBERT's cleaning process and segmentation to ARCD or
# any SQUAD-like structured files and "naively" re-alligns the answers start positions

import sys

sys.path.append("..")

import tensorflow.compat.v1 as tf

import json

from fuzzysearch import find_near_matches
from pyarabic import araby
from tqdm import tqdm

from arabert.arabert.tokenization import BasicTokenizer
from arabert.preprocess import ArabertPreprocessor

flags = tf.flags
FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
    "input_file", None, "The input json file with a SQUAD like structure."
)

flags.DEFINE_string(
    "output_file", None, "The ouput json file with AraBERT preprocessing applied."
)

flags.DEFINE_string("model_name", None, "Model name same as HuggingFace library")

flags.DEFINE_bool(
    "filter_tydiqa",
    False,
    "If the input dataset is tydiqa, then only process arabic examples",
)


bt = BasicTokenizer(do_lower_case=False)


def clean_preprocess(text, processor):
    text = " ".join(bt._run_split_on_punc(text))
    text = processor.preprocess(text)
    text = " ".join(text.split())  # removes extra whitespaces
    return text


def get_start_pos(old_context, old_answer_start, processor):
    new_context = clean_preprocess(old_context[:old_answer_start], processor)
    num_of_pluses = new_context.count("+")
    return old_answer_start + num_of_pluses * 2 - 20


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = tf.get_logger()
    logger.propagate = False

    print(FLAGS.model_name)
    arabert_prep = ArabertPreprocessor(
        model_name=FLAGS.model_name, remove_html_markup=False
    )

    with tf.gfile.Open(FLAGS.input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    new_answers_count = 0
    no_answers_found_count = 0
    trunc_ans_count = 0
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:

            if FLAGS.filter_tydiqa:
                # this will only apply farasa segmentation to Arabic Data
                if "arabic" not in paragraph["qas"][0]["id"]:
                    continue
            old_context = paragraph["context"]
            paragraph["context"] = clean_preprocess(paragraph["context"], arabert_prep)
            for qas in paragraph["qas"]:
                qas["question"] = clean_preprocess(qas["question"], arabert_prep)

                for i in range(len(qas["answers"])):
                    temp_text = clean_preprocess(
                        qas["answers"][i]["text"], arabert_prep
                    )

                    if temp_text == "":
                        temp_text = qas["answers"][i]["text"]

                    answer_location = paragraph["context"].find(temp_text)
                    if answer_location == -1:

                        search_start_pos = get_start_pos(
                            old_context, qas["answers"][i]["answer_start"], arabert_prep
                        )
                        search_end_pos = min(
                            len(paragraph["context"]),
                            search_start_pos + len(temp_text) + 20,
                        )
                        answer_match = find_near_matches(
                            temp_text,
                            paragraph["context"][search_start_pos:search_end_pos],
                            max_l_dist=min(10, len(temp_text) // 2),
                        )
                        if len(answer_match) > 0:
                            tf.logging.warning(
                                "Found new answer for question '%s' :\n '%s' \nvs old.\n '%s'\norig:\n'%s'\ncontext:\n'%s'\n==================",
                                qas["id"],
                                answer_match[i].matched,
                                temp_text,
                                qas["answers"][i]["text"],
                                paragraph["context"],
                            )
                            temp_text = answer_match[i].matched
                            qas["answers"][i]["answer_start"] = answer_match[i].start
                            new_answers_count += 1

                        else:
                            tf.logging.warning(
                                "Could not find answer for question '%s' :\n '%s' \nvs.\n '%s'\norig answer:\n '%s'\n==================",
                                qas["id"],
                                paragraph["context"],
                                temp_text,
                                qas["answers"][i]["text"],
                            )
                            qas["answers"][i]["answer_start"] = -1
                            no_answers_found_count += 1
                    else:
                        qas["answers"][i]["answer_start"] = answer_location

                    if len(temp_text) + qas["answers"][i]["answer_start"] < (
                        len(paragraph["context"]) + 1
                    ):
                        qas["answers"][i]["text"] = temp_text
                    else:
                        tf.logging.warning(
                            "answer truncated for question '%s' :\n context:\n'%s' \nanswer:\n '%s'\n orig_answer:\n'%s'\nanswer start: %d\nlength of answer: %d\nlength of paragraph: %d\n=================================",
                            qas["id"],
                            paragraph["context"],
                            temp_text,
                            qas["answers"][i]["text"],
                            qas["answers"][0]["answer_start"],
                            len(temp_text),
                            len(paragraph["context"]),
                        )
                        qas["answers"][0]["text"] = temp_text[
                            0 : len(paragraph["context"])
                            - (len(temp_text) + qas["answers"][0]["answer_start"])
                        ]
                        trunc_ans_count += 1

    tf.logging.warning("Found %d new answers: ", new_answers_count)
    tf.logging.warning("Found %d with no answers: ", no_answers_found_count)
    tf.logging.warning("Found %d with trunc answers: ", trunc_ans_count)

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
