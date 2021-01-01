import argparse
import glob
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)
    args = parser.parse_args()

    gpt2_tok = ByteLevelBPETokenizer(add_prefix_space=True)

    files = glob.glob(args.data_files)
    if len(files) > 10:
        print(files[0:10])
    else:
        print(files)

    gpt2_tok.train(
        files=files,
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["<|endoftext|>", "<s>", "<pad>", "</s>"],
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    gpt2_tok.save(
            os.path.join(args.output_dir,"tokenizer.json"), pretty=True
        )  # FIX Access is denied. (os error 5)
    gpt2_tok.save_model(args.output_dir, args.output_file_name)

    # tokenizer = GPT2TokenizerFast(
    #     vocab_file=os.path.join(args.output_dir, args.output_file_name) + "-vocab.json",
    #     merges_file=os.path.join(args.output_dir, args.output_file_name)
    #     + "-merges.txt",
    #     add_prefix_space=True,
    # )

    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": "<|endoftext|>",
    #         "bos_token": "<|endoftext|>",
    #         "unk_token": "<|endoftext|>",
    #         "pad_token": "<|endoftext|>",
    #         "mask_token": "<|endoftext|>",
    #     }
    # )

    # tokenizer.save_pretrained(
    #     args.output_dir, legacy_format=False, filename_prefix=args.output_file_name
    # )
