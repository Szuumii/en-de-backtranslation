import argparse
import os

import backtranslation_lab.constants as C
import torch


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Generate augmented with backtranslation technique dataset")

    parser.add_argument("--en_sentences", type=str, required=True, help="Path to txt file with English sentences.")
    parser.add_argument("--de_sentences", type=str, required=True, help="Path to txt file with corresponding German sentences.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to directory where new dataset will be saved.")
    parser.add_argument("--en_lm_beam_size", type=int, default=1, help="De-En language model beam size.")
    parser.add_argument("--de_lm_beam_size", type=int, default=1, help="En-De language model beam size.")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # fmt: off
    en2de = torch.hub.load("pytorch/fairseq", C.EN2DE_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    de2en = torch.hub.load("pytorch/fairseq", C.DE2EN_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    # fmt: on

    with open(args.en_sentences) as eng_dset, open(args.de_sentences) as de_dset:
        for eng_sentence, de_sentence in zip(eng_dset, de_dset):
            eng_sentence, de_sentence = eng_sentence.strip(), de_sentence.strip()

            translation = en2de.translate(eng_sentence, beam_size=args.de_lm_beam_size)
            backtranslation = de2en.translate(
                translation, beam_size=args.en_lm_beam_size
            )

            new_sample = (backtranslation, translation)
            print(new_sample)
            break


if __name__ == "__main__":
    main()
