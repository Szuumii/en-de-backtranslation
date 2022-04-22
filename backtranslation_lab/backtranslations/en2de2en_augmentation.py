import argparse
import os

import backtranslation_lab.constants as C
import torch


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Generate augmented with backtranslation technique dataset")

    parser.add_argument("--de_sentences", type=str, required=True, help="Path to txt file with German sentences.")
    parser.add_argument("--en_sentences", type=str, required=True, help="Path to txt file with corresponding English sentences.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to directory where new dataset will be saved.")
    parser.add_argument("--de_lm_beam_size", type=int, default=1, help="Path to txt file with corresponding English sentences.")
    parser.add_argument("--en_lm_beam_size", type=int, default=1, help="Path to txt file with corresponding English sentences.")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # fmt: off
    en2de = torch.hub.load("pytorch/fairseq", C.EN2DE_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    de2en = torch.hub.load("pytorch/fairseq", C.DE2EN_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    # fmt: on

    for sentence in ["PyTorch Hub is an awesome interface!"]:
        translation = en2de.translate(sentence, beam_size=args.de_lm_beam_size)
        backtranslation = de2en.translate(translation, beam_size=args.en_lm_beam_size)

        new_sample = (backtranslation, translation)
        print(new_sample)


if __name__ == "__main__":
    main()
