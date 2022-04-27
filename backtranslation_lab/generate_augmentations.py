import argparse
import os

import backtranslation_lab.constants as C
import torch
from loguru import logger
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Generate augmented with backtranslation technique dataset")

    parser.add_argument("--en_sentences", type=str, required=True, help="Path to txt file with English sentences.")
    parser.add_argument("--de_sentences", type=str, required=True, help="Path to txt file with corresponding German sentences.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to directory where new dataset will be saved.")
    parser.add_argument("--en_lm_beam_size", type=int, default=1, help="De-En language model beam size.")
    parser.add_argument("--de_lm_beam_size", type=int, default=1, help="En-De language model beam size.")
    parser.add_argument("--num_sentences", type=int, default=10_000, help="Number of sentences picked to perform backtranslation.")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    logger.info("✌️ Parsing arguments")
    args = parse_args()

    # fmt: off
    logger.info("🕸 Loading NMT models")
    en2de = torch.hub.load("pytorch/fairseq", C.EN2DE_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    de2en = torch.hub.load("pytorch/fairseq", C.DE2EN_NMT_MODEL_NAME, tokenizer="moses", bpe="fastbpe")
    # fmt: on

    original_dset = []
    en2de2en_augs = []
    de2en2en_augs = []
    logger.info("🤹 Generating backtranslations")
    with open(args.en_sentences) as eng_dset, open(args.de_sentences) as de_dset:
        generated_backtranslations = 0
        for eng_sentence, de_sentence in tqdm(zip(eng_dset, de_dset)):
            eng_sentence, de_sentence = eng_sentence.strip(), de_sentence.strip()
            if eng_sentence == "" or de_sentence == "":
                continue

            original_dset.append((eng_sentence, de_sentence))

            if generated_backtranslations == args.num_sentences:
                continue

            translation = en2de.translate(eng_sentence, beam_size=args.de_lm_beam_size)
            backtranslation = de2en.translate(
                translation, beam_size=args.en_lm_beam_size
            )

            en2de2en_new_sample = (backtranslation, translation)
            en2de2en_augs.append(en2de2en_new_sample)

            translation = de2en.translate(de_sentence, beam_size=args.en_lm_beam_size)
            backtranslation = en2de.translate(
                translation, beam_size=args.de_lm_beam_size
            )

            de2en2de_new_sample = (translation, backtranslation)
            de2en2en_augs.append(de2en2de_new_sample)

            generated_backtranslations += 1

    logger.info("💾 Saving to output directory")
    os.makedirs(args.outdir, exist_ok=True)

    logger.info("☝️ Saving original data")
    with open(os.path.join(args.outdir, "original.en"), "w") as orig_en_f:
        with open(os.path.join(args.outdir, "original.de"), "w") as orig_de_f:
            for original_en, original_de in tqdm(original_dset):
                orig_en_f.write(f"{original_en}\n")
                orig_de_f.write(f"{original_de}\n")

    logger.info("✌️ Saving en2de2en backtranslations")
    with open(os.path.join(args.outdir, "backtranslations.en"), "w") as en_file:
        with open(os.path.join(args.outdir, "backtranslations.de"), "w") as de_file:
            for en_sentence, de_sentence in tqdm([*en2de2en_augs, *de2en2en_augs]):
                en_file.write(f"{en_sentence}\n")
                de_file.write(f"{de_sentence}\n")

    logger.info("👯 Removing duplications.")
    augmented_dset = set([*original_dset, *de2en2en_augs, *en2de2en_augs])

    logger.info("📚 Saving original dataset with backtranslations.")
    with open(os.path.join(args.outdir, "aug_dataset.en"), "w") as en_file:
        with open(os.path.join(args.outdir, "aug_dataset.de"), "w") as de_file:
            for en_sentence, de_sentence in tqdm(augmented_dset):
                en_file.write(f"{en_sentence}\n")
                de_file.write(f"{de_sentence}\n")


if __name__ == "__main__":
    main()
