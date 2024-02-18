from typing import Any, Tuple, List
from dataclasses import dataclass, field
import re
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, T5ForConditionalGeneration, T5Tokenizer


@dataclass
class Arguments:
    """
    Arguments for generating answer based on question and database.
    """

    generated_answer: str = field(
        metadata={
            "help": "Generated answer for the question.",
            "required": True,
        },
    )

    true_answer: str = field(
        metadata={
            "help": "True answer for the question.",
            "required": True,
        },
    )

    model_path: str = field(
        default="google/flan-t5-large",
        metadata={
            "help": "Name of the evaluation model.",
            "required": False,
        },
    )


def get_args() -> Tuple[Any, ...]:
    """
    Parse command line arguments.

    :return: Arguments
    """

    parser = HfArgumentParser(Arguments)

    return parser.parse_args_into_dataclasses()


def scale_score(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: List[str],
    hypothesis: List[str],
) -> List[float]:
    """
    Compute the SCALE scores to evaluate the generated answers.

    :param model: FlanT5 model
    :param tokenizer: FlanT5 tokenizer
    :param premise: list of reference sentences
    :param hypothesis: list of target sentences

    :return the score averaged over target sentences
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yes_no_tokens = [tokenizer("Yes").input_ids[0], tokenizer("No").input_ids[0]]
    h_results = []
    for h in tqdm(hypothesis, total=len(hypothesis)):
        p_results = []
        for p in tqdm(premise, total=len(premise)):
            prompt = f'{p} Question: Does this imply that "{h}"? Yes or No?'
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
            )
            scores = outputs["scores"][0][0][yes_no_tokens]
            p_results.append(torch.nn.functional.softmax(scores, dim=0)[0].item())
        h_results.append(max(p_results))
    return sum(h_results) / len(h_results)


def split_text(text: str):
    """
    Split a string into sentences.

    :param text: the full string

    :return list of sentences split from the string
    """

    text = text.replace("\n", "  ").replace("\t", "  ").replace(" v.", " v.s")

    text = re.sub("\s{2,}", "  ", text)
    text = re.sub(r"\.+", ".", text)

    text = (
        text.replace(":  ", ": ")
        .replace(". ", ".***")
        .replace("? ", "?***")
        .replace("! ", "!***")
        .replace("  ", "***")
    )
    return [s.strip() for s in text.split("***") if len(s.strip()) > 0]


def main(args: Arguments):
    generated_sentences = split_text(args.generated_answer)
    true_sentences = split_text(args.true_answer)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto"
    )
    results = {}
    results["SCALE_precision"] = scale_score(
        model, tokenizer, true_sentences, generated_sentences
    )
    results["SCALE_recall"] = scale_score(
        model, tokenizer, generated_sentences, true_sentences
    )
    print(results)


if __name__ == "__main__":
    args = get_args()
    main(args[0])
