from dataclasses import dataclass, field
import pickle
from typing import Any, Tuple
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments for loading data and saving vector database.
    """

    data_path: str = field(
        metadata={
            "help": "Path to the dataset.",
            "required": True,
        },
    )

    output_db_path: str = field(
        metadata={
            "help": "Path to the output vector database.",
            "required": True,
        },
    )

    sent_trans: str = field(
        default="all-mpnet-base-v2",
        metadata={
            "help": "Name of the sentence transformer model that is used to encode chunks.",
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


def main(args: Arguments):
    sent_trans_model = SentenceTransformer(args.sent_trans)
    with open(args.data_path, "r") as f:
        text = f.read()
    # split into chunks then encode each chunk using a sentence transformer
    chunks = [s.strip() for s in text.split("\n") if len(s.strip()) > 0]
    embeddings = sent_trans_model.encode(chunks)
    # save the vector database
    with open(args.output_db_path, "wb") as f:
        pickle.dump((chunks, embeddings), f)


if __name__ == "__main__":
    args = get_args()
    main(args[0])
