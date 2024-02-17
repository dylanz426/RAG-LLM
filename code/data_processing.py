from dataclasses import dataclass, field
import wikipedia
from typing import List, Any, Tuple
from transformers import HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments for loading data and saving vector database.
    """

    wikipedia_title: str = field(
        metadata={
            "help": "Wikipedia title name.",
            "required": True,
        },
    )

    output_data_path: str = field(
        metadata={
            "help": "Path to the output processed dataset.",
            "required": True,
        },
    )


def get_args() -> Tuple[Any, ...]:
    """
    Parse command line arguments.

    :return: Arguments
    """

    parser = HfArgumentParser(Arguments)

    return parser.parse_args_into_dataclasses()


def process_data(content: str) -> List[str]:
    """
    Split the Wikipedia page content into chunks and process each chunk by
    adding subtitle information.

    :param content: the entire Wikipedia page content as a string

    :return List of processed chunks
    """

    # split into chunks based on new_line
    chunks = [s.strip() for s in content.split("\n") if len(s.strip()) > 0]

    subtitle = None
    subsubtitle = None
    new_chunks = []
    for line in chunks:
        if line.startswith("==="):  # new subsubtitle line
            subsubtitle = line.replace("=", "").strip()
        elif line.startswith("=="):  # new subtitle line
            subtitle = line.replace("=", "").strip()
            subsubtitle = None
        else:  # concatenate subtitle, subsubtitle, and the chunk
            sequence = "\t".join(
                [part for part in [subtitle, subsubtitle, line] if part]
            )
            new_chunks.append(sequence)

    return new_chunks


def main(args: Arguments):
    content = wikipedia.page(args.wikipedia_title).content
    chunks = process_data(content)
    with open(args.output_data_path, "w") as f:
        f.write("\n".join(chunks))


if __name__ == "__main__":
    args = get_args()
    main(args[0])
