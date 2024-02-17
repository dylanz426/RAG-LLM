import wikipedia
from typing import List


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


def main(title: str, output_dir: str):
    content = wikipedia.page(title).content
    chunks = process_data(content)
    with open(output_dir, "w") as f:
        f.write("\n".join(chunks))


if __name__ == "__main__":
    title = "List of landmark court decisions in the United States"
    output_dir = "data/landmark-court-decisions-in-the-US.txt"
    main(title, output_dir)
