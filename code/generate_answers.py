from dataclasses import dataclass, field
import pickle
import pandas as pd
import openai
from typing import Any, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments for loading data and saving vector database.
    """

    question: str = field(
        metadata={
            "help": "The question to generate answers for.",
            "required": True,
        },
    )

    db_path: str = field(
        metadata={
            "help": "Path to the vector database.",
            "required": True,
        },
    )

    answer_path: str = field(
        metadata={
            "help": "Path to the output answer.",
            "required": True,
        },
    )

    openai_api_key: str = field(
        metadata={
            "help": "OpenAI key to access the API.",
            "required": True,
        },
    )

    sent_trans: str = field(
        default="multi-qa-mpnet-base-cos-v1",
        metadata={
            "help": "Name of the sentence transformer model that is used to encode chunks.",
            "required": False,
        },
    )

    top_k: int = field(
        default=10,
        metadata={
            "help": "The top k chunks based on the semantic matching scores.",
            "required": False,
        },
    )

    thre: float = field(
        default=0.5,
        metadata={
            "help": "The threshold on the semantic matching scores to select context.",
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


def call_api(prompt: str, openai_api_key: str) -> str:
    """
    Call the OpenAI API to generate the answer given a prompt.

    :param prompt: the prompt of concatenated question and chunks
    :param openai_api_key: OpenAI key to access the API

    :return the answer
    """

    openai.api_key = openai_api_key
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the following question with details based on your knowledge and the provided context.",
        },
        {"role": "user", "content": prompt},
    ]
    response = (
        openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0, max_tokens=1024
        )
        .choices[0]
        .message.content
    )
    return response


def get_prompt(question: str, df_chunks: pd.DataFrame, thre: float) -> str:
    """
    Get the prompt by concatenating question and chunks that pass a semantic
    score threshold.

    :param question: the question
    :param df_chunks: chunks and semantic scores in a dataframe
    :param thre: the threshold to select chunks

    :return the prompt of concatenated question and chunks
    """

    chunks = []
    for _, row in df_chunks.iterrows():
        if len(chunks) == 0 or row["scores"] > thre:
            text = row["chunks"].split("\t")[-1]
            chunks.append(text)
    context = "\n".join(chunks)
    return f"question:\n{question}\n\ncontext:\n{context}\n\nanswer:"


def main(args: Arguments):
    # load the database
    with open(args.db_path, "rb") as f:
        chunks, chunk_embeddings = pickle.load(f)
    # load the sentence transformer and encode the question
    sent_trans_model = SentenceTransformer(args.sent_trans)
    question_embedding = sent_trans_model.encode([args.question])
    # semantic search for the top k chunks
    preds_dict = util.semantic_search(
        question_embedding, chunk_embeddings, top_k=args.top_k
    )
    selected = {"chunks": [], "scores": []}
    for i in range(args.top_k):
        dic = preds_dict[0][i]
        selected["chunks"].append(chunks[dic["corpus_id"]])
        selected["scores"].append(dic["score"])
    df = pd.DataFrame(selected)
    # get the prompt by concatenating question and chunks
    prompt = get_prompt(args.question, df, args.thre)
    # call the api to generate answers
    answer = call_api(prompt, args.openai_api_key)
    # save the question and the generated answer
    output = f"{prompt}\n{answer}"
    with open(args.answer_path, "w") as f:
        f.write(output)


if __name__ == "__main__":
    args = get_args()
    main(args[0])
