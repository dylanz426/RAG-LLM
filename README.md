# RAG-LLM

In this repo we build a simple prototype for a question-answering system that uses LLMs and retrieval augmented generation.

## Install
Before running the scripts, please install the required packages.
```
pip install -r requirements.txt
```

## Dataset
The dataset is based on the Wikipedia page [List of landmark court decisions in the United States](https://en.wikipedia.org/wiki/List_of_landmark_court_decisions_in_the_United_States). The page content is split into chunks and each chunk is processed by adding subtitle information. The processed dataset `landmark-court-decisions-in-the-US.txt` is saved in the `data` folder. We can replicate data processing by
```
python code/data_processing.py \
    --wikipedia_title "List of landmark court decisions in the United States" \
    --output_data_path "data/landmark-court-decisions-in-the-US.txt"
```

## Vector Database
We vectorize each chunk using a `sentence transformer`. All the vectors and their corresponding chunks are saved as the database `landmark-court-decisions-in-the-US-db.pkl` in the `data` folder. We can replicate the data vectorization by
```
python code/save_vector_database.py \
    --data_path "data/landmark-court-decisions-in-the-US.txt" \
    --output_db_path "data/landmark-court-decisions-in-the-US-db.pkl" \
    --sent_trans "multi-qa-mpnet-base-cos-v1"
```

## Answer Generation
Given a question, we vectorize it using the same `sentence transformer` then compare the semantic similarities with the vectors in the database. Based on the semantic similarity scores, the top k most relevant chunks are retrieved from the database. These top k chunks are then paired up with the question and reranked by a large reranking model `BAAI/bge-reranker-large`. The chunks with reranker scores higher than a threshold are used as the context for the question. Then we can call the OpenAI API to generate an answer given the question and retrieved context. The prompt used here is `You are a helpful assistant. Answer the following question with details based on your knowledge and the provided context.` This whole process can be run by
```
python code/generate_answers.py \
    --question "What are the most important supreme court cases to do with gun rights in the USA?" \
    --db_path "data/landmark-court-decisions-in-the-US-db.pkl" \
    --answer_path "results/answer1.txt" \
    --openai_api_key $KEY \
    --sent_trans "multi-qa-mpnet-base-cos-v1" \
    --reranker_path "BAAI/bge-reranker-large" \
    --top_k 20 \
    --thre 0.01
```
The answers for the three example questions are saved in the `results` folder. These three questions are:
1. What are the most important supreme court cases to do with gun rights in the USA?
2. Can police in the USA search my house? What court cases determine this?
3. What was the recent scotus ruling on affirmative action?

## Answer Evaluation
The evaluation metric `SCALE` is based on one of my recent papers [Fast and Accurate Factual Inconsistency Detection Over Long Documents](https://aclanthology.org/2023.emnlp-main.105.pdf). Compared to traditional evaluation metrics such as BLEU, ROUGE, or F1 scores, SCALE much better approximates the human evaluations of the actual task. The original SCALE scores were designed to accommodate long documents, here I'm using a simplified version focusing on the entitlement relationship of generated answers and true answers only. By answering a Yes or No question, the resulting logits from `Flan-T5` are used to compute the entailment score. `SCALE-recall` is used for this analysis to compare each sentence in the true answer with the generated answer. Given the generated answer and the true answer, we can get the scores by
```
python code/evaluate_answers.py \
    --generated_answer "Some of the most important Supreme Court cases related to gun rights in the USA include:\n\n1. District of Columbia v. Heller (2008): This case established that the Second Amendment protects an individual's right to possess a firearm for self-defense within the home, unrelated to militia service.\n\n2. McDonald v. City of Chicago (2010): This case extended the individual right to keep and bear arms for self-defense to the states through the Fourteenth Amendment.\n\n3. United States v. Miller (1939): This case clarified that the Second Amendment does not protect all types of weapons, but rather those with a reasonable relationship to a well-regulated militia.\n\n4. New York State Rifle & Pistol Association, Inc. v. Bruen (2022): This recent case affirmed the individual right to carry a handgun for self-defense in public, outside the home, and emphasized that firearms regulations must be evaluated based on historical traditions in the U.S.\n\nThese cases have played significant roles in shaping the interpretation and application of gun rights in the United States." \
    --true_answer "Some of the most important Supreme Court cases to do with gun rights in the USA include:\n\n1. District of Columbia v. Heller (2008): The court ruled in favor of Heller, affirming an individual right to keep handguns in the home for self-defense.\n2. New York State Rifle & Pistol Association, Inc. v. Bruen (2022): This case ruled on the ability to carry concealed handguns in public.\n\nThese cases have been pivotal in shaping the interpretation of the Second Amendment and the scope of gun rights in the United States." \
    --model_path "google/flan-t5-large"
```
### SCALE Results
| Method | Proposed | No Retrieval | No Reranking |
| ------ | ------ | ------ | ------ |
| Question 1 | **0.7500** | 0.5288 | 0.7473 |
| Question 2 | **0.6721** | 0.6157 | 0.6617 |
| Question 3 | **0.5890** | 0.2922 | 0.5558 |

In the above results, `proposed` represents the approach in this repo, `No Retrieval` means directly asking the question to OpenAI API without providing the retrieved context, and `No Reranking` means directly using the top k chunks as the retrieved context without reranking. As we can see, `No Retrieval` has the lowest scores across all three questions, indicating the effectiveness of our retrieval augmented generation. The proposed method achieves similar (slightly better) scores as `No Reranking` while using less context information that are filtered out based on reranking scores.
