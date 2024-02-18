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
Given a question, we vectorize it using the same `sentence transformer` then compare the semantic similarities with the vectors in the database. Based on the semantic similarity scores, we select the top k most relevant chunks from the database. These top k chunks are paired up with the question and reranked by a large reranking model `BAAI/bge-reranker-large`. The chunks with reranker scores higher than a threshold are used as the context for the question. Then we can call the OpenAI API to generate an answer given the question and context. This whole process can be run by
```
python code/generate_answers.py \
    --question "What was the recent scotus ruling on affirmative action?" \
    --db_path "data/landmark-court-decisions-in-the-US-db.pkl" \
    --answer_path "results/answer3.txt" \
    --openai_api_key $KEY \
    --sent_trans "multi-qa-mpnet-base-cos-v1" \
    --reranker_path "BAAI/bge-reranker-large" \
    --top_k 10 \
    --thre 0.01
```
The answers for the three example questions are saved in the `results` folder. These three questions are:
1. What are the most important supreme court cases to do with gun rights in the USA?
2. Can police in the USA search my house? What court cases determine this?
3. What was the recent scotus ruling on affirmative action?

## Answer Evaluation
The evaluation metric `SCALE` is based on one of my recent papers [Fast and Accurate Factual Inconsistency Detection Over Long Documents](https://aclanthology.org/2023.emnlp-main.105.pdf). Compared to traditional evaluation metrics such as BLEU, ROUGE, or F1 scores, SCALE much better approximates the human evaluations of the actual task. The original SCALE scores were designed to accommodate long documents, here I'm using a simplified version focusing on the entitlement relationship of generated answers and true answers only. By answering a Yes or No question, the resulting logits from `Flan-T5` are used to compute the entailment score. Given the generated answer and the true answer, we can get the `SCALE-precision` and `SCALE-recall` scores by
```
python code/evaluate_answers.py \
    --generated_answer "The recent SCOTUS ruling on affirmative action was in the cases of Students for Fair Admissions v. Harvard and Students for Fair Admissions v. University of North Carolina in 2023. The ruling stated that race-based affirmative action programs in civilian college admissions processes violate the Equal Protection Clause. This decision has significant implications for how universities can consider race in their admissions processes moving forward." \
    --true_answer "The recent SCOTUS ruling on affirmative action was in the cases SFFA v. Harvard and SFFA v. UNC, where the Supreme Court effectively eliminated the use of affirmative action in college admissions. In a 6-3 ruling, the Court held that the admissions programs of Harvard and UNC, which considered race at various stages, violated the Equal Protection Clause of the Fourteenth Amendment to the U.S. Constitution. This ruling has significant implications for race-based affirmative action in higher education in the United States" \
    --model_path "google/flan-t5-large" 
```
