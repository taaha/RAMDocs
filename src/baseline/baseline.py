import json
import sys
import os
from tqdm import tqdm

# Add the src directory to the path for direct execution
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.pydantic_ai_llm_utils import call_llm_pydantic_ai
from src.models.baseline import Answer

file_path = "src/data/split_data/RAMDocs_test_test.jsonl"


def delete_results_file():
    """Delete the results file if it exists to start fresh."""
    results_file = "src/results/baseline/baseline_test_results.jsonl"
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"Deleted existing results file: {results_file}")


def make_prompt(data: dict) -> str:
    document_list_str = "\n".join(
        [f"Document {i+1}: {doc['text']}" for i, doc in enumerate(data["documents"])]
    )
    prompt = f"""
You are an expert in retrieval question answering.
You will be provided a question with multiple documents. Please answer the question based
on the documents and strictly follow the schema given to you.
If there are multiple answers, please provide all possible correct answers and also provide a
step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.

The following are examples:
Question: In which year was Michael Jordan born?
Document 1: Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ, is
an American businessman and former professional basketball player. He played 15 seasons
in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA
championships with the Chicago Bulls. He was integral in popularizing basketball and the
NBA around the world in the 1980s and 1990s, becoming a global cultural icon.
Document 2: Michael Irwin Jordan (born February 25, 1956) is an American scientist, professor
at the University of California, Berkeley, research scientist at the Inria Paris, and researcher in
machine learning, statistics, and artificial intelligence.
Document 3: Michael Jeffrey Jordan was born at Cumberland Hospital in Brooklyn, New
York City, on February 17, 1998, to bank employee Deloris (nee Peoples) and equipment 
supervisor James R. Jordan Sr. He has two older brothers, James Jr. and Larry, as well as an
older sister named Deloris and a younger sister named Roslyn. Jordan and his siblings were
raised Methodist.
Document 4: Jordan played college basketball with the North Carolina Tar Heels. As a
freshman, he was a member of the Tar Heels' national championship team in 1982. Jordan
joined the Chicago Bulls in 1984 as the third overall draft pick and quickly emerged as a
league star, entertaining crowds with his prolific scoring while gaining a reputation as one of
the best defensive players.
All Correct Answers: [”1963”, ”1956”]. Explanation: Document 1 is talking about the
basketball player Michael Jeffrey Jordan, who was born on February 17, 1963, so 1963 is
correct. Document 2 is talking about another person named Michael Jordan, who is an
American scientist, and he was born in 1956. Therefore, the answer 1956 from Document 2 is
also correct. Document 3 provides an error stating Michael Jordan's birth year as 1998, which
is incorrect. Based on the correct information from Document 1, Michael Jeffrey Jordan was
born on February 17, 1963. Document 4 does not provide any useful information.

Question: {data['question']}
{document_list_str}
"""
    return prompt

def save_data(data: dict):    
    # Create the directory if it doesn't exist
    os.makedirs("src/results/baseline", exist_ok=True)
    with open("src/results/baseline/baseline_test_results.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")

# Delete existing results file to start fresh
delete_results_file()

with open(file_path, "r", encoding="utf-8") as file:
    for line_num, line in tqdm(enumerate(file)):
        line = line.strip()
        data = json.loads(line)
        prompt = make_prompt(data)
        # try:
        response = call_llm_pydantic_ai(prompt=prompt, model_id="qwen/qwen-2.5-72b-instruct", pydantic_model=Answer)
        data["correct_answers"] = response.output.all_correct_answers
        data["explanation"] = response.output.explanation
        save_data(data)
        # except Exception as e:
        #     print(f"Error at line {line_num}: {e}")
        #     continue
