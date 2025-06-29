import json
from tqdm import tqdm



def evaluate(file_path: str):
    correct_count = 0
    with open(file_path, "r", encoding="utf-8") as file:
        total_count = len(file.readlines())
        file.seek(0)
        for line_num, line in tqdm(enumerate(file)):
            line = line.strip()
            data = json.loads(line)
            llm_answers = [answer.lower() for answer in data["correct_answers"]]
            gold_answers = [answer.lower() for answer in data["gold_answers"]]
            wrong_answers = [answer.lower() for answer in data["wrong_answers"]]
            print(llm_answers)
            print(gold_answers)
            print(wrong_answers)
            
            # Check if llm_answers contains all gold answers and no wrong answers
            all_gold_included = all(gold in llm_answers for gold in gold_answers)
            no_wrong_included = all(wrong not in llm_answers for wrong in wrong_answers)
            
            if all_gold_included and no_wrong_included:
                correct_count += 1
                print(f"Correct answer - all gold included, no wrong answers")
            else:
                print(f"Incorrect answer")

    print(f"Total count: {total_count}")
    print(f"Correct count: {correct_count}")
    print(f"Accuracy: {correct_count / total_count}")

if __name__ == "__main__":
    evaluate("src/results/baseline/baseline_test_results.jsonl")