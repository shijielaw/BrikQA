import json


def evaluate(input_f, output_f):
    total = 0
    correct = 0
    format_errors = 0
    empty_responses = 0

    with open(input_f, "r", encoding="utf-8") as file:
        data = json.load(file)

        for item in data:
            answer = item.get("answer", "").strip()
            prediction = item.get("prediction", "").strip()

            if not prediction:
                empty_responses += 1
                continue

            if answer == prediction:
                correct += 1

            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    with open(output_f, "w", encoding="utf-8") as fw:
        fw.write("=" * 40 + "\n")
        fw.write(f"File dir: \t{input_f}" + "\n")
        fw.write(f"Line Count: \t{len(data)}" + "\n")
        fw.write("-" * 40 + "\n")
        fw.write(f"Valid: \t{total}" + "\n")
        fw.write(f"Error: \t{format_errors}" + "\n")
        fw.write(f"Null: \t{empty_responses}" + "\n")
        fw.write("=" * 40 + "\n")
        fw.write(f"Correct: \t{correct}" + "\n")
        fw.write(f"Accuracy: \t{accuracy:.2f}%" + "\n")
        fw.write("=" * 40 + "\n")

    print("=" * 40)
    print(f"File dir: \t{input_f}")
    print(f"Line Count: \t{len(data)}")
    print("-" * 40)
    print(f"Valid: \t{total}")
    print(f"Error: \t{format_errors}")
    print(f"Null: \t{empty_responses}")
    print("=" * 40)
    print(f"Correct: \t{correct}")
    print(f"Accuracy: \t{accuracy:.2f}%")
    print("=" * 40)
