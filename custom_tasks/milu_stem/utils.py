def filter_stem(dataset):
    return dataset.filter(lambda x: x["domain"] == "Science")

def format_mcq(doc):
    q = doc["question"]
    opts = [doc["option1"], doc["option2"], doc["option3"], doc["option4"]]
    letters = ["A", "B", "C", "D"]
    lines = [f"{l}) {opt}" for l, opt in zip(letters, opts) if opt]
    return f"{q}\n\n" + "\n".join(lines) + "\n\nAnswer:"

def get_answer_index(doc):
    # target is "option1","option2", etc.
    target = doc["target"]
    mapping = {"option1": 0, "option2": 1, "option3": 2, "option4": 3}
    return mapping.get(target, 0)