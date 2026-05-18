STEM_CATEGORIES = {"गणित", "भौतिकी", "रसायन विज्ञान", "जीव विज्ञान"}

def filter_stem(dataset):
    def _filter(example):
        cat = example.get("category", "")
        return cat.strip() in STEM_CATEGORIES if cat else False
    return dataset.filter(_filter)

def get_choices(doc):
    options = doc["options"]
    letters = "ABCDEFGHIJ"
    choices = []
    for i, opt in enumerate(options):
        if i >= len(letters):
            break
        if opt and opt.strip():
            choices.append(letters[i])
    return choices

def format_prompt(doc):
    question = doc["question"].strip()
    options = doc["options"]
    letters = "ABCDEFGHIJ"
    lines = []
    for i, opt in enumerate(options):
        if i >= len(letters):
            break
        if opt and opt.strip():
            lines.append(f"{letters[i]}) {opt.strip()}")
    prompt = f"{question}\n\n" + "\n".join(lines) + f"\n\nAnswer:"
    return prompt

def get_target_index(doc):
    answer_letter = doc["answer"].strip().upper()
    choices = get_choices(doc)
    try:
        return choices.index(answer_letter)
    except ValueError:
        # Should never happen if answer is a valid letter
        return doc["answer_index"]