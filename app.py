from flask import Flask, request, jsonify, render_template
import openai
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import re

client = OpenAI()

app = Flask(__name__)


def find_and_return_match(list1, list2):
    for item in list1:
        if item in list2:
            return item


def get_valid_response(client, message, model_name, answer_string_list, temperature=None):
    while True:
        if temperature is None:
            expert_decision = client.chat.completions.create(
                model=model_name,
                messages=message,
                logprobs=True
            )
        else:
            expert_decision = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                logprobs=True
            )
        reply = expert_decision.choices[0].message.content
        print(reply)
        tokens = re.split(r'\s+|[,.!?;:\'\"]+', reply.lower())

        if any(item in answer_string_list for item in tokens):
            return find_and_return_match(tokens, answer_string_list)

        print(f"Invalid reply '{reply}', generating a new one...")


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/compare', methods=['POST'])
def compare():
    results = []
    method = request.form['generateMethod']
    pair_count = int(request.form.get('pairCount'))
    print(method)
    for i in range(pair_count):
        sentence1 = request.form.get(f'sentence1-{i}')
        sentence2 = request.form.get(f'sentence2-{i}')

        if method == 'huggingface-model':
            label = compare_with_huggingface(sentence1, sentence2)
            print("label")
        else:
            label = gpt_prompt(sentence1, sentence2)
        results.append({'pair_index': i, 'label': label})
    return jsonify(results)


def gpt_prompt(sentence1, sentence2):
    prompt = f"""Task: Determine the relationship between the following premise and hypothesis.

Example 1:
Premise: "All birds can fly."
Hypothesis: "Penguins can fly."
Output: contradiction

Example 2:
Premise: "She adopted a young Labrador."
Hypothesis: "She has a dog."
Output: entailment

Example 3:
Premise: "He moved to New York for a new job."
Hypothesis: "He lives in Los Angeles."
Output: contradiction

Example 4:
Premise: "The sun rises in the east."
Hypothesis: "The sun sets in the west."
Output: neutral

Now, analyze the following statements:
Premise: "{sentence1}"
Hypothesis: "{sentence2}"

What is the relationship between these two statements? Output only one of the following: entailment, neutral, or contradiction."""

    answer_string_list = ["entailment", "neutral", "contradiction"]
    message = [{"role": "user", "content": prompt}]

    reply1 = get_valid_response(client, message, "gpt-3.5-turbo", answer_string_list)
    reply2 = get_valid_response(client, message, "gpt-3.5-turbo-0125", answer_string_list)
    reply3 = get_valid_response(client, message, "gpt-3.5-turbo-1106", answer_string_list)
    reply4 = get_valid_response(client, message, "gpt-4-turbo", answer_string_list)
    reply5 = get_valid_response(client, message, "gpt-4-turbo-preview", answer_string_list)

    replies = [reply1, reply2, reply3, reply4, reply5]
    print(replies)
    most_frequent_reply = max(set(replies), key=replies.count)

    return most_frequent_reply


def compare_with_huggingface(sentence1, sentence2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    premise = sentence1
    hypothesis = sentence2
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(inputs["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    label = max(prediction, key=prediction.get)

    return label





if __name__ == '__main__':
    app.run(debug=True)
