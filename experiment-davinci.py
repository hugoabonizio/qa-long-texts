import os
import openai
import re
import sys
import json
import random
random.seed(1)
from tqdm.auto import tqdm
from statistics import mean, stdev
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

def find_index_window(context, window):
    for i in range(len(context)):
        if context[i:i + len(window)] == window:
            return i, i + len(window)
    raise Exception(f'Window "{window}" not found in the context: "{context}"')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

max_total_tokens = 3980

prompt_template = '''Read the following text and answer the question: "{question}".

{article}

Question: {question}
Answer: {answer}'''

prompt_tokens = len(tokenizer.tokenize(prompt_template))

examples = open('QuALITY/QuALITY.v0.9.htmlstripped.dev').readlines()
lengths = [len(tokenizer.encode(json.loads(example)['article'])) for example in tqdm(examples)]
selected_examples = [example for (example, length) in tqdm(zip(examples, lengths)) if length <= (max_total_tokens - prompt_tokens)]
print('mean', mean(lengths), stdev(lengths))
print('selected_examples', len(selected_examples))
sampled = random.sample(selected_examples, 20)

lengths = []

engine = 'text-davinci-002'

with open(f'results/experiment-{engine}.jsonl', 'w') as f:
    for example in tqdm(sampled):
        row = json.loads(example)
        article = row['article'].strip()
        soup = BeautifulSoup(article, features="html.parser")
        article = soup.get_text()
        article = article.replace('\n', ' ').replace('\t', ' ')
        article = re.sub(' +', ' ', article)

        article_id = row['article_id']

        question = random.choice(row['questions'])
        question_text = question['question'].strip()
        question_unique_id = question['question_unique_id']

        answer_tokens = []
        answer_logprobs = []

        for option in question['options']:
            prompt = prompt_template.format(article=article, question=question_text, answer=option.strip())
            lengths.append(len(tokenizer.tokenize(prompt)))

            output = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=1,
                logprobs=1,
                echo=True,
                temperature=0,
                top_p=1.0,
            )
            tokens = output['choices'][0]['logprobs']['tokens']
            logprobs = output['choices'][0]['logprobs']['token_logprobs']

            window = ['\n', 'Answer', ':']
            start_window, end_window = find_index_window(tokens, window)
            answer_tokens.append(tokens[end_window:])
            answer_logprobs.append(logprobs[end_window:])


        f.write(json.dumps({
            'article_id': article_id,
            'question_unique_id': question_unique_id,
            'writer_label': question['writer_label'],
            'gold_label': question['gold_label'],
            'difficult': question['difficult'],

            'answer_1_tokens': answer_tokens[0],
            'answer_1_logprobs': answer_logprobs[0],

            'answer_2_tokens': answer_tokens[1],
            'answer_2_logprobs': answer_logprobs[1],

            'answer_3_tokens': answer_tokens[2],
            'answer_3_logprobs': answer_logprobs[2],

            'answer_4_tokens': answer_tokens[3],
            'answer_4_logprobs': answer_logprobs[3],
        }) + '\n')

print('total', sum(lengths))
print(lengths)
