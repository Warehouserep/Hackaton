#!bank/bin/python3
import json
import torch

from flask import abort, Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = 'alignment-checkpoint-18'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)


@app.route('/assist', methods=['POST'],)
def assist():
    try:
        base_question = request.json['query']

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        assistant_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        questions = [
            base_question + ' Почему ты так думаешь?',
            base_question + ' Убеди меня, что ты прав.',
            base_question + ' Напиши, как бы ты отвечал по шагам, лучше всего списком.',
            base_question + ' Дай ответ и усомнись в нем.',
            base_question + ' Сформулируй ответ не более, чем в 10 словах.',
        ]
        answers = []

        for question in questions:
            inputs = tokenizer(question, return_tensors='pt',).to(DEVICE)
            outputs = model.generate(**inputs, prompt_lookup_num_tokens=9, use_cache=True, max_new_tokens=350)
            answers.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        response = {
            "thoughts": {
                "text": answers[0],
                "reasoning": answers[1],
                "plan": answers[2],
                "criticism": answers[3],
                "speak": answers[4]
            },
            "command": {
                "name": "command name",
                "args": {
                    "arg name": "value"
                }
            }
        }
        return jsonify(response), 200
    except:
        abort(422)


if __name__ == '__main__':
    app.run(debug=True)
