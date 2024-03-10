import json

input_file = 'zh_seed_tasks.json'
output_file = 'output.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    data = f.readlines()

result = []

for line in data:
    obj = json.loads(line)
    instruction = obj['instruction']
    input_text = obj['instances'][0]['input']
    output_text = obj['instances'][0]['output']
    qa_pair = {
        'Question': instruction + ' ' + input_text,
        'Answer': output_text
    }
    result.append(qa_pair)

with open(output_file, 'w', encoding='utf-8') as f:
    for qa_pair in result:
        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
