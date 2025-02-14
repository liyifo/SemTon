import json
import random
from tqdm import tqdm
from openai import OpenAI
import logging
name = 'llmv2'
print(name)
# 配置log文件将控制台输出保存到文件
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"log/{name}.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger()


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

client = OpenAI(
    api_key='EMPTY',
    base_url='http://127.0.0.1:8000/v1',
)
model_type = client.models.list().data[0].id
logger.info(f'model_type: {model_type}')
patients = []
patient_dict = {}
# 读取 CSV 文件
data_path = 'data/llm_test.jsonl'
output_path = f'output/{name}.jsonl'
# data = json.load(open(data_path, 'r', encoding='utf-8'))
data = read_jsonl(data_path)
logger.info(f'总病历数：{len(data)}')
# 使用tqdm显示进度条
output_file = ''
for idx, item in tqdm(enumerate(data)):
    query = item['query']
    logger.error(f'=========================={id}=====================================\n\n') 
    # logger.info(f'查询：{query}')
    messages = []
    messages.append({
        'role': 'user',
        'content': query
    })
    count = 0
    resp = client.chat.completions.create(
        model=model_type,
        messages=messages,
        max_tokens=256,
        seed=42
    )
    response = resp.choices[0].message.content
    herb_list = response.split(',')

    data_item = {
        'query':query,
        'pred':herb_list,
    }
    logger.info(f'预测结果：{herb_list}')
    writer = open(output_path, 'a', encoding='utf-8')
    writer.write(json.dumps(data_item, ensure_ascii=False) + '\n')
    writer.close()