from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json

# CUDA_VISIBLE_DEVICES=0,1 python chat.py

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./SemTon-TMD", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("./SemTon-TMD")


# 数据格式与Appendix中的Prompt一致，在json里是列表
with open("./data/llm_input.json", "r") as json_file:
    data = json.load(json_file)



print('数据长度：', len(data))

for index, item in enumerate(data):
    print(f'{index}/{len(data)}')
    id = item['id']
    messages = item['message']

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as e:
        print(id, e)
        continue
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    data[index]['output'] = output_text
    print(f'{id} result: {output_text}')

with open('./output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)