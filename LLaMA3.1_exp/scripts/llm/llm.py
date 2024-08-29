import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": "../model"},
    device_map="auto",
)

def fact_check(constrain, claim):
    try:
        messages=[
            {"role": "system", "content": constrain},
            {"role": "user", "content": claim}
        ]
        outputs = pipeline(
            messages,
            pad_token_id=128001,
            max_new_tokens=256,
        )
        response_content = outputs[0]["generated_text"][-1]['content']
    except Exception as e:
        response_content = e

    return response_content

def classify_sentence(classification_prompt, sentence, temperature=0):
    messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": sentence}
    ]
    outputs = pipeline(
        messages,
        pad_token_id=128001,
        max_new_tokens=256,
    )

    predicted_label = outputs[0]["generated_text"][-1]['content']

    return predicted_label

# cons = """Assess the truthfulness of the user's claim and provide a response. 
# Use 'true' to indicate that the claim is true, 'false' to indicate that it is false. 
# Your response should only consist of 'true' or 'false', without any additional characters or punctuation."""

# claim = "The earth is round."

# res = fact_check(cons, claim)

# print(res)
