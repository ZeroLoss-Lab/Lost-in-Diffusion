# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import openai
import random
import time
import requests
import sys

if "GEMINI_API_KEY" in os.environ:
    openai.api_key = os.environ["GEMINI_API_KEY"]
if "GEMINI_API_URL" in os.environ:
    openai.api_base = os.environ["GEMINI_API_URL"]

'''
NOTE: 
    Available functions:
        - call_vllm_api: using vllm self-served models
        - openai_generate: using openai models
'''
########################################################################################################
def custom_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):

    raise NotImplementedError()

def generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):

    # TODO: You need to use your own inference method
    # return custom_api(prompt, model, temperature, top_p, max_tokens, port)
    if 'LLaDA' in model:
        return call_llada_api(prompt, model, temperature, top_p, max_tokens, port, i)
    if 'Dream' in model:
        return call_dream_api(prompt, model, temperature, top_p, max_tokens, port, i)
    if 'Fast_dLLM' in model:
        return call_fast_dllm_api(prompt, model, temperature, top_p, max_tokens, port, i)
    if model.startswith("gemini-"):
        return call_gemini_api(prompt, model, temperature, top_p, max_tokens)
    return call_vllm_api(prompt, model, temperature, top_p, max_tokens, port, i)

CUSTOM_SERVER = "0.0.0.0" # you may need to change the port

model_map = {   'meta-llama/Llama-3.1-405B-Instruct-FP8': {'name': 'llama3.1_405B',
                                                            'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
                'meta-llama/Llama-3.3-70B-Instruct': {'name': 'llama3.3_70B',
                                                    'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
                'meta-llama/Llama-3.1-70B-Instruct': {'name': 'llama3.1_70B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                'mistralai/Mistral-7B-Instruct-v0.2': {'name': 'mistral7B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                "mistralai/Mistral-Nemo-Instruct-2407": {'name': 'Mistral-Nemo-Instruct-2407',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                "Qwen/Qwen2.5-3B-Instruct": {'name': 'Qwen2.5-3B-Instruct',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:9000/v1"],
                                                    },
                "Qwen/Qwen2.5-7B-Instruct": {'name': 'Qwen2.5-7B-Instruct',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:9002/v1"]
                },
                "Qwen/Qwen2.5-7B": {'name': 'Qwen2.5-7B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:9002/v1"]
                },
                'meta-llama/Meta-Llama-3-8B-Instruct': {'name': 'llama3_8B',
                                        'server_urls': [f"http://{CUSTOM_SERVER}:9003/v1"],
                                    },
                'meta-llama/Meta-Llama-3-8B': {'name': 'llama3_8B',
                                        'server_urls': [f"http://{CUSTOM_SERVER}:9003/v1"],
                                    },
                'GSAI-ML/LLaDA-8B-Instruct': {'name': 'llada_8b',
                                              'server_urls': [f"http://{CUSTOM_SERVER}:8088/v1"],
                                             },
                'GSAI-ML/LLaDA-8B-Base': {'name': 'llada_8b',
                                              'server_urls': [f"http://{CUSTOM_SERVER}:8088/v1"],
                                             },
                'Dream-org/Dream-v0-Instruct-7B': {'name': 'dream_7b',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:8067/v1"]
                                                },
                'Dream-org/Dream-v0-Base-7B': {'name': 'dream_7b',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:8067/v1"]
                                                },
                'Dream-org/Dream-v0-Base-7B-2': {'name': 'dream_7b',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:9067/v1"]
                                                },
                'Efficient-Large-Model/Fast_dLLM_v2_7B': {'name': 'fast_dllm_7b',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:8070/v1"]
                                                },
                'meta-llama/Llama-3.1-8B-Instruct': {'name': 'llama3.1_8B',
                                        'server_urls': [f"http://{CUSTOM_SERVER}:8071/v1"],
                                    },
                'inclusionAI/LLaDA2.0-mini': {'name': 'llada2.0_mini',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:9090/v1"]
                                                },
                'inclusionAI/LLaDA2.0-mini-2': {'name': 'llada2.0_mini',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:9091/v1"]
                                                },
                'inclusionAI/LLaDA2.0-mini-3': {'name': 'llada2.0_mini',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:9092/v1"]
                                                },
                'inclusionAI/Ling-mini-2.0': {'name': 'ling_mini_2',
                                                'server_urls': [f"http://{CUSTOM_SERVER}:9093/v1"]
                                                },
                                            
            }
########################################################################################################

def call_llada_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    if port == None:
        port = model_map[model]["server_urls"][i]
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "steps": max_tokens,
        "gen_length": max_tokens,
        "block_length": 32,
        "temperature": temperature,
        "cfg_scale": 0.0,
        "remasking": "low_confidence"
    }
    
    try:
        # print(f"Calling LLaDA API at {port} with payload: {payload}")
        # sys.stdout.flush()
        response = requests.post(f"{port}/", json=payload)
        response.raise_for_status()
        # print(response.json()["choices"][0]["message"]["content"])
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLaDA API: {e}")
        return ""

def call_dream_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    if port == None:
        port = model_map[model]["server_urls"][i]
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "steps": max_tokens,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 0,
        "alg": "entropy",
        "alg_temp": 0.1
    }
    
    try:
        response = requests.post(f"{port}/", json=payload)
        response.raise_for_status()
        # print(response.json())
        # print(response.json()["response"])
        return response.json()["response"]

    except Exception as e:
        print(f"Error calling Dream API: {e}")
        return ""

def call_vllm_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    if port == None:
        port = model_map[model]["server_urls"][i]

    client = openai.OpenAI(
        base_url=f"{port}",
        api_key="NOT A REAL KEY",
    )
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return completion.choices[0].text


def openai_generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    # Create a client object
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_KEY"],
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content


def call_gemini_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    client = openai.OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url=os.environ.get("GEMINI_API_URL"),
    )

    max_retries = 10
    base_delay = 2
    max_delay = 600

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                input=prompt,
            )
            if attempt > 5:
                print("input prompt:", prompt)
                print("API response:", response)
            if response and response.output_text:
                return response.output_text
        except Exception as e:
            print(f"Error during API call (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"Retrying API call (attempt {attempt + 1}) in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    print("Failed to get a valid response after multiple attempts.")
    return "NO_RESPONSE"

def call_fast_dllm_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    if port == None:
        port = model_map[model]["server_urls"][i]
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "block_size": 32,
        "small_block_size": 32,
        "max_new_tokens": max_tokens,
        "use_block_cache": False,
        "threshold": 0.9,
        "top_p": top_p,
        "temperature": temperature
    }
    
    try:
        response = requests.post(f"{port}/", json=payload)
        response.raise_for_status()
        # print(response.json()["response"])
        return response.json()["response"]
    except Exception as e:
        print(f"Error calling Fast-dLLM API: {e}")
        return ""