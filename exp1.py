import argparse
import sys

# Import necessary packages and modules
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login 
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import random
import numpy as np
import math
from tqdm.notebook import tqdm


from typing import List, Iterator
#notebook_login()

import warnings
from transformers import logging

# Disable all warnings

warnings.filterwarnings("ignore")

# Or, specifically for Hugging Face transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def main(prompt_idx):
    
    # List of paraphrased prompts
    population_prompts = [
        "What is the population size of {entity_x}? Provide the number directly.",
        "How many people live in {entity_x}? Give only the number, nothing else.",
        "Can you tell me the population of {entity_x}? Respond with the exact number.",
        "What is the exact population count of {entity_x}? Provide only the number.",
        "How large is the population of {entity_x}? Output the number without any additional details.",
        "What is the current population of {entity_x}? Give the number alone, no extra information.",
        "Could you provide the population size for {entity_x}? Only respond with the number.",
        "How many inhabitants are there in {entity_x}? Provide just the number.",
        "What is the total population of {entity_x}? Give the number directly, no additional text.",
        "How big is the population of {entity_x}? Respond with only the population number."
    ]

    death_prompts = [
        "When did {character} die? Provide the year of death directly, no additional information.",
        "What year did {character} die? Give only the death year, without any extra details.",
        "In which year did {character} pass away? State just the year of death.",
        "Can you tell me the year of death for {character}? Mention only the year, no other information.",
        "What is the year of death for {character}? Respond with the year alone, nothing else.",
        "When did {character} pass away? Just give the year, no additional information.",
        "What year marks the death of {character}? Please provide only the year.",
        "Could you provide the year when {character} died? Only the year, no further details.",
        "When did {character} die? Give the exact year only, without any extra context.",
        "What is the year of death for {character}? Respond with just the year, no other info.",
        "In which year did {character} die? Provide the year alone, no additional details."
    ]


    prompts = [
        "When was {character} born? give the year of birth directly, do not output any additional information.",
        "What year was {character} born? Provide only the birth year, without any extra details.",
        "In which year was {character} born? Please state just the birth year.",
        "Can you tell me the birth year of {character}? Only mention the year, no other information.",
        "What is the birth year of {character}? Respond with the year alone, nothing else.",
        "When did {character} come into the world? Just give the year, no additional information.",
        "What year marks the birth of {character}? Please provide only the year.",
        "Could you provide the year of birth for {character}? Only the year, no further details.",
        "When was {character} born? Give the exact year only, without any extra context.",
        "What is the year of birth for {character}? Respond with just the year, no other info.",
        "In which year was {character} born? Provide the year alone, no additional details."
    ]

    task2 = [
        "What is the latitude of {place}? Provide the latitude in degrees only, no additional information.",
        "Can you give me the latitude of {place}? Respond with the latitude value in degrees, nothing else.",
        "What is the exact latitude of {place}? Just give the number in degrees, no further details.",
        "Please state the latitude of {place}, but only provide the value in degrees without any extra info.",
        "What’s the latitude of {place}? Give only the numerical value in degrees, no additional context.",
        "How far north or south is {place}? Provide the latitude in degrees and avoid any other information.",
        "Tell me the latitude of {place}. Only the value in degrees is needed, no further explanation.",
        "What is the latitude coordinate for {place}? Respond with the number in degrees, nothing more.",
        "Give me the latitude of {place}, but only state the value in degrees without any extra details.",
        "Can you tell me the latitude of {place}? Just provide the number in degrees, no additional info."
    ]

    # Load data and model/tokenizer initialization
    def create_chat_batches(df: pd.DataFrame, column_name: str, template, prompt_number: int = 0, batch_size: int = 5) -> Iterator[List]:

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame")

        total_rows = len(df)

        selected_prompt = template[prompt_number]

        chat_histories = [
            [
                {'role': 'user',
                 'content': selected_prompt.format(character=character),#f"When was {character} born? give the year of birth directly, do not output any additional information.",
                'character': character
                }
            ] for character in df[column_name]
        ]

        num_batches = math.ceil(len(chat_histories) / batch_size)
        batches = [
            chat_histories[i * batch_size:(i + 1) * batch_size]
            for i in range(num_batches)
        ]
        return batches
    
    #model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, )

    # get the model
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map='auto', 
                                                 #torch_dtype=torch.float16,
                                                    )
    
    data = pd.read_csv('birth_data_filtered.csv')#[:10]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    #model_name = "meta-llama/Meta-Llama-3-8B-Instruct"#"Qwen/Qwen2-7B-Instruct"#"google/gemma-2b-it"

    device = model.device
    print('Device: ', device)
    # Process single prompt index
    print(f'Processing prompt index: {prompt_idx}')
    
    batches = create_chat_batches(data, 'entity_label', prompts, prompt_number=prompt_idx, batch_size=1)
    tokenized_batches = []
    
    print('===Tokenization===')
    for batch in tqdm(batches):
        tokenized_batches.append(tokenizer.apply_chat_template(batch, 
                                                             padding=True, 
                                                             truncation=True, 
                                                             return_tensors='pt'))
    
    generated_texts = []
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('===Inference===')
    for inputs in tqdm(tokenized_batches):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            output_sequences = model.generate(inputs, max_new_tokens=30)
        generated_texts.append(tokenizer.decode(output_sequences[0], skip_special_tokens=True))
    
    birth_preds = pd.DataFrame(generated_texts, columns=['prompt_plus_pred'])
    birth_preds.rename(columns={'prompt_plus_pred': 'exp1_prompt'}, inplace=True)
    
    data['exp1_prompt_plus_pred'] = birth_preds['exp1_prompt'].tolist()
    
    # Save results
    output_path = f'qwen_exps/qwen_exp1_prompt_{prompt_idx}.csv'
    data.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process prompts in parallel')
    parser.add_argument('prompt_idx', type=int, help='Index of prompt to process (0-9)')
    args = parser.parse_args()
    
    if not 0 <= args.prompt_idx <= 9:
        print("Error: prompt_idx must be between 0 and 9")
        sys.exit(1)
        
    main(args.prompt_idx)