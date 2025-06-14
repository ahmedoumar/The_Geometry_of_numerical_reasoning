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
from tqdm import tqdm
import itertools
import re
import argparse
from typing import List, Iterator


# seed everything
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed for reproducibility
seed = 42
set_seed(seed)

population_prompts = [
    "Is the population of {entity_x} larger than that of {entity_y}? Answer only Yes or No.",
    "Does {entity_x} have a bigger population than {entity_y}? Only respond with Yes or No.",
    "Is {entity_x} more populous compared to {entity_y}? Give just Yes or No.",
    "Does {entity_x} have a greater population size than {entity_y}? Reply with only Yes or No.",
    "Is the population of {entity_x} higher than that of {entity_y}? Answer with Yes or No, nothing else.",
    "Does {entity_x} have more people living in it than {entity_y}? Only Yes or No, please.",
    "Is the population count of {entity_x} larger than {entity_y}? Only Yes or No.",
    "Is the number of inhabitants in {entity_x} greater than in {entity_y}? Give only Yes or No.",
    "Does {entity_x} have a larger population than {entity_y}? Just Yes or No, no extras.",
    "Is {entity_x} more heavily populated than {entity_y}? Answer only Yes or No."
]

death_prompts = [
    "Did {entity_x} die before {entity_y}? Answer with Yes or No.",
    "Did {entity_x} pass away earlier than {entity_y}? Respond with Yes or No.",
    "Was {entity_x}'s death prior to {entity_y}? Provide only Yes or No.",
    "Did {entity_x} pass on before {entity_y}? Answer Yes or No.",
    "Did {entity_x} die first compared to {entity_y}? Respond only with Yes or No.",
    "Was {entity_x}'s death earlier than {entity_y}'s? Answer with Yes or No.",
    "Did {entity_x} precede {entity_y} in death? Reply only with True or False.",
    "Did {entity_x} pass before {entity_y}? Respond only with True or False.",
    "Did {entity_x} die earlier than {entity_y}? Answer only with Yes or No.",
    "Did {entity_x} pass away first compared to {entity_y}? Reply with Correct or Incorrect."
]


latitude_prompts = [
    "Is {entity_x} located at a higher latitude than {entity_y}? Answer Yes or No.",
    "Is {entity_x} farther north than {entity_y}? Answer Yes or No.",
    "Does {entity_x} have a higher latitude value than {entity_y}? Answer Yes or No.",
    "Comparing latitudes, is {entity_x} north of {entity_y}? Answer Yes or No.",
    "In terms of latitude, is {entity_x} above {entity_y}? Answer Yes or No.",
    "Is the latitude of {entity_x} greater than the latitude of {entity_y}? Answer Yes or No.",
    "Geographically, is {entity_x} at a more northern latitude than {entity_y}? Answer Yes or No.",
    "Does {entity_x} have a more northerly latitude compared to {entity_y}? Answer Yes or No.",
    "Is {entity_x} positioned at a latitude north of {entity_y}? Answer Yes or No.",
    "Considering only latitude, is {entity_x} more northward than {entity_y}? Answer Yes or No."
]



# Sanity check prompts: Experiment 6
sanity_check_prompts = [
    "When comparing dates, is {year_x} prior to {year_y}? Give a Yes or No answer.",
    "Is someone born in {year_x} older than someone born in {year_y}? Answer only Yes or No",
    "Was the year {year_x} chronologically before {year_y}? Answer only Yes or No.",
    "Did {year_x} occur earlier in history than {year_y}? Respond with Yes or No.",
    "In the timeline of years, does {year_x} come before {year_y}? Answer Yes or No.",
    "Chronologically speaking, is {year_x} prior to {year_y}? Reply with Yes or No only.",
    "When comparing dates, is {year_x} earlier than {year_y}? Provide a Yes or No answer.",
    "On a historical timeline, would {year_x} appear to the left of {year_y}? Yes or No?",
    "In terms of time progression, does {year_x} precede {year_y}? Answer Yes or No.",
    "If arranging years in order, would {year_x} come before {year_y}? Respond Yes or No.",
    "Is the numerical value of year {year_x} less than that of year {year_y}? Yes or No only.",
    "In the sequence of time, is {year_x} followed by {year_y}? Answer with Yes or No."
]
sanity_check_prompts_birthyears = [
    "Is a person born in {year_x} older than one born in {year_y}? Answer Yes/No",
    "Between someone born {year_x} and {year_y}, is the first older? Respond Yes/No",
    "Does {year_x} indicate an earlier birth year than {year_y}? Reply Yes/No",
    "{year_x} birth vs {year_y} birth: Is the former older? Answer Yes/No",
    "Born {year_x} older than born {year_y}? Give only Yes or No",
    "Comparing birthdates: {year_x} before {year_y}? Respond exclusively Yes/No",
    "Is {year_x} an earlier birth year than {year_y}? Answer with a single word: Yes/No",
    "{year_x} birth older than {year_y} birth? Provide only a Yes or No response.",
    "Born in {year_x} vs {year_y}: First one older? Reply with just Yes/No",
    "Birthdate comparison: {year_x} earlier than {year_y}? Answer with only Yes/No"
]



birth_prompts = [
    "Did {entity_x} come into the world earlier than {entity_y}? Answer with Yes or No.",
    "Is {entity_x}'s birthdate before {entity_y}'s? Respond with Yes or No.",
    "Was {entity_x} born prior to {entity_y}? Output only Yes or No.", # 82% Accuracy
    "Did {entity_x} enter life before {entity_y}? Answer with Yes or No.",
    "Was {entity_x}'s birth earlier than {entity_y}'s? Output only Yes or No.", #69%
    "Was {entity_x} born first compared to {entity_y}? Respond with Yes or No.",
    "Is {entity_x} older than {entity_y}? Reply only with True or False.",
    "Did {entity_x} precede {entity_y} in birth? Respond only with True or False.",
    "Did {entity_x} arrive before {entity_y}? Answer only with True or False.",
    "Is {entity_x} senior to {entity_y}? Reply only with Correct or Incorrect.",
]

two_shot_prompt = ["Was Albert Einstein born earlier than Barack Obama? Yes",
"Was Donald Trump born earlier than Isaac Newton? No",
"Was {entity_x} born earlier than {entity_y}? "]

two_shot_prompt_string = "\n".join(two_shot_prompt)


cot_prompts = [
    "Did {entity_x} come into the world earlier than {entity_y}? think step by step and Answer only with Yes or No.",
    "Is {entity_x}'s birthdate before {entity_y}'s? think step by step and Respond only with Yes or No.",
    "Was {entity_x} born prior to {entity_y}? think step by step and Output only Yes or No.", # 66% Accyracy
    "Did {entity_x} enter life before {entity_y}? think step by step and Answer only with Yes or No.",
    "Was {entity_x}'s birth earlier than {entity_y}'s? think step by step and  Output only Yes or No.", # 69.5% Accuracy
    "Was {entity_x} born first compared to {entity_y}? think step by step and  Respond only with Yes or No.",
    "Is {entity_x} older than {entity_y}? think step by step and  Reply only with True or False.", # 73.5% Accuracy
    "Did {entity_x} precede {entity_y} in birth? think step by step and  Respond only with True or False.",
    "Did {entity_x} arrive before {entity_y}? think step by step and  Answer only with True or False.",
    "Is {entity_x} senior to {entity_y}? think step by step and  Reply only with Correct or Incorrect.",
]
    

def generate_comparison_column(df, template=latitude_prompts, prompt_number=0, n_samples=5000, seed=seed):
    """ 
    n_samples: number of Yes/No combinations to generate. 
               This function will generate a balanced dataset with equal Yes and No answers.
    """
    random.seed(seed)
    pairs = list(itertools.combinations(df.index, 2))
    random.shuffle(pairs)

    exp2_prompts = []
    answers = []
    
    entity1_birth = []
    entity2_birth = [] 

    entity1 = []
    entity2 = []

    selected_prompt = template[prompt_number]
    
    yes_count = 0
    no_count = 0
    target_yes_no_count = n_samples // 2  # Half Yes, half No
    
    for (i, j) in tqdm(pairs):
        if yes_count >= target_yes_no_count and no_count >= target_yes_no_count:
            break  # Stop once both Yes and No counts reach the target

        entity_x = df.at[i, 'entity_label']
        entity_y = df.at[j, 'entity_label']

        year_x = df.at[i, 'value']
        year_y = df.at[j, 'value']

        prompt = selected_prompt.format(entity_x=entity_x, entity_y=entity_y)
        answer = 'Yes' if year_x < year_y else "No"

        # Only add to the lists if we still need more 'Yes' or 'No' answers
        if answer == 'Yes' and yes_count < target_yes_no_count:
            yes_count += 1
        elif answer == 'No' and no_count < target_yes_no_count:
            no_count += 1
        else:
            continue  # Skip if we've reached the target for this answer

        exp2_prompts.append(prompt)
        answers.append(answer)
        
        entity1_birth.append(year_x)
        entity2_birth.append(year_y)

        entity1.append(entity_x)
        entity2.append(entity_y)

    # Create the final DataFrame
    exp2_df = pd.DataFrame({
                'exp2_prompt': exp2_prompts,
                'answer': answers,
                'entity1': entity1,
                'entity2': entity2,
                'entity1_value': entity1_birth,
                'entity2_value': entity2_birth
    })

    # Shuffle the balanced df
    balanced_df = exp2_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return balanced_df



def create_chat_batches(df: pd.DataFrame, column_name: str, batch_size: int = 5) -> Iterator[List]:

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")

    total_rows = len(df)
    
    chat_histories = [
        [
            {'role': 'user',
             'content': prompt}
        ] for prompt in df[column_name]
    ]

    num_batches = math.ceil(len(chat_histories) / batch_size)
    batches = [
        chat_histories[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]
    return batches



def tokenize(batches):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_batches = []
    print('===Tokenization===')
    for batch in tqdm(batches):
        tokenized_batches.append(tokenizer.apply_chat_template(batch, padding=True, add_generation_prompt=True,
                                               truncation=True, return_tensors='pt'))
    return tokenized_batches



def inference(tokenized_batches, model):
    
    generated_texts = []
    #output_sequences = []
    print('===Inference===')
    for inputs in tqdm(tokenized_batches):
        # Tokenize and generate for this batch
        #inputs = 
        inputs = inputs.to(model.device)
        with torch.no_grad():
            output_sequences = model.generate(inputs, max_new_tokens=150
                                             )
            
        # Empty cache
        #torch.cuda.empty_cache()
        decoded = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print("Output sequence: ", decoded)
        print("=====")
        generated_texts.append(decoded)

    return generated_texts


# Function to clean the column
def clean_yes_no(column):
    return column.str.strip().str.replace(r'[^A-Za-z]', '', regex=True).str.capitalize().replace({'Yes': 'Yes', 'No': 'No'})





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data processing script for birth or death predictions")
    parser.add_argument("-n", "--number", type=int, default=1, help="prompt number, 0-indexed")
    parser.add_argument("-t", "--task", type=str, choices=['birth', 'death', 'population'], required=True, help="Task type: 'birth' or 'death'")
    
    args = parser.parse_args()
    i = args.number
    task = args.task

    #device = torch.cuda.current_device()

    # Model name
    model_name = "google/gemma-2-9b-it"
    
    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    
    # get the model
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map='auto', 
                                                 )
    
    device = model.device
    print(f"*****====Device: {device} ===****")
    # Load data based on task
    #data = pd.read_csv(f'./more_unique_entities_llama3_exp1_{task}_preds_on_cleaned_bigger_dataset.csv')
    data = pd.read_csv(f'gemma_exps/more_unique_entities_gemma_exp1_{task}_preds_on_cleaned_bigger_dataset.csv')
    
    # Generate the comparisons for experiment 2: P(X, Y)
    prompts = globals()[f"{task}_prompts"]  # This assumes you have birth_prompts and death_prompts defined
    yes_no_df = generate_comparison_column(data, template=prompts, prompt_number=i)
    
    # Experiment 4: Consistency check: P(Y, X)
    #yes_no_df['exp4_prompt'] = [prompts[i].format(entity_x=entity_x, entity_y=entity_y)
    #                            for entity_x, entity_y in zip(yes_no_df['entity2'], yes_no_df['entity1'])]
    
    #yes_no_df['exp4_answer'] = ['Yes' if value=='No' else 'No' for value in yes_no_df.answer.tolist()]
    
    p_x_y_batches = create_chat_batches(yes_no_df, 'exp2_prompt', batch_size=1)
    #p_y_x_batches = create_chat_batches(yes_no_df, 'exp4_prompt', batch_size=1)
    
    tokenized_p_x_y_batches = tokenize(p_x_y_batches)
    #tokenized_p_y_x_batches = tokenize(p_y_x_batches)

    p_x_y_preds = inference(tokenized_p_x_y_batches, model)
    #p_y_x_preds = inference(tokenized_p_y_x_batches, model)
    
    p_x_y_preds = pd.DataFrame(p_x_y_preds, columns=['exp2_prompt_plus_pred'])
    #p_x_y_preds[['exp2_prompt', 'exp2_pred']] = p_x_y_preds['exp2_prompt_plus_pred'].str.split('assistant\n\n', expand=True)
    
    #p_y_x_preds = pd.DataFrame(p_y_x_preds, columns=['exp4_prompt_plus_pred'])
    #p_y_x_preds[['exp4_prompt', 'exp4_pred']] = p_y_x_preds['exp4_prompt_plus_pred'].str.split('assistant\n\n', expand=True)
    #p_y_x_preds.drop(columns=['exp4_prompt_plus_pred'], inplace=True)
    
    yes_no_df['exp2_pred'] = p_x_y_preds['exp2_prompt_plus_pred'].tolist()
    #yes_no_df['exp4_pred'] = p_y_x_preds['exp4_pred'].tolist()

    yes_no_df.to_csv(f'gemma_exps/yes_no_experiment/{task}_exp2_exp4_prompt_{i}_preds_on_cleaned_bigger_dataset.csv')
