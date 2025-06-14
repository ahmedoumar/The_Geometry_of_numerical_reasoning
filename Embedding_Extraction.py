import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
import umap
from umap import UMAP

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Union
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from tqdm import tqdm

from typing import Iterator
import math
from itertools import chain
import argparse
from transformers import logging

logging.set_verbosity_error()


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
    #torch.use_deterministic_algorithms(True)
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

# Set the seed for reproducibility
seed = 42
generator = set_seed(seed)









def preprocess_hidden_states(hidden_states):
    """
    Preprocess hidden states to handle NaN and infinite values.
    
    Args:
    hidden_states (np.array): Hidden states array.
    
    Returns:
    np.array: Preprocessed hidden states.
    """
    # Replace infinity with NaN
    hidden_states = np.where(np.isinf(hidden_states), np.nan, hidden_states)
    
    # Reshape to 2D for imputation
    original_shape = hidden_states.shape
    hidden_states_2d = hidden_states.reshape(-1, original_shape[-1])
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    hidden_states_imputed = imputer.fit_transform(hidden_states_2d)
    
    # Reshape back to original shape
    return hidden_states_imputed.reshape(original_shape)

def train_pls_per_layer(hidden_states_path: str, df: str, target_column: str, plot_components=True, n_components: int = 50, test_size: float = 0.1, random_state: int = seed):
    """
    Train a PLS model for each layer of hidden states to predict a binary class.

    Args:
    hidden_states_path (str): Path to the numpy file containing hidden states.
    df_path (str): Path to the CSV file containing the dataframe with the target column.
    target_column (str): Name of the binary target column in the dataframe.
    n_components (int): Number of components for PLS. Default is 2.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
    random_state (int): Random state for reproducibility. Default is 42.

    Returns:
    dict: Dictionary containing PLS models, accuracies, and evaluation metrics for each layer.
    """
    print("Loading data...")
    
    hidden_states = np.load(hidden_states_path)
    
    if hidden_states_path.endswith('.npz'):
        
        hidden_states = hidden_states['data']
        a, b, c, d = hidden_states.shape
        hidden_states = np.transpose(hidden_states, (1, 2, 0, 3)).reshape(-1, a, d)

        
        print("Preprocessing data...")
        n_samples, n_layers, n_features = hidden_states.shape
        
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
        print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")

        hidden_states = preprocess_hidden_states(hidden_states)

        print("After preprocessing:")
        print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
        print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")

        y = np.empty(2*len(df))
        
        if hidden_states_path.endswith('Y_X.npz'):
            y[0::2] = df.entity2_value
            y[1::2] = df.entity1_value
        else:
            y[0::2] = df.entity1_value
            y[1::2] = df.entity2_value
        
        
        
    else:
        
        df = df#pd.read_csv(df_path)

        assert hidden_states.shape[0] == df.shape[0], "Number of samples in hidden states and dataframe don't match"

        print("Preprocessing data...")
        n_samples, n_layers, n_features = hidden_states.shape

        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
        print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")

        hidden_states = preprocess_hidden_states(hidden_states)

        print("After preprocessing:")
        print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
        print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")

        le = LabelEncoder()
        y = le.fit_transform(df[target_column].values)

        print("Label Encoding:")
        for cls in le.classes_:
            print(f"{cls} -> {le.transform([cls])[0]}")

    X_train, X_test, y_train, y_test = train_test_split(hidden_states, y, test_size=test_size, random_state=random_state, ) # stratify=y in case of binary classification(yes/no)

    results = {}
    print('PLS components: ', n_components)
    for layer in tqdm(range(n_layers), desc="Training PLS models for each layer"):
        print(f"\nProcessing layer {layer + 1}/{n_layers}")
        
        X_train_layer = X_train[:, layer, :]
        X_test_layer = X_test[:, layer, :]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_layer)
        X_test_scaled = scaler.transform(X_test_layer)

        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train_scaled, y_train)
            
            
            if hidden_states_path.endswith('npz'):
                
                y_train_pred = pls.predict(X_train_scaled)
                y_test_pred = pls.predict(X_test_scaled)

                train_mse = mean_absolute_error(y_train, y_train_pred)
                test_mse = mean_absolute_error(y_test, y_test_pred)

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                print(f"Layer {layer + 1} - Train MAE: {train_mse:.4f}, Test MAE: {test_mse:.4f}")
                print(f"Layer {layer + 1} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

                results[layer] = {
                    'model': pls,
                    'scaler': scaler,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse
                }
            
            else:
                y_train_pred = (pls.predict(X_train_scaled) > 0.5).astype(int)
                y_test_pred = (pls.predict(X_test_scaled) > 0.5).astype(int)

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                print(f"Layer {layer + 1} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

                train_report = classification_report(y_train, y_train_pred, target_names=le.classes_, output_dict=True)
                test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)

                results[layer] = {
                    'model': pls,
                    'scaler': scaler,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_report': train_report,
                    'test_report': test_report
                }
                
                
            """if plot_components:
                # Get PLS components
                pls_components = pls.transform(X_train_scaled)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(pls_components[:, 0], pls_components[:, 1], c=y_train, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Birth Year')
                plt.xlabel('First PLS Component')
                plt.ylabel('Second PLS Component')
                plt.title(f'Layer {layer + 1}: PLS Components')
                plt.tight_layout()
                plt.show() """
            
            if plot_components:
                # Get PLS components
                pls_components = pls.transform(X_train_scaled)

                plt.figure(figsize=(10, 6))
                plt.hist(pls_components[:, 0], bins=50, edgecolor='black')
                plt.xlabel('First PLS Component')
                plt.ylabel('Frequency')
                plt.title(f'Layer {layer + 1}: Distribution of First PLS Component')
                plt.tight_layout()
                plt.show()
                
                
        except Exception as e:
            print(f"Error in layer {layer + 1}: {str(e)}")
            print(f"X_train_scaled shape: {X_train_scaled.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"X_train_scaled sample:\n{X_train_scaled[:5, :5]}")
            print(f"y_train sample: {y_train[:5]}")
    
    if hidden_states_path.endswith('.npz'):
        return results, None
    
    return results, le

def print_classification_report(report):
    """
    Print a formatted classification report.

    Args:
    report (dict): Classification report dictionary.
    """
    headers = ["precision", "recall", "f1-score", "support"]
    row_format = "{:>12}" * (len(headers) + 1)
    print(row_format.format("", *headers))
    
    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        row = [f"{metrics[h]:.2f}" if h != 'support' else f"{metrics[h]}" for h in headers]
        print(row_format.format(label, *row))
    
    # Print averages
    print()
    for avg in ['accuracy', 'macro avg', 'weighted avg']:
        if avg in report:
            row = [f"{report[avg][h]:.2f}" if h in report[avg] and h != 'support' else f"{report[avg].get(h, '')}" for h in headers]
            print(row_format.format(avg, *row))

def print_layer_results(results):
    """
    Print detailed results for each layer.

    Args:
    results (dict): Dictionary containing results for each layer.
    """
    for layer, layer_results in results.items():
        print(f"\n--- Layer {layer + 1} Results ---")
        print(f"Train Accuracy: {layer_results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {layer_results['test_accuracy']:.4f}")
        
        try:
            print("\nTraining Set Classification Report:")
            print_classification_report(layer_results['train_report'])
        except Exception as e:
            print(f"Error printing training report: {str(e)}")
            print("Training report structure:", layer_results['train_report'])
        
        try:
            print("\nTest Set Classification Report:")
            print_classification_report(layer_results['test_report'])
        except Exception as e:
            print(f"Error printing test report: {str(e)}")
            print("Test report structure:", layer_results['test_report'])
            

            
            


#print_layer_results(results)


def load_and_preprocess_hidden_states(hidden_states_path):
    """
    Load and preprocess hidden states from a file.
    
    Args:
    hidden_states_path (str): Path to the numpy file containing hidden states.
    
    Returns:
    np.array: Preprocessed hidden states.
    """
    hidden_states = np.load(hidden_states_path)
    
    if hidden_states_path.endswith('.npz'):
        hidden_states = hidden_states['data']
        a, b, c, d = hidden_states.shape
        hidden_states = np.transpose(hidden_states, (1, 2, 0, 3)).reshape(-1, a, d)
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
    print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")
    
    hidden_states = preprocess_hidden_states(hidden_states)
    
    print("After preprocessing:")
    print(f"Number of NaN values: {np.isnan(hidden_states).sum()}")
    print(f"Number of infinite values: {np.isinf(hidden_states).sum()}")
    
    return hidden_states

def pls_inverse_transform_global(pls_model, scaler):
    """
    Generate a global vector that maps the 1-dimensional PLS component to the original feature space.

    Args:
    pls_model (PLSRegression): Trained PLS model.
    scaler (StandardScaler): Fitted StandardScaler used for the PLS model.

    Returns:
    numpy.ndarray: A vector representing the global mapping from PLS space to original feature space.
    """
    # The first column of x_weights_ represents the contribution of each feature to the first PLS component
    global_vector = pls_model.x_weights_[:, 0]
    #print('GLOBAL VECTOR SHAPE: ', global_vector.shape)
    # Adjust the global vector for the scaling that was applied
    global_vector *= scaler.scale_
    
    return global_vector

class ActivationPatcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def logit_lens(self, input_text, layer_indices=None):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        if layer_indices is None:
            layer_indices = range(len(self.model.transformer.h))
        
        logits_per_layer = []
        
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            logits = self.model.lm_head(hidden_states)
            logits_per_layer.append(logits)
        
        handles = []
        for i in layer_indices:
            if hasattr(self.model, 'transformer'):
                layer = self.model.transformer.h[i]
            else:
                layer = self.model.model.layers[i]
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        for handle in handles:
            handle.remove()
        
        return logits_per_layer

    def analyze_logit_lens(self, input_text, top_k=5):
        logits_per_layer = self.logit_lens(input_text)
        
        results = []
        for layer, logits in enumerate(logits_per_layer):
            probs = F.softmax(logits[0, -1], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices)
            
            layer_result = {
                "layer": layer,
                "top_tokens": top_tokens,
                "top_probs": top_probs.tolist()
            }
            results.append(layer_result)
        
        return results
    
    def patch_activation(self, token_index, new_value, alpha):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                hidden_states = output
                return_tensor = True
            else:
                hidden_states = output[0]
                return_tensor = False
            patched_hidden_states = hidden_states.clone()
            new_value_tensor = new_value.to(hidden_states.device).expand(hidden_states.size(0), -1)
            patched_hidden_states[:, token_index, :] += alpha * torch.tensor(new_value_tensor, dtype=torch.float32)
            if return_tensor:
                return patched_hidden_states
            else:
                return (patched_hidden_states,) + output[1:]
        return hook

    def run_with_patching(self, input_text, patch_layer, new_value, alpha):
        # Apply chat template
        #chat_input = self.tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
        #input_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids.to(self.device)
        #print(input_ids)
        new_value_tensor = torch.tensor(new_value, dtype=torch.float32, device=self.device).clone().detach()
        
        model.eval()
        
        original_texts, patched_texts = [], []
        
        for input_ids in tqdm(input_text):
            
            #token_index = torch.nonzero(input_ids == 949).squeeze()[-1].item()  - 1
            #print("Token index:", token_index)
            token_indices = (input_ids == 1654).nonzero(as_tuple=True)[1]#(input_ids == 949).nonzero(as_tuple=True)[1]
            if len(token_indices) == 0:
                print(f"Warning: Target token {target_token_id} not found in input.")
                continue
            token_index = token_indices[-1].item() - 1
            # Generate original output
            with torch.no_grad():
                original_output = self.model.generate(input_ids, max_new_tokens=1, 
                                                      do_sample=False, num_beams=1,
                                                     #top_p=None, temperature=None
                                                     )

            # Apply patching
            if patch_layer == "embed":
                layer = self.model.get_input_embeddings()
            else:
                layer_num = patch_layer#int(patch_layer.split("_")[1])
                layer = self.model.transformer.h[layer_num] if hasattr(self.model, 'transformer') else self.model.model.layers[layer_num]

            handle = layer.register_forward_hook(self.patch_activation(token_index, new_value_tensor, alpha))

            # Generate patched output
            with torch.no_grad():
                patched_output = self.model.generate(input_ids, max_new_tokens=1, 
                                                     do_sample=False, num_beams=1,
                                                     #top_p=None, temperature=None
                                                    )

            handle.remove()

            # Decode outputs
            original_text = self.tokenizer.decode(original_output[0], skip_special_tokens=True)
            patched_text = self.tokenizer.decode(patched_output[0], skip_special_tokens=True)
            #print(original_text)
            #print(patched_text)
            original_texts.append(original_text)
            patched_texts.append(patched_text)
        
        return original_texts, patched_texts

    
def tokenize(batches):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_batches = []
    print('===Tokenization===')
    for batch in tqdm(batches):
        
        chat_input = tokenizer.apply_chat_template([{'role': 'user', 'content': batch}], tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(chat_input, return_tensors="pt").input_ids.to(model.device)
        tokenized_batches.append(input_ids)
    return tokenized_batches    
   
def check_unique_values(base_column, df):
    columns = df.columns.tolist()
    columns.remove(base_column)
    
    answer = []
    
    for col in columns:
        overlap = df[df[base_column] != df[col]].index.tolist()
        answer.append(overlap)
    #answer = list(chain(*answer))
    answer = [item for sublist in answer for item in sublist]
    return len(set(answer))

def extract_after_assistant(text):
    match = re.search(r'assistant\s*\n\s*(\w+)', text)
    return match.group(1) if match else None

import warnings
from transformers import logging

# Disable all warnings

warnings.filterwarnings("ignore")

# Or, specifically for Hugging Face transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="A script that accepts one layer integer index.")
    parser.add_argument("-taskid", type=int, help="An integer to be processed")
    parser.add_argument('-vec_type', type=str, help="[pls, random]")
    parser.add_argument('-model', type=str, help="[llama, mistral, gemma, qwen]")
    
    args = parser.parse_args()
    taskid = args.taskid
    vec_type = args.vec_type
    model = args.model
    
    if model=='gemma' and taskid==1:
        new_file="gemma_exps/yes_no_experiment/birth_exp2_exp4_prompt_2_preds_on_cleaned_bigger_dataset_processed_yes_no.csv"
        hidden_states_path = "gemma_birth_hidden_states_filtered_on_cleaned_bigger_dataset.npz"
    
    elif model=='qwen' and taskid==1:
        new_file = "qwen_exps/yes_no_experiment/birth_exp2_exp4_prompt_1_preds_on_cleaned_bigger_dataset_processed_yes_no.csv"
        hidden_states_path = "qwen_birth_hidden_states_filtered_on_cleaned_bigger_dataset.npz"
    
    elif taskid == 1:
        new_file = "exp2_exp4_prompt_2_preds_on_cleaned_bigger_dataset.csv"
        hidden_states_path = "birth_years_hidden_states_filtered_on_cleaned_bigger_dataset.npz"
    elif taskid == 2:
        new_file = "death_exp2_exp4_prompt_2_preds_on_cleaned_bigger_dataset.csv"
        hidden_states_path = "death_years_hidden_states_filtered_on_cleaned_bigger_dataset.npz"
    else:
        new_file = "latitude_exp2_exp4_prompt_2_preds_on_cleaned_bigger_dataset.csv"
        hidden_states_path = "latitude_hidden_states_filtered_on_cleaned_bigger_dataset.npz"
       
        ood_df = pd.read_csv('latitude_exp2_exp4_prompt_9_preds_on_cleaned_bigger_dataset.csv')
        ood_df['exp2_prompt'] = ood_df['exp2_prompt'].apply(lambda x: re.sub(r'(?<!\s)(?=[.)])?\?', r' ?', x))
        yes_ = ood_df[ood_df['answer'] == 'Yes'].sample(n=50, random_state=seed)
        no_ = ood_df[ood_df['answer'] == 'No'].sample(n=50, random_state=seed)
        new_df = pd.concat([yes_, no_],)
    
    # Example texts
    #old_file = 'exp2_exp4_prompt_2_preds.csv'
    #new_file = "latitude_exp2_exp4_prompt_2_preds_on_cleaned_bigger_dataset.csv"
    df = pd.read_csv(new_file)
    
    df['exp2_prompt'] = df['exp2_prompt'].apply(lambda x: re.sub(r'(?<!\s)(?=[.)])?\?', r' ?', x))
    #print(len(df[(df['answer'] == 'Yes') & (df['answer'] != df['exp2_pred'])]))
    # for test on out of distribution data
    yes_ = df[(df['answer'] == 'Yes') & (df['answer'] != df['exp2_pred'])].sample(n=50, replace = True, random_state=seed)
    no_ = df[(df['answer'] == 'No') & (df['answer'] != df['exp2_pred'])].sample(n=50, replace = True, random_state=seed)
    new_df = pd.concat([yes_, no_],)
    #print("Length of OOD File:", len(new_df))
    if taskid != 3:
        df = df[df['exp2_pred'] == df['answer'].str.lower()]
    
    #df = df.drop_duplicates(subset='entity1')
    #df = df.drop_duplicates(subset='entity2')

    

    #df = df[:10]

    df.head()
    
    
    model_name = "google/gemma-2-9b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')#.to('cuda')

    print("Starting PLS training process for each layer...")
    

    #df_path = "your_dataframe.csv"
    target_column = "answer"

    results, label_encoder = train_pls_per_layer(hidden_states_path, df, target_column)

    print("\nPLS training completed for all layers.")
    
    

    tokenized_batches = tokenize(new_df.exp2_prompt)
    
    
    hidden_size = model.config.hidden_size  
    num_layers = int(model.config.num_hidden_layers)
    random_global_vector = torch.randn(hidden_size, generator=generator).to(model.device)

     

    answers_df = pd.DataFrame()
    
    all_originals, all_patcheds = [], []

    alphas =  torch.arange(-30, 30, 1.5)#[:25]#[-30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, 
               #-8,  -6,  -4, -2,   -1,  2,   4,   6,   8,  10,  12,  14,  16,  18] # #torch.arange(-20, 20, 1.5) 

    for layer in tqdm(range(0, num_layers), desc='Progress bar of Layers: '):
        print(f'\t===Intervening at Layer: {layer}\t===')
        all_originals, all_patcheds = [], []
        answers_df = pd.DataFrame()

        pls_model = results[layer]['model']
        scaler = results[layer]['scaler']

        # Generate the global vector
        pls_global_vector = pls_inverse_transform_global(pls_model, scaler)
        norm = torch.norm(torch.tensor(eval(f"{vec_type}_global_vector"), dtype=torch.float32, device=model.device))

        for alpha in tqdm(alphas, desc='Progress bar of Alphas: '):


            patcher = ActivationPatcher(model, tokenizer)
            original, patched = patcher.run_with_patching(tokenized_batches, layer, eval(f"{vec_type}_global_vector"), alpha/norm)
            all_originals.append(original)
            all_patcheds.append(patched)


            # Assuming you have the lists 'original' and 'patched'
            answers_df[f'patched_{alpha}'] = patched#[extract_after_assistant(text) for text in patched]

            

            answers_df['original'] = original #[extract_after_assistant(text) for text in original]

            print(f"Number of flipped values for Alpha {alpha}: ", check_unique_values('original', answers_df))
        print(f"====== Saving file: ./gemma_exps/task{taskid}/task{taskid}_{vec_type}_vec_layer_{layer}.csv ==== ")
        answers_df.to_csv(f'./gemma_exps/task{taskid}/task{taskid}_{vec_type}_vec_layer_{layer}.csv')
