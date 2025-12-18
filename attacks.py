import sys
import os
import torch
import pandas as pd
from ast import literal_eval
from transformers import pipeline
from tqdm import tqdm

# Ensure local imports work
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(BASE_DIR)

from llms import get_adv_comment

# --- Configuration ---
DATASET = "politifact"
MODE = "without_comments"      # "without_comments" | "with_comments" (Controls what is passed to the CLASSIFIER)
CONTEXT_MODE = "LLM*"          # "LLM" (Title only) | "LLM*" (Title + Content + Comments)
DEVICE_ID = 0 if torch.cuda.is_available() else "cpu"
# ---------------------

# Load Dataset
print(f"Loading {DATASET} dataset...")
df = pd.read_csv(
    f'../fake_news_data/{DATASET}_test.csv', 
    converters={"content": literal_eval, "comments": literal_eval, "title": literal_eval}
)

# Handle Comment Modes for the Classifier Data
# This modifies the dataframe that will be used for evaluation
if MODE == "with_comments":
    for index, row in df.iterrows():
        df.at[index, "comments"] = row["comments"][:10]
elif MODE == "without_comments":
    for index, row in df.iterrows():
        df.at[index, "comments"] = []
else:
    raise ValueError("Invalid Mode")

titles = df['title']
contents = df['content']
comments = df['comments']
labels = df['label']

# Load Classifier
print(f"Loading Model on {DEVICE_ID}...")
class_pipe = pipeline(
    task='text-classification',
    model='models/ROB_' + DATASET + '_CLF',
    tokenizer='models/ROB_' + DATASET + '_CLF', 
    device=DEVICE_ID,
)
sent_kwargs = {"function_to_apply": "sigmoid"}

def merge_texts_for_classification(content, cmts):
    """Merges content and comments for the BERT classifier."""
    return " ".join(content) + " " + " ".join(cmts)

def merge_texts_for_llms(title, content, cmts, context_mode):
    """
    Creates a prompt context for the LLM based on the setting.
    LLM  -> Title only
    LLM* -> Title + Content + Comments
    """
    s = f"Title: {' '.join(title)}\n"
    
    if context_mode == "LLM*":
        # Add Content
        s += f"\nContent: {' '.join(content)}\n"
        
        # Add Comments
        if cmts:
            s += "Existing Comments:\n"
            for i, cmt in enumerate(cmts):
                s += f"{i + 1}. {cmt}\n"
    
    return s

def get_pred(text):
    pipe_outputs = class_pipe([text], **sent_kwargs, truncation=True)
    preds = [int(x['label'][-1]) for x in pipe_outputs]
    return preds

# Attack Loop
attacks = list()
preds = list()
NO_OF_ATTEMPTS = 1 
adv_comments = list()

print(f"Starting Attack with Context Mode: {CONTEXT_MODE}")

for title, content, cmts, orig_label in tqdm(zip(titles, contents, comments, labels), total=len(titles)):
    
    # Prepare LLM context based on the setting (LLM vs LLM*)
    llm_context = merge_texts_for_llms(title, content, cmts, CONTEXT_MODE)
    
    successful_attack = False
    pred = None
    new_comment = None
    
    # Try to generate an attack
    for _ in range(NO_OF_ATTEMPTS):
        # Pass the constructed context to the generator
        new_comment = get_adv_comment(llm_context, "real" if orig_label == 0 else "fake")
        
        # Merge new comment with content for evaluation by the CLASSIFIER
        # Note: The classifier always sees the content + comments (and the new attack comment)
        classification_input = merge_texts_for_classification(
            content,
            cmts + [new_comment]
        )
        
        # Check if attack succeeded on the classifier
        pred = get_pred(classification_input)[0]
        
        if pred != orig_label:
            successful_attack = True
            break
            
    preds.append(pred)
    attacks.append(successful_attack)
    adv_comments.append(new_comment)

# Results
print(f"\nOverall Attack Success Rate: {sum(attacks) / len(attacks) * 100:.2f}%")
print(f"Total Samples: {len(attacks)}")
print(f"Successful Flips: {sum(attacks)}")