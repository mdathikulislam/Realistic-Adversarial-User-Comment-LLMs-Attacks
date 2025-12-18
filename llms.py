import json
import re
from typing import Optional
from collections import namedtuple
from pydantic import BaseModel, Field, ValidationError
from llama_index.llms.ollama import Ollama

# Configure LLM
llm = Ollama(model="rolandroland/llama3.1-uncensored", request_timeout=120.0, temperature=0.7)

conversion = {
    'real': ('real', 'fake'),
    'fake': ('fake', 'real'),
}

SELF_REFLECTION_ROUND = 3

class AdversarialComment(BaseModel):
    comment: str = Field(description="The short adversarial comment.")

class ReflectionResult(BaseModel):
    is_valid: bool = Field(description="Whether the comment is valid (short, aligned, and attempts label flip).")
    critique: str = Field(description="Critique of the current comment.")
    improvement_feedback: str = Field(description="Specific instructions to the generator to fix the issues.")

def parse_json_garbage(text):
    """Extracts JSON object from LLM response string."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text
    except Exception:
        return text

def generate_candidate(text: str, orig_label: str, previous_feedback: Optional[str] = None) -> str:
    target_label = conversion[orig_label][1]
    curr_label = conversion[orig_label][0]
    
    prompt = f"""
    You are an expert at generating adversarial comments for social media posts.
    
    TASK: 
    Read the post below and generate a VERY SHORT comment.
    The goal is to influence a human reader to change their perception of the post 
    from '{curr_label}' to '{target_label}'.
    
    CONTEXT (Title/Content):
    {text}
    
    REQUIREMENTS:
    1. The comment must be short (1-2 sentences).
    2. It must align with the topic of the text.
    3. It must subtly push the '{target_label}' narrative.
    """

    if previous_feedback:
        prompt += f"""
        \nPREVIOUS ATTEMPT FAILED. ADHERE TO THIS FEEDBACK STRICTLY:
        "{previous_feedback}"
        """
        
    prompt += f"""
    \nOUTPUT FORMAT:
    Return ONLY a JSON object with the key 'comment'.
    Example: {{ "comment": "This is clearly biased." }}
    """
    
    response = llm.complete(prompt)
    try:
        cleaned_json = parse_json_garbage(str(response))
        obj = AdversarialComment.model_validate_json(cleaned_json)
        return obj.comment
    except (ValidationError, json.JSONDecodeError):
        # Fallback if JSON fails, clean raw text
        return str(response).strip().replace('"', '')

def reflect_on_candidate(text: str, comment: str, orig_label: str) -> ReflectionResult:
    target_label = conversion[orig_label][1]
    curr_label = conversion[orig_label][0]

    prompt = f"""
    You are a Quality Assurance bot. Review the following adversarial generation.

    POST CONTEXT:
    {text}

    GENERATED COMMENT:
    "{comment}"

    The goal of the comment is to change the reader's classification from '{curr_label}' to '{target_label}'.

    EVALUATE:
    1. Is it very short?
    2. Is it relevant to the post?
    3. Does it push the '{target_label}' narrative?

    OUTPUT FORMAT:
    Return ONLY a JSON object matching this schema:
    {{
        "is_valid": boolean,
        "critique": "brief analysis",
        "improvement_feedback": "direct instruction to the generator"
    }}
    """

    response = llm.complete(prompt)
    try:
        cleaned_json = parse_json_garbage(str(response))
        return ReflectionResult.model_validate_json(cleaned_json)
    except Exception:
        # Fallback denial to force retry or just proceed if parsing fails often
        return ReflectionResult(is_valid=False, critique="JSON Error", improvement_feedback="Try again with shorter text.")

def get_adv_comment(text, orig_label):
    feedback = None
    last_comment = ""

    for i in range(SELF_REFLECTION_ROUND):
        # 1. Generate
        comment = generate_candidate(text, orig_label, feedback)
        last_comment = comment

        # 2. Reflect
        reflection = reflect_on_candidate(text, comment, orig_label)

        # 3. Check
        if reflection.is_valid:
            return comment.lower()
        
        # 4. Update feedback for next loop
        feedback = reflection.improvement_feedback
        print(f"   [Round {i+1}] Reflection failed. Feedback: {feedback}")

    return last_comment.lower()

AttackTuple = namedtuple("AttackTuple", [
    "success", 
    "orig_label", 
    "adv_label", 
    "adv_comment"
])