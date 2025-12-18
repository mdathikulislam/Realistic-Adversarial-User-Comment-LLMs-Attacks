# LLM / LLM* Comment Attacks for RoBERTa

## Overview
This repository is used to run LLM and LLM* comment attacks against a RoBERTa-based fake–news classifier.  
These attacks generate adversarial user comments intended to flip the model’s prediction.

## Prerequisites
First install Ollama:

https://ollama.com/download

Then clone and set up the RoBERTa surrogate model located at:

https://github.com/ChandlerU11/RoBERTa_Surrogate

Follow the installation steps provided in that repository.

## Setup Instructions

1. Clone the RoBERTa_Surrogate repository and install its requirements.
2. From this repository:
   - Copy `attacks.py` into the `RoBERTa_Surrogate` directory.
   - Copy `llms.py` one directory above `RoBERTa_Surrogate` (the parent directory).

## Configuration
All experimental settings can be edited inside `attacks.py`, including:

- Dataset selection
- Number of initial comments
- Target attack type (fake to real, or real to fake)
- Model mode (LLM or LLM*)

Adjust configuration values before executing the attack script.

## Running the Attack
To perform the attack, move into the directory containing `attacks.py` and run:

`python attacks.py`

