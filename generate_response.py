import openai
import pandas as pd
import numpy as np
import datetime
import random
import re
from tqdm import tqdm
import time
from collections import defaultdict
import pickle
import os
 
openai.api_key = 'sk-xxxxx'

def read_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    paragraphs = content.split('\n\n')
    return [para.strip() for para in paragraphs if para.strip()]

def tune_prompt(previous_prompts,evaluation,dataset_name='ml-1m'):
    if dataset_name == 'ml-1m':
        optimizer_system_prompt = (
            "You are a prompt optimizer for alleviating unfairness issues caused by gender in movie recommendation. "
            "You will be asked to read and understand the prompts, then creatively and critically improve prompts which are used to generate data for improving male users' fairness. "
            "You will receive some evaluation values, and improve prompts according the evaluation. "
            "Please note that YOUR GOAL is to POLISH THE PROMPTS so that it can generate movie data for users that IMPROVES their FAIRNESS. "
            "This is very important: You MUST give your response by sending the improved prompts between <IMPROVED_PROMPTS> {{improved prompts}} </IMPROVED_PROMPTS> tags. "
            "DO NOT generate data, that will be the job of data generator."
            "The text you send between the tags will directly replace the prompts.\n"
            f"Here is the template of prompts to generate data for each male user which you will improve: <CUR_PROMPTS>{previous_prompts}</CUR_PROMPTS>.\n\n"
            "Here is the evaluation we got for the cur_prompts:\n\n"
            "We evaluate the generated data based on whether it matches the user's interests and the diversity of genres. "
            "Specifically, we calculate the percentage of generated movies that fall within the users' genre of interest for a set of male users, i.e., inactive users. "
            "Additionally, we calculate the diversity of the generated movie data."
            f"<EVALUATION>Matching scores with user interests:{evaluation[0]}</EVALUATION>\n\n" 
            f"<EVALUATION>Diversity scores of movie genres: {evaluation[1]}</EVALUATION>\n\n" 
            "Improve the prompts according to the evaluation provided in <EVALUATION> tags.\n"
            "The improved prompts should increase the degree to which the generated data matches users' interests to improve the evaluation, and alleviate the unfairness. "
            "Send the improved prompts "
            "in the following format:\n\n<IMPROVED_PROMPTS>{{improved prompts}}</IMPROVED_PROMPTS>\n\n"
            "Send ONLY the improved prompts between the <IMPROVED_PROMPTS> tags, and nothing else."
        )
    elif dataset_name == 'lastfm':
        optimizer_system_prompt = (
            "You are a prompts optimizer for alleviating unfairness issues caused by gender in music artist recommendation. "
            "You will be asked to creatively and critically improve prompts which are used to generate data. "
            "You will receive some evaluation values, and improve prompts according the evaluation. "
            "This is very important: You MUST give your response by sending the improved prompts between <IMPROVED_PROMPTS> {{improved prompts}} <\IMPROVED_PROMPTS> tags. "
            "DO NOT generate data, that will be the job of data generator."
            "The text you send between the tags will directly replace the prompts.\n"
            f"Here is the template of prompts to generate data for each female user which you will improve: <CUR_PROMPTS>{previous_prompts}</CUR_PROMPTS>.\n\n"
            "Here is the evaluation score we got for the cur_prompts using gpt-4o:\n\n"
            "We evaluate the generated data in terms of user preferences and fairness. "
            "Specifically, we determine whether the generated data matches users' preferences and consider two fairness metrics, Demographic Parity (DP) and Equalized of Opportunity (EO)."
            f"The score is <EVALUATION>{evaluation}</EVALUATION>\n\n"
            "Improve the prompts according to the evaluation score provided in <EVALUATION> tags.\n"
            "The improved prompts should increase the degree to which the generated data matches users' preferences to improve the evaluation score, and alleviate the unfairness."
            "Send the improved prompts in the following format:\n\n<IMPROVED_PROMPTS>{{improved prompts}}<\IMPROVED_PROMPTS>\n\n"
            "Send ONLY the improved prompts between the <IMPROVED_PROMPTS> tags, and nothing else."
        )

    message = [{"role":"system","content":optimizer_system_prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=message,
        temperature=0.3,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

def extract_first_lines(paragraphs):
    first_lines = []
    processed_paragraphs = []

    for para in paragraphs:
        lines = para.split('\n')
        if lines:
            first_lines.append(lines[0].strip())
            processed_paragraphs.append('\n'.join(lines[1:]).strip())

    return first_lines, processed_paragraphs

def openai_reply(message):
    print('create chatcompletion')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message,
        temperature=0.0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print('Get responses')
    return response.choices[0].message.content

def generate_response(dataset_name):
    if dataset_name == 'ml-1m':
        paragraphs = read_paragraphs(f'./Data/{dataset_name}/prompt_response/prompts.txt')
        user_list, prompt_list = extract_first_lines(paragraphs)
        ans_list = []

        with open(f'Data/{dataset_name}/prompt_response/response.txt', 'a', encoding ='utf-8') as f:
            i = 0
            for prompt in tqdm(prompt_list):
                message = [{"role": "system", "content": "You are a data generator for alleviating unfairness issues caused by gender in movie recommendation."},
                           {"role": "user", "content": prompt}]
                ans = openai_reply(message)
                f.write(f'user:{user_list[i]}\n')
                f.write(ans)
                f.write('\n\n')
                ans_list.append(ans)
                if (i + 1) % 5 == 0:
                    time.sleep(5)
                i += 1

    elif dataset_name == 'lastfm':
        paragraphs = read_paragraphs(f'./Data/{dataset_name}/prompt_response/prompts.txt')
        user_list, prompt_list = extract_first_lines(paragraphs)
        ans_list = []

        with open(f'Data/{dataset_name}/prompt_response/response.txt', 'a', encoding='utf-8') as f:
            i = 0
            for prompt in tqdm(prompt_list):
                message = [{"role": "system",
                            "content": "You are a data generator for alleviating unfairness issues caused by gender in music artist recommendation."},
                           {"role": "user", "content": prompt}]
                ans = openai_reply(message)
                f.write(f'user:{user_list[i]}\n')
                f.write(ans)
                f.write('\n\n')
                ans_list.append(ans)
                if (i + 1) % 5 == 0:
                    time.sleep(5)
                i += 1

if __name__ == '__main__':
    generate_response('ml-1m')



