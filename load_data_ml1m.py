import pickle
import re
import time
import openai
import pandas as pd
import numpy as np
import datetime
import random
import os
from collections import defaultdict
import itertools
from generate_response import *
import torch


openai.api_key = 'sk-xxxxx'

prompt_template = (
    "User #{minority_users} is a male user who may be treated unfairly in the movie recommendation system.\n"
    "In the movie recommendation system, we mainly consider two fairness metrics: Demographic Parity (DP) and Equalized of Opportunity (EO).\n"
    "Demographic Parity aims to measures the difference in recommendation rates between male and female groups. "
    "Ideally, the recommendation rates should be equal, resulting in DP approaching 0. A higher DP value indicates greater disparity and therefore greater unfairness in recommendations. \n"
    "We also consider Equalized Opportunity, which aims to ensure that recommendation systems have the same predictive accuracy in recommending relevant movies to male and female groups. "
    "Male and female groups should have the same recommendation accuracy for movies of actual interest, resulting in EO approaching 0.\n"
    "The unfairness is divided into 3 levels: weakly unfair, unfair, strongly unfair.\n"
    "User #{minority_user} {minority_user_unfairness_level}\n"
    "And User #{minority_user} watched the following movies in the past:\n{minority_user_history}\n"
    "Additionally, here are {fair_users_number} male users who were treated relatively fairly by recommendation system and their watching records for reference:\n{fair_users_history}\n"
    "Here are {unfair_users_number} male users who were treated unfairly by recommendation system and their watching records for reference:\n{unfair_users_history}\n"
    "And there is one male user with moderate fairness: \n{moderate_users_history}\n"
    "The augmented data should be able to decrease DP and EO, and reflect User #{minority_user}'s personalized preferences.\n"
    "Now there are {candidates_number} candidate movies for augmentation that User #{minority_user} can watch next:\n{candidates_information}\n"
    "Please rank these {candidates_number} candidate movies based on their potential to alleviate the unfairness suffered by User #{minority_user}, according to the given records of User #{minority_user} and other users. "
    "Please think step by step, but don't out the process.\n"
    "Please only show me your top 5 results with the format: {{Top1. movie id; Top2. movie id; ......}}. DO NOT output movie title and release year. You MUST rank the given candidate movies. You CAN NOT generate movies that are not in the given candidate list."
)

def interest_hit_percent(trainSet,user,items):
    with open(f'./Data/ml-1m/item_title.pkl', 'rb') as f:
        item_title = pickle.load(f)
    with open(f'./Data/ml-1m/item_genre.pkl', 'rb') as f:
        item_genre = pickle.load(f)
    personal_popular_items = get_personal_popular_items(trainSet,user,30)
    personal_popular_genres = set()
    for m in personal_popular_items:
        for g in item_genre[m]:
            personal_popular_genres.add(g)
    hit = 0
    genre_set_augment = set()
    for m in items:
        print(item_title[m], item_genre[m])
        if set(item_genre[m]) <= personal_popular_genres:
            hit += 1
        for g in item_genre[m]:
            genre_set_augment.add(g)
    return hit/len(items), len(genre_set_augment)/len(personal_popular_genres)

def get_similar_users(trainset, u, k):
    cur_user_interaction = set(trainset[u])
    users_inter_num = defaultdict(int)
    for u in set(trainset.keys()) - {u}:
        inter_num = len(cur_user_interaction & set(trainset[u]))
        union_set = cur_user_interaction.union(set(trainset[u]))
        users_inter_num[u] = int(inter_num/len(union_set))
    users_inter_num = dict(sorted(users_inter_num.items(),key=lambda x:x[1],reverse=True))
    user_candidates = list(users_inter_num.keys())
    user_selected = []

    for u in user_candidates:
        user_selected.append(u)
        if len(user_selected)==k:
            break
    return user_selected

def get_personal_popular_items(trainSet,u,k):
    group_users = get_similar_users(trainSet,u,30)
    item_personal_popularity = defaultdict(float)
    with open('./Data/ml-1m/item_title.pkl', 'rb') as f:
        item_title = pickle.load(f)
    for i in set(item_title.keys()):
        count = 0
        for user in group_users:
            history = trainSet[user]
            if i in history:
                count += 1
        item_personal_popularity[i] = count / len(group_users)
    item_personal_popularity = dict(sorted(item_personal_popularity.items(),key=lambda x:x[1],reverse=True))
    return list(item_personal_popularity.keys())[:k]

def get_various_type_items(genre_items,k):
    selected_movies = []
    genre_cycle = itertools.cycle(genre_items.keys())
    while len(selected_movies) < k:
        genre = next(genre_cycle)
        selected_movies.extend(random.sample(genre_items[genre],1))
    return selected_movies

def apply_10_core_principle(data):
    while True:
        user_checkins = defaultdict(int)
        poi_visits = defaultdict(int)

        for row in data:
            user_id = row[0]
            poi_id = row[1]
            user_checkins[user_id] += 1
            poi_visits[poi_id] += 1

        filtered_data = [
            row for row in data
            if user_checkins[row[0]] >= 10 and poi_visits[row[1]] >= 10
        ]
        if len(filtered_data) == len(data):
            break
        data = filtered_data

    print('apply 10-core over')
    return data

def create_prompts(dataset_name, trainSet, item_genre):
    if dataset_name == 'ml-1m':
        with open(f'./Data/{dataset_name}/item_title.pkl','rb') as f:
            item_title = pickle.load(f)
        with open(f'./Data/{dataset_name}/male_traindata.pkl','rb') as f:
            male_traindata = pickle.load(f)
        with open(f'./Data/{dataset_name}/genre_items.pkl','rb') as f:
            genre_items = pickle.load(f)
        with open(f'./Data/{dataset_name}/male_impact_values.pkl','rb') as f:
            male_impact_values = pickle.load(f)

        augment_males = [u for u,l in male_traindata.items() if len(l) < 20]
        print(len(augment_males))
        per_user_augnum = defaultdict(int)
        for u, movies in male_traindata.items():
            per_user_augnum[u] = 20 - len(movies)

        personal_unfairness = defaultdict(str)
        for u,v in enumerate(male_impact_values):
            if u in augment_males:
                if v.item() >= 1:
                    personal_unfairness[u] = "suffered a <strongly unfair> treatment from the recommendation system, the movies recommended for her didn't match with her preferences very well."
                elif v.item() >= .5:
                    personal_unfairness[u] = "suffered a <unfair> treatment from the recommendation system, the movies recommended for her didn't match with her preferences."
                elif v.item() > 0.:
                    personal_unfairness[u] = "suffered a <weakly unfair> treatment from the recommendation system, the movies recommended for her didn't match with her preferences slightly."
        augment_males = list(personal_unfairness.keys())
        with open(f'./Data/{dataset_name}/personal_unfairness.pkl','wb') as f:
            pickle.dump(personal_unfairness,f)


        '''
        # prompt tuning
        cur_eval_value = -np.inf
        global prompt_template
        best_prompt_template = ""
        candidate_items_dict = defaultdict(list)
        
        for epoch in range(5):
            print('epoch: ',epoch)
            interest_hit = []
            diversity_hit = []
            for u in augment_males:
                candidate_items_dict[u] = random.sample(set(item_title.keys()), 30)
            for u in tqdm(augment_males):
                most_fair_males = [411,5631,851] 
                most_unfair_males = [2866,820,4012] 
                medium_fair_males = random.sample(set(male_traindata.keys()) - set(most_fair_males) - set(most_unfair_males), 1)
                prompts = construct_prompt_with_template(dataset_name, prompt_template, u, item_title, trainSet, most_fair_males,
                                           most_unfair_males, medium_fair_males, item_genre, candidate_items_dict[u])
                preference_score, diversity_score = generate_single_response(trainSet,u,prompts)
                print(preference_score, diversity_score)
                interest_hit.append(preference_score)
                diversity_hit.append(diversity_score)
            evaluation_value_preference = np.mean(interest_hit)
            evaluation_value_diversity = np.mean(diversity_hit)
            print('evaluation preference value:',evaluation_value_preference)
            print('evaluation preference diversity:', evaluation_value_diversity)
            if (evaluation_value_preference + evaluation_value_diversity) > cur_eval_value:
                cur_eval_value = evaluation_value_preference + evaluation_value_diversity
                best_prompt_template = prompt_template
                print('Update prompt template')
            improved_prompts = tune_prompt(prompt_template, (evaluation_value_preference,evaluation_value_diversity))
            pattern = r"(?<=<IMPROVED_PROMPTS>)(.*?)(?=</IMPROVED_PROMPTS>)"
            improved_prompts = re.findall(pattern, improved_prompts, re.DOTALL)[0]
            print(improved_prompts)
            prompt_template = improved_prompts
        with open('Data/ml-1m/best_prompt_template.pkl', 'wb') as f:
            pickle.dump(best_prompt_template,f)
        '''

        with open(f'./Data/{dataset_name}/prompt_response/prompts.txt', 'w', encoding="utf-8") as file:
            for u in augment_males:
                for e in range(per_user_augnum[u]):
                    most_fair_males = [411,5631,851] # obtained by Fair representation model
                    most_unfair_males = [2866,820,4012] # obtained by Fair representation model
                    medium_fair_males = random.sample(set(male_traindata.keys()) - set(most_fair_males) - set(most_unfair_males), 1)

                    personal_popular_items = get_personal_popular_items(trainSet,u,100)
                    personal_popular_items = random.sample(personal_popular_items,15)
                    other_items = get_various_type_items(genre_items,15)
                    candidates = personal_popular_items + other_items

                    prompts = construct_prompt(dataset_name, u, item_title, trainSet, most_fair_males, most_unfair_males, medium_fair_males, item_genre, candidates)
                    file.write(f'user:{u}\n')
                    file.write(prompts)
                    file.write('\n\n')


def construct_prompt(dataset_name, minority_user, id2title, user_checkin_time, fair_users, unfair_users, medium_users, item_category, candidates):
    if dataset_name in ['ml-1m']:
        with open(f'./Data/{dataset_name}/movie_genres_text.pkl', 'rb') as f:
            movie_genres_text = pickle.load(f)
        with open(f'./Data/{dataset_name}/personal_unfairness.pkl','rb') as f:
            personal_unfairness = pickle.load(f)

        minority_user_history_text = [f"{i + 1}. {id2title[item]}, {movie_genres_text[item]}."
                                      for i, item in enumerate(user_checkin_time[minority_user][-20:])]

        fair_users_history_text = []
        for u in fair_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{i+1}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            fair_users_history_text.append(text)
        unfair_users_history_text = []
        for u in unfair_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{i + 1}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            unfair_users_history_text.append(text)
        medium_users_history_text = []
        for u in medium_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{i + 1}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            medium_users_history_text.append(text)

        candidate_text_order = [f'{i+1}. {id2title[item]}, {movie_genres_text[item]}.' for i,item in enumerate(candidates)]
        prompt = f"User #{minority_user} is a male user who may be treated unfairly in the movie recommendation system.\n" \
                f"In the movie recommendation system, we mainly consider two fairness metrics: Demographic Parity (DP) and Equalized of Opportunity (EO).\n" \
                f"Demographic Parity aims to measures the difference in recommendation rates between male and female groups. " \
                f"Ideally, the recommendation rates should be equal, resulting in DP approaching 0. A higher DP value indicates greater disparity and therefore greater unfairness in recommendations. \n" \
                f"We also consider Equalized Opportunity, which aims to ensure that recommendation systems have the same predictive accuracy in recommending relevant movies to male and female groups. " \
                f"Male and female groups should have the same recommendation accuracy for movies of actual interest, resulting in EO approaching 0.\n" \
                f"The unfairness is divided into 3 levels: weakly unfair, unfair, strongly unfair.\n" \
                f"User #{minority_user} {personal_unfairness[minority_user]}\n" \
                f"And User #{minority_user} watched the following movies in the past in order:\n{minority_user_history_text}\n" \
                f"Additionally, here are {len(fair_users)} male users who were treated relatively fairly by recommendation system and their watching records for reference:\n{fair_users_history_text}\n" \
                f"Here are {len(unfair_users)} male users who were treated unfairly by recommendation system and their watching records for reference:\n{unfair_users_history_text}\n" \
                f"And there is one male user with moderate fairness: \n{medium_users_history_text}\n" \
                f"The augmented data should be able to decrease DP and EO, and reflect User #{minority_user}'s personalized preferences.\n" \
                f"Now there are {len(candidates)} candidate movies for augmentation that User #{minority_user} can watch next:\n{candidate_text_order}\n" \
                f"Please rank these {len(candidates)} candidate movies based on their potential to alleviate the unfairness suffered by User #{minority_user}, according to the given records of User #{minority_user} and other users. " \
                f"Prioritize movies that closely match User #{minority_user}'s genre preferences and consider the diversity of genres to ensure fairness. "\
                f"Ensure that the top-ranked movies align strongly with User #{minority_user}'s past watched genres while also introducing some variety to promote fairness. Additionally, consider the genre preferences of the fair and moderately fair users to guide the ranking. " \
                f"Please think step by step, but don't out the process.\n" \
                "Please only show me your top 3 results with the format: {{Top1. movie title; Top2. movie title; ......}}. DO NOT omit the release year in movie title. You MUST rank the given candidate movies. You CAN NOT generate movies that are not in the given candidate list."

    else:
        raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
    return prompt

def construct_prompt_with_template(dataset_name, prompt_template, minority_user, id2title, user_checkin_time, fair_users, unfair_users, medium_users, item_category, candidates):
    if dataset_name in ['ml-1m']:
        with open(f'./Data/{dataset_name}/movie_genres_text.pkl', 'rb') as f:
            movie_genres_text = pickle.load(f)
        with open(f'./Data/{dataset_name}/personal_unfairness.pkl','rb') as f:
            personal_unfairness = pickle.load(f)

        minority_user_history_text = [f"{item}. {id2title[item]}, {movie_genres_text[item]}."
                                      for i, item in enumerate(user_checkin_time[minority_user][-20:])]

        fair_users_history_text = []
        for u in fair_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{item}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            fair_users_history_text.append(text)
        unfair_users_history_text = []
        for u in unfair_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{item}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            unfair_users_history_text.append(text)
        medium_users_history_text = []
        for u in medium_users:
            text = f"User #{u}: "
            for i, item in enumerate(user_checkin_time[u][-20:]):
                text += f"{item}. {id2title[item]}, {movie_genres_text[item]}; "
            text += '\n'
            medium_users_history_text.append(text)

        candidate_text_order = [f'{item}. {id2title[item]}, {movie_genres_text[item]}.' for i, item in enumerate(candidates)]
        prompt = prompt_template.format(minority_user=minority_user, minority_user_unfairness_level=personal_unfairness[minority_user],
                                         minority_user_history = minority_user_history_text, fair_users_number = len(fair_users),
                                         fair_users_history = fair_users_history_text, unfair_users_number = len(unfair_users),
                                         unfair_users_history = unfair_users_history_text, moderate_users_history = medium_users_history_text,
                                         candidates_number = len(candidates), candidates_information = candidate_text_order)
    else:
        raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
    return prompt

def file_exists_in_path(path, filename):
    if os.path.exists(path):
        files_in_path = os.listdir(path)
        return filename in files_in_path
    return False

def generate_single_response(trainSet,u,prompts):
    message = [{"role": "system",
                "content": "You are a data generator for alleviating unfairness issues caused by gender in movie recommendation."},
               {"role": "user", "content": prompts}]
    ans = openai_reply(message)
    print('ans:',ans)
    pattern = r'\b\d+\b'
    movies = re.findall(pattern, ans)
    print(movies)
    movies = [int(mid) for mid in movies]
    hit_percent = interest_hit_percent(trainSet,u,movies)
    return hit_percent

def introduce_movie_genre(movie_genre):
    movie_genres_text = defaultdict(str)
    for movie_id, genres in movie_genre.items():
        if len(genres) == 1:
            movie_genres_text[movie_id] = f"it's a/an {genres[0]} genre movie"
        elif len(genres) == 2:
            movie_genres_text[movie_id] = f"it's a/an {genres[0]} and {genres[1]} genre movie"
        else:
            genre_str = ", ".join(genres[:-1])
            genre_str += f" and {genres[-1]}" 
            movie_genres_text[movie_id] = f"it's a/an {genre_str} genre movie"
    return movie_genres_text

def read_data(dataset_name):
    if dataset_name == 'ml-1m':
        allData = defaultdict(list)
        trainSet = defaultdict(list)
        validSet = defaultdict(list)
        testSet = defaultdict(list)
        item_title = defaultdict(str)
        item_genre = defaultdict(list)
        male_users = set()
        female_users = set()
        user_gender = defaultdict(str)
        male_data = defaultdict(list)
        female_data = defaultdict(list)
        genre_items = defaultdict(set)
        user2id = {}
        item2id = {}
        id2user = {}

        count = 0
        for line in open(f'./Data/{dataset_name}/users.dat','r'):
            l = line.strip().split('::')
            userId, gender, age, occupation = l[0], l[1], l[2], l[3]
            user_gender[userId] = gender
        count_f = 0
        count_m = 0
        for line in open(f'./Data/{dataset_name}/ratings.dat', 'r'):
            userId, itemId, rating, _ = line.strip().split('::')
            if int(rating) > 3:
                if userId not in user2id:
                    user2id[userId] = len(user2id)
                    id2user[user2id[userId]] = userId
                if itemId not in item2id:
                    item2id[itemId] = len(item2id)
                allData[user2id[userId]].append(item2id[itemId])
                count += 1
                gender = user_gender[userId]
                if gender == 'M':
                    male_users.add(user2id[userId])
                    male_data[user2id[userId]].append(item2id[itemId])
                    count_m += 1
                elif gender == 'F':
                    female_users.add(user2id[userId])
                    female_data[user2id[userId]].append(item2id[itemId])
                    count_f += 1
        print('female user num: %d, male user num: %d'%(len(female_data.keys()),len(male_data.keys())))
        female_data = dict(sorted(female_data.items(),key=lambda x:len(x[1])))
        male_data = dict(sorted(male_data.items(), key=lambda x: len(x[1])))

        n_items = len(item2id.keys())
        print('data number is %d, n_users is %d, n_items is %d'%(count, len(allData.keys()), n_items))

        for line in open(f'./Data/{dataset_name}/movies.dat','r',encoding='ISO-8859-1'):
            l = line.strip().split('::')
            itemId = l[0]
            if itemId not in item2id:
                continue
            item_title[item2id[itemId]] = l[1]
            item_genre[item2id[itemId]] = l[2].split('|')
            for g in l[2].split('|'):
                genre_items[g].add(item2id[itemId])
        movie_genres_text = introduce_movie_genre(item_genre)

        with open(f'./Data/{dataset_name}/movie_genres_text.pkl','wb') as f:
            pickle.dump(movie_genres_text,f)
        with open(f'./Data/{dataset_name}/item_title.pkl','wb') as f:
            pickle.dump(item_title,f)
        with open(f'./Data/{dataset_name}/genre_items.pkl','wb') as f:
            pickle.dump(genre_items, f)
        with open(f'./Data/{dataset_name}/item_genre.pkl','wb') as f:
            pickle.dump(item_genre, f)

        data = []
        for u, movies in allData.items():
            for i in movies:
                data.append([u,i])
        np.random.shuffle(data)
        print('data length:',len(data))
        # data for BPR
        trainData = data[:int(len(data) * 0.6)]
        validData = data[int(len(data) * 0.6) : int(len(data) * 0.7)]
        testData = data[int(len(data) * 0.7):]
        with open(f'./Data/{dataset_name}/train.txt', 'w') as f_t, \
                open(f'./Data/{dataset_name}/valid.txt', 'w') as f_v,\
                open(f'./Data/{dataset_name}/test.txt', 'w') as f_te:
            for d in trainData:
                f_t.write(str(d[0])+'\t'+str(d[1])+'\n')
            for d in validData:
                f_v.write(str(d[0])+'\t'+str(d[1])+'\n')
            for d in testData:
                f_te.write(str(d[0])+'\t'+str(d[1])+'\n')

        male_traindata = defaultdict(list)
        female_traindata = defaultdict(list)
        for d in trainData:
            if d[0] in male_users:
                male_traindata[d[0]].append(d[1])
            elif d[0] in female_users:
                female_traindata[d[0]].append(d[1])
            u,i = d[0],d[1]
            trainSet[u].append(i)

        print(len(male_traindata.keys()),len(female_traindata.keys()))
        female_traindata = dict(sorted(female_traindata.items(),key=lambda x:len(x[1])))
        male_traindata = dict(sorted(male_traindata.items(), key=lambda x: len(x[1])))

        for d in validData:
            u, i = d[0], d[1]
            validSet[u].append(i)

        for d in testData:
            u, i = d[0], d[1]
            testSet[u].append(i)

        with open(f'./Data/{dataset_name}/male_users.pkl', 'wb') as f:
            pickle.dump(male_users, f)
        with open(f'./Data/{dataset_name}/female_users.pkl', 'wb') as f:
            pickle.dump(female_users, f)
        with open(f'./Data/{dataset_name}/male_traindata.pkl', 'wb') as f:
            pickle.dump(male_traindata, f)
        with open(f'./Data/{dataset_name}/female_traindata.pkl', 'wb') as f:
            pickle.dump(female_traindata, f)
        with open(f'./Data/{dataset_name}/user_gender.pkl', 'wb') as f:
            pickle.dump(user_gender, f)

        print('max record length in female: %d, max record length in male: %d' % (max([len(l) for u, l in female_data.items()]), max([len(l) for u, l in male_data.items()])))
        print('mean of female interaction: ', np.mean([len(l) for u, l in female_data.items()]))
        print('mean of male interaction: ', np.mean([len(l) for u, l in male_data.items()]))

        train_u2i = trainSet
        train_i2u = defaultdict(list)
        test_u2i = testSet
        test_i2u = defaultdict(list)
        valid_u2i = validSet
        train_set = {'userid':np.array([d[0] for d in trainData]),'itemid':np.array([d[1] for d in trainData])}
        test_set = {'userid':np.array([d[0] for d in testData]),'itemid':np.array([d[1] for d in testData])}
        valid_set = {'userid':np.array([d[0] for d in validData]),'itemid':np.array([d[1] for d in validData])}
        user_side_features = {}
        n_users = len(allData.keys())
        for d in trainData:
            train_i2u[d[1]].append(d[0])
        for d in testData:
            test_i2u[d[1]].append(d[0])
        gender_list = []
        for u in range(n_users):
            if u in male_users:
                gender_list.append(1)
            elif u in female_users:
                gender_list.append(0)
        user_side_features['gender'] = np.array(gender_list)
        with open('./Data/ml-1m/data.pkl', 'wb') as f:
            pickle.dump(train_u2i, f)
            pickle.dump(train_i2u, f)
            pickle.dump(test_u2i, f)
            pickle.dump(test_i2u, f)
            pickle.dump(train_set, f)
            pickle.dump(test_set, f)
            pickle.dump(valid_u2i, f)
            pickle.dump(valid_set, f)
            pickle.dump(user_side_features, f)
            pickle.dump((n_users,n_items), f)

        return trainSet, testSet, item_title, item_genre

def extract_augment_data(dataset_name):
    if dataset_name == 'ml-1m':
        with open(f'./Data/{dataset_name}/male_traindata.pkl', 'rb') as f:
            male_traindata = pickle.load(f)
        with open(f'./Data/{dataset_name}/augment_males.pkl','rb') as f:
            minority_users = pickle.load(f)
        with open(f'./Data/{dataset_name}/prompt_response/response.txt', 'r', encoding="utf-8") as f:
            response = f.read()
        with open(f'./Data/{dataset_name}/item_title.pkl','rb') as f:
            item_title = pickle.load(f)
        title2item = {v.strip(): k for k, v in item_title.items()}
        paragraphs = response.split('user:user:')
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        paragraphs = ['user:user:' + para for para in paragraphs]
        user_augment_watching = []
        pattern1 = r'\{(.*?)\}'
        pattern2 = r'Top\s*\d+\.\s(.*?\(\d{4}\))'
        for i,user_augment_text in enumerate(paragraphs):
            movies = re.findall(pattern1,user_augment_text)
            movies = movies[0]
            movies = re.findall(pattern2,movies)
            user_augment_watching.append(movies)

        per_user_augnum = defaultdict(int)
        for u,movies in male_traindata.items():
            per_user_augnum[u] = 20 - len(movies)
        minority_users = list(minority_users)
        expanded_minority_users = [u for u in minority_users for i in range(per_user_augnum[u])]
        augData = defaultdict(list)
        for i,u in enumerate(expanded_minority_users):

            augData[u].extend(user_augment_watching[i][:1])
        for u in augData:
            augData[u] = list(set(augData[u]))
        with open(f'./Data/{dataset_name}/AugmentData/augmented_data.txt','w') as f:
            for u,movies in augData.items():
                f.write(str(u))
                for m in movies[:per_user_augnum[u]]:
                    if m in title2item.keys():
                        mid = title2item[m]
                        f.write('\t' + str(mid))
                        continue
                    if m.startswith('The '):
                        years = m[-7:]
                        m = m[4:-7] + ', The' + years
                    if m.startswith('A '):
                        years = m[-7:]
                        m = m[2:-7] + ', A' + years
                    mid = title2item[m]
                    f.write('\t' + str(mid))
                f.write('\n')

if __name__ == '__main__':
    trainSet, testSet, item_title, item_genre = read_data('ml-1m') # step 1
    create_prompts('ml-1m',trainSet, None) # step 2
    # extract_augment_data('ml-1m') # step 3: First run generate_response.py to augment data, then execute the extract_augment_data function to extract augmented data. (Please first comment out the above two lines)

