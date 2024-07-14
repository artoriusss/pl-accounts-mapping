import numpy as np
from fuzzywuzzy import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity

def fuzzy_match(pl_account, master_list, token_threshold=95, partial_threshold=95):
    match, score = process.extractOne(pl_account.lower(), master_list, scorer=fuzz.token_sort_ratio)
    if score >= token_threshold:
        return match
    match, score = process.extractOne(pl_account.lower(), master_list, scorer=fuzz.partial_ratio)
    if score >= partial_threshold:
        intermediate_match, intermediate_score = process.extractOne(pl_account, master_list, scorer=fuzz.token_set_ratio)
        if intermediate_score >= token_threshold:
            return intermediate_match
        return 'Unrecognized account'
    
    return 'Unrecognized account'

def get_embedding(open_ai_client, text):
    '''Get the embedding of a text using OpenAI API'''
    response = open_ai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    return np.array(response)

def find_best_match_embedding(embedding, master_list, master_embeddings, threshold=0.6):
    similarities = cosine_similarity([embedding], master_embeddings)[0]
    max_similarity = similarities.max()
    if max_similarity >= threshold:
        best_match_index = similarities.argmax()
        return master_list[best_match_index]
    return 'Unrecognized account'