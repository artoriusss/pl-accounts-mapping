import os 
import logging
import random

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
from sklearn.decomposition import PCA


tqdm.pandas()

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
client = OpenAI()

from .helpers import *

def read_and_preprocess():
    master = pd.read_excel('data/master-categories.xlsx')
    trans1 = pd.read_csv('data/transactions1.csv')
    trans2 = pd.read_csv('data/transactions2.csv')

    transactions = pd.concat([trans1, trans2])

    transactions.drop(['Description', 'Counterparty'], axis=1, inplace=True)
    transactions['Date'] = pd.to_datetime(transactions['Date'], format='mixed')

    master.rename(columns={'Master categories': 'master_category'}, inplace=True)
    transactions.rename(columns={'Date': 'date', 'Amount': 'amount', 'PL Account': 'pl_account'}, inplace=True)

    transactions['clean_pl_account'] = transactions['pl_account'].apply(lambda x: x.replace(' (deleted)', ''))

    return master, transactions

def get_embeddings(master_list, account_list):
    master_embeddings = np.array([get_embedding(client, cat) for cat in tqdm(master_list, desc='Generating master embeddings')])
    account_embeddings = np.array([get_embedding(client, account) for account in tqdm(account_list, desc='Generating embeddings for unrecognized accounts')])
    return master_embeddings, account_embeddings

def PCA_transform(master_embeddings, account_embeddings, n_components=0.7):
    pca = PCA(n_components=n_components)
    reduced_master_emb = pca.fit_transform(master_embeddings)
    reduced_unrec_emb = pca.transform(account_embeddings)
    return reduced_master_emb, reduced_unrec_emb

def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Started mapping pipeline")
    logging.info("Reading and preprocessing data...")
    master, transactions = read_and_preprocess()

    master_list = master['master_category'].tolist()
    unique_clean_accounts = transactions['clean_pl_account'].unique()

    logging.info("Stage 1: Fuzzy matching")
    fuzzy_account_mapping = {account: fuzzy_match(account, master_list) for account in tqdm(unique_clean_accounts, desc='Mapping accounts')}

    transactions['master_category'] = transactions['clean_pl_account'].map(fuzzy_account_mapping)
    rec_trans = transactions[transactions['master_category'] != 'Unrecognized account']
    unrec_trans = transactions[transactions['master_category'] == 'Unrecognized account']

    unique_unrec_clean_accs = unrec_trans['clean_pl_account'].unique()

    logging.info("Stage 2: Generating embeddings")
    master_embeddings, unrec_embeddings = get_embeddings(master_list, unique_unrec_clean_accs)
    reduced_master_emb, reduced_unrec_emb = PCA_transform(master_embeddings, unrec_embeddings)

    logging.info("Matching embeddings by cosine similarity...")
    unrec_acc_map = {account: find_best_match_embedding(embedding, master_list, reduced_master_emb, threshold=0.7)
                                for account, embedding in zip(unique_unrec_clean_accs, reduced_unrec_emb)}

    unrec_trans.loc[:, 'master_category'] = unrec_trans['clean_pl_account'].map(unrec_acc_map)

    final_rec_trans = pd.concat([rec_trans, unrec_trans])
    final_rec_trans.to_csv('data/final_transactions.csv')
    logging.info("Pipeline completed successfully")
    logging.info("Saved final transactions to data/final_transactions.csv")

if __name__ == "__main__":
    main()
