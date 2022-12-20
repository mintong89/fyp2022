from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

import torch
import pickle
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, XLMRobertaTokenizer, XLMRobertaConfig, GPT2Config
from modeling_bert import BertForSequenceClassification
from modeling_xlm_roberta import XLMRobertaForSequenceClassification
from modeling_gpt2 import GPT2ForSequenceClassification
from language_tokens import get_lang_tokens
import numpy as np
import pandas as pd
import json

from preprocessing_data import clean_text, normalise_text

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_TYPE = 'bert'

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-multilingual-uncased', do_lower_case=True)

config = BertConfig.from_pretrained(
    'bert-base-multilingual-uncased',
    num_labels=64,
    output_attentions=False,
    output_hidden_states=False,
    num_hidden_layers=5,
    num_attention_heads=8,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
    ignore_mismatched_sizes=True
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-uncased',
    config=config
)

device = torch.device('cpu')
model.to(device)

with open('../../saved_model/model_bert+li_sentiment.bin', 'rb') as fp:
    states = torch.load(fp,
                        map_location=torch.device('cpu'))
    model.load_state_dict(states['model'])

model.eval()
torch.set_grad_enabled(False)


def change_model(model_type: str):
    global MODEL_TYPE
    global tokenizer
    global model

    if model_type == 'bert':

        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-uncased', do_lower_case=True)

        config = BertConfig.from_pretrained(
            'bert-base-multilingual-uncased',
            num_labels=64,
            output_attentions=False,
            output_hidden_states=False,
            num_hidden_layers=5,
            num_attention_heads=8,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            ignore_mismatched_sizes=True
        )

        model = BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-uncased',
            config=config
        )

        with open('../../saved_model/model_bert+li_sentiment.bin', 'rb') as fp:
            states = torch.load(fp,
                                map_location=torch.device('cpu'))
            model.load_state_dict(states['model'])

        model.eval()
        torch.set_grad_enabled(False)

    elif model_type == 'xlmr':

        tokenizer = XLMRobertaTokenizer.from_pretrained(
            'xlm-roberta-base', do_lower_case=True)

        config = XLMRobertaConfig.from_pretrained(
            'xlm-roberta-base',
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
            num_hidden_layers=5,
            num_attention_heads=8,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            ignore_mismatched_sizes=True
        )

        model = XLMRobertaForSequenceClassification.from_pretrained(
            'xlm-roberta-base',
            config=config
        )

        with open('../../saved_model/model_xlmr+li_sentiment.bin', 'rb') as fp:
            states = torch.load(fp,
                                map_location=torch.device('cpu'))
            model.load_state_dict(states['model'])

        model.eval()
        torch.set_grad_enabled(False)

    elif model_type == 'gpt2':

        with open('../../dictionary/tokenizer-gpt2.bin', 'rb') as fp:
            tokenizer = pickle.load(fp)

        tokenizer.padding_side = 'left'

        model_config = GPT2Config.from_pretrained(
            'gpt2',
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
            num_hidden_layers=5,
            num_attention_heads=8,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            ignore_mismatched_sizes=True,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id
        )
        model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', config=model_config)
        model.resize_token_embeddings(len(tokenizer))

        model.eval()
        torch.set_grad_enabled(False)

    MODEL_TYPE = model_type


def run_script(sentence: str):
    # ====================================
    # get data

    clean_sentence = normalise_text(clean_text(sentence))
    df_sentence = pd.concat([pd.DataFrame([clean_sentence], columns=['text'])],
                            ignore_index=True)

    # ====================================
    class TestDataset(Dataset):

        def __init__(self, df):
            self.df_data = df

        def __getitem__(self, index):

            # get the sentence from the dataframe
            features = self.df_data.loc[index, 'text']

            # Process the sentence
            # ---------------------

            encoded_dict = tokenizer.encode_plus(
                features,           # Sentence to encode.
                add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
                truncation=True,
                max_length=256,           # Pad or truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )

            # These are torch tensors already.
            input_ids = encoded_dict['input_ids'][0]
            att_mask = encoded_dict['attention_mask'][0]

            language_ids = torch.tensor(get_lang_tokens(
                [x.replace(' ', '')
                 for x in tokenizer.batch_decode(input_ids.tolist())]
            ))

            if MODEL_TYPE == 'bert':
                token_type_ids = encoded_dict['token_type_ids'][0]
                sample = (input_ids, att_mask, token_type_ids,
                          language_ids, features)
            else:
                sample = (input_ids, att_mask,
                          language_ids, features)

            return sample

        def __len__(self):
            return len(self.df_data)

    test_data = TestDataset(df_sentence)

    b_input_ids = []
    b_input_mask = []
    b_language_ids = []

    b_input_ids.append(test_data[0][0].tolist())
    b_input_mask.append(test_data[0][1].tolist())

    if MODEL_TYPE == 'bert':
        b_token_type_ids = []
        b_token_type_ids.append(test_data[0][2].tolist())
        b_token_type_ids = torch.tensor(b_token_type_ids)

        b_language_ids.append(test_data[0][3].tolist())
    else:
        b_language_ids.append(test_data[0][2].tolist())


    b_input_ids = torch.tensor(b_input_ids)
    b_input_mask = torch.tensor(b_input_mask)
    b_language_ids = torch.tensor(b_language_ids)

    # ====================================

    model_preds_list = []

    for _ in range(5):
        stacked_val_preds = None
        model.eval()
        torch.set_grad_enabled(False)

        if MODEL_TYPE == 'bert':
            outputs = model(b_input_ids.to(device),
                            token_type_ids=b_token_type_ids.to(device),
                            language_ids=b_language_ids.to(device),
                            attention_mask=b_input_mask.to(device))
        else:
            outputs = model(b_input_ids.to(device),
                            language_ids=b_language_ids.to(device),
                            attention_mask=b_input_mask.to(device))

        preds = outputs[0]
        val_preds = preds.detach().cpu().numpy()

        if stacked_val_preds is None:
            stacked_val_preds = val_preds

        else:
            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

        model_preds_list.append(stacked_val_preds)

    print('\nPrediction complete.')

    # ====================================

    for i, item in enumerate(model_preds_list):

        if i == 0:
            preds = item
        else:
            preds = item + preds

    avg_preds = preds/(len(model_preds_list))
    test_preds = np.argmax(avg_preds, axis=1)

    # ====================================

    return {'output': str(test_preds[0])}


@app.post("/")
async def root(data=Form()):

    data_list = json.loads(data)

    if data_list['function'] == 'run_script':
        return run_script(data_list['sentence'])

    elif data_list['function'] == 'change_model':
        return change_model(data_list['model_type'])
