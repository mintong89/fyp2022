from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import BertTokenizer, BertConfig
from modeling_bert import BertForSequenceClassification
from language_tokens import get_lang_tokens
import numpy as np

from unidecode import unidecode
import re

punctuation = '‘’“”!$%&\()*+,./:;<=>?[\\]^_`{|}~•@…'


def clean_text(text):
    # convert characters to ascii
    text = unidecode(text)

    # remove words that is hashtags, mentions and links
    text = re.sub(r'^([@#]|http|https)[^\s]*', '', text)

    # remove punctuation
    text = text.translate(text.maketrans('', '', punctuation))

    # remove next line
    text = re.sub('\n', '', text)

    # lowercasing text
    text = text.lower()

    # stripping text
    text = text.strip()

    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    return text


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def root(sentence: str = Form()):
    MODEL_TYPE = 'bert-base-multilingual-uncased'
    test_ids = []
    test_attention_mask = []
    test_token_type_ids = []
    test_language_ids = []

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_TYPE, do_lower_case=True)

    encoded_dict = tokenizer.encode_plus(
        sentence,           # Sentences to encode.
        add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
        max_length=256,           # Pad or truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,   # Construct attn. masks.
        return_tensors='pt',          # Return pytorch tensors.
    )

    padded_token_list = encoded_dict['input_ids']
    att_mask = encoded_dict['attention_mask']
    token_type_ids = encoded_dict['token_type_ids']
    language_ids = torch.tensor(get_lang_tokens(
        [x.replace(' ', '')
         for x in tokenizer.batch_decode(padded_token_list.tolist())]
    ))

    num_to_sentiment = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }

    config = BertConfig.from_pretrained(
        MODEL_TYPE,
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
        MODEL_TYPE,
        config=config
    )

    device = torch.device('cpu')
    model.to(device)

    with open('../../saved_model/model_bert+li.bin', 'rb') as fp:
        states = torch.load('../../saved_model/model_bert+li.bin',
                            map_location=torch.device('cpu'))
        model.load_state_dict(states['model'])

    model.eval()
    torch.set_grad_enabled(False)

    with torch.no_grad():
        outputs = model(padded_token_list.to(device),
                        token_type_ids=att_mask.to(device),
                        attention_mask=token_type_ids.to(device),
                        language_ids=language_ids.to(device)
                        )

    preds = outputs[0]

    val_preds = preds.detach().cpu().numpy()

    final_preds = np.argmax(val_preds, axis=1)

    return {'output': num_to_sentiment[final_preds[0]]}
