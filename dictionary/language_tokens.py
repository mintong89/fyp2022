# Importing language detection
# For Malay, we use dictionaries from IPA-Dict and Dewan Bahasa.
# For English, we use NLTK corpus.

# import json
import nltk
nltk.download('words')

# my_raw1 = json.load(open('../dictionary/200k-english-malay.json'))
# my_raw2 = open('../dictionary/en-ms.txt', encoding="utf8")
# my_raw3 = open('../dictionary/malay-ipa-dict.txt', encoding="utf8")

# my_raw1 = [x[1] for x in my_raw1]
# my_raw2 = [x.split('\t')[1].strip() for x in my_raw2.readlines()]
# my_raw3 = [x.split('\t')[0] for x in my_raw3.readlines()]

# with open('../dictionary/combined-malay-dict.txt', 'w', encoding="utf8") as fp:
#     for item in sorted(list(dict.fromkeys(my_raw1 + my_raw2 + my_raw3))):
#         if item:
#             fp.write("%s\n" % item)

with open('combined-malay-dict.txt', encoding="utf8") as fp:
    malay_dict = set([x.strip() for x in fp.readlines()])

eng_dict = set(nltk.corpus.words.words())


def detect_malay(text): return text in malay_dict
def detect_english(text): return text in eng_dict


special_token_list = set(
    ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]', '<s>', '</s>', '<pad>', '<unk>', '<mask>', '<|endoftext|>', '<|beginoftext|>', '<|pad|>', '<|sep|>', '<|mask|>'])

lang_id2num = {'special_token': 0, 'english': 1, 'malay': 2, 'other': 3}
lang_num2id = {v: k for k, v in lang_id2num.items()}


def detect_lang(text):
    if text in special_token_list:
        return 'special_token'
    elif detect_malay(text):
        return 'malay'
    elif detect_english(text):
        return 'english'
    else:
        return 'other'


def get_lang_tokens(decoded_input_tokens):
    language_ids = []

    full_sentence = ''
    token_count = 0
    for token in decoded_input_tokens:
        if '##' in token:
            full_sentence = token[2:] + full_sentence
            token_count += 1
            continue

        full_sentence = token + full_sentence
        token_count += 1
        lang_token = lang_id2num[detect_lang(full_sentence)]
        for _ in range(token_count):
            language_ids.append(lang_token)

        full_sentence = ''
        token_count = 0

    return language_ids
