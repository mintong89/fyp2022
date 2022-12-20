# Performing MLM Transformers for inter-sentential Malay-English code-switching sentences
Creator: Ong Chen Xiang
<br>From: TARUMT Final Year Project 2022

## Source details
Source code of transformer models with added language embeddings.
<br>Everything used in this project are for research purpose.

### Model List

- BERT + Language Embeddings (BERT+LE)
- XLM-R + Language Embeddings (XLM-R+LE)
- GPT2 + Language Embeddings (GPT2+LE)

### Source Codes

- `modeling_bert.py`: BERT+LE
- `modeling_xlm_roberta.py`: XLM-R+LE
- `modeling_gpt2.py`: GPT2+LE

### Notebook Examples

**Note: The jupyter notebook files are optimised to suited with Google Colab, additional changes (e.g. save files) are needed in order to perform successfully.**

- `sentiment-analysis.ipynb`: Sentiment Analysis (BERT and XLM-Roberta only)
- `masked-language-modeling.ipynb`: Masked Language Modeling (BERT and XLM-Roberta only)
- `sentiment-analysis-gpt2.ipynb`: Sentiment Analysis (GPT2 only)
- `language-modeling-gpt2.ipynb`: Model transformer with a language modeling head (GPT2 only)

## Credits

### Datasets gathered in combined_data.csv

| Source | Datasets |
| -------- | ---- |
| [Malay-Dataset](https://github.com/huseinzol05/malay-dataset) | [Local News](https://github.com/huseinzol05/malay-dataset/blob/master/sentiment/news-sentiment) <br> [Semisupervised Twitter](https://github.com/huseinzol05/malay-dataset/blob/master/sentiment/semisupervised-twitter-3class) <br> [Supervised Twitter](https://github.com/huseinzol05/malay-dataset/blob/master/sentiment/supervised-twitter) <br> [Supervised Twitter Politics](https://github.com/huseinzol05/malay-dataset/blob/master/sentiment/supervised-twitter-politics) |
| Twitter Scrapping | Please refer to [`get_twitter_data.ipynb`](https://github.com/mintong89/fyp2022/blob/master/data/get_twitter_data.ipynb) |

### Dictionaries used in combined-malay-dict.txt

| Source | Dictionaries |
| ------ | ------------ |
| [Malay-Dataset](https://github.com/huseinzol05/malay-dataset) | [200k English-Malay](https://github.com/huseinzol05/malay-dataset/blob/master/dictionary/200k-english-malay) |
| Dewan Bahasa | [73k English-Malay](https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ms.txt) |
| [ipa-dict](https://github.com/open-dict-data/ipa-dict) | [IPA Dictionary](https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ma.txt) |

### Normalise Data

| Source | Link |
| ------ | --------- |
| Cilisos | https://cilisos.my/bahasa-sms-shortforms-glossary/ |
