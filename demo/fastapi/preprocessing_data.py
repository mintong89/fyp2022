from unidecode import unidecode
import string
import re
import pandas as pd

malaya_sf = pd.read_csv(r'../../normalise/malaya.csv')
cilisos_sf = pd.read_csv(r'../../normalise/cilisos.csv', encoding='ISO-8859-1')

combined_sf = {x[0]: x[1] for x in malaya_sf.values.tolist() + cilisos_sf.values.tolist()}

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
    


def normalise_text(text):
    return ' '.join([combined_sf[x] if x in combined_sf.keys() else x for x in text.split()])