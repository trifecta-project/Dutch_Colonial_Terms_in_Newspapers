import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

dutch_stopwords = set(stopwords.words('dutch'))

def clean_dutch_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and word not in dutch_stopwords and len(word) > 3
    ]
    return " ".join(cleaned_tokens)

input_file = "/data/groups/trifecta/teresa/CHR/1860_1899.txt" #your input file
output_file = "/data/groups/trifecta/teresa/CHR/1860_1899_clean.txt" #output cleaned file

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        cleaned = clean_dutch_text(line.strip())
        if cleaned:
            outfile.write(cleaned + "\n")