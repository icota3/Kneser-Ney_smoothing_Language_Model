import re
import nltk
# code uses the pickle module to serialize Python objects (such as lists and dictionaries) 
# to binary files, making it easier to load and manipulate them later
import pickle 
import collections

with open('output_extracted.txt','r', encoding='utf-8') as input_file:
    content = input_file.read()
    content = re.sub(r'(?:http|www)\S+', ' ', content) # remove urls
    content = content.lower()  # lowercase all letters
    content = re.sub(r'\d+', ' ', content)  # remove numbers
    content = re.sub(r'(\.\s*){2,}', ' ', content) # remove consecutive dots
    # remove percentages and other special characters
    content = re.sub(r'[^\w\s.!?]+', ' ', content)
    content = re.sub(r'[_-]', ' ', content)  # remove underscores
    content = re.sub(r'\s+', ' ', content)  # remove extra whitespaces
    sentences = re.split(r'[.!?]+|\n', content)
    content = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            content += "<s> " + sentence + " </s> "

with open('output_preprocessed.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(content)

input_file.close()
output_file.close()