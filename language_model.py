import nltk
import pickle 
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

# Define a function to calculate the Kneser-Ney smoothed probability of a bigram
def bigram_function(words, count, unigram_counts, bigram_counts, discount, firstWords, secondWords):
    word1 = words[0]
    word2 = words[1]
    w1CountAs1 = firstWords[word1]
    w2CountAs2 = secondWords[word2]
    total_bigrams = len(bigram_counts)

    # Calculate the probability of the bigram using Kneser-Ney smoothing
    l = (discount / unigram_counts[word1]) * w1CountAs1
    continuation = w2CountAs2 / total_bigrams
    pkn = (max(count - discount, 0) / unigram_counts[word1]) + l * continuation
    return pkn

# Define a function to calculate the Kneser-Ney smoothed probability of a trigram
def trigram_function(words, count, bigram_counts, trigram_counts, discount):
    word1 = words[0]
    word2 = words[1]
    word3 = words[2]
    w1w2CountAs1 = bigram_counts[(word1, word2)]
    w2w3CountAs2 = bigram_counts[(word2, word3)]
    total_trigrams = len(trigram_counts)

    # Calculate the probability of the trigram using Kneser-Ney smoothing
    l1 = (discount / bigram_counts[(word1, word2)]) * w1w2CountAs1
    l2 = (discount / unigram_counts[word2]) * w2w3CountAs2
    continuation = (max(count - discount, 0) / bigram_counts[(word1, word2)]) * w2w3CountAs2 / total_trigrams
    pkn = (max(count - discount, 0) / bigram_counts[(word1, word2)]) + l1 * l2 * continuation
    return pkn

# Read in the text file
with open('output_preprocessed.txt', 'r', encoding='utf-8') as input_file:
    text = input_file.read()

# Define a custom tokenizer that splits the text into tokens based on <s>, </s>, and blank lines
tokenizer = RegexpTokenizer(r'<s>|</s>|\n\n+|\S+')
words = tokenizer.tokenize(text)

# Add <s> and </s> to the beginning and end of the token list
words = ['<s>'] + words + ['</s>']

# Build unigram, bigram, and trigram models
unigram_counts = collections.Counter(words)
bigram_counts = collections.Counter(ngrams(words, 2))
trigram_counts = collections.Counter(ngrams(words, 3))

# Calculate the number of times each word appears as the first or second word in a bigram
firstWords = collections.Counter([bigram[0] for bigram in bigram_counts])
secondWords = collections.Counter([bigram[1] for bigram in bigram_counts])

# Apply Kneser-Ney smoothing to the bigram and trigram models
discount = 0.75
bigram_probs = {}
for bigram, count in bigram_counts.items():
    bigram_probs[bigram] = bigram_function(bigram, count, unigram_counts, bigram_counts, discount, firstWords, secondWords)

trigram_probs = {}
for trigram, count in trigram_counts.items():
    trigram_probs[trigram] = trigram_function(trigram, count, bigram_counts, trigram_counts, discount)

# Save the resulting models to files using the pickle module
with open('unigrams.pkl', 'wb') as output_file:
    pickle.dump(unigram_counts, output_file)

with open('bigrams.pkl', 'wb') as output_file:
    pickle.dump(bigram_probs, output_file)

with open('trigrams.pkl', 'wb') as output_file:
    pickle.dump(trigram_probs, output_file)