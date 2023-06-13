import pickle

# Load the saved models
with open('unigrams.pkl', 'rb') as input_file:
    unigram_counts = pickle.load(input_file)

with open('bigrams.pkl', 'rb') as input_file:
    bigram_probs = pickle.load(input_file)

with open('trigrams.pkl', 'rb') as input_file:
    trigram_probs = pickle.load(input_file)

# Function to calculate the probability of a sentence
def sentence_probs(sentence, unigram_counts, bigram_probs, trigram_probs):
    words = sentence.split()
    words = ['<s>'] + words + ['</s>']
    probability = 1.0

    # Calculate the probability using trigram model
    for i in range(2, len(words)): # Start at 2 since we need the previous two words
        trigram = (words[i - 2], words[i - 1], words[i]) # Get the trigram
        bigram = (words[i - 2], words[i - 1]) # Get the bigram
        if trigram in trigram_probs: 
            probability *= trigram_probs[trigram] 
        elif bigram in bigram_probs: # If trigram not found, try using the bigram 
            probability *= bigram_probs[bigram] 
        else:                      # If bigram not found, try using the unigram
            probability *= (unigram_counts[words[i]] / sum(unigram_counts.values()))

    return probability

# Test sentences
sentence = input("Type your sentence without '?.!':")

probability = sentence_probs(sentence, unigram_counts, bigram_probs, trigram_probs)
print(f"Sentence: {sentence}")
print(f"Probability: {probability}")
