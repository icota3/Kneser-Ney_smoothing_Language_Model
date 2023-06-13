import math
import pickle

# Load the trigram model from the file
with open('trigrams.pkl', 'rb') as input_model:
    trigram_probs = pickle.load(input_model)

# Preprocess the text (tokenize into words or appropriate units)
with open('output_preprocessed.txt', 'r', encoding='utf-8') as input_text:
    text=input_text.read()
    
# Preprocess the text based on your language model (e.g., tokenize into words)

# Initialize variables for perplexity calculation
log_perplexity = 0
total_words = 0

# Iterate through the preprocessed text
for i in range(1, len(text)):
    # Get the current trigram (previous two words and the current word)
    trigram = (text[i-2], text[i-1], text[i])
    
    # Check if the trigram exists in the trigram probabilities
    if trigram in trigram_probs:
        # Get the trigram probability from the loaded model
        trigram_prob = trigram_probs[trigram]
        
        # Compute the log-perplexity of the current word
        log_perplexity += -math.log2(trigram_prob)
        total_words += 1

# Compute the average perplexity
average_log_perplexity = log_perplexity / total_words
perplexity = 2 ** average_log_perplexity

# Print the perplexity
print("Perplexity using trigram model:", perplexity)

