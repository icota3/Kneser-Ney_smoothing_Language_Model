import pickle

# Load the saved models
with open('unigrams.pkl', 'rb') as input_file:
    unigram_counts = pickle.load(input_file)

with open('bigrams.pkl', 'rb') as input_file:
    bigram_probs = pickle.load(input_file)

with open('trigrams.pkl', 'rb') as input_file:
    trigram_probs = pickle.load(input_file)

# Function to generate text
def generate_text(prefix, n=10, unigram_counts=unigram_counts, bigram_probs=bigram_probs, trigram_probs=trigram_probs):
    words = prefix.split()
    words = ['<s>'] + words[-2:]  # Consider only the last two words as context
    generated_text = words.copy()

    # Generate words
    for _ in range(n):
        trigram = tuple(words)  # Convert context words to a tuple
        if trigram in trigram_probs:
            next_word = max(trigram_probs[trigram], key=trigram_probs[trigram].get)
        else:
            bigram = tuple(words[-2:])  # Use the last two words as a fallback
            if bigram in bigram_probs:
                next_word = max(bigram_probs[bigram].keys(), key=bigram_probs[bigram].get)
            else:
                next_word = max(unigram_counts.keys(), key=unigram_counts.get)
        
        generated_text.append(next_word)
        words = words[1:] + [next_word]

    return ' '.join(generated_text[2:])  # Exclude start tokens from generated text

# Prompt user for a sentence
user_input = input("Enter the start of a sentence: ")

# Generate text
generated_sentence = generate_text(user_input)

# Print generated sentence
print(f"Generated sentence: {generated_sentence}")
