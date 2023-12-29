"""""
import gensim
from gensim.models import KeyedVectors

# Replace 'path_to_downloaded_file' with the actual path to the downloaded Word2Vec binary file.
file_path = 'C:\\Users\\Saku\\Downloads\\GoogleNews-vectors-negative300.bin.gz'

# Load the word embeddings for the first million vectors from their binary form.
wv = KeyedVectors.load_word2vec_format(file_path, binary=True, limit=1000000)

# Save the word embeddings as a flat file (Word2Vec format).
flat_file_path = 'vectors.txt'
wv.save_word2vec_format(flat_file_path)
"""""
"""""
import gensim
from gensim.models import KeyedVectors
import pandas as pd

# Load the previously saved Word2Vec vectors as a KeyedVectors model.
flat_file_path = 'vectors.txt'
wv = KeyedVectors.load_word2vec_format(flat_file_path, binary=False)

# Load the phrases from the CSV file.
csv_file_path = 'phrases (1).csv'
df = pd.read_csv(csv_file_path)

# Function to calculate the similarity of two phrases.
def calculate_similarity(phrase1, phrase2):
    words1 = phrase1.split()
    words2 = phrase2.split()

    # Filter out words without embeddings
    words1 = [word for word in words1 if word in wv]
    words2 = [word for word in words2 if word in wv]

    if not words1 or not words2:
        # Return a default similarity if no common words with embeddings
        return 0.0

    # Calculate the similarity using cosine similarity
    similarity = wv.n_similarity(words1, words2)
    return similarity

# Calculate and print the similarity for each pair of phrases.
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        phrase1 = df.iloc[i]['Phrase']
        phrase2 = df.iloc[j]['Phrase']
        similarity = calculate_similarity(phrase1, phrase2)
        print(f"Similarity between '{phrase1}' and '{phrase2}': {similarity:.4f}")

"""""

import gensim
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


flat_file_path = 'vectors.txt'
wv = KeyedVectors.load_word2vec_format(flat_file_path, binary=False)


csv_file_path = 'text_distance.csv'
#df = pd.read_csv(csv_file_path)
df = pd.read_csv('C:\\Users\\Saku\\Desktop\\phrases (1).csv')

# Function to calculate the approximate phrase vector.
def calculate_phrase_vector(phrase):
    words = phrase.split()
    # Filter out words without embeddings
    words = [word for word in words if word in wv]
    if not words:
        
        return np.zeros(wv.vector_size)
    
    word_vectors = [wv[word] for word in words]
    normalized_sum = np.sum(word_vectors, axis=0) / np.linalg.norm(np.sum(word_vectors, axis=0))
    return normalized_sum


phrase_vectors = np.array([calculate_phrase_vector(phrase) for phrase in df['Phrase']])


l2_distances = euclidean_distances(phrase_vectors)

cosine_distances = cosine_distances(phrase_vectors)

results_df = pd.DataFrame({
    'Phrase1': df['Phrase'].repeat(len(df)),
    'Phrase2': np.tile(df['Phrase'], len(df)),
    'L2_Distance': l2_distances.flatten(),
    'Cosine_Distance': cosine_distances.flatten()
})

results_df.to_csv('distances_results.csv', index=False)

def find_closest_match(user_input):
    user_vector = calculate_phrase_vector(user_input)
    if np.all(user_vector == 0):
        return "No matching phrase found (no valid embeddings for input)"


    user_distances = cosine_distances([user_vector], phrase_vectors.flatten().reshape(1, -1))

    closest_match_index = np.argmin(user_distances)

    closest_match_phrase = df.loc[closest_match_index, 'Phrase']
    closest_match_distance = user_distances[0, closest_match_index]

    return closest_match_phrase, closest_match_distance


user_input_phrase = "your input phrase here"
closest_match, distance = find_closest_match(user_input_phrase)
print(f"Closest Match: {closest_match}")
print(f"Distance: {distance:.4f}")

