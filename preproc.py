import nltk
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk.tag import pos_tag

import json

five_letter_words = set(filter(lambda w: len(w) == 5, words.words()))

pos = pos_tag(five_letter_words)
pos_by_word = {}
for (word, p) in pos:
    pos_by_word[word] = p

filtered_dictionary = set(w for w in five_letter_words if 'NNP' not in pos_by_word[w])

frequencies_file = open('wiki_corpus_frequency.txt')
frequencies_raw = frequencies_file.read()
frequencies = frequencies_raw.splitlines()

frequency_by_word = {}
for word in filtered_dictionary:
    frequency_by_word[word] = 0

for line in frequencies:
    word, frequency_text = line.split(' ')
    if word in filtered_dictionary:
        frequency_by_word[word] = int(frequency_text)

sorted_by_freq = list(filtered_dictionary)
sorted_by_freq.sort(key=lambda w: -frequency_by_word[w])

top_5k = sorted_by_freq[:5000]

serialized = [{
    "word": w,
    "frequency": frequency_by_word[w]
} for w in top_5k]

out_file = open('top_5k_words.json', 'w')
json.dump(serialized, out_file)
