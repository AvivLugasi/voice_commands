from metaphone import doublemetaphone
import Levenshtein
from typing import Literal

SOUNDEX_CODING_DICT = {"b": "1",
                       "f": "1",
                       "p": "1",
                       "v": "1",
                       "c": "2",
                       "g": "2",
                       "j": "2",
                       "k": "2",
                       "q": "2",
                       "s": "2",
                       "x": "2",
                       "z": "2",
                       "d": "3",
                       "t": "3",
                       "l": "4",
                       "m": "5",
                       "n": "5",
                       "r": "6"}
VOWELS_ENCODING = "."
SOUNDEX_HASH_LEN = 4


def soundex_encoder(input_word: str):
    hash_code = input_word[0]
    if len(input_word) > 1:
        input_word = input_word[1:]
        previous_hash_value = SOUNDEX_CODING_DICT[hash_code] if hash_code in SOUNDEX_CODING_DICT else '.'
        for char in input_word:
            if char in SOUNDEX_CODING_DICT:
                hash_value = SOUNDEX_CODING_DICT[char]
            else:
                hash_value = VOWELS_ENCODING
            if hash_value != previous_hash_value:
                previous_hash_value = hash_value
                hash_code = hash_code + hash_value

    hash_code = hash_code.replace(VOWELS_ENCODING, '')

    if len(hash_code) < SOUNDEX_HASH_LEN:
        while len(hash_code) != SOUNDEX_HASH_LEN:
            hash_code = hash_code + "0"
    elif len(hash_code) > SOUNDEX_HASH_LEN:
        hash_code = hash_code[:4]

    return hash_code

def calc_corpus_phonetic_codes(corpus,
                               method: Literal["double metaphone", "soundex"] = "double metaphone"):
    word_codes_dict = {}
    for word in corpus:
        if method == "double metaphone":
            word_codes_dict[word] = doublemetaphone(word)
        elif method == "soundex":
            word_codes_dict[word] = soundex_encoder(word)
        else:
            raise ValueError(f"Invalid value: {method}. must be one of: double metaphone, soundex.")

    return word_codes_dict

def find_closest_word(input_word,
                      word_codes_dict: dict,
                      method: Literal["double metaphone", "soundex"] = "double metaphone"):
    if method == "double metaphone":
        input_word_code = doublemetaphone(input_word)
    elif method == "soundex":
        input_word_code = soundex_encoder(input_word)
    else:
        raise ValueError(f"Invalid value: {method}. must be one of: double metaphone, soundex.")

    smallest_distance = float('inf')
    most_similar = []
    for word, corpus_word_code in word_codes_dict.items():
        if method == "double metaphone":
            distance = calc_distance_double_metaphone(input_word_code, corpus_word_code)
        else:
            distance = _hamming_distance(input_word_code, corpus_word_code)
        if distance < smallest_distance:
            most_similar = [word]
            smallest_distance = distance
            # calc confidence level in the current most phonetic similar word
            if method == "double metaphone":
                combined_codes_list = corpus_word_code + list(input_word_code)
                max_length = max(len(element) for element in combined_codes_list)
                confidence = 1 - smallest_distance / max_length
            else:
                confidence = 1 - smallest_distance / len(input_word_code)
        elif distance == smallest_distance:
            most_similar.append(word)

    return most_similar, smallest_distance, confidence

def calc_distance_double_metaphone(input_word_code, corpus_word_code):
    # check prime key to prime key match
    smallest_distance = Levenshtein.distance(input_word_code[0], corpus_word_code[0])
    # check prime key to secondary key match
    if input_word_code[1] != '':
        distance = Levenshtein.distance(input_word_code[1], corpus_word_code[0])
        smallest_distance = _update_smallest_distance(smallest_distance, distance)
    if corpus_word_code[1] != '':
        distance = Levenshtein.distance(input_word_code[0], corpus_word_code[1])
        smallest_distance = _update_smallest_distance(smallest_distance, distance)
    # check secondary key to secondary key match
    if corpus_word_code[1] != '' and input_word_code[1] != '':
        distance = Levenshtein.distance(input_word_code[1], corpus_word_code[1])
        smallest_distance = _update_smallest_distance(smallest_distance, distance)

    return smallest_distance

def _update_smallest_distance(smallest_distance, distance):
    if distance < smallest_distance:
        smallest_distance = distance
    return smallest_distance

def _hamming_distance(input_code, corpus_word_code):
    if len(input_code) != len(corpus_word_code):
        raise ValueError(f"strings lengths not match!.")

    distance = 0
    for input_code_char, corpuse_word_code_char in zip(input_code, corpus_word_code):
        if input_code_char != corpuse_word_code_char:
            distance+=1

    return distance