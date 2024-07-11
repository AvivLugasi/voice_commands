import phonetics
from phonetics import metaphone, nysiis

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

# soundex
print("soundex")
print(soundex_encoder("rich"))
print(soundex_encoder("breach"))
print(soundex_encoder("bleach"))
print(soundex_encoder("collect"))
print(soundex_encoder("correct"))
print(soundex_encoder("stick"))
print(soundex_encoder("stack"))
print(soundex_encoder("god"))
print(soundex_encoder("gold"))

#  metaphone
print("metaphone")
print(metaphone("rich"))
print(metaphone("breach"))
print(metaphone("bleach"))
print(metaphone("collect"))
print(metaphone("correct"))
print(metaphone("stick"))
print(metaphone("stack"))
print(metaphone("god"))
print(metaphone("gold"))

# nysiis
print("nysiis")
print(nysiis("rich"))
print(nysiis("breach"))
print(nysiis("bleach"))
print(nysiis("collect"))
print(nysiis("correct"))
print(nysiis("stick"))
print(nysiis("stack"))
print(nysiis("god"))
print(nysiis("gold"))
