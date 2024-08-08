import json
import numpy as np
import ast

# formatted commands default json file path
FORMATTED_COMMANDS_FILE_PATH = "Assets/Data/ReadyOrNot/ReadyOrNotCommandsFormatted.json"
# json keys
PROFILE_KEY = "profile name"
COMMANDS_GROUPS_KEY = "commands groups"
GROUP_KEY = "group name"
COMMANDS_LIST_KEY = "commands list"
COMMAND_KEY = "command name"
COMMAND_VARIATIONS_KEY = "variations"
COMMAND_KEYS_SEQUENCE_KEY = "keys_sequence"

# commands variations and embeddings default file path
COMMANDS_VARIATIONS_AND_EMBEDDING_FILE_PATH = "Assets/Data/ReadyOrNot/CommandsAndEmbedding"
# variation, embedding seperator
VARIATION_EMBEDDING_SEPERATOR = ":"


class DataIO:
    def __init__(self,
                 formatted_commands_file_path=FORMATTED_COMMANDS_FILE_PATH,
                 commands_variations_and_embedding_file_path=COMMANDS_VARIATIONS_AND_EMBEDDING_FILE_PATH):

        self.formatted_commands_file_path = formatted_commands_file_path
        self.commands_variations_and_embedding_file_path = commands_variations_and_embedding_file_path
        self.formatted_commands_dict = self.load_commands_json()

    def load_commands_json(self):
        try:
            with open(self.formatted_commands_file_path, 'r') as file:
                commands_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {self.formatted_commands_file_path} does not exist.")
            commands_data = {}
        except json.JSONDecodeError:
            print(f"Error: The file {self.formatted_commands_file_path} contains invalid JSON.")
            commands_data = {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            commands_data = {}

        return commands_data

    def write_commands_json(self):
        with open(self.formatted_commands_file_path, 'w') as json_file:
            json.dump(self.formatted_commands_dict, json_file, indent=4)

    def get_commands_list(self):
        # List to hold all variations
        commands_list = []

        # Iterate through the commands groups and collect variations
        for group in self.formatted_commands_dict.get(COMMANDS_GROUPS_KEY, []):
            for command in group.get(COMMANDS_LIST_KEY, []):
                commands_list.append(command)

        return commands_list

    def get_commands_variations(self):
        # List to hold all commands
        commands_list = self.get_commands_list()
        # List to hold all variations
        variations_list = []

        for command in commands_list:
            variations = command.get(COMMAND_VARIATIONS_KEY, [])
            variations_list.extend(variations)

        return variations_list

    def get_commands_variations_words_set(self):
        # List to hold all commands
        commands_list = self.get_commands_list()
        # List to hold all variations
        words_set = set()

        for command in commands_list:
            variations = command.get(COMMAND_VARIATIONS_KEY, [])
            for variation in variations:
                words_set.update(variation.split())

        return words_set

    def get_variations_keys_sequence_dict(self):
        # List to hold all commands
        commands_list = self.get_commands_list()
        # List to hold all variations
        variations_dict = {}

        for command in commands_list:
            variations = command.get(COMMAND_VARIATIONS_KEY, [])
            keys_sequence = command.get(COMMAND_KEYS_SEQUENCE_KEY, [])
            for variation in variations:
                variations_dict[variation] = keys_sequence

        return variations_dict

    def load_variations_embedding_dict(self,
                                       file_path=None):
        if file_path is None:
            file_path = self.commands_variations_and_embedding_file_path
        variations_embedding_dict = {}
        with open(file_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line_splited = line.strip().split(VARIATION_EMBEDDING_SEPERATOR)
                variation = line_splited[0]
                embedding_tensor = line_splited[1]
                variations_embedding_dict[variation] = np.array(ast.literal_eval(embedding_tensor))

        return variations_embedding_dict

    def write_variations_embedding_dict(self,
                                        variations_embedding_dict:dict,
                                        file_path=None):
        if file_path is None:
            file_path = self.commands_variations_and_embedding_file_path

        for command_variation, embedding in variations_embedding_dict.items():
            # Convert the numpy tensor to a string
            tensor_string = np.array2string(embedding,
                                            separator=',',
                                            formatter={'all': lambda x: str(x)},
                                            max_line_width=np.inf)
            # Create the line with the string and tensor string separated by a colon
            line_to_write = f"{command_variation}:{tensor_string}"

            # Write the line to a file
            with open(file_path, 'a') as file:
                file.write(line_to_write + '\n')

    def write_words_phonetic_codes_dict(self,
                                        words_phonetic_cods_dict: dict,
                                        file_path=None):
        if file_path is None:
            file_path = self.commands_variations_and_embedding_file_path

        for word, phonetic_code in words_phonetic_cods_dict.items():
            # if phonetic_code is tuple(double metaphone case)
            if isinstance(phonetic_code, tuple):
                line_to_write = f"{word}:{phonetic_code[0]}:{phonetic_code[1]}"
            # else if it is a string(soundex case)
            else:
                line_to_write = f"{word}:{phonetic_code}"

            # Write the line to a file
            with open(file_path, 'a') as file:
                file.write(line_to_write + '\n')

    def load_words_phonetic_codes_dict(self,
                                       file_path=None):
        if file_path is None:
            file_path = self.commands_variations_and_embedding_file_path

        words_codes_dict = {}
        with open(file_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line_splited = line.strip().split(VARIATION_EMBEDDING_SEPERATOR)
                word = line_splited[0]
                # double metaphone case
                if len(line_splited) == 3:
                    phonetic_codes = [line_splited[1],line_splited[2]]
                else:
                    # soundex case
                    phonetic_codes = line_splited[1]

                words_codes_dict[word] = phonetic_codes

        return words_codes_dict
