from Model.BertModel import SentenceEmbedderModel, find_top_n_similar_sentences, SequenceClassificationModel
from Model.Utils import cosine_sim

fine_tuned_bert = SequenceClassificationModel()

base_bert = SentenceEmbedderModel(model_name='bert-base-uncased')


sentence_1 = "breach with explosives and clear throw stun grenade"
sentence_2 = "open and clear use charges throw stinger grenade"
sentence_3 = "open and clear use explosive throw stun grenade"
print(f"sentence 1:{sentence_1}")
print(f"sentence 2:{sentence_2}")
print(f"sentence 3:{sentence_3}")

sentence_vectors_np_1 = base_bert.sentence_to_vector(processed_sentence=sentence_1)
sentence_vectors_np_2 = base_bert.sentence_to_vector(processed_sentence=sentence_2)
sentence_vectors_np_3 = base_bert.sentence_to_vector(processed_sentence=sentence_3)

print(f"base model similarity sentence 1 and 3: {cosine_sim(sentence_vectors_np_1, sentence_vectors_np_3, is_1d = True)}")
print(f"base model similarity sentence 2 and 3: {cosine_sim(sentence_vectors_np_2, sentence_vectors_np_3, is_1d = True)}")

sentence_vectors_np_1 = fine_tuned_bert.sentence_to_vector(sentence_1)
sentence_vectors_np_2 = fine_tuned_bert.sentence_to_vector(sentence_2)
sentence_vectors_np_3 = fine_tuned_bert.sentence_to_vector(sentence_3)

print(f"fine tuned model similarity sentence 1 and 3: {cosine_sim(sentence_vectors_np_1, sentence_vectors_np_3, is_1d = True)}")
print(f"fine tuned model similarity sentence 2 and 3: {cosine_sim(sentence_vectors_np_2, sentence_vectors_np_3, is_1d = True)}")