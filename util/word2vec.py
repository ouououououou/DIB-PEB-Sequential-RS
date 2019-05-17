# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#
# import gensim
#
# # Load Google's pre-trained Word2Vec model
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#
# # Checking vocabulary size
# vocab = model.vocab.keys()
# print("Vocabulary size is: " + str(len(vocab)))
#
# # Similarity between words
# print(model.similarity("the", "The"))

import gensim
import logging
import multiprocessing
import os
import sys
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from time import time
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', raw_html)
  return cleantext


class MySentences(object):
  def __init__(self, dirname):
    self.dirname = dirname

  def __iter__(self):
    word_tokenizer = RegexpTokenizer(r'\w+')
    for root, dirs, files in os.walk(self.dirname):
      for filename in files:
        file_path = root + '/' + filename
        for line in open(file_path):
          sline = line.strip()
          if sline == "":
            continue
          json_data = json.loads(line)
          raw_review = str(json_data['reviewText'])

          #tokenized_line = ' '.join(word_tokenizer.tokenize(raw_review))
          # is_alpha_word_line = [word for word in
          #                       tokenized_line.lower().split()
          #                       if word.isalpha()]
          # yield is_alpha_word_line
          raw_review = raw_review.strip().lower()

          yield word_tokenizer.tokenize(raw_review)

def runWord2vec(fileName, vec_size, min_count=10):
  data_path = '../dataset/raw_datasets/' + fileName + '/word2vec/raw_data'
  begin = time()

  sentences = MySentences(data_path)
  model = gensim.models.Word2Vec(sentences,
                                 size=vec_size,
                                 window=10,
                                 min_count=min_count,
                                 workers=multiprocessing.cpu_count(),
                                 sg=1)

  model.wv.save_word2vec_format('../dataset/raw_datasets/' + fileName + '/word2vec/model_file/word2vec_org',
                                '../dataset/raw_datasets/' + fileName + '/word2vec/model_file/vocabulary',
                                binary=True)

  end = time()
  print("Total procesing time: %d seconds" % (end - begin))


def checkResults(fileName, test_words):
  loaded_model = gensim.models.KeyedVectors.load_word2vec_format(
    '../dataset/raw_datasets/' + fileName + '/word2vec/model_file/word2vec_org', binary=True)
  vocab = loaded_model.vocab.keys()
  for test_word in test_words:
    if test_word in vocab:
      print(test_word + " is in vocab")
      print(loaded_model[test_word])
      print(loaded_model.most_similar(test_word))
    else:
      print(test_word + " is not in vocab")



if __name__ == '__main__':
  # musical_instruments_raw
  # instant_video_raw   sports_outdoors_raw digital_music_5_core_100
  # 'books_5', 'kindle_5', 'movies_and_tv_5', 'electronics_5'
  # 'cd_vinyl_5', 'kindle_5', 'movies_tv_5', 'electronics_5', 'video_games_5'
  fileNames = ['books_5']
  for fileName in fileNames:
      runWord2vec(fileName=fileName, vec_size=200, min_count=20)
      # check the results
      checkResults(fileName=fileName, test_words=['10', '20', '30', 'great', 'album'])





