from __future__ import division
import logging
import string
import sys
import math
import re
import timeit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_to_dctnry(wrd, dctnry):
    if wrd in dctnry:
        dctnry[wrd] += 1
    else:
        dctnry[wrd] = 1
    return dctnry


class Review:
    sentiment = '0'
    text = []
    dctnry = {}
    nwords = 0
    posprob = 0
    negprob = 0

    def __init__(self, sentiment, text):
        self.sentiment = sentiment
        self.text = text
        self.dctnry = {}
        self.nwords = len(self.text)
        self.posprob = 0
        self.negprob = 0
        for i in range(len(self.text)):
            # self.text[i] = self.stem_word(self.text[i])
            self.dctnry = add_to_dctnry(self.text[i], self.dctnry)

    def stem_word(self, word):
        if word[-1] == 's':
            if word[-4:] == 'sses':
                word = word[:-2]
            elif word[-3:] == 'ies':
                word = word[:-2]
            elif word[len(word) - 2] != 's':
                word = word[:-1]
        if word[-3:] == 'eed':
            if len(word[:-3]) > 1:
                word = word[:-1]
        elif sum(letter in 'aeiou' for letter in word[:-3]) > 0 and (word[-2:] == 'ed' or word[-3:] == 'ing'):
            if word[-2:] == 'ed':
                word = word[:-2]
            elif word[-3:] == 'ing':
                word = word[:-3]
            if word[-2:] == 'at' or word[-2:] == 'bl' or word[-2:] == 'iz':
                word += 'e'
            elif word[-1] == word[-2] and word[-1] != 'l' and word[-1] != 's' and word[-1] != 'z':
                word = word[:-1]
            elif len(word) > 2 and word[-1] != 'w' and word[-1] != 'x' and word[-1] != 'y':
                # check for a consonant
                if sum(letter in 'bcdfghjklmnpqrstvwxyz' for letter in word[-3:]) > 0:
                    # check for a vowel
                    if sum(letter in 'aeiou' for letter in word[-3:]) > 0:
                        word += 'e'
        return word

class Classifier:
    digit = re.compile('\d')
    negTF = {}
    posTF = {}
    negIDF = {}
    posIDF = {}
    nposwords = 0
    nnegwords = 0
    nposdocs = 0
    nnegdocs = 0
    stopwords = []
    trainingdata = []
    testingdata = []

    def __init__(self, training, testing):
        self.stopwords = self.load_stop_words()
        self.trainingdata = self.load_file(training)
        self.testingdata = self.load_file(testing)

    def load_file(self, filename):
        file = open(filename, 'r')

        data = []
        trans = string.maketrans('', '')
        for line in file:
            review = line[1:].translate(trans, string.punctuation)
            review = review.lower().split()
            review = [word for word in review if not self.invalid_word(word)]
            data.append(Review(line[0], review))

        # print "time :",timeit.Timer('f(s)', 'from __main__ import s,loadTrainingFile as f').timeit(1000000)
        return data

    # load stop words from file into list
    def load_stop_words(self):
        return open('stopWords.txt', 'r').read().split()

    # check if word is too long, contains digits or is in the list of stop words
    def invalid_word(self, word):
        toolong = len(word) > 20
        hasdigit = self.digit.search(word) is not None
        # stopword = word in self.stopwords
        # return toolong or hasdigit or stopword
        return toolong or hasdigit

    def train_classifier(self):
        for review in self.trainingdata:
            dlist = []
            if review.sentiment == '1':
                self.nposdocs += 1
            else:
                self.nnegdocs += 1
            for word in review.text:
                if word not in dlist:
                    dlist.append(word)
                    if review.sentiment == '1':
                        self.nposwords += 1
                        if word in self.posTF:
                            self.posTF[word] += review.dctnry[word] / review.nwords
                        else:
                            self.posTF[word] = review.dctnry[word] / review.nwords
                    else:
                        self.nnegwords += 1
                        if word in self.negTF:
                            self.negTF[word] += review.dctnry[word] / review.nwords
                        else:
                            self.negTF[word] = review.dctnry[word] / review.nwords

            for word in dlist:
                if review.sentiment == '1':
                    self.posIDF = add_to_dctnry(word, self.posIDF)
                else:
                    self.negIDF = add_to_dctnry(word, self.negIDF)
            # logger.debug('n dlist : %d', len(dlist))
        return

    def test_data(self):
        trainingacc = self.test_accuracy(self.trainingdata)
        testingacc = self.test_accuracy(self.testingdata)

        print 'training accuracy : ' + str(trainingacc)
        print 'testing accuracy  : ' + str(testingacc)

        return

    def test_accuracy(self, data):
        acc = 0
        totalwords = self.nnegwords + self.nposwords
        probposword = math.log(1 + self.nposwords / totalwords)
        probnegword = math.log(1 + self.nnegwords / totalwords)

        for review in data:
            posprob = probposword
            negprob = probnegword
            dlist = []
            for word in review.text:
                if word not in dlist:
                    dlist.append(word)
                    if word in self.posTF:
                        posprob += math.log(1 + self.posTF[word]) * (1 + math.log(self.nposdocs / self.posIDF[word]))
                        logger.debug('%s posIDF : %f', word, self.nposdocs / self.posIDF[word])
                    elif word in self.negTF:
                        negprob += math.log(1 + self.negTF[word]) * (1 + math.log(self.nnegdocs / self.negIDF[word]))
                        logger.debug('%s negIDF : %f', word, self.nnegdocs / self.negIDF[word])
            # logger.debug('posprob test : %f', posprob)
            # logger.debug('negprob test : %f', negprob)
            if posprob > negprob and review.sentiment == '1':
                acc += 1
            elif posprob < negprob and review.sentiment == '0':
                acc += 1
        return acc / len(data)

    def print_dicts(self):
        logger.debug("positive IDF : ")
        for word, freq in self.posIDF.items():
            logger.debug("%s : %f", word, freq)
        logger.debug("negative IDF : ")
        for word, freq in self.negIDF.items():
            logger.debug("%s : %f", word, freq)

        logger.debug('nposIDF : %d', len(self.posIDF))
        logger.debug('nnegIDF : %d', len(self.negIDF))
        return


# main method
if __name__ == '__main__':
    NaiveBayes = Classifier(sys.argv[1], sys.argv[2])
    NaiveBayes.train_classifier()
    # NaiveBayes.print_dicts()
    NaiveBayes.test_data()

