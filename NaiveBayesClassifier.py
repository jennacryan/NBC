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

    def __init__(self, classifier, sentiment, text):
        self.sentiment = sentiment
        self.posprob = 0
        self.negprob = 0
        self.nwords = 0
        self.text = text.lower().split()
        for word in self.text:
            if not classifier.invalid_word(word):
                self.nwords += 1
                # self.dctnry = add_to_dctnry(word, self.dctnry)
                if word in self.dctnry:
                    self.dctnry[word] += 1
                else:
                    self.dctnry[word] = 1


class Classifier:
    digit = re.compile('\d')
    negdctnry = {}
    posdctnry = {}
    numposwords = 0
    numnegwords = 0
    stopwords = []
    traindata = []
    testdata = []
    trainacc = 0
    testacc = 0

    def __init__(self, training, testing):
        self.stopwords = self.load_stop_words()
        self.traindata = self.load_file(training)
        self.testdata = self.load_file(testing)

    def load_file(self, filename):
        file = open(filename, 'r')
        data = []
        trans = string.maketrans('', '')
        for line in file:
            text = line[1:].translate(trans, string.punctuation)
            review = Review(self, line[0], text)
            data.append(review)
        # print "time :",timeit.Timer('f(s)', 'from __main__ import s,loadTrainingFile as f').timeit(1000000)
        return data

    # load stop words from file into list
    def load_stop_words(self):
        return open('stopWords.txt', 'r').read().split()

    # check if word is too long, contains digits or is in the list of stop words
    def invalid_word(self, word):
        toolong = len(word) > 20
        hasdigit = self.digit.search(word) is not None
        stopword = word in self.stopwords
        return toolong or hasdigit or stopword

    def train_classifier(self):
        for review in self.traindata:
            for word in review.dctnry:
                if review.sentiment == '1':
                    self.numposwords += 1   # review.dctnry[word] ?
                    self.posdctnry[word] = review.dctnry[word] / review.nwords
                    # logger.debug('self.posdctnry[%s] = %f', word, self.posdctnry[word])
                else:
                    self.numnegwords += 1   # review.dctnry[word] ?
                    self.negdctnry[word] = review.dctnry[word] / review.nwords
        return

    def test_data(self):
        totalwords = self.numnegwords + self.numposwords
        probposword = math.log(1 + self.numposwords / totalwords)
        probnegword = math.log(1 + self.numnegwords / totalwords)
        logger.debug('probposword : %f', probposword)
        logger.debug('probnegword : %f', probnegword)

        for review in self.traindata:
            review.posprob = probposword
            review.negprob = probnegword
            for word in review.dctnry:
                if word in self.posdctnry:
                    review.posprob += math.log(1 + self.posdctnry[word])
                elif word in self.negdctnry:
                    review.negprob += math.log(1 + self.negdctnry[word])
            # logger.debug('posprob train : %f', review.posprob)
            # logger.debug('negprob train : %f', review.negprob)
            if review.posprob > review.negprob and review.sentiment == '1':
                self.trainacc += 1
            elif review.posprob < review.negprob and review.sentiment == '0':
                self.trainacc += 1
        self.trainacc /= len(self.traindata)
        logger.debug('training accuracy : %f', self.trainacc)

        for review in self.testdata:
            review.posprob = probposword
            review.negprob = probnegword
            for word in review.text:
                if word in self.posdctnry:
                    # logger.debug('positive word[freq] : %s[%d]', word, self.poswords[word])
                    review.posprob += math.log(1 + self.posdctnry[word] / self.numposwords)
                    # logger.debug('%s prob += log(1 + %d / %d) = %f', word, self.poswords[word], self.numposwords, review.posprob)
                elif word in self.negdctnry:
                    # logger.debug('negative word[freq] : %s[%d]', word, self.negwords[word])
                    review.negprob += math.log(1 + self.negdctnry[word] / self.numnegwords)
            logger.debug('posprob test : %f', review.posprob)
            logger.debug('negprob test : %f', review.negprob)
            if review.posprob > review.negprob and review.sentiment == '1':
                self.testacc += 1
            elif review.posprob < review.negprob and review.sentiment == '0':
                self.testacc += 1
        self.testacc /= len(self.testdata)
        logger.debug('numposwords  : %d', self.numposwords)
        logger.debug('numnegwords  : %d', self.numnegwords)
        logger.debug('testing accuracy : %f', self.testacc)

        return

    def print_debug(self):
        logger.debug('positive words : %d', self.numposwords)
        for word, freq in self.posdctnry.items():
            logger.debug("%s : %f", word, freq)
        logger.debug('negative words : %d', self.numnegwords)
        for word, freq in self.negdctnry.items():
            logger.debug('%s : %f', word, freq)
        return


NaiveBayes = Classifier(sys.argv[1], sys.argv[2])
NaiveBayes.train_classifier()
NaiveBayes.print_debug()
# NaiveBayes.test_data()
