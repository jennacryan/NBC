from __future__ import division
import logging
import string
import sys
import math
import re
import timeit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_to_dctnry(self, wrd, dctnry):
        if wrd in dctnry:
            dctnry[wrd] += 1
        else:
            dctnry[wrd] = 1
        return dctnry


class Review:
    sentiment = '0'
    text = []
    posprob = 0
    negprob = 0

    def __init__(self, sentiment, text):
        self.sentiment = sentiment
        self.text = text
        self.posprob = 0
        self.negprob = 0


class Classifier:
    digit = re.compile('\d')
    negdctnry = {}
    posdctnry = {}
    numposwords = 0
    numnegwords = 0
    stopwords = []
    trainingdata = []
    testingdata = []
    trainingacc = 0
    testingacc = 0

    def __init__(self, training, testing):
        self.stopwords = self.load_stop_words()
        self.trainingdata = self.load_file(training)
        self.testingdata = self.load_file(testing)

    def load_file(self, filename):
        file = open(filename, "rb")

        data = []
        trans = string.maketrans("", "")
        for line in file:
            review = line[1:].translate(trans, string.punctuation)
            review = review.lower().split()
            review = [word for word in review if not self.invalid_word(word)]
            data.append(Review(line[0], review))

        # print "time :",timeit.Timer('f(s)', 'from __main__ import s,loadTrainingFile as f').timeit(1000000)
        return data

    # load stop words from file into list
    def load_stop_words(self):
        return open("stopWords.txt", "rb").read().split()

    # check if word is too long, contains digits or is in the list of stop words
    def invalid_word(self, word):
        toolong = len(word) > 20
        hasdigit = self.digit.search(word) is not None
        stopword = word in self.stopwords
        return toolong or hasdigit or stopword


    def train_classifier(self):
        for review in self.trainingdata:
            for word in review.text:
                if review.sentiment == '1':
                    self.numposwords += 1
                    self.posdctnry = add_to_dctnry(word, self.posdctnry)
                else:
                    self.numnegwords += 1
                    self.negdctnry = add_to_dctnry(word, self.negdctnry)
        return

    def test_data(self):
        totalwords = self.numnegwords + self.numposwords
        ppositive = math.log(1 + self.numposwords / totalwords)
        pnegative = math.log(1 + self.numnegwords / totalwords)

        for review in self.trainingdata:
            review.posprob = ppositive
            review.negprob = pnegative
            for word in review.text:
                if word in self.posdctnry:
                    review.posprob += math.log(1 + self.poswords[word] / self.numposwords)
                elif word in self.negdctnry:
                    review.negprob += math.log(1 + self.negwords[word] / self.numnegwords)
            # logger.debug('posprob train : %f', review.posprob)
            # logger.debug('negprob train : %f', review.negprob)
            if review.posprob > review.negprob and review.sentiment == '1':
                self.trainingacc += 1
            elif review.posprob < review.negprob and review.sentiment == '0':
                self.trainingacc += 1
        self.trainingacc /= len(self.trainingdata)
        logger.debug('ppositive : %f', ppositive)
        logger.debug('pnegative : %f', pnegative)
        logger.debug('training accuracy : %f', self.trainingacc)

        for review in self.testingdata:
            review.posprob = ppositive
            review.negprob = pnegative
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
            logger.debug('numposwords  : %d', self.numposwords)
            logger.debug('numnegwords  : %d', self.numnegwords)
            if review.posprob > review.negprob and review.sentiment == '1':
                self.testingacc += 1
            elif review.posprob < review.negprob and review.sentiment == '0':
                self.testingacc += 1
        self.testingacc /= len(self.testingdata)
        logger.debug("testing accuracy : %f", self.testingacc)

        return

    def print_dicts(self):
        logger.debug("positive words : ")
        for word, freq in self.posdctnry.items():
            logger.debug("%s : %d", word, freq)
        logger.debug("negative words : ")
        for word, freq in self.negdctnry.items():
            logger.debug("%s : %d", word, freq)
        return


NaiveBayes = Classifier(sys.argv[1], sys.argv[2])
NaiveBayes.train_classifier()
# NaiveBayes.print_dicts()
NaiveBayes.test_data()