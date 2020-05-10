"""
Jacob Baginski Doar, Edward Christenson, Daniel Xiong
nGramModels.py
Due 1/26/2020
"""

import math

def tokenize(sentence):
    """
    Return a tokenized list of the sentence
    Appends a <stop> token to the end of each sentence

    Params:
    sentence: str
        a sentence as a string

    Returns:
    tokens: list(str)
        list of tokens in sentence string
    """
    sentence = sentence.strip('\n')
    tokens = sentence.split(" ")
    tokens.append("<STOP>")
    return tokens

def countTokenOccurences(sentenceList):
    """
    Return a dictionary of each token and how many times it occurs

    Params:
    sentenceList: list(list(str))
        list of sentences, where a sentence is a list of token strings

    Returns:
    tokenOccurences: dict
        dict[token] = # of occurences of a unique token in sentences
    """
    tokenOccurences = {}

    for sentence in sentenceList:
        for token in sentence:
            if token in tokenOccurences:
                tokenOccurences[token] += 1
            else:
                tokenOccurences[token] = 1

    return tokenOccurences

def parseTrainingOOV(sentenceList, margin):
    """
    Return sentenceList with tokens with margin or less than occurences replaced by <UNK>

    Params:
    sentenceList: list(list(str))
        list of sentences, where a sentence is a list of token strings
    margin: int
        threshold for 'rare' tokens to be replaced by <UNK>

    Returns:
    newSentenceList: list(list(str))
        list of sentences, with 'rare' tokens replaced by <UNK>
    vocab: set
        set of unique tokens, including <UNK>
    """
    newSentenceList = []
    vocab = set()
    tokenOccurences = countTokenOccurences(sentenceList)

    # Remove tokens with more than margin occurences from tokenOccurences
    for key, value in tokenOccurences.copy().items():
        if value > margin:
            tokenOccurences.pop(key)
            vocab.add(key) # adds keys to vocab set
    vocab.add("<UNK>")  # add <UNK> to vocab set

    # Replace tokens with less then margin occurences with UNK
    for sentence in sentenceList:
        newSentence = []
        for token in sentence:
            if token not in tokenOccurences:
                newSentence.append(token)
            else:
                newSentence.append("<UNK>")
        newSentenceList.append(newSentence)

    return newSentenceList, vocab

def parseOOV(sentenceList, vocab):
    """
    Return sentenceList with words that dont occur in vocab replaced by <UNK>

    Params:
    sentenceList: list(list(str))
        list of sentences, where a sentence is a list of token strings
    vocab: set(str)
        set of unique tokens

    Returns:
    newSentenceList: list(list(str))
        list of sentences, with all tokens not found in vocab replaced by <UNK>
    """
    newSentenceList = []
    for sentence in sentenceList:
        newSentence = []
        for token in sentence:
            if token not in vocab:
                newSentence.append("<UNK>")
            else:
                newSentence.append(token)
        newSentenceList.append(newSentence)

    return newSentenceList

def ngram(sentence, n):
    """
    Return sentence broken into a list of n-grams

    Params:
    sentence: list(str)
        list of token strings
    n: int
        size of each n-gram

    Returns:
    ngramList: list(list(str))
        list of n-grams
    """
    ngramList = []

    for i in range(len(sentence)-n+1):
        ngramList.append(sentence[i:i+n])

    return ngramList

def ngramProbability(languageModel, word, context):
    """
    Return probability of word given context

    Params:
    languageModel: dict[context: dict[token: count]]
        A dictionary of contexts where each context is a dictionary of tokens that follow context and their occurences
    word: str
        next word in the sentence
    context: tuple(word_i, ... , word_i+n-1)
        previous words in the sentence

    Returns:
    probability: float
        probability of word following context
    """
    numerator = 0
    denominator = 1

    if context in languageModel:
        denominator = sum(languageModel[context].values())

        if word in languageModel[context]:
            numerator = languageModel[context][word]

    return float(numerator) / float(denominator)

def unigramModelTrain(sentenceList):
    """
    Trains a unigram model

    Params:
    sentenceList: list(list(str))
        list of sentences, where a sentence is a list of token strings

    Returns:
    modelProbabilities: dict[token: probability]
        A dictionary of tokens and their probability of occuring

    comments here
    """
    model = {}
    modelProbabilities = {}
    N = 0

    for sentence in sentenceList:
        N += len(sentence)
        for token in sentence:
            if token not in model:
                model[token] = 1
            else:
                model[token] += 1

    for key in model:
        modelProbabilities[key] = model.get(key) / N

    return modelProbabilities

def ngramModelTrain(sentenceList, n):
    """
    Trains a bigram model

    Params:
    sentenceList: list(list(str))
        list of sentences, where a sentence is a list of token strings

    n: int
        size of ngrams
    Returns:
    modelProbabilities: dict[context: dict[token: probability]]
        A dictionary of contexts where each context is a dictionary of tokens that follow context and their probability of occuring
    """
    model = {}
    modelProbabilities = {}

    # Construct the model
    for sentence in sentenceList:
        for ng in ngram(sentence, n):
            context = tuple(ng[0:-1])
            word = ng[-1]

            if context not in model:
                model[context] = {}

            if word not in model[context]:
                model[context][word] = 1
            else:
                model[context][word] += 1

    # Construct the model probabilities
    for context, nextWord in model.items():
        for word in nextWord.keys():
            if context not in modelProbabilities:
                modelProbabilities[context] = {}

            modelProbabilities[context][word] = ngramProbability(model, word, context)

    return modelProbabilities

def unigramModelPerplexity(sentenceList, model):
    """
    Calculates perplexity of a trained unigram model

    Params:
    sentenceList: list(list(str))
        testing data
    model: dict[token: probability]
        trained model

    Returns:
    perplexity: float
        perplexity of model
    """
    m = 0
    l = 0

    for sentence in sentenceList:
        for word in sentence:
            m += 1
            if word not in model:
                p = 0
            else:
                p = model[word]

            if p > 0:
                l += math.log((p), 2)
            else:
                return math.inf

    perplexity = pow(2, -l / m)

    return perplexity


def ngramModelPerplexity(sentenceList, model):
    """
    Calculates perplexity of a trained ngram model

    Params:
    sentenceList: list(list(str))
        testing data
    model: dict[context: dict[token: probability]]
        trained model

    Returns:
    perplexity: float
        perplexity of model
    """
    m = 0
    l = 0

    n = len(next(iter(model))) + 1

    for sentence in sentenceList:
        ng = ngram(sentence, n)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            if context not in model:
                p = 0
            elif word not in model[context]:
                p = 0
            else:
                p = model[context][word]

            if p > 0:
                l += math.log((p), 2)
            else:
                return math.inf

    perplexity = pow(2, -l / m)

    return perplexity

def smoothedModelPerplexity(sentenceList, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3):
    """
    Implements linear interpolation smoothing

    Params:
    sentenceList: list(list(str))
        list of sentences
    unigramModel: dict[token: probability]
        trained unigram model
    bigramModel, trigramModel: dict[context: dict[token: probability]]
        trained language models
    lambda_1, lambda_2, lambda_3: float
        hyperparameters

    Returns:
    perplexity: float
        perplexity score on sentenceList
    """
    m = 0
    l = 0

    for sentence in sentenceList:
        ng = ngram(sentence, 3)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            p = 0

            if context in trigramModel:
                if word in trigramModel[context]:
                    p += trigramModel[context][word] * lambda_3
            if context in bigramModel:
                if word in bigramModel[context]:
                    p += bigramModel[context[1:2]][word] * lambda_2
            p += unigramModel[word] * lambda_1

            l += math.log((p), 2)

    perplexity = pow(2, -l / m)

    return perplexity

def main():
    sentenceList_train = []
    with open("../A1-Data/1b_benchmark.train.tokens") as training_file:
        for line in training_file:
            sentenceList_train.append(tokenize(line))

    sentenceList_test = []
    with open("../A1-Data/1b_benchmark.test.tokens") as testing_file:
        for line in testing_file:
            sentenceList_test.append(tokenize(line))

    sentenceList_dev = []
    with open("../A1-Data/1b_benchmark.dev.tokens") as dev_file:
        for line in dev_file:
            sentenceList_dev.append(tokenize(line))

    # training
    print("Training models...")
    sentenceList_train_, vocab = parseTrainingOOV(sentenceList_train, 2)
    unigramModel = unigramModelTrain(sentenceList_train_)
    bigramModel = ngramModelTrain(sentenceList_train_, 2)
    trigramModel = ngramModelTrain(sentenceList_train_, 3)
    print("Training complete.")

    print("")
    print("====================")
    print("")

    # get perplexity on training data
    print("Calculating perplexity on training set...")
    unigramPerplexity_train = unigramModelPerplexity(sentenceList_train_, unigramModel)
    bigramPerplexity_train = ngramModelPerplexity(sentenceList_train_, bigramModel)
    trigramPerplexity_train = ngramModelPerplexity(sentenceList_train_, trigramModel)
    print("unigram Perplexity on training set: {}".format(unigramPerplexity_train))
    print("bigram Perplexity on training set: {}".format(bigramPerplexity_train))
    print("trigram Perplexity on training set: {}".format(trigramPerplexity_train))

    print("")
    print("====================")
    print("")

    # get perplexity on dev data
    print("Calculating perplexity on dev set...")
    sentenceList_dev_ = parseOOV(sentenceList_dev, vocab)
    unigramPerplexity_dev = unigramModelPerplexity(sentenceList_dev_, unigramModel)
    bigramPerplexity_dev = ngramModelPerplexity(sentenceList_dev_, bigramModel)
    trigramPerplexity_dev = ngramModelPerplexity(sentenceList_dev_, trigramModel)
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev))

    print("")
    print("====================")
    print("")

    # get perplexity on test data
    print("Calculating perplexity on test set...")
    sentenceList_test_ = parseOOV(sentenceList_test, vocab)
    unigramPerplexity_test = unigramModelPerplexity(sentenceList_test_, unigramModel)
    bigramPerplexity_test = ngramModelPerplexity(sentenceList_test_, bigramModel)
    trigramPerplexity_test = ngramModelPerplexity(sentenceList_test_, trigramModel)
    print("unigram Perplexity on test set: {}".format(unigramPerplexity_test))
    print("bigram Perplexity on test set: {}".format(bigramPerplexity_test))
    print("trigram Perplexity on test set: {}".format(trigramPerplexity_test))

    print("")
    print("====================")
    print("")

    print("Training smoothed models...")
    lambdas = [(0.1, 0.3, 0.6), (0.7, 0.15, 0.15), (0.15, 0.7, 0.15), (0.15, 0.15, 0.7), (0.33, 0.33, 0.33)]
    for (lambda_1, lambda_2, lambda_3) in lambdas:
        print("====================")
        print("Calculating smoothed model perplexities with hyperparameters lambda_1={}, lambda_2={}, lambda_3={}".format(lambda_1, lambda_2, lambda_3))
        smoothedPerplexity_training = smoothedModelPerplexity(sentenceList_train_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        smoothedPerplexity_dev = smoothedModelPerplexity(sentenceList_dev_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        smoothedPerplexity_test = smoothedModelPerplexity(sentenceList_test_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        print("smoothed model perplexity on training set: {}".format(smoothedPerplexity_training))
        print("smoothed model perplexity on dev set: {}".format(smoothedPerplexity_dev))
        print("smoothed model perplexity on test set: {}".format(smoothedPerplexity_test))


if __name__ == '__main__':
    main()
