# -*- coding: UTF-8 -*-

import re
import nltk  
import nltk.data
import nltk.stem
# from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer

stop_word = ["i",
            "me",
            "my",
            "myself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "ought",
            "i'm",
            "you're",
            "he's",
            "she's",
            "it's",
            "we're",
            "they're",
            "i've",
            "you've",
            "we've",
            "they've",
            "i'd",
            "you'd",
            "he'd",
            "she'd",
            "we'd",
            "they'd",
            "i'll",
            "you'll",
            "he'll",
            "she'll",
            "we'll",
            "they'll",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "doesn't",
            "don't",
            "didn't",
            "won't",
            "wouldn't",
            "shan't",
            "shouldn't",
            "can't",
            "cannot",
            "couldn't",
            "mustn't",
            "let's",
            "that's",
            "who's",
            "what's",
            "here's",
            "there's",
            "when's",
            "where's",
            "why's",
            "how's",
            "daren't",
            "needn't",
            "oughtn't ",
            "mightn't",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "ever",
            "also",
            "just",
            "whether",
            "like",
            "even",
            "still",
            "since",
            "another",
            "however",
            "please",
            "much",
            "many"]

pattern = r"""(?x)                   
              (?:[a-zA-Z]\.)+           
              |\d+(?:\.\d+)?%?       
              |\w+(?:[-']\w+)*       
              |\.\.\.               
              |(?:[.,;"'?():-_`])    
            """

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
# wordtokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')

codePattern = re.compile(r'<code>([\s\S]*?)</code>', re.I)
singleCodePattern = re.compile(r'`([\s\S]*?)`', re.I)
multiCodePattern = re.compile(r'```([\s\S]*?)```', re.I)
termPattern = re.compile(r'[A-Za-z]+.*[A-Z]+.*')
camelCase1 = re.compile(r'^[A-Z]+[a-z]+.*[A-Z]+.*$')
camelCase2 = re.compile(r'^[a-z]+.*[A-Z]+.*$')
upperCase = re.compile(r'^[A-Z]+[0-9]*$')
upperExtCase = re.compile(r'^[A-Z]*(_+[A-Z]*)+[0-9]*$')
methodInvocationCase = re.compile(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)')

LINK_TAG = 'rhlinkrh'
CODE_TAG = 'rhcoderh'


def isDelete(word):
    if word in stop_word:
        return True
    if not re.match('.*[a-zA-Z]+.*', word, re.I):
        return True
    return False        


def preprocess(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern) 
        temp = []
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            else:
                toDeal.append(word)
            for deal in toDeal:
                    if not isDelete(deal.lower()):
                        temp.append(stemmer.stem(deal))
        result.append(temp)
    return result


def preprocessBad(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = WordPunctTokenizer().tokenize(sentence)
        temp = []
        for word in words:
            if not isDelete(word.lower()):
                temp.append(stemmer.stem(word.lower()))
        result.append(temp)
    return result


def preprocessNoCamel(paragraph):
    result = []
    paragraph = re.sub(r'(\[[\s\S]*?\])', '', paragraph, 0, re.I)
    paragraph = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', LINK_TAG, paragraph, 0, re.I)
    paragraph = re.sub(r'`[\s\S]*?`', CODE_TAG, paragraph, 0, re.I)
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        temp = []
        for word in words:
            if not isDelete(word.lower()) and len(word) > 1:
                # temp.append(stemmer.stem(word.lower()))
                temp.append(lemmatizer.lemmatize(word.lower()))
        result.append(temp)
    return result


def processHTMLNoCamel(html):
    texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
    texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
    texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
    return preprocessNoCamel(texts)


def preprocessToWord(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            else:
                toDeal.append(word)
            for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(stemmer.stem(deal))
    return result


def processDiffCode(code):
    code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
    code = re.sub(r'(@@[\s\S]*?\n)', '', code, 0, re.I)
    code = re.sub(r'(-[\s\S]*?\n)', '', code, 0, re.I)
    result = []
    mis = methodInvocationCase.findall(code)
    for mi in mis:
        miWords = mi.split('\.')
        for miWord in miWords:
            toDeal = []
            if camelCase1.match(miWord) or camelCase2.match(miWord):
                toDeal = splitCode(miWord)
            elif upperExtCase.match(miWord):
                toDeal = splitFinalExt(miWord)
            elif upperCase.match(miWord):
                toDeal.append(miWord)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))

    code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
    sentences = tokenizer.tokenize(code)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            elif upperCase.match(word):
                toDeal.append(word)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))
    return result


def processPreDiffCode(code):
    code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
    code = re.sub(r'(@@[\s\S]*?\n)', '', code, 0, re.I)
    code = re.sub(r'(\+[\s\S]*?\n)', '', code, 0, re.I)
    result = []
    mis = methodInvocationCase.findall(code)
    for mi in mis:
        miWords = mi.split('\.')
        for miWord in miWords:
            toDeal = []
            if camelCase1.match(miWord) or camelCase2.match(miWord):
                toDeal = splitCode(miWord)
            elif upperExtCase.match(miWord):
                toDeal = splitFinalExt(miWord)
            elif upperCase.match(miWord):
                toDeal.append(miWord)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))

    code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
    sentences = tokenizer.tokenize(code)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            elif upperCase.match(word):
                toDeal.append(word)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))
    return result


def processHTML(html):
      multiCodes = multiCodePattern.findall(html)
      texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
      singleCodes = singleCodePattern.findall(html)
      texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(`)', '', texts, 0, re.I)
      preText = preprocessToWord(texts)
      result = []
      codes = multiCodes + singleCodes
      for code in codes:
          code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
          mis = methodInvocationCase.findall(code)
          for mi in mis:
              miWords = mi.split('.')
              for miWord in miWords:
                  toDeal = []
                  if camelCase1.match(miWord) or camelCase2.match(miWord):
                      toDeal = splitCode(miWord)
                  elif upperExtCase.match(miWord):
                      toDeal = splitFinalExt(miWord)
                  elif upperCase.match(miWord):
                      toDeal.append(miWord)
                  else:
                      toDeal.append(miWord)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
          code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
          sentences = tokenizer.tokenize(code)
          for sentence in sentences:
              words = nltk.regexp_tokenize(sentence, pattern)
              for word in words:
                  toDeal = []
                  if camelCase1.match(word) or camelCase2.match(word):
                      toDeal = splitCode(word)
                  elif upperExtCase.match(word):
                      toDeal = splitFinalExt(word)
                  elif upperCase.match(word):
                      toDeal.append(word)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
      return result, preText


def getIssueCode(html):
    multiCodes = multiCodePattern.findall(html)
    texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
    singleCodes = singleCodePattern.findall(texts)
    result = []
    codes = multiCodes + singleCodes
    for code in codes:
        code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
        mis = methodInvocationCase.findall(code)
        for mi in mis:
            miWords = mi.split('.')
            for miWord in miWords:
                toDeal = []
                if camelCase1.match(miWord) or camelCase2.match(miWord):
                    toDeal = splitCode(miWord)
                elif upperExtCase.match(miWord):
                    toDeal = splitFinalExt(miWord)
                elif upperCase.match(miWord):
                    toDeal.append(miWord)
                else:
                    toDeal.append(miWord)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))
        code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
        sentences = tokenizer.tokenize(code)
        for sentence in sentences:
            words = nltk.regexp_tokenize(sentence, pattern)
            for word in words:
                toDeal = []
                if camelCase1.match(word) or camelCase2.match(word):
                    toDeal = splitCode(word)
                elif upperExtCase.match(word):
                    toDeal = splitFinalExt(word)
                elif upperCase.match(word):
                    toDeal.append(word)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))
    return result


def processHTMLByTag(html):
      codes = codePattern.findall(html)
      texts = re.sub(r'(<pre>\s*?<code>[\s\S]*?</code>\s*?</pre>)', '', html, 0, re.I)
      texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
      preText = preprocessToWord(texts)
      result = []
      for code in codes:
          code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
          mis = methodInvocationCase.findall(code)
          for mi in mis:
              miWords = mi.split('.')
              for miWord in miWords:
                  toDeal = []
                  if camelCase1.match(miWord) or camelCase2.match(miWord):
                      toDeal = splitCode(miWord)
                  elif upperExtCase.match(miWord):
                      toDeal = splitFinalExt(miWord)
                  elif upperCase.match(miWord):
                      toDeal.append(miWord)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
          code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
          sentences = tokenizer.tokenize(code)
          for sentence in sentences:
              words = nltk.regexp_tokenize(sentence, pattern)
              for word in words:
                  toDeal = []
                  if camelCase1.match(word) or camelCase2.match(word):
                      toDeal = splitCode(word)
                  elif upperExtCase.match(word):
                      toDeal = splitFinalExt(word)
                  elif upperCase.match(word):
                      toDeal.append(word)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
      return result, preText


def splitCode(code):
    res = []
    words = re.split(r"([A-Z]+[a-z]*)", code)
    for word in words:
        if word:
            res.append(word)
    return res


def splitFinalExt(ext):
    res = []
    words = ext.split('_')
    for word in words:
        if word:
            res.append(word)
    return res