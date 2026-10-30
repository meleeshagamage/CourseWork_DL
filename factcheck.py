
import torch
from typing import List
import numpy as np
import spacy
import gc
import re

from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from nltk.stem import PorterStemmer


# FactExample

class FactExample(object):

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr(
            "fact=" + repr(self.fact)
            + "; label=" + repr(self.label)
            + "; passages=" + repr(self.passages)
        )


# Entailment Model

class EntailmentModel(object):

    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):

        with torch.no_grad():

            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                padding=True
            )

            if self.cuda:
                inputs = {
                    key: value.to('cuda')
                    for key, value in inputs.items()
                }

            outputs = self.model(**inputs)
            logits = outputs.logits

        # TODO: Implement entailment prediction

        raise Exception("Not implemented")


# Base FactChecker


class FactChecker(object):

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


# Baseline FactCheckers


class RandomGuessFactChecker(FactChecker):

    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"



# Text Preprocessing

STOP_WORDS = set(SPACY_STOP_WORDS)

# Preserve negation words because they can change meaning
STOP_WORDS -= {"not", "no", "never"}

_stemmer = PorterStemmer()


def preprocess_text(text: str) -> List[str]:

    # 1. Remove special sentence markers
    text = re.sub(r"</?s>", " ", text)

    # 2. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Tokenization
    words = re.findall(r"[a-z0-9]+", text)

    # 5. Stopword removal
    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    # 6. Apply Porter stemming
    words = [
        _stemmer.stem(word)
        for word in words
    ]

    return words

# Part 1: Word Recall Threshold FactChecker



class WordRecallThresholdFactChecker(FactChecker):

    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold

    def score(self, fact: str, passages: List[dict]) -> float:

        # Preprocess the fact
        fact_words = set(preprocess_text(fact))

        if not fact_words:
            return 0.0

        # Preprocess all retrieved passages
        passage_words = set()

        for passage in passages:
            passage_words.update(
                preprocess_text(passage["text"])
            )

        # Calculate word overlap
        overlap = len(fact_words.intersection(passage_words))

        # Calculate word recall
        recall = overlap / len(fact_words)

        return recall

    def predict(self, fact: str, passages: List[dict]) -> str:

        score = self.score(fact, passages)

        if score >= self.threshold:
            return "S"

        return "NS"





# Optional: Dependency Recall FactChecker

class DependencyRecallThresholdFactChecker(FactChecker):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:

        # TODO: Implement dependency-based classification
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):

        processed_sent = self.nlp(sent)

        relations = set()

        for token in processed_sent:

            ignore_dep = [
                'punct', 'ROOT', 'root', 'det',
                'case', 'aux', 'auxpass', 'dep',
                'cop', 'mark'
            ]

            if token.is_punct or token.dep_ in ignore_dep:
                continue

            head = (
                token.head.lemma_
                if token.head.pos_ == 'VERB'
                else token.head.text
            )

            dependent = (
                token.lemma_
                if token.pos_ == 'VERB'
                else token.text
            )

            relation = (head, token.dep_, dependent)

            relations.add(relation)

        return relations
