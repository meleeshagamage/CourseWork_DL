
import torch
from typing import List
import numpy as np
import spacy
import gc
import re

from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Metrics used to compare the fact with the passage
OVERLAP_METRICS = [
    "rouge-1",
    "rouge-2",
    "rouge-l",
    "bleu",
    "cosine_similarity",
    "jaccard_similarity"
]


# Text preprocessing
STOP_WORDS = set(SPACY_STOP_WORDS)

# Keep negation words because they can change the meaning of a sentence
STOP_WORDS -= {"not", "no", "never"}

_stemmer = PorterStemmer()


def preprocess_text(text: str, remove_stop_words: bool = True, stem: bool = True) -> List[str]:

    # Remove sentence markers
    text = re.sub(r"</?s>", " ", text)

    # Remove extra spaces and convert text to lowercase
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    # Split the text into individual words
    words = re.findall(r"[a-z0-9]+", text)

    # Remove common words if stop word removal is enabled
    if remove_stop_words:
        words = [word for word in words if word not in STOP_WORDS]

    # Convert words into their root forms
    if stem:
        words = [_stemmer.stem(word) for word in words]

    return words


# Metric calculations

def compute_rouge_1(fact_tokens: List[str], passage_tokens: List[str]) -> float:

    if not fact_tokens:
        return 0.0

    # Find the common words between the fact and passage
    fact_set = set(fact_tokens)
    passage_set = set(passage_tokens)
    overlap = fact_set.intersection(passage_set)

    return len(overlap) / len(fact_set)


def compute_rouge_2(fact_tokens: List[str], passage_tokens: List[str]) -> float:

    if len(fact_tokens) < 2 or len(passage_tokens) < 2:
        return 0.0

    # Compare pairs of consecutive words
    fact_bigrams = set(zip(fact_tokens[:-1], fact_tokens[1:]))
    passage_bigrams = set(zip(passage_tokens[:-1], passage_tokens[1:]))

    overlap = fact_bigrams.intersection(passage_bigrams)
    return len(overlap) / len(fact_bigrams)


def compute_rouge_l(fact_tokens: List[str], passage_tokens: List[str]) -> float:

    if not fact_tokens or not passage_tokens:
        return 0.0

    # Create a table to find the longest common word sequence
    m, n = len(fact_tokens), len(passage_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if fact_tokens[i - 1] == passage_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    return lcs_length / len(fact_tokens)


def compute_bleu(fact_tokens: List[str], passage_tokens: List[str], weights=(0.5, 0.5)) -> float:

    if not fact_tokens or not passage_tokens:
        return 0.0

    # Calculate the matching unigram precision
    p1 = len(set(fact_tokens).intersection(set(passage_tokens))) / len(fact_tokens)

    # Calculate the matching bigram precision
    if len(fact_tokens) > 1 and len(passage_tokens) > 1:
        f_bi = set(zip(fact_tokens[:-1], fact_tokens[1:]))
        p_bi = set(zip(passage_tokens[:-1], passage_tokens[1:]))
        p2 = len(f_bi.intersection(p_bi)) / len(f_bi) if f_bi else 0.0
    else:
        p2 = p1

    # Combine the scores using a geometric average
    eps = 1e-7
    log_p1 = np.log(p1 + eps)
    log_p2 = np.log(p2 + eps)

    geom_avg = np.exp(weights[0] * log_p1 + weights[1] * log_p2)

    # Apply a penalty when the passage is shorter than the fact
    fact_len = len(fact_tokens)
    pass_len = len(passage_tokens)
    bp = 1.0 if pass_len >= fact_len else np.exp(1 - fact_len / pass_len)

    return float(bp * geom_avg)


def compute_jaccard_similarity(fact_tokens: List[str], passage_tokens: List[str]) -> float:
    """Calculate similarity using the intersection and union of words."""

    fact_set = set(fact_tokens)
    passage_set = set(passage_tokens)

    intersection = fact_set.intersection(passage_set)
    union = fact_set.union(passage_set)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compute_cosine_similarity(fact_tokens: List[str], passage_tokens: List[str]) -> float:
    """Calculate TF-IDF cosine similarity between the two texts."""

    if not fact_tokens or not passage_tokens:
        return 0.0

    fact_text = " ".join(fact_tokens)
    passage_text = " ".join(passage_tokens)

    # Convert both texts into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False
    )

    try:
        vectors = vectorizer.fit_transform([fact_text, passage_text])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(score)
    except ValueError:
        return 0.0


def overlap_score(metric: str, fact_tokens: List[str], passage_tokens: List[str]) -> float:

    # Select the metric based on the given name
    metric = metric.lower()

    if metric == "rouge-1":
        return compute_rouge_1(fact_tokens, passage_tokens)
    elif metric == "rouge-2":
        return compute_rouge_2(fact_tokens, passage_tokens)
    elif metric == "rouge-l":
        return compute_rouge_l(fact_tokens, passage_tokens)
    elif metric == "bleu":
        return compute_bleu(fact_tokens, passage_tokens)
    elif metric == "cosine_similarity":
        return compute_cosine_similarity(fact_tokens, passage_tokens)
    elif metric == "jaccard_similarity":
        return compute_jaccard_similarity(fact_tokens, passage_tokens)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of {OVERLAP_METRICS}")


# Stores one fact, its passages and the correct label
class FactExample(object):
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


# Base class for all fact-checking methods
class FactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


# Task 1: Word overlap fact checker
class WordRecallThresholdFactChecker(FactChecker):

    def __init__(self, metric: str = 'rouge-1', threshold: float = 0.625, remove_stop_words: bool = True, stem: bool = True):
        self.metric = metric.lower()
        self.threshold = threshold
        self.remove_stop_words = remove_stop_words
        self.stem = stem

    def predict(self, fact: str, passages: List[dict]) -> str:

        # Prepare the fact before comparing it with the passages
        fact_tokens = preprocess_text(fact, remove_stop_words=self.remove_stop_words, stem=self.stem)

        max_score = 0.0

        # Compare the fact with each available passage
        for passage in passages:
            passage_text = passage.get('text', '')
            passage_tokens = preprocess_text(passage_text, remove_stop_words=self.remove_stop_words, stem=self.stem)

            score = overlap_score(self.metric, fact_tokens, passage_tokens)

            # Keep the highest similarity score
            if score > max_score:
                max_score = score

        # Classify the fact using the selected threshold
        return "S" if max_score >= self.threshold else "NS"


# Returns a random prediction
class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


# Always predicts that the fact is supported
class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# Phase 2: Textual entailment using DeBERTa-v3
class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

        # Set the model to evaluation mode
        self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str):

        # Disable gradient calculation during prediction
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move input data to GPU when CUDA is enabled
            if self.cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}

            logits = self.model(**inputs).logits

        return logits


# Checks whether the passage supports the given fact
class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

        print("Model Labels:", self.ent_model.model.config.id2label)

        self.pruning_threshold = 0.20
        self.entailment_threshold = 0.40
        self.last_best_score = 0.0
        self.last_best_evidence = ""

    @staticmethod
    def _extract_sentences(text: str) -> List[str]:

        # Convert sentence markers into line breaks
        text = text.replace(r"\<s>", "<s>").replace(r"\</s>", "</s>")
        text = re.sub(r"</?s>", "\n", text)

        pieces = []

        # Split the passage into individual sentences
        for chunk in text.splitlines():
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk.strip()):
                sentence = sentence.strip()
                if sentence:
                    pieces.append(sentence)

        return pieces

    def predict(self, fact: str, passages: List[dict]) -> str:

        best_score = 0.0
        best_evidence = ""
        fact_tokens = preprocess_text(fact)

        # Check each passage for possible supporting evidence
        for passage in passages:
            sentences = self._extract_sentences(passage.get("text", ""))

            # Also consider pairs of neighbouring sentences
            candidates = list(sentences)
            candidates.extend(
                sentences[i] + " " + sentences[i + 1]
                for i in range(len(sentences) - 1)
            )

            for evidence in candidates:

                # Skip evidence with very low word overlap
                overlap = compute_rouge_1(fact_tokens, preprocess_text(evidence))
                if overlap < self.pruning_threshold:
                    continue

                # Get the entailment score from the model
                logits = self.ent_model.check_entailment(evidence, fact)
                score = torch.softmax(logits, dim=-1)[0][0].item()

                # Save the evidence with the highest entailment score
                if score > best_score:
                    best_score = score
                    best_evidence = evidence

        self.last_best_score = best_score
        self.last_best_evidence = best_evidence

        # Return the final prediction based on the threshold
        return "S" if best_score >= self.entailment_threshold else "NS"


# Placeholder class for a dependency-based fact checker
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")
