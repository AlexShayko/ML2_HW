from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import re
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)

    root = ET.fromstring(text)

    sentence_pairs = []
    alignments = []

    for sent in root.findall('s'):
        eng_text = sent.find('english').text
        czech_text = sent.find('czech').text
        sentence_pairs.append(SentencePair(source=eng_text.split(), target=czech_text.split()))

        sure = sent.find('sure').text
        if sure is None:
            sure = []
        else:
            sure = [tuple(map(int, pair.split('-'))) for pair in sure.split()]

        pos = sent.find('possible').text
        if pos is None:
            pos = []
        else:
            pos = [tuple(map(int, pair.split('-'))) for pair in pos.split()]

        alignments.append(LabeledAlignment(sure=sure, possible=pos))


    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter = Counter()
    target_counter = Counter()

    for texts in sentence_pairs:
        source_counter.update(texts.source)
        target_counter.update(texts.target)

    if freq_cutoff is None:
        source_tokens = list(source_counter.keys())
        target_tokens = list(target_counter.keys())
    else:
        source_tokens=[token for token, _ in source_counter.most_common(freq_cutoff)]
        target_tokens=[token for token, _ in target_counter.most_common(freq_cutoff)]

    source_dict = {source_token: id for id, source_token in enumerate(source_tokens)}
    target_dict = {target_token: id for id, target_token in enumerate(target_tokens)}

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for texts in sentence_pairs:
        source = texts.source
        target = texts.target
        source_tokens = [source_dict[word] for word in source if word in source_dict]
        target_tokens = [target_dict[word] for word in target if word in target_dict]
        if (len(source_tokens) == 0) or (len(target_tokens) == 0):
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens=np.array(source_tokens, dtype=np.int32), target_tokens=np.array(target_tokens, dtype=np.int32)))

    return tokenized_sentence_pairs
