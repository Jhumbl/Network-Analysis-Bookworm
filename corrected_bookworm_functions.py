'''
This document contains corrected and updated functions for the bookworm module to better help
with sanitizing and working with the network data.
'''
from bookworm import *

import community
import matplotlib.pyplot as plt
import networkx as nx
import csv
import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
import string


def bookworm_sanitized(book_path, charaters_path=None, threshold=2, exclude_words = None):
    '''
    Wraps the full bookworm analysis from the raw .txt file's path, to
    production of the complete interaction dataframe. Updated function allows
    user to exclude characters which are known to be incorrectly included due to the roughness
    of the nlp. The returned dataframe is directly analysable by networkx using:

    nx.from_pandas_dataframe(interaction_df,
                             source='source',
                             target='target')

    Parameters
    ----------
    book_path : string (required)
        path to txt file containing full text of book to be analysed
    charaters_path : string (optional)
        path to csv file containing full list of characters to be examined
    exclude_words : list
        list of words which are known to have been incorrectly included in the characters variable

    Returns
    -------
    interaction_df : pandas.DataFrame
        DataFrame enumerating the strength of interactions between charcters.
        source = character one
        target = character two
        value = strength of interaction between character one and character two
    '''
    book = load_book(book_path)
    sequences = get_sentence_sequences(book)

    if charaters_path is None:
        characters = extract_character_names(book)
    else:
        characters = load_characters(charaters_path)
    
    # Exclude words known to be incorrectly implemented or included.
    # if exclude_words is not None:
    #    for word in exclude_words:
    #        if word in characters:
    #            characters.remove(word)

    df = find_connections(sequences, characters)
    cooccurence = calculate_cooccurence(df)
    int_df = get_interaction_df(cooccurence, threshold)
    
    # Remove trailing whitespace from strings in column
    int_df['source'] = int_df['source'].str.strip()
    int_df['target'] = int_df['target'].str.strip()
    
    # Exclude words known to be incorrectly implemented or included.
    int_df = int_df[~int_df['source'].isin(exclude_words)]
    int_df = int_df[~int_df['target'].isin(exclude_words)]
    
    return int_df

def d3_dict_corrected(interaction_df, group):
    '''
    Reformats a DataFrame of interactions into a dictionary which is
    interpretable by the Mike Bostock's d3.js force directed graph script
    https://bl.ocks.org/mbostock/4062045
    
    This updated function has corrected the evaluation of the 'nodes' variable.

    Parameters
    ----------
    interaction_df : pandas.DataFrame (required)
        DataFrame enumerating the strength of interactions between charcters.
        source = character one
        target = character two
        value = strength of interaction between character one and character two
    
    group : dictionary (required)
        Dictionary of the node as the key value and the group number it is assigned to.

    Returns
    -------
    d3_dict : dict
        a dictionary of nodes and links in a format which is immediately
        interpretable by the d3.js script
    '''
    nodes = [{"id": str(id), "group": group.get(id)} for id in set(list(interaction_df['source']) + list(interaction_df['target']))]
    links = interaction_df.to_dict(orient='records')
    return {'nodes': nodes, 'links': links}


def chronological_network_corrected(book_path, n_sections=10, cumulative=True):
    '''
    Split a book into n equal parts, with optional cumulative aggregation, and
    return a dictionary of assembled character graphs

    Parameters
    ----------
    book_path : string (required)
        path to the .txt file containing the book to be split
    n_sections :  (optional)
        the number of sections which we want to split our book into
    cumulative : bool (optional)
        If true, the returned sections will be cumulative, ie all will start at
        the book's beginning and end at evenly distributed points throughout
        the book

    Returns
    -------
    graph_dict : dict
        a dictionary containing the graphs of each split book section
        keys = section index
        values = nx.Graph describing the character graph in the specified book
                 section
    '''
    book = load_book(book_path)
    sections = split_book(book, n_sections, cumulative)
    graph_dict = {}

    for i, section in enumerate(sections):
        characters = extract_character_names(' '.join(section))
        df = find_connections(sequences=section, characters=characters)
        cooccurence = calculate_cooccurence(df)
        interaction_df = get_interaction_df(cooccurence, threshold=2)

        graph_dict[i] = nx.from_pandas_edgelist(interaction_df,
                                                 source='source',
                                                 target='target')
    return graph_dict


def chronological_network_sanitized(book_path, n_sections=10, cumulative=True, exclude_words=None):
    '''
    Split a book into n equal parts, with optional cumulative aggregation, and
    return a dictionary of assembled character graphs

    Parameters
    ----------
    book_path : string (required)
        path to the .txt file containing the book to be split
    n_sections :  (optional)
        the number of sections which we want to split our book into
    cumulative : bool (optional)
        If true, the returned sections will be cumulative, ie all will start at
        the book's beginning and end at evenly distributed points throughout
        the book

    Returns
    -------
    graph_dict : dict
        a dictionary containing the graphs of each split book section
        keys = section index
        values = nx.Graph describing the character graph in the specified book
                 section
    '''
    book = load_book(book_path)
    sections = split_book(book, n_sections, cumulative)
    graph_dict = {}

    for i, section in enumerate(sections):
        characters = extract_character_names(' '.join(section))
        df = find_connections(sequences=section, characters=characters)
        cooccurence = calculate_cooccurence(df)
        int_df = get_interaction_df(cooccurence, threshold=2)
        
        # Remove trailing whitespace from strings in column
        int_df['source'] = int_df['source'].str.strip()
        int_df['target'] = int_df['target'].str.strip()
    
        # Exclude words known to be incorrectly implemented or included.
        int_df = int_df[~int_df['source'].isin(exclude_words)]
        int_df = int_df[~int_df['target'].isin(exclude_words)]

        graph_dict[i] = nx.from_pandas_edgelist(int_df,
                                                 source='source',
                                                 target='target')
    return graph_dict