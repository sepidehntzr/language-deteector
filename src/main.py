# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.
#from asyncio.windows_events import NULL
import glob
import csv
from random import shuffle  # For writing to train.tsv file
import numpy as np
import argparse
from operator import le
from nltk.lm import NgramCounter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import nltk
#nltk.download('punkt')
from random import shuffle


def create_argument_parser():
    """ Create a parser and which have 3 arguments and 1 optional argument
    First argument: input_data_path like "data/train/"
    Second argument: test_data_path like "data/dev/"
    Third argument: output_path like "output/results_dev_laplace.csv"
    Optional argument: like "--unsmoothed", "--laplace", or "--interpolation"

    Returns:
        args: args
    """
    parser = argparse.ArgumentParser(
        "character n-grams")
    parser.add_argument("input_data_path", help="Path to the data directory")
    parser.add_argument(
        "test_data_path", help="Path to the test data directory")
    parser.add_argument("output_path", help="Path to the output CSV file")
    parser.add_argument('--unsmoothed', action='store_true',
                        help="the type of smoothing")
    parser.add_argument('--laplace',  action='store_true',
                        help="the type of smoothing")
    parser.add_argument('--interpolation',  action='store_true',
                        help="the type of smoothing")
    args = parser.parse_args()
    return args


def deleted_interpolation(n, all_n_grams,total_freq):
    lambdas = {}
    for i in range(n):
       lambdas[i+1] = 0
    lambdas_temp = {}

    ngrams_with_freq = all_n_grams.get(n)
    for n_gram, freq in ngrams_with_freq.items():# grams
        if freq > 0 and n_gram != 'unk':
            for j in range(len(n_gram)):
                
                new_gram_numerator = n_gram[j:] 
                ngrams_with_freq_numerator = all_n_grams.get(len(new_gram_numerator))
                if new_gram_numerator in ngrams_with_freq_numerator:
                  numerator = ngrams_with_freq_numerator[new_gram_numerator] - 1

                if len(new_gram_numerator) != 1:
                  new_gram_denominator = new_gram_numerator[:-1]
                  ngrams_with_freq_denominator  = all_n_grams.get(len(new_gram_denominator))
                  denominator = ngrams_with_freq_denominator[new_gram_denominator] - 1
                else:
                    denominator = total_freq - 1
                lambdas_temp[j+1] = numerator/denominator
            m = max(lambdas_temp, key=lambdas_temp.get) 
            lambdas.update({m: lambdas.get(m) + freq})
           

    sum_values = sum(lambdas.values())
    for i in lambdas.keys():    
        lambdas.update({i: lambdas.get(i) / sum_values })         
    #TODOo
    return lambdas
    #pass


def calc_probability(ngrams_with_freq, n, data, threshold, laplace=False, interpolation=False):
    unk_chars = find_unk_chr(data, threshold)
    total_freq = 0
    for gram in ngrams_with_freq:
        total_freq += ngrams_with_freq[gram]
    num_all_chrs = len(ngrams_with_freq)
    probs = {}
    if laplace == False and interpolation == False:
        if n == 1:
            for gram in ngrams_with_freq:
                probs[gram] = ngrams_with_freq[gram]/total_freq
            # ngrams_with_freq
        else:
            n_1_gram_with_freq = create_n_gram_with_freq(n-1, data, threshold)

            for gram in ngrams_with_freq.keys():
                if gram != 'unk':
                    if gram[:-1] not in unk_chars:
                        probs[gram] = ngrams_with_freq[gram] / \
                            n_1_gram_with_freq[gram[:-1]]
                    else:

                        probs[gram] = ngrams_with_freq[gram] / \
                            n_1_gram_with_freq['unk']
    elif laplace:
        if n == 1:

            for gram in ngrams_with_freq:
                probs[gram] = ngrams_with_freq[gram] + \
                    1/total_freq+num_all_chrs
            # ngrams_with_freq
        else:
            n_1_gram_with_freq = create_n_gram_with_freq(n-1, data, threshold)

            for gram in ngrams_with_freq.keys():
                if gram != 'unk':
                    if gram[:-1] not in unk_chars:
                        probs[gram] = ngrams_with_freq[gram]+1 / \
                            n_1_gram_with_freq[gram[:-1]]+num_all_chrs
                    else:

                        probs[gram] = ngrams_with_freq[gram] + 1 / \
                            n_1_gram_with_freq['unk']+num_all_chrs

    elif interpolation:
        #TODOo
        all_n_grams = {}
        for i in range(n,0,-1):
            all_n_grams[i] = create_n_gram_with_freq(i, data, threshold)   
        lambdas = deleted_interpolation(n, all_n_grams, total_freq)

        if n == 1:
            for gram in ngrams_with_freq:
                    unigram_prob = ngrams_with_freq[gram]  \
                        /total_freq
                    probs[gram] =  lambdas.get(1) * unigram_prob
            

            ##########################
        else:
            for i in range(n,0,-1):# grams
               ngrams_with_freq = all_n_grams.get(i)
               #if i != 1:
                  #n_1_gram_with_freq = all_n_grams.get(i-1)

               for n_gram, freq in ngrams_with_freq.items():# grams
                  if n_gram != 'unk':
                    linear_sum = 0
                    for j in range(len(n_gram),0,-1):
                        new_gram_numerator = n_gram[len(n_gram)-j:] 
                        ngrams_with_freq_numerator = all_n_grams.get(len(new_gram_numerator))
                        if new_gram_numerator in ngrams_with_freq_numerator:
                            numerator = ngrams_with_freq_numerator[new_gram_numerator] 

                        if len(new_gram_numerator) != 1:
                            new_gram_denominator = new_gram_numerator[:-1]
                            ngrams_with_freq_denominator = all_n_grams.get(len(new_gram_denominator))
                            denominator = ngrams_with_freq_denominator[new_gram_denominator] 
                        else:
                            denominator = total_freq
                        linear_sum += (numerator/denominator) * lambdas.get(j)   
                    probs[n_gram] =  linear_sum   
                            



            #    for gram in ngrams_with_freq.keys():
            #         if i == n:#first time
            #            probs[gram] = 0
            #         if gram != 'unk':
            #             if (gram[:-1] not in unk_chars) and i != 1:
            #                 probs[gram] += (ngrams_with_freq[gram] / \
            #                     n_1_gram_with_freq[gram[:-1]]) * lambdas.get(i)
            #             elif i == 1:    
            #                 probs[gram] += (ngrams_with_freq[gram]  / \
            #                     total_freq) * lambdas.get(i)   

            #             elif (gram[:-1] in unk_chars) and i != 1:
            #                 probs[gram] += (ngrams_with_freq[gram]  / \
            #                     n_1_gram_with_freq['unk']) * lambdas.get(i)
                              
            

        #probs = {}

    print(probs)
    return probs


def build_lm(n, data, threshold, laplace=False, interpolation=False):
    ngrams_with_freq = create_n_gram_with_freq(n, data, threshold)
    probs = calc_probability(ngrams_with_freq, n, data,
                             threshold, laplace, interpolation)
    return ngrams_with_freq, probs


def find_unk_chr(data, threshold):
    """finding the list of OOC (Out of Character)

    Args:
        data (str): string of data
        threshold (int): the number for determining whether the character has enough frequency to be in their characters list

    Returns:
        unk_chars: list of OOC
    """
    unk_chars = []
    n_grams = {"unk": 0}
    list_of_words = nltk.word_tokenize(data)
    for word in list_of_words:
        word = '-'+word+'_'
        for j in range(len(word)):
            temp_ngram = word[j:j+1]
            if temp_ngram not in list(n_grams.keys()):
                n_grams[temp_ngram] = 1
            else:
                n_grams[temp_ngram] += 1

    for gram in list(n_grams.keys()):
        if gram != 'unk':
            if n_grams[gram] < threshold:
                n_grams['unk'] += n_grams[gram]
                n_grams.pop(gram)
                unk_chars.append(gram)
    return unk_chars


def create_n_gram_with_freq(n, data, threshold):
    n_grams_with_freq = {"unk": 0}
    list_of_words = nltk.word_tokenize(data)
    for word in list_of_words:
        word = '-'+word+'_'
        for j in range(len(word)-n+1):
            temp_ngram = word[j:j+n]
            #temp_ngram = tuple(word[j:j+n])
            
            if temp_ngram not in list(n_grams_with_freq.keys()):
                n_grams_with_freq[temp_ngram] = 1
            else:
                n_grams_with_freq[temp_ngram] += 1

    for gram in list(n_grams_with_freq.keys()):
        if gram != 'unk':

            if n_grams_with_freq[gram] < threshold:
                n_grams_with_freq['unk'] += n_grams_with_freq[gram]
                n_grams_with_freq.pop(gram)

    #print(n_grams_with_freq)
    return n_grams_with_freq


def read_file(path):
    f = open(path, "r")
    text = f.read()
    f.close()
    return text


def write_output_files(path):
    # TODO
    pass


def preprocess_data(text):

    return


def calculate_perplexity(test_data, lm_probs):

    # test data
    # TODO
    return


def split_train_set_to_train_and_heldout(ratio, path):
    all_paths = glob.glob(path+"/*")
    shuffle(all_paths)
    N = len(all_paths)
    heldout_set_paths = all_paths[:int(N*ratio)]
    train_set_paths = all_paths[int(N*ratio):]
    return train_set_paths, heldout_set_paths


if __name__ == "__main__":
    args = create_argument_parser()
    # Heldout set for tuning hyper parameters
    train_set_paths, heldout_set_paths = split_train_set_to_train_and_heldout( #???
        0.7, args.input_data_path)
    for single_input_path in train_set_paths:
        text_train = read_file(single_input_path)
        #build LM for each train file (55)
        ngrams_with_freq, probs = build_lm(n= 3, data = text_train, threshold= 5, laplace=False, interpolation=True)
        break
        # TODO deleted interpolation + tune hyper parameters on heldout set + loop through all N possible and all files + calc perplexity on test set

    #validation_set_paths = args.test_data_path 
    #for single_dev_input_path in validation_set_paths:  
       #text_test = read_file(single_dev_input_path)
        #calculate_perplexity (text_test, probs) 
