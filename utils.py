import numpy as np
import pickle, csv, itertools, nltk


def preprocess(vocabulary_size, training_data_raw, \
    unknown_token="UNKNOWN_TOKEN", \
    sentence_start_token="SENTENCE_START", \
    sentence_end_token="SENTENCE_END"):
    
    print("Reading raw input file")
    with open(training_data_raw, "rU") as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8') \
                    .lower()) for x in reader])
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) \
                for x in sentences]
    tokenized_sentences \
        = [nltk.word_tokenize(sentence) for sentence in sentences]
    print("Created tokenized sentences")

    #Count the word frequences
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique word tokens." % len(word_freq.items()))

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print("Created index_to_word and word_to_index")

    #Replace all words not in out vocabulary with unknown_token
    for i, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]
    print("Tokenized sentences created")

    print("\nExample sentence before preprocessing: '%s'\n" % sentences[0])
    print("\nExample sentence after preprocessing: '%s'\n" % tokenized_sentences[0])

    #Create training data
    X_train= np.asarray([[word_to_index[w] for w in sentence[:-1]] for sentence in tokenized_sentences])
    Y_train= np.asarray([[word_to_index[w] for w in sentence[1:]] for sentence in tokenized_sentences]) 
    print("Successfully created X, Y pairs from tokenized sentences")

    return X_train, Y_train, index_to_word, word_to_index


    

def softmax(z):
    """
    Algebraically equivalent to e^x/sum(e^x), but numerically more stable
    """
    zt = np.exp(z-np.max(z))
    return zt/np.sum(zt)




