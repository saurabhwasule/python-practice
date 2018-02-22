###NLP - Python - Introduction 
http://www.nltk.org/book/
http://www.nltk.org/api/nltk.html

$ pip install numpy 
$ pip install nltk 

#Download data/corpus with nltk (has option for proxy server)
>>> import nltk
>>> nltk.download()  #set c:\nltk_data else set NLTK_DATA env var 

#Download book 
>>> from nltk.corpus import brown
>>> brown.words()
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]


>>> from nltk.book import *
>>> text1

>>> type(text1)
<class 'nltk.text.Text'>

###nltk Reference - class nltk.text.Text(tokens, name=None)
    Bases: object
    A wrapper around a sequence of simple (string) tokens,
    #Example of creating Text 
    f=open('my-file.txt','rU')
    raw=f.read()
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)
    #OR 
    from nltk.corpus import PlaintextCorpusReader
    # RegEx or list of file names
    files = ".*\.txt"
    corpus0 = PlaintextCorpusReader("/path/", files)
    corpus  = nltk.Text(corpus0.words())
    #OR 
    import nltk.corpus
    from nltk.text import Text
    moby = Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
    #Methods 
    collocations(num=20, window_size=2)
        Print collocations derived from the text, ignoring stopwords.
    common_contexts(words, num=20)
        Find contexts where the specified words appear; 
        list most frequent common contexts first.
    concordance(word, width=79, lines=25)
        Print a concordance for word with the specified context window. 
        Word matching is not case-sensitive
        A concordance view shows us every occurrence of a given word
    count(word)
        Count the number of times this word appears in the text.
    dispersion_plot(words)
        Produce a plot showing the distribution of the words through the text. 
        Requires pylab(matplotlib) to be installed.
    findall(regexp)
        Find instances of the regular expression in the text
        angle brackets as non-capturing parentheses
        >>> text5.findall("<.*><.*><bro>")
        you rule bro; telling you bro; u twizted bro
    generate(words)
    index(word)
        Find the index of the first occurrence of the word in the text.
    plot(*args)
        for FreqDist.plot() 
    readability(method)
    similar(word, num=20)
        Distributional similarity: find other words which appear in the same contexts as the specified word; 
        list most similar words first.
    unicode_repr()
    vocab()

###nltk Reference - class nltk.text.TextCollection(source)
    Bases: nltk.text.Text
    A collection of texts, which can be loaded with list of texts, 
    or with a corpus consisting of one or more texts  
    Iterating over a TextCollection produces all the tokens of all the texts in order.
        >>> import nltk.corpus
        >>> from nltk.text import TextCollection
        >>> print('hack'); from nltk.book import text1, text2, text3
        hack...
        >>> gutenberg = TextCollection(nltk.corpus.gutenberg)
        >>> mytexts = TextCollection([text1, text2, text3])

    idf(term)
        The number of texts in the corpus divided by the number of texts that the term appears in. 
        If a term does not appear in the corpus, 0.0 is returned.
    tf(term, text)
        The frequency of the term in text.
    tf_idf(term, text)
    
    
##Examples 

#A concordance view shows us every occurrence of a given word
>>> text1.concordance("monstrous")

#Similar term 
>>> text1.similar("monstrous")

#common_contexts allows us to examine the contexts that are shared by two or more words,
>>> text2.common_contexts(["monstrous", "very"])

#dispersion plot(requires matplotlib)
#Each stripe represents an instance of a word, and each row represents the entire text
#location of a word in the text: how many words from the beginning it appears
>>> text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


#Counting Vocabulary
>>> len(text3)
>>> sorted(set(text3)) 
>>> len(set(text3)) 

#calculate a measure of the lexical richness of the text
>>> len(set(text3)) / len(text3)  #number of distinct words is just 6% of the total number of words
0.06230453042623537

#count how often a word occurs in a text, and compute what percentage of the text is taken up by a specific word
>>> text3.count("smote")
5
>>> 100 * text4.count('a') / len(text4)
1.4643016433938312
>>>

>>> def lexical_diversity(text): 
        return len(set(text)) / len(text) 

>>> def percentage(count, total): 
        return 100 * count / total

#dummy data 
>>> sent1 = ['Call', 'me', 'Ishmael', '.']

>>> len(sent1) 
4
>>> lexical_diversity(sent1)
1.0
>>>
#for Text data 
>>> lexical_diversity(text3)
0.06230453042623537
>>> lexical_diversity(text5)
0.13477005109975562
>>> percentage(4, 5)
80.0
>>> percentage(text4.count('a'), len(text4))
1.4643016433938312
>>>





##Frequency Distributions

#Frequency of words 
>>> fdist1 = FreqDist(text1) 
>>> print(fdist1) 
<FreqDist with 19317 samples and 260819 outcomes>
>>> fdist1.most_common(50)
[(',', 18713), ('the', 13721), ('.', 6862), ('of', 6536), ('and', 6024),
('a', 4569), ('to', 4542), (';', 4072), ('in', 3916), ('that', 2982),
("'", 2684), ('-', 2552), ('his', 2459), ('it', 2209), ('I', 2124),
('s', 1739), ('is', 1695), ('he', 1661), ('with', 1659), ('was', 1632),
('as', 1620), ('"', 1478), ('all', 1462), ('for', 1414), ('this', 1280),
('!', 1269), ('at', 1231), ('by', 1137), ('but', 1113), ('not', 1103),
('--', 1070), ('him', 1058), ('from', 1052), ('be', 1030), ('on', 1005),
('so', 918), ('whale', 906), ('one', 889), ('you', 841), ('had', 767),
('have', 760), ('there', 715), ('But', 705), ('or', 697), ('were', 680),
('now', 646), ('which', 640), ('?', 637), ('me', 627), ('like', 624)]
>>> fdist1['whale']
906
>>>


#to find the words from the vocabulary of the text that are more than 7 characters long
#and count of that word more than 7 
>>> fdist5 = FreqDist(text5)
>>> sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)
['#14-19teens', '#talkcity_adults', '((((((((((', '........', 'Question',
'actually', 'anything', 'computer', 'cute.-ass', 'everyone', 'football',
'innocent', 'listening', 'remember', 'seriously', 'something', 'together',
'tomorrow', 'watching']


##Collocations and Bigrams
#A collocation is a sequence of words that occur together unusually often. 
#Thus 'red wine' is a collocation, whereas 'the wine' is not

#extract  from a text a list of word pairs, also known as bigrams
>>> list(bigrams(['more', 'is', 'said', 'than', 'done']))
[('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]
#words than-done is a bigram,

>>> text4.collocations()
United States; fellow citizens; four years; years ago; Federal
Government; General Government; American people; Vice President; Old
World; Almighty God; Fellow citizens; Chief Magistrate; Chief Justice;
God bless; every citizen; Indian tribes; public debt; one another;
foreign nations; political parties


#Counting Other Things
>>> [len(w) for w in text1] 
[1, 4, 4, 2, 6, 8, 4, 1, 9, 1, 1, 8, 2, 1, 4, 11, 5, 2, 1, 7, 6, 1, 3, 4, 5, 2, ...]
#frequency of lengths of words 
>>> fdist = FreqDist(len(w) for w in text1) 
>>> print(fdist)  
<FreqDist with 19 samples and 260819 outcomes>
>>> fdist
FreqDist({3: 50223, 1: 47933, 4: 42345, 2: 38513, 5: 26597, 6: 17111, 7: 14399,
  8: 9966, 9: 6428, 10: 3528, ...})
>>>

>>> fdist.most_common()
[(3, 50223), (1, 47933), (4, 42345), (2, 38513), (5, 26597), (6, 17111), (7, 14399),
(8, 9966), (9, 6428), (10, 3528), (11, 1873), (12, 1053), (13, 567), (14, 177),
(15, 70), (16, 22), (17, 12), (18, 1), (20, 1)]
>>> fdist.max()
3
>>> fdist[3]
50223
>>> fdist.freq(3)  # 20% of the words of length 3  making up the book
0.19255882431878046
>>>




###nltk Reference - Details of Module probability
FreqDist  
    frequency distributions - which count the number of times that each outcome of an experiment occurs.

ProbDistI 
    probability distributions -  encode the probability of each outcome for an experiment. 
    
#There are two types of probability distribution:
    derived probability distributions - models the probability distribution that generated the frequency distribution.
    analytic probability distributions - created directly from parameters (such as variance).



###nltk Reference - FreqDist
class nltk.probability.FreqDist(samples=None)
    Bases: collections.Counter
    A frequency distribution records the number of times 
    each outcome of an experiment has occurred
#Example 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
sent = 'This is an example sentence'
fdist = FreqDist(word.lower() for word in word_tokenize(sent))

##FreqDist - Methods 
B()
    Return the total number of sample values (or “bins”) 
    that have counts greater than zero. 
    FreqDist.B() is the same as len(FreqDist)

N()
    Return the total number of sample outcomes that have been recorded by this FreqDist.

Nr(r, bins=None)

copy()
    Create a copy of this frequency distribution.

freq(sample)
    Return the frequency of a given sample. 
    The frequency of a sample is defined as the count of that sample divided by the total number of sample outcomes that have been recorded by this FreqDist. 

hapaxes()
    Return a list of all samples that occur once (hapax legomena)

max()
    Return the sample with the greatest number of outcomes in this frequency distribution. 

pformat(maxlen=10)
    Return a string representation of this FreqDist.

plot(*args, **kwargs)
    Plot samples from the frequency distribution displaying the most frequent sample first. 
    If an integer parameter is supplied, stop after this many samples have been plotted. 
    For a cumulative plot, specify cumulative=True. 
        Parameters:
        title (bool) – The title for the graph
        cumulative – A flag to specify whether the plot is cumulative (default = False)
     
pprint(maxlen=10, stream=None)
    Print a string representation of this FreqDist to ‘stream’
 
r_Nr(bins=None)
    Return the dictionary mapping r to Nr, the number of samples with frequency r, where Nr > 0.

setdefault(key, val)

tabulate(*args, **kwargs)
    Tabulate the given samples from the frequency distribution (cumulative), 
    displaying the most frequent sample first. 
    If an integer parameter is supplied, stop after this many samples have been plotted.
    Parameters:
    samples (list) – The samples to plot (default is all samples)
    cumulative – A flag to specify whether the freqs are cumulative (default = False)
 
unicode_repr()
    Return a string representation of this FreqDist.


update(*args, **kwargs)



###nltk Reference - ProbDistI
class nltk.probability.ProbDistI
    Bases: object
    A ProbDist is used to model the probability distribution based on frequency distribution.
    Iterate to get all the samples 
    SUM_TO_ONE = True
        True if the probabilities of the samples in this probability distribution 
        will always sum to one.

    discount()
        Return the ratio by which counts are discounted on average: 

    generate()
        Return a randomly selected sample from this probability distribution. 
        The probability of returning each sample samp is equal to self.prob(samp).

    logprob(sample)
        Return the base 2 logarithm of the probability for a given sample.

    max()
        Return the sample with the greatest probability. 

    prob(sample)
        Return the probability for a given sample. 

    samples()
        Return a list of all samples that have nonzero probabilities.
        Use prob to find the probability of each sample.


    
    
##Subclasses of ProbDistI - Note most of these take FreqDist as constructors 
class nltk.probability.CrossValidationProbDist(freqdists, bins)
    Bases: nltk.probability.ProbDistI
    The “cross-validation estimate” for the probability of a sample is found 
    by averaging the held-out estimates for the sample in each pair of frequency distributions.

class nltk.probability.DictionaryProbDist(prob_dict=None, log=False, normalize=False)
    Bases: nltk.probability.ProbDistI
    A probability distribution whose probabilities are directly specified by a given dictionary. 
    The given dictionary maps samples to probabilities.

class nltk.probability.LidstoneProbDist(freqdist, gamma, bins=None)
    Bases: nltk.probability.ProbDistI
    This is equivalent to adding gamma to the count for each bin, 
    and taking the maximum likelihood estimate of the resulting frequency distribution.
  
class nltk.probability.ELEProbDist(freqdist, bins=None)
    Bases: nltk.probability.LidstoneProbDist
    The “expected likelihood estimate” approximates the probability of a sample 
    with count c from an experiment with N outcomes and B bins as (c+0.5)/(N+B/2). 
    This is equivalent to adding 0.5 to the count for each bin, 
    and taking the maximum likelihood estimate of the resulting frequency distribution.
    
class nltk.probability.SimpleGoodTuringProbDist(freqdist, bins=None)
    Bases: nltk.probability.ProbDistI
    SimpleGoodTuring ProbDist approximates from frequency to frequency of frequency into a linear line under log space by linear regression.

class nltk.probability.HeldoutProbDist(base_fdist, heldout_fdist, bins=None)
    Bases: nltk.probability.ProbDistI
    The heldout estimate for the probability distribution of the experiment used to generate two frequency distributions. 
    These two frequency distributions are called the “heldout frequency distribution” and the “base frequency distribution.” 
 
class nltk.probability.LaplaceProbDist(freqdist, bins=None)
    Bases: nltk.probability.LidstoneProbDist
    The “Laplace estimate” approximates the probability of a sample with count c from an experiment with N outcomes and B bins as (c+1)/(N+B). 
    This is equivalent to adding one to the count for each bin, and taking the maximum likelihood estimate of the resulting frequency distribution.

class nltk.probability.MLEProbDist(freqdist, bins=None)
    Bases: nltk.probability.ProbDistI
    The “maximum likelihood estimate” approximates the probability of each sample as the frequency of that sample in the frequency distribution.

class nltk.probability.MutableProbDist(prob_dist, samples, store_logs=True)
    Bases: nltk.probability.ProbDistI
    An mutable probdist where the probabilities may be easily modified.

class nltk.probability.KneserNeyProbDist(freqdist, bins=None, discount=0.75)
    Bases: nltk.probability.ProbDistI
    Kneser-Ney estimate of a probability distribution. 
    
class nltk.probability.UniformProbDist(samples)
    Bases: nltk.probability.ProbDistI
    A probability distribution that assigns equal probability to each sample in a given set
    
class nltk.probability.WittenBellProbDist(freqdist, bins=None)
    Bases: nltk.probability.ProbDistI
    The Witten-Bell estimate of a probability distribution 
    

    
#Example 
from nltk import corpus
emma_words = corpus.gutenberg.words('austen-emma.txt')
fd = FreqDist(emma_words)
sgt = SimpleGoodTuringProbDist(fd)
print('%18s %8s  %14s'  % ("word", "freqency", "SimpleGoodTuring"))
fd_keys_sorted=(key for key, value in sorted(fd.items(), key=lambda item: item[1], reverse=True))
for key in fd_keys_sorted:
    print('%18s %8d  %14e'  % (key, fd[key], sgt.prob(key))) #key, count of that key, prob of that key 
            
##Example 
def _create_rand_fdist(numsamples, numoutcomes):
    """
    Create a new frequency distribution, with random samples.  The
    samples are numbers from 1 to ``numsamples``, and are generated by
    summing two numbers, each of which has a uniform distribution.
    """
    import random
    fdist = FreqDist()
    for x in range(numoutcomes):
        y = (random.randint(1, (1 + numsamples) // 2) +
             random.randint(0, numsamples // 2))
        fdist[y] += 1
    return fdist

def _create_sum_pdist(numsamples):
    """
    Return the true probability distribution for the experiment
    ``_create_rand_fdist(numsamples, x)``.
    """
    fdist = FreqDist()
    for x in range(1, (1 + numsamples) // 2 + 1):
        for y in range(0, numsamples // 2 + 1):
            fdist[x+y] += 1
    return MLEProbDist(fdist)
    
    
numsamples=6
numoutcomes=500
# Randomly sample a stochastic process three times.
fdist1 = _create_rand_fdist(numsamples, numoutcomes)
fdist2 = _create_rand_fdist(numsamples, numoutcomes)
fdist3 = _create_rand_fdist(numsamples, numoutcomes)

# Use our samples to create probability distributions.
pdists = [
    MLEProbDist(fdist1),
    LidstoneProbDist(fdist1, 0.5, numsamples),
    HeldoutProbDist(fdist1, fdist2, numsamples),
    HeldoutProbDist(fdist2, fdist1, numsamples),
    CrossValidationProbDist([fdist1, fdist2, fdist3], numsamples),
    SimpleGoodTuringProbDist(fdist1),
    SimpleGoodTuringProbDist(fdist1, 7),
    _create_sum_pdist(numsamples),
]

# Find the probability of each sample.
vals = []
for n in range(1,numsamples+1):
    vals.append(tuple([n, fdist1.freq(n)] + [pdist.prob(n) for pdist in pdists]))
print('Generating:')
for pdist in pdists:
    fdist = FreqDist(pdist.generate() for i in range(5000))
    print('%20s %s' % (pdist.__class__.__name__[:20], ("%s" % fdist)[:55]))
print()
        
    
    
    
    
    
    
    
    
    
###nltk Reference - ConditionalFreqDist - base - dict 
class ConditionalFreqDist(condition_samples=None)
    condition_samples = list of tuples => (condition, variable)
    For same condition , the variable is incremented 
    
    
#For example, the following code will produce a conditional frequency distribution 
#of words based on length of the word 
    
from nltk.probability import ConditionalFreqDist  
from nltk.tokenize import word_tokenize
sent = "the the the dog dog some other words that we do not care about"
cfdist = ConditionalFreqDist((len(word), word) for word in word_tokenize(sent))
    
#Access a particular condition - ie length of the word = 3
>>> cfdist[3]
FreqDist({'the': 3, 'dog': 2, 'not': 1})
>>> cfdist[3].freq('the') #% of total 
0.5
>>> cfdist[3]['dog']
2

##ConditionalFreqDist - methods 
N()
    Return the total number of sample outcomes 

conditions()
    Return a list of the conditions that have been accessed for this ConditionalFreqDist. 

plot(*args, **kwargs)  #with matplotlib 
    Plot the given samples from the conditional frequency distribution. 
    For a cumulative plot, specify cumulative=True. 
    Parameters:
        samples (list) – The samples to plot
        title (str) – The title for the graph
        conditions (list) – The conditions to plot (default is all)
 
 
tabulate(*args, **kwargs)
    Tabulate the given samples 
    Parameters:
        samples (list) – The samples to plot
        conditions (list) – The conditions to plot (default is all)
        cumulative – A flag to specify whether the freqs are cumulative (default = False)
     
unicode_repr()
    Return a string representation of this ConditionalFreqDist.


###nltk Reference - ConditionalProbDist - Prob dist based on ConditionalFreqDist
class nltk.probability.ConditionalProbDist(cfdist, probdist_factory, *factory_args, **factory_kw_args)
    Bases: nltk.probability.ConditionalProbDistI, dict 
    cfdist = ConditionalFreqDist
    probdist_factory = function that takes a condition’s frequency distribution, 
                       and returns its probability distribution(Instance of ProbDist)
                       OR ProbDist class's name (such as MLEProbDist or HeldoutProbDist) 

#For example, the following code constructs a ConditionalProbDist, 
#where the probability distribution for each condition is an ELEProbDist with 10 bins

from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, ELEProbDist
cfdist = ConditionalFreqDist(brown.tagged_words()[:5000]) #tagged_words = list of (word, tag), so condition is word , variable is tag 
cpdist = ConditionalProbDist(cfdist, ELEProbDist, 10)
>>> cpdist['passed'].max()
'VBD'
>>> cpdist['passed'].prob('VBD')
0.423...

#Methods 
conditions()
    Return a list of the conditions that are represented by this ConditionalProbDist. 

unicode_repr()
    Return a string representation of this ConditionalProbDist.

##Subclasses of ConditionalProbDistI
class nltk.probability.DictionaryConditionalProbDist(probdist_dict)
    Bases: nltk.probability.ConditionalProbDistI
    wraps a dictionary of ProbDists rather than creating these from FreqDists.

   
    
##Utility from probability modules 
nltk.probability.add_logs(logx, logy)
    Given two numbers logx = log(x) and logy = log(y), return log(x+y).
    
nltk.probability.sum_logs(logs)
    Returns log(x+y+...) given logx, logy,...

nltk.probability.log_likelihood(test_pdist, actual_pdist)
    def log_likelihood(test_pdist, actual_pdist):
        if (not isinstance(test_pdist, ProbDistI) or
            not isinstance(actual_pdist, ProbDistI)):
            raise ValueError('expected a ProbDist.')
        # Is this right?
        return sum(actual_pdist.prob(s) * math.log(test_pdist.prob(s), 2)  for s in actual_pdist)


nltk.probability.entropy(pdist)
    def entropy(pdist):
        probs = (pdist.prob(s) for s in pdist.samples())
        return -sum(p * math.log(p,2) for p in probs)

    
    

##NLTK - Probability - Examples 

##NLTK's Frequency Distributions: commonly-used methods                            Description
fdist = FreqDist(samples)           create a frequency distribution containing the given samples 
fdist[sample] += 1                  increment the count for this sample 
fdist['monstrous']                  count of the number of times a given sample occurred 
fdist.freq('monstrous')             frequency of a given sample (%)
fdist.N()                           total number of samples 
fdist.most_common(n)                the n most common samples and their frequencies 
for sample in fdist:                iterate over the samples 
fdist.max()                         sample with the greatest count 
fdist.tabulate()                    tabulate the frequency distribution 
fdist.plot()                        graphical plot of the frequency distribution 
fdist.plot(cumulative=True)         cumulative plot of the frequency distribution 
fdist1 |= fdist2                    update fdist1 with counts from fdist2 
fdist1 < fdist2                     test if samples in fdist1 occur less frequently than in fdist2 

##NLTK's Conditional Frequency Distributions: commonly-used methods 

cfdist = ConditionalFreqDist(pairs)         create a conditional frequency distribution from a list of pairs 
cfdist.conditions()                         the conditions 
cfdist[condition]                           the frequency distribution for this condition 
cfdist[condition][sample]                   frequency for the given sample for this condition 
cfdist.tabulate()                           tabulate the conditional frequency distribution 
cfdist.tabulate(samples, conditions)        tabulation limited to the specified samples and conditions 
cfdist.plot()                               graphical plot of the conditional frequency distribution 
cfdist.plot(samples, conditions)            graphical plot limited to the specified samples and conditions 
cfdist1 < cfdist2                           test if samples in cfdist1 occur less frequently than in cfdist2 

##diff between conditionalFreq and FreqDistr
text = "cow cat mouse cat tiger"

fDist = FreqDist(word for word in word_tokenize(text))

for word in fDist:
    print "Frequency of", word, fDist.freq(word)

This will result in:
Frequency of tiger 0.2
Frequency of mouse 0.2
Frequency of cow 0.2
Frequency of cat 0.4

cfdist = ConditionalFreqDist((len(word), word) for word in word_tokenize(sent))

for condition in cfdist:
    for word in cfdist[condition]:
        print "Cond. frequency of", word, cfdist[condition].freq(word), "[condition is word-length=", condition, "]"

This will print:
Cond. frequency of cow 0.333333333333 [condition is word-length= 3 ]
Cond. frequency of cat 0.666666666667 [condition is word-length= 3 ]
Cond. frequency of tiger 0.5 [condition is word-length= 5 ]
Cond. frequency of mouse 0.5 [condition is word-length= 5 ]


##Example 
  
>>> import nltk
>>> from nltk.probability import *



##FreqDist

>>> text1 = ['no', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '!']
>>> text2 = ['no', 'good', 'porpoise', 'likes', 'to', 'fish', 'fish', 'anywhere', '.']

>>> fd1 = nltk.FreqDist(text1)
>>> fd1 == nltk.FreqDist(text1)
True

#Note that items are sorted in order of decreasing frequency; 
#two items of the same frequency appear in indeterminate order.

>>> import itertools
>>> both = nltk.FreqDist(text1 + text2)
>>> both_most_common = both.most_common()
>>> list(itertools.chain(*(sorted(ys) for k, ys in itertools.groupby(both_most_common, key=lambda t: t[1]))))
[('fish', 3), ('anywhere', 2), ('good', 2), ('no', 2), ('porpoise', 2), ('!', 1), ('.', 1), ('a', 1), ('goes', 1), ('likes', 1), ('to', 1), ('without', 1)]

>>> both == fd1 + nltk.FreqDist(text2)
True
>>> fd1 == nltk.FreqDist(text1) # But fd1 is unchanged
True

>>> fd2 = nltk.FreqDist(text2)
>>> fd1.update(fd2)
>>> fd1 == both
True

>>> fd1 = nltk.FreqDist(text1)
>>> fd1.update(text2)
>>> fd1 == both
True

>>> fd1 = nltk.FreqDist(text1)
>>> fd2 = nltk.FreqDist(fd1)
>>> fd2 == fd1
True


##nltk.FreqDist can be pickled

>>> import pickle
>>> fd1 = nltk.FreqDist(text1)
>>> pickled = pickle.dumps(fd1)
>>> fd1 == pickle.loads(pickled)
True

## Example of Prob distribution - SimpleGoodTuringProbDist
>>> from nltk import SimpleGoodTuringProbDist, FreqDist
>>> fd = FreqDist({'a':1, 'b':1, 'c': 2, 'd': 3, 'e': 4, 'f': 4, 'g': 4, 'h': 5, 'i': 5, 'j': 6, 'k': 6, 'l': 6, 'm': 7, 'n': 7, 'o': 8, 'p': 9, 'q': 10})
>>> p = SimpleGoodTuringProbDist(fd)
>>> p.prob('a')
0.017649766667026317...
>>> p.prob('o')
0.08433050215340411...
>>> p.prob('z')
0.022727272727272728...
>>> p.prob('foobar')
0.022727272727272728...


#MLEProbDist, ConditionalProbDist
#DictionaryConditionalProbDist and ConditionalFreqDist can be pickled:

>>> import pickle
>>> pd = MLEProbDist(fd)
>>> sorted(pd.samples()) == sorted(pickle.loads(pickle.dumps(pd)).samples())
True
>>> dpd = DictionaryConditionalProbDist({'x': pd})
>>> unpickled = pickle.loads(pickle.dumps(dpd))
>>> dpd['x'].prob('a')
0.011363636...
>>> dpd['x'].prob('a') == unpickled['x'].prob('a')
True
>>> cfd = nltk.probability.ConditionalFreqDist()
>>> cfd['foo']['hello'] += 1
>>> cfd['foo']['hello'] += 1
>>> cfd['bar']['hello'] += 1
>>> cfd2 = pickle.loads(pickle.dumps(cfd))
>>> cfd2 == cfd
True
>>> cpd = ConditionalProbDist(cfd, SimpleGoodTuringProbDist)
>>> cpd2 = pickle.loads(pickle.dumps(cpd))
>>> cpd['foo'].prob('hello') == cpd2['foo'].prob('hello')
True

  

###nltk - Using ProbDist - Testing some HMM(Hidden Markove Model) estimators
#Hidden Markov model class, a generative model for labelling sequence data. 
#These models define the joint probability of a sequence of symbols  and their labels (state transitions) 

##nltk reference - HiddenMarkovModelTrainer
class nltk.tag.hmm.HiddenMarkovModelTrainer(states=None, symbols=None)
    Algorithms for learning HMM parameters from training data. 
        states (sequence of any) – the set of state labels
        symbols (sequence of any) – the set of observation symbols
    train(labeled_sequences=None, unlabeled_sequences=None, **kwargs)
        Trains the HMM using both (or either of) supervised and unsupervised techniques.
        Returns: the trained model
        Return type:HiddenMarkovModelTagger
        Parameters:	
            labelled_sequences (list) – the supervised training data, a set of labelled sequences of observations
            unlabeled_sequences (list) – the unsupervised training data, a set of sequences of observations
            kwargs – additional arguments to pass to the training methods
    train_supervised(labelled_sequences, estimator=None)
        Supervised training maximising the joint probability of the symbol 
        and state sequences.
        Return type:        HiddenMarkovModelTagger
        Parameters:	
            labelled_sequences (list) – the training data, a set of labelled sequences of observations
            estimator – a function taking a FreqDist and a number of bins and returning a CProbDistI; 
                        otherwise a MLE estimate is used
    train_unsupervised(unlabeled_sequences, update_outputs=True, **kwargs)
        Trains the HMM using the Baum-Welch algorithm to maximise the probability of the data sequence. 
        Return type:	HiddenMarkovModelTagger
        Parameters:	unlabeled_sequences (list) – the training data, a set of sequences of observations
 
class nltk.tag.hmm.HiddenMarkovModelTagger(symbols, states, transitions, outputs, priors, transform=<function _identity>)
    Hidden Markov model class, a generative model for labelling sequence data. 
    These models define the joint probability of a sequence of symbols 
    and their labels (state transitions) 
    as the product of the starting state probability, 
    the probability of each state transition, 
    and the probability of each observation being generated from each state. 
 
    best_path(unlabeled_sequence)
        Returns the state sequence of the optimal (most probable) path through the HMM. 
        Uses the Viterbi algorithm to calculate this part by dynamic programming.
        Returns:	the state sequence
        Return type:	sequence of any
        Parameters:	unlabeled_sequence (list) – the sequence of unlabeled symbols

    best_path_simple(unlabeled_sequence)
        Returns the state sequence of the optimal (most probable) path through the HMM. 
        Uses the Viterbi algorithm to calculate this part by dynamic programming. 
        This uses a simple, direct method, and is included for teaching purposes.
        Returns:	the state sequence
        Return type:	sequence of any
        Parameters:	unlabeled_sequence (list) – the sequence of unlabeled symbols

    entropy(unlabeled_sequence)
        Returns the entropy over labellings of the given sequence. 
        
    log_probability(sequence)
        Returns the log-probability of the given symbol sequence. 
        If the sequence is labelled, then returns the joint log-probability of the symbol, state sequence. 
        Otherwise, uses the forward algorithm to find the log-probability over all label sequences.
        Returns:	the log-probability of the sequence
        Return type:	float
        Parameters:	sequence (Token) – the sequence of symbols which must contain the TEXT property, and optionally the TAG property

    point_entropy(unlabeled_sequence)
        Returns the pointwise entropy over the possible states at each position in the chain, given the observation sequence.

    probability(sequence)
        Returns the probability of the given symbol sequence. 
        If the sequence is labelled, then returns the joint probability of the symbol, state sequence. 
        Otherwise, uses the forward algorithm to find the probability over all label sequences.
        Returns:	the probability of the sequence
        Return type:	float
        Parameters:	sequence (Token) – the sequence of symbols which must contain the TEXT property, and optionally the TAG property

    random_sample(rng, length)
        Randomly sample the HMM to generate a sentence of a given length. 
        Returns:
        the randomly created state/observation sequence, generated according to the HMM’s probability distributions. The SUBTOKENS have TEXT and TAG properties containing the observation and state respectively.
        Return type:
        list
        Parameters:	
            rng (Random (or any object with a random() method)) – random number generator
            length (int) – desired output length

    reset_cache()

    tag(unlabeled_sequence)
        Tags the sequence with the highest probability state sequence. 
        This uses the best_path method to find the Viterbi path.
        Determine the most appropriate tag sequence for the given token sequence, and return a corresponding list of tagged tokens. 
        A tagged token is encoded as a tuple (token, tag).
        Returns:	a labelled sequence of symbols
        Return type:	list(tuple(str, str))
        Parameters:	unlabeled_sequence (list) – the sequence of unlabeled symbols   

    test(test_sequence, verbose=False, **kwargs)
        Tests the HiddenMarkovModelTagger instance.
        Parameters:	
            test_sequence (list(list)) – a sequence of labeled test instances
            verbose (bool) – boolean flag indicating whether training should be verbose or include printed output

    classmethod train(labeled_sequence, test_sequence=None, unlabeled_sequence=None, **kwargs)
        Train a new HiddenMarkovModelTagger using the given labeled 
        and unlabeled training instances. 
        Testing will be performed if test instances are provided.
        Returns:
        a hidden markov model tagger
        Return type:	
        HiddenMarkovModelTagger
        Parameters:	
            labeled_sequence (list(list)) – a sequence of labeled training instances, i.e. a list of sentences represented as tuples
            test_sequence (list(list)) – a sequence of labeled test instances
            unlabeled_sequence (list(list)) – a sequence of unlabeled training instances, i.e. a list of sentences represented as words
            transform (function) – an optional function for transforming training instances, defaults to the identity function, see transform()
            estimator (class or function) – an optional function or class that maps a condition’s frequency distribution to its probability distribution, defaults to a Lidstone distribution with gamma = 0.1
            verbose (bool) – boolean flag indicating whether training should be verbose or include printed output
            max_iterations (int) – number of Baum-Welch interations to perform

    unicode_repr()
    
    evaluate(gold)
        Score the accuracy of the tagger against the gold standard. 
        Strip the tags from the gold standard text, retag it using the tagger, 
        then compute the accuracy score.
        Parameters:	gold (list(list(tuple(str, str)))) – The list of tagged sentences to score the tagger on.
        Return type:	float

    tag_sents(sentences)
        Apply self.tag() to each element of sentences. I.e.:
            return [self.tag(sent) for sent in sentences]

            
#Example 
#corpus is tagged for their part-of-speech
>>> nltk.corpus.brown.tagged_words()
[('The', 'AT'), ('Fulton', 'NP-TL'), ...]
>>> nltk.corpus.brown.tagged_words(tagset='universal')
[('The', 'DET'), ('Fulton', 'NOUN'), ...]

#extract a small part (500 sentences) of the Brown corpus
#tagged_sents: divides up sentence into tagged words(tagged for their part-of-speech) 
>>> corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]
[[('Dan', 'NP'), ('Morgan', 'NP'), ('told', 'VBD'), ('himself', 'PPL')
, ('he', 'PPS'), ('would', 'MD'), ('forget', 'VB'), ('Ann', 'NP'), ('Turner', 'NP'), ('.', '.')], 
[('He', 'PPS'), ('was', 'BEDZ'), ('well','RB'), ('rid', 'JJ'), ('of', 'IN'), ('her', 'PPO'), ('.', '.')], ...]
>>> print(len(corpus))
500


#create a HMM trainer - note that we need the tags and symbols from the whole corpus, 
#not just the training corpus

>>> from nltk.util import unique_list
>>> tag_set = unique_list(tag for sent in corpus for (word,tag) in sent)
>>> print(len(tag_set))
92
>>> symbols = unique_list(word for sent in corpus for (word,tag) in sent)
>>> print(len(symbols))
1464

>>> trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)

   
    
    
#We divide the corpus into 90% training and 10% testing

>>> train_corpus = []
>>> test_corpus = []
>>> for i in range(len(corpus)):
        if i % 10:
            train_corpus += [corpus[i]]
        else:
            test_corpus += [corpus[i]]
>>> print(len(train_corpus))
450
>>> print(len(test_corpus))
50


#And now we can test the estimators

>>> def train_and_test(est):
        hmm = trainer.train_supervised(train_corpus, estimator=est) #retrns nltk.tag.hmm.HiddenMarkovModelTagger
        print('%.2f%%' % (100 * hmm.evaluate(test_corpus)))



##Maximum Likelihood Estimation
>>> mle = lambda fd, bins: MLEProbDist(fd)
>>> train_and_test(mle)
22.75%


#with Laplace (= Lidstone with gamma==1)
>>> train_and_test(LaplaceProbDist)
66.04%


#Expected Likelihood Estimation (= Lidstone with gamma==0.5)

>>> train_and_test(ELEProbDist)
73.01%


#Lidstone Estimation, for gamma==0.1, 0.5 and 1 
#(the later two should be exactly equal to MLE and ELE above)

>>> def lidstone(gamma):
        return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)
>>> train_and_test(lidstone(0.1))
82.51%
>>> train_and_test(lidstone(0.5))
73.01%
>>> train_and_test(lidstone(1.0))
66.04%



##Witten Bell Estimation
>>> train_and_test(WittenBellProbDist)
88.12%


#Good Turing Estimation
>>> gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
>>> train_and_test(gt)
86.93%



##Kneser Ney Estimation
#Since the Kneser-Ney distribution is best suited for trigrams, 
#we must adjust our testing accordingly.

>>> corpus = [[((x[0],y[0],z[0]),(x[1],y[1],z[1]))
        for x, y, z in nltk.trigrams(sent)]
            for sent in corpus[:100]]

#We will then need to redefine the rest of the training/testing variables
>>> tag_set = unique_list(tag for sent in corpus for (word,tag) in sent)
>>> len(tag_set)
906

>>> symbols = unique_list(word for sent in corpus for (word,tag) in sent)
>>> len(symbols)
1341

>>> trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
>>> train_corpus = []
>>> test_corpus = []

>>> for i in range(len(corpus)):
        if i % 10:
            train_corpus += [corpus[i]]
        else:
            test_corpus += [corpus[i]]

>>> len(train_corpus)
90
>>> len(test_corpus)
10

>>> kn = lambda fd, bins: KneserNeyProbDist(fd)
>>> train_and_test(kn)
0.86%


    
    
    
    
    
    
    
    
    

    
###NLP terminologies and challenges 

##Word Sense Disambiguation

#alone 'serve/dish' is ambiguous 
a.  serve: help with food or drink; hold an office; put ball into play 
b.  dish: plate; course of a meal; communications device 


#interpret the meaning of by.
a.  The lost children were found by the searchers (agentive) 
b.  The lost children were found by the mountain (locative) 
c.  The lost children were found by the afternoon (temporal) 

 
##Pronoun Resolution
#"who did what to whom" — i.e., to detect the subjects and objects of verbs
#Consider three possible following sentences 
#and try to determine what was sold, caught, and found (one case is ambiguous).
a.  The thieves stole the paintings. They were subsequently sold. 
b.  The thieves stole the paintings. They were subsequently caught. 
c.  The thieves stole the paintings. They were subsequently found. 

 
#Answering this question involves finding the antecedent of the pronoun they, either thieves or paintings. 
#Computational techniques for tackling this problem include 
#anaphora resolution — identifying what a pronoun or noun phrase refers to — 
#and semantic role labeling — identifying how a noun phrase relates to the verb (as agent, patient, instrument, and so on).

##Generating Language Output - Machine Translation
#Checking - equilibrium
#REF- http://www.translationparty.com/
#Machine Translation systems have some serious shortcomings, which are starkly revealed 
#by translating a sentence back and forth between a pair of languages until equilibrium is reached


##Spoken Dialog Systems
#In the history of artificial intelligence, the chief measure of intelligence has been a linguistic one, 
#namely the Turing Test: 
#can a dialogue system, responding to a user's text input, perform so naturally that we cannot distinguish it from a human-generated response?
#Check the Architecture of Dialog Systems examples/data/dialogue.png

#For an example of a primitive dialogue system, 
#try having a conversation with an NLTK chatbot. 
#To see the available chatbots, run nltk.chat.chatbots().

#These are implemented as RE tokens of questions and ansers, check code nltk/chat/rude.py



###nltk - Exploring corpora - large bodies of linguistic data ie text 

#Download data/corpus with nltk (has option for proxy server)
>>> import nltk
>>> nltk.download()  #set c:\nltk_data else set NLTK_DATA env var 


##Accessing Text Corpora
#NLTK includes a small selection of texts from the Project Gutenberg electronic text archive, 
#which contains some 25,000 free electronic books, hosted at http://www.gutenberg.org/

>>> nltk.corpus.gutenberg.fileids()
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']


#Emma by Jane Austen — find out how many words it contains:
>>> emma = nltk.corpus.gutenberg.words('austen-emma.txt')
>>> len(emma)
192427
 
#concordancing 
#A concordance view shows us every occurrence of a given word
>>> emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
>>> emma.concordance("surprize")

    

##average word length, average sentence length, 
#and the number of times each vocabulary item appears in the text on average (lexical diversity score).
    
>>> for fileid in gutenberg.fileids():
        num_chars = len(gutenberg.raw(fileid)) 
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
        print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
    
5 25 26 austen-emma.txt
5 26 17 austen-persuasion.txt
5 28 22 austen-sense.txt
4 34 79 bible-kjv.txt
5 19 5 blake-poems.txt
4 19 14 bryant-stories.txt
4 18 12 burgess-busterbrown.txt
4 20 13 carroll-alice.txt
5 20 12 chesterton-ball.txt
5 23 11 chesterton-brown.txt
5 18 11 chesterton-thursday.txt
4 21 25 edgeworth-parents.txt
5 26 15 melville-moby_dick.txt
5 52 11 milton-paradise.txt
4 12 9 shakespeare-caesar.txt
4 12 8 shakespeare-hamlet.txt
4 12 7 shakespeare-macbeth.txt
5 36 12 whitman-leaves.txt
    
#The raw() function gives us the contents of the file without any linguistic processing. 
#The sents() function divides the text up into its sentences, 
#where each sentence is a list of words:
>>> macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
>>> macbeth_sentences
[['[', 'The', 'Tragedie', 'of', 'Macbeth', 'by', 'William', 'Shakespeare',
'1603', ']'], ['Actus', 'Primus', '.'], ...]
>>> macbeth_sentences[1116]
['Double', ',', 'double', ',', 'toile', 'and', 'trouble', ';',
'Fire', 'burne', ',', 'and', 'Cauldron', 'bubble']
>>> longest_len = max(len(s) for s in macbeth_sentences)
>>> [s for s in macbeth_sentences if len(s) == longest_len]
[['Doubtfull', 'it', 'stood', ',', 'As', 'two', 'spent', 'Swimmers', ',', 'that',
'doe', 'cling', 'together', ',', 'And', 'choake', 'their', 'Art', ':', 'The',
'mercilesse', 'Macdonwald', ...]]


##NLTK's small collection of web text includes content from a Firefox discussion forum, 
#conversations overheard in New York, the movie script of Pirates of the Carribean, 
#personal advertisements, and wine reviews:

>>> from nltk.corpus import webtext
>>> for fileid in webtext.fileids():
        print(fileid, webtext.raw(fileid)[:65], '...')
...
firefox.txt Cookie Manager: "Don't allow sites that set removed cookies to se...
grail.txt SCENE 1: [wind] [clop clop clop] KING ARTHUR: Whoa there!  [clop...
overheard.txt White guy: So, do you have any plans for this evening? Asian girl...
pirates.txt PIRATES OF THE CARRIBEAN: DEAD MAN'S CHEST, by Ted Elliott & Terr...
singles.txt 25 SEXY MALE, seeks attrac older single lady, for discreet encoun...
wine.txt Lovely delicate, fragrant Rhone wine. Polished leather and strawb...
 
 
##There is also a corpus of instant messaging chat sessions
#The filename contains the date, chatroom, and number of posts; e.g., 10-19-20s_706posts.xml contains 706 posts gathered from the 20s(age) chat room on 10/19/2006.

>>> from nltk.corpus import nps_chat
>>> chatroom = nps_chat.posts('10-19-20s_706posts.xml')
>>> chatroom[123]
['i', 'do', "n't", 'want', 'hot', 'pics', 'of', 'a', 'female', ',',
'I', 'can', 'look', 'in', 'a', 'mirror', '.']  
    
##Brown Corpus
#The Brown Corpus was the first million-word electronic corpus (English)
#categorized by genre, such as news, editorial, http://icame.uib.no/brown/bcm-los.html
    
>>> from nltk.corpus import brown
>>> brown.categories()
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
'science_fiction']
>>> brown.words(categories='news')
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
>>> brown.words(fileids=['cg22'])
['Does', 'our', 'society', 'have', 'a', 'runaway', ',', ...]
>>> brown.sents(categories=['news', 'editorial', 'reviews'])
[['The', 'Fulton', 'County'...], ['The', 'jury', 'further'...], ...]
    
##The Brown Corpus is a convenient resource for studying systematic differences between genres, 
#a kind of linguistic inquiry known as stylistics. 

#Let's compare genres in their usage of modal verbs. 
#The first step is to produce the counts for a particular genre. 

>>> from nltk.corpus import brown
>>> news_text = brown.words(categories='news')
>>> fdist = nltk.FreqDist(w.lower() for w in news_text)
>>> modals = ['can', 'could', 'may', 'might', 'must', 'will']
>>> for m in modals:
        print(m + ':', fdist[m], end=' ')

can: 94 could: 87 may: 93 might: 38 must: 53 will: 389

#Next, we need to obtain counts for each genre of interest 
>>> cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))
>>> genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
>>> modals = ['can', 'could', 'may', 'might', 'must', 'will']

#Observe that the most frequent modal in the news genre is will, 
#while the most frequent modal in the romance genre is could
>>> cfd.tabulate(conditions=genres, samples=modals)
                 can could  may might must will
           news   93   86   66   38   50  389
       religion   82   59   78   12   54   71
        hobbies  268   58  131   22   83  264
science_fiction   16   49    4   12    8   16
        romance   74  193   11   51   45   43
          humor   16   30    8    8    9   13 
    

    
    
###nltk-reference - Text Corpus Structure 
help(nltk.corpus.reader)

fileids()                           the files of the corpus 
fileids([categories])               the files of the corpus corresponding to these categories 
categories()                        the categories of the corpus 
categories([fileids])               the categories of the corpus corresponding to these files 
raw()                               the raw content of the corpus 
raw(fileids=[f1,f2,f3])             the raw content of the specified files 
raw(categories=[c1,c2])             the raw content of the specified categories 
words()                             the words of the whole corpus 
words(fileids=[f1,f2,f3])           the words of the specified fileids 
words(categories=[c1,c2])           the words of the specified categories 
sents()                             the sentences of the whole corpus 
sents(fileids=[f1,f2,f3])           the sentences of the specified fileids 
sents(categories=[c1,c2])           the sentences of the specified categories 
abspath(fileid)                     the location of the given file on disk 
encoding(fileid)                    the encoding of the file (if known) 
open(fileid)                        open a stream for reading the given corpus file 
root                                if the path to the root of locally installed corpus 
readme()                            the contents of the README file of the corpus 



class nltk.corpus.reader.api.CorpusReader(root, fileids, encoding='utf8', tagset=None)¶
    Bases: object
    A base class for “corpus reader” classes,  
    
    abspath(fileid)
        Return the absolute path for the given file.
        Parameters:	fileid (str) – The file identifier for the file whose path should be returned.
        Return type:	PathPointer

    abspaths(fileids=None, include_encoding=False, include_fileid=False)
        Return a list of the absolute paths(PathPointer) for all fileids in this corpus

    citation()
        Return the contents of the corpus citation.bib file, if it exists.

    encoding(file)
        Return the unicode encoding for the given corpus file, if known. 
        If the encoding is unknown, or if the given file should be processed using byte strings (str), then return None.

    ensure_loaded()
        Load this corpus (if it has not already been loaded). 
  
    fileids()
        Return a list of file identifiers for the fileids that make up this corpus.

    license()
        Return the contents of the corpus LICENSE file, if it exists.

    open(file)
        Return an open stream that can be used to read the given file. 
        If the file’s encoding is not None, then the stream will automatically decode the file’s contents into unicode.
        Parameters:	file – The file identifier of the file to read.

    readme()
        Return the contents of the corpus README file, if it exists.

    root
        The directory where this corpus is stored.
        Type:	PathPointer
        
    unicode_repr()

    
class nltk.corpus.reader.plaintext.PlaintextCorpusReader(root, fileids, w
                ord_tokenizer=WordPunctTokenizer(pattern='\w+|[^\w\s]+', 
                gaps=False, discard_empty=True, flags=56), 
                sent_tokenizer=<nltk.tokenize.punkt.PunktSentenceTokenizer object>, 
                para_block_reader=<function read_blankline_block>, encoding='utf8')
    Bases: nltk.corpus.reader.api.CorpusReader

    Reader for corpora that consist of plaintext documents. 
    Paragraphs are assumed to be split using blank lines. 
    Sentences and words can be tokenized using the default tokenizers, 
    or by custom tokenizers specificed as parameters to the constructor.

    paras(fileids=None)
        Returns:	the given file(s) as a list of paragraphs, 
                    each encoded as a list of sentences, which are in turn encoded as lists of word strings.
        Return type:	list(list(list(str)))

    raw(fileids=None)
        Returns:	the given file(s) as a single string.
        Return type:	str

    sents(fileids=None)
        Returns:	the given file(s) as a list of sentences or utterances, 
                    each encoded as a list of word strings.
        Return type:	list(list(str))

    words(fileids=None)
        Returns:	the given file(s) as a list of words and punctuation symbols.
        Return type:	list(str)
        
    
class nltk.corpus.reader.api.SyntaxCorpusReader(root, fileids, encoding='utf8', tagset=None)
    Bases: nltk.corpus.reader.api.CorpusReader  
    parsed_sents(fileids=None)
    raw(fileids=None)
    sents(fileids=None)
    tagged_sents(fileids=None, tagset=None)
    tagged_words(fileids=None, tagset=None)
    words(fileids=None)   
    
class nltk.corpus.reader.bracket_parse.BracketParseCorpusReader(root, fileids, 
        comment_char=None, detect_blocks='unindented_paren', encoding='utf8', 
        tagset=None)
    Bases: nltk.corpus.reader.api.SyntaxCorpusReader
    Reader for corpora that consist of parenthesis-delineated parse trees, 
    like those found in the “combined” section of the Penn Treebank, 
    e.g. “(S (NP (DT the) (JJ little) (NN dog)) (VP (VBD barked)))”.

    
##Example 
#The simplest kind lacks any structure: it is just a collection of texts. 
#Often, texts are grouped into categories that might correspond to genre, source, author, language, etc. 
#Occasionally, text collections have temporal structure

>>> nltk.corpus.gutenberg.fileids()
>>> raw = gutenberg.raw("burgess-busterbrown.txt")
>>> raw[1:20]
'The Adventures of B'
>>> words = gutenberg.words("burgess-busterbrown.txt")
>>> words[1:20]
['The', 'Adventures', 'of', 'Buster', 'Bear', 'by', 'Thornton', 'W', '.',
'Burgess', '1920', ']', 'I', 'BUSTER', 'BEAR', 'GOES', 'FISHING', 'Buster',
'Bear']
>>> sents = gutenberg.sents("burgess-busterbrown.txt")
>>> sents[1:20]
[['I'], ['BUSTER', 'BEAR', 'GOES', 'FISHING'], ['Buster', 'Bear', 'yawned', 'as',
'he', 'lay', 'on', 'his', 'comfortable', 'bed', 'of', 'leaves', 'and', 'watched',
'the', 'first', 'early', 'morning', 'sunbeams', 'creeping', 'through', ...], ...] 
    
##Loading your own Corpus - any list of text files  
#NLTK's PlaintextCorpusReader. 
>>> from nltk.corpus import PlaintextCorpusReader
>>> corpus_root = '/usr/share/dict'  #directory 

#The second parameter of the PlaintextCorpusReader initializer can be a list of fileids, like ['a.txt', 'test/b.txt'], 
#or a pattern that matches all fileids, like '[abc]/.*\.txt' 


>>> wordlists = PlaintextCorpusReader(corpus_root, '.*') 
>>> wordlists.fileids()
['README', 'connectives', 'propernames', 'web2', 'web2a', 'words']
>>> wordlists.words('connectives')
['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', ...]


#Corpus reader for corpora that consist of parenthesis-delineated parse trees.
#suppose you have your own local copy of Penn Treebank (release 3), in C:\corpora
>>> from nltk.corpus import BracketParseCorpusReader
>>> corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj" 
>>> file_pattern = r".*/wsj_.*\.mrg" 
>>> ptb = BracketParseCorpusReader(corpus_root, file_pattern)
>>> ptb.fileids()
['00/wsj_0001.mrg', '00/wsj_0002.mrg', '00/wsj_0003.mrg', '00/wsj_0004.mrg', ...]
>>> len(ptb.sents())
49208
>>> ptb.sents(fileids='20/wsj_2013.mrg')[19]
['The', '55-year-old', 'Mr.', 'Noriega', 'is', "n't", 'as', 'smooth', 'as', 'the',
'shah', 'of', 'Iran', ',', 'as', 'well-born', 'as', 'Nicaragua', "'s", 'Anastasio',
'Somoza', ',', 'as', 'imperial', 'as', 'Ferdinand', 'Marcos', 'of', 'the', 'Philippines',
'or', 'as', 'bloody', 'as', 'Haiti', "'s", 'Baby', Doc', 'Duvalier', '.']


##Generating Random Text with Bigrams (word pairs). 
>>> sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.']
>>> list(nltk.bigrams(sent))
[('In', 'the'), ('the', 'beginning'), ('beginning', 'God'), ('God', 'created'),
('created', 'the'), ('the', 'heaven'), ('heaven', 'and'), ('and', 'the'),
('the', 'earth'), ('earth', '.')]


#Example 
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()  #set word as max frequency of current word 

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)  #two words, first is condition, 2nd is word 
 
>>> cfd['living']  #living creature, living thing .....
FreqDist({'creature': 7, 'thing': 4, 'substance': 2, ',': 1, '.': 1, 'soul': 1})
>>> generate_model(cfd, 'living')
living creature that he said , and the land of the land of the land
 
 
##lexical_diversity
#% of unique words 

>>> def lexical_diversity(my_text_data):
        word_count = len(my_text_data)
        vocab_size = len(set(my_text_data))
        diversity_score = vocab_size / word_count
        return diversity_score
 
 


>>> from nltk.corpus import genesis
>>> kjv = genesis.words('english-kjv.txt')
>>> lexical_diversity(kjv)
0.06230453042623537
 
 
#Creating plural 
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'
 
 

>>> plural('fairy')
'fairies'
>>> plural('woman')
'women'
 
 
###nltk - Lexical Resources
#A lexicon, or lexical resource, is a collection of words and/or phrases 
#along with associated information such as part of speech and sense definitions. 

#Lexical resources are secondary to texts, 
#and are usually created and enriched with the help of texts. 

#For example, if we have defined a text my_text, 
#then vocab = sorted(set(my_text)) builds the vocabulary of my_text, 
#while word_freq = FreqDist(my_text) counts the frequency of each word in the text. 
#Both of vocab and word_freq are simple lexical resources.

#A lexical entry consists of a headword (also known as a lemma) 
#along with additional information such as the part of speech and the sense definition. 

#Two distinct words having the same spelling are called homonyms.

                                headword or lemma   part of speech or lexical category      sense definition or gloss
            lexical entry ->    saw                 [verb]                                  past test of see
homonyms
            lexical entry ->    saw                 [noun]                                  cutting instrument
                        
    

#The simplest kind of lexicon is nothing more than a sorted list of words. 
#Sophisticated lexicons include complex structure 
#within and across the individual entries    
    
    
    
    
### nltk -  nltk.corpus.words - contains english words as vocabs 

>>len(nltk.corpus.words.words())


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

>>> unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
['abbeyland', 'abhorred', 'abilities', 'abounded', 'abridgement', 'abused', 'abuses',
'accents', 'accepting', 'accommodations', 'accompanied', 'accounted', 'accounts',
'accustomary', 'aches', 'acknowledging', 'acknowledgment', 'acknowledgments', ...]
>>> unusual_words(nltk.corpus.nps_chat.words())
['aaaaaaaaaaaaaaaaa', 'aaahhhh', 'abortions', 'abou', 'abourted', 'abs', 'ack',
'acros', 'actualy', 'adams', 'adds', 'adduser', 'adjusts', 'adoted', 'adreniline',
'ads', 'adults', 'afe', 'affairs', 'affari', 'affects', 'afk', 'agaibn', 'ages', ...]
    
    
#Example- how many words can you make from below    
#Each letter must be used once , center letter must be there always 
E G I
V R V
O N L 

>>> nltk.FreqDist('egivrvonl')
FreqDist({'v': 2, 'e': 1, 'o': 1, 'i': 1, 'g': 1, 'l': 1, 'r': 1, 'n': 1})

>>> puzzle_letters = nltk.FreqDist('egivrvonl')
>>> obligatory = 'r'
>>> wordlist = nltk.corpus.words.words()
>>> [w for w in wordlist if len(w) >= 6 
            and obligatory in w 
            and nltk.FreqDist(w) <= puzzle_letters]
['glover', 'gorlin', 'govern', 'grovel', 'ignore', 'involver', 'lienor',
'linger', 'longer', 'lovering', 'noiler', 'overling', 'region', 'renvoi',
'revolving', 'ringle', 'roving', 'violer', 'virole']

    
    
    
### nltk -  nltk.corpus.stopwords - corpus of stopwords
#that is, high-frequency words like the, to and also   
    
>>> from nltk.corpus import stopwords
>>> stopwords.words('english')
 
#Example - Let's define a function to compute 
#what fraction of words in a text are not in the stopwords list:
>>> def content_fraction(text):
        stopwords = nltk.corpus.stopwords.words('english')
        content = [w for w in text if w.lower() not in stopwords]
        return len(content) / len(text)
    
>>> content_fraction(nltk.corpus.reuters.words())
0.7364374824583169
 
 
### nltk -  nltk.corpus.names

#One more wordlist corpus is the Names corpus, 
#containing 8,000 first names categorized by gender. 
#The male and female names are stored in separate files. 

#Example - Let's find names which appear in both files, 
#i.e. names that are ambiguous for gender:



>>> names = nltk.corpus.names
>>> names.fileids()
['female.txt', 'male.txt']
>>> male_names = names.words('male.txt')
>>> female_names = names.words('female.txt')
>>> [w for w in male_names if w in female_names]
['Abbey', 'Abbie', 'Abby', 'Addie', 'Adrian', 'Adrien', 'Ajay', 'Alex', 'Alexis',
'Alfie', 'Ali', 'Alix', 'Allie', 'Allyn', 'Andie', 'Andrea', 'Andy', 'Angel',
'Angie', 'Ariel', 'Ashley', 'Aubrey', 'Augustine', 'Austin', 'Averil', ...]
 
 
#Example - It is well known that names ending in the letter a are almost always female. 
#Remember that name[-1] is the last letter of name.

>>> cfd = nltk.ConditionalFreqDist(
            (fileid, name[-1])
            for fileid in names.fileids()
            for name in names.words(fileid))
>>> cfd.plot()
 
 
### nltk -  A Pronouncing Dictionary
#is a table (or spreadsheet), containing a word plus some properties in each row. 

#NLTK includes the CMU Pronouncing Dictionary for US English, 
#which was designed for use by speech synthesizers.

>>> entries = nltk.corpus.cmudict.entries()
>>> len(entries)
133737
>>> for entry in entries[42371:42379]:
        print(entry)
...
('fir', ['F', 'ER1'])
('fire', ['F', 'AY1', 'ER0'])
('fire', ['F', 'AY1', 'R'])
('firearm', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M'])
('firearm', ['F', 'AY1', 'R', 'AA2', 'R', 'M'])
('firearms', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M', 'Z'])
('firearms', ['F', 'AY1', 'R', 'AA2', 'R', 'M', 'Z'])
('fireball', ['F', 'AY1', 'ER0', 'B', 'AO2', 'L'])
 
#For each word, this lexicon provides a list of phonetic codes 
#— distinct labels for each contrastive sound — known as phones. 

#Observe that fire has two pronunciations (in US English): 
#the one-syllable F AY1 R, and the two-syllable F AY1 ER0. 

#The symbols in the CMU Pronouncing Dictionary are from the Arpabet, 
#described in more detail at http://en.wikipedia.org/wiki/Arpabet

#Example - scans the lexicon looking for entries 
#whose pronunciation consists of three phones and begining and ending with P and T 
>>> for word, pron in entries: 
        if len(pron) == 3: 
            ph1, ph2, ph3 = pron 
            if ph1 == 'P' and ph3 == 'T':
                print(word, ph2, end=' ')
    
pait EY1 pat AE1 pate EY1 patt AE1 peart ER1 peat IY1 peet IY1 peete IY1 pert ER1
pet EH1 pete IY1 pett EH1 piet IY1 piette IY1 pit IH1 pitt IH1 pot AA1 pote OW1
pott AA1 pout AW1 puett UW1 purt ER1 put UH1 putt AH1

#Example - finds all words whose pronunciation ends with a syllable sounding like nicks. 
#You could use this method to find rhyming words.

>>> syllable = ['N', 'IH0', 'K', 'S']
>>> [word for word, pron in entries if pron[-4:] == syllable]
["atlantic's", 'audiotronics', 'avionics', 'beatniks', 'calisthenics', 'centronics',
'chamonix', 'chetniks', "clinic's", 'clinics', 'conics', 'conics', 'cryogenics',
'cynics', 'diasonics', "dominic's", 'ebonics', 'electronics', "electronics'", ...]
 
#Notice that the one pronunciation is spelt in several ways: nics, niks, nix, even ntic's with a silent t, for the word atlantic's. 
 

#The phones contain digits to represent primary stress (1), secondary stress (2) and no stress (0). 
#Example -  define a function to extract the stress digits 
#and then scan our lexicon to find words having a particular stress pattern.



>>> def stress(pron):
        return [char for phone in pron for char in phone if char.isdigit()]
>>> [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
['abbreviated', 'abbreviated', 'abbreviating', 'accelerated', 'accelerating',
'accelerator', 'accelerators', 'accentuated', 'accentuating', 'accommodated',
'accommodating', 'accommodative', 'accumulated', 'accumulating', 'accumulative', ...]
>>> [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]
['abbreviation', 'abbreviations', 'abomination', 'abortifacient', 'abortifacients',
'academicians', 'accommodation', 'accommodations', 'accreditation', 'accreditations',
'accumulation', 'accumulations', 'acetylcholine', 'acetylcholine', 'adjudication', ...]
 
 
#Example - use a conditional frequency distribution to help us find minimally-contrasting sets of words. 
#Here we find all the p-words consisting of three sounds [2], 
#and group them according to their first and last sounds [1].

>>> p3 = [(pron[0]+'-'+pron[2], word) #[1]
        for (word, pron) in entries
        if pron[0] == 'P' and len(pron) == 3] #[2]
>>> cfd = nltk.ConditionalFreqDist(p3)
>>> for template in sorted(cfd.conditions()):
        if len(cfd[template]) > 10:
            words = sorted(cfd[template])
            wordstring = ' '.join(words)
            print(template, wordstring[:70] + "...")
    
P-CH patch pautsch peach perch petsch petsche piche piech pietsch pitch pit...
P-K pac pack paek paik pak pake paque peak peake pech peck peek perc perk ...
P-L pahl pail paille pal pale pall paul paule paull peal peale pearl pearl...
P-N paign pain paine pan pane pawn payne peine pen penh penn pin pine pinn...
P-P paap paape pap pape papp paup peep pep pip pipe pipp poop pop pope pop...
P-R paar pair par pare parr pear peer pier poor poore por pore porr pour...
P-S pace pass pasts peace pearse pease perce pers perse pesce piece piss p...
P-T pait pat pate patt peart peat peet peete pert pet pete pett piet piett...
P-UW1 peru peugh pew plew plue prew pru prue prugh pshew pugh...
 
 
#access it by looking up particular words.
>>> prondict = nltk.corpus.cmudict.dict()
>>> prondict['fire'] 
[['F', 'AY1', 'ER0'], ['F', 'AY1', 'R']]
>>> prondict['blog'] 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'blog'
>>> prondict['blog'] = [['B', 'L', 'AA1', 'G']] 
>>> prondict['blog']
[['B', 'L', 'AA1', 'G']]

#We can use any lexical resource to process a text, 
#e.g., to filter out words having some lexical property (like nouns), 
#or mapping every word of the text. 

#example, the following text-to-speech function looks up each word of the text in the pronunciation dictionary.

>>> text = ['natural', 'language', 'processing']
>>> [ph for w in text for ph in prondict[w][0]]
['N', 'AE1', 'CH', 'ER0', 'AH0', 'L', 'L', 'AE1', 'NG', 'G', 'W', 'AH0', 'JH',
'P', 'R', 'AA1', 'S', 'EH0', 'S', 'IH0', 'NG']
 
 
 

### nltk -  Comparative Wordlists
#NLTK includes Swadesh wordlists, lists of about 200 common words in several languages. 
#The languages are identified using an ISO 639 two-letter code.

>>> from nltk.corpus import swadesh
>>> swadesh.fileids()
['be', 'bg', 'bs', 'ca', 'cs', 'cu', 'de', 'en', 'es', 'fr', 'hr', 'it', 'la', 'mk',
'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'uk']
>>> swadesh.words('en')
['I', 'you (singular), thou', 'he', 'we', 'you (plural)', 'they', 'this', 'that',
'here', 'there', 'who', 'what', 'where', 'when', 'how', 'not', 'all', 'many', 'some',
'few', 'other', 'one', 'two', 'three', 'four', 'five', 'big', 'long', 'wide', ...]
 
 
#We can access cognate words from multiple languages using the entries() method, 
#specifying a list of languages. 

>>> fr2en = swadesh.entries(['fr', 'en'])  #fr to en 
>>> fr2en
[('je', 'I'), ('tu, vous', 'you (singular), thou'), ('il', 'he'), ...]
>>> translate = dict(fr2en)
>>> translate['chien']
'dog'
>>> translate['jeter']
'throw'
 
 
#Let's get the German-English and Spanish-English pairs, 
#convert each to a dictionary using dict(), 
#then update our original translate dictionary with these additional mappings:
>>> de2en = swadesh.entries(['de', 'en'])    # German-English
>>> es2en = swadesh.entries(['es', 'en'])    # Spanish-English
>>> translate.update(dict(de2en))
>>> translate.update(dict(es2en))
>>> translate['Hund']
'dog'
>>> translate['perro']
'dog'
 
 

#Example - compare words in various Germanic and Romance languages:
>>> languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
>>> for i in [139, 140, 141, 142]:
        print(swadesh.entries(languages)[i])

('say', 'sagen', 'zeggen', 'decir', 'dire', 'dizer', 'dicere')
('sing', 'singen', 'zingen', 'cantar', 'chanter', 'cantar', 'canere')
('play', 'spielen', 'spelen', 'jugar', 'jouer', 'jogar, brincar', 'ludere')
('float', 'schweben', 'zweven', 'flotar', 'flotter', 'flutuar, boiar', 'fluctuare')
 
 
 
### nltk -  Shoebox and Toolbox Lexicons
#tool used by linguists for managing data is Toolbox, previously known as Shoebox 
#since it replaces the field linguist's traditional shoebox full of file cards. 

#Toolbox is freely downloadable from http://www.sil.org/computing/toolbox/.

#A Toolbox file consists of a collection of entries, 
#where each entry is made up of one or more fields. 
#Most fields are optional or repeatable, which means that this kind of lexical resource cannot be treated as a table or spreadsheet.

#Here is a dictionary for the Rotokas language. 
#for the word kaa meaning "to gag"

#The Rotokas language is spoken on the island of Bougainville, Papua New Guinea. 
#Rotokas is notable for having an inventory of just 12 phonemes (contrastive sounds), http://en.wikipedia.org/wiki/Rotokas_language


>>> from nltk.corpus import toolbox
>>> toolbox.entries('rotokas.dic')
[('kaa', [('ps', 'V'), ('pt', 'A'), ('ge', 'gag'), ('tkp', 'nek i pas'),
('dcsv', 'true'), ('vx', '1'), ('sc', '???'), ('dt', '29/Oct/2005'),
('ex', 'Apoka ira kaaroi aioa-ia reoreopaoro.'),
('xp', 'Kaikai i pas long nek bilong Apoka bikos em i kaikai na toktok.'),
('xe', 'Apoka is gagging from food while talking.')]), ...]
 
#Entries consist of a series of attribute-value pairs, like ('ps', 'V') 
#to indicate that the part-of-speech is 'V' (verb), 
#and ('ge', 'gag') to indicate that the gloss-into-English is 'gag'. 

#The last three pairs contain an example sentence in Rotokas 
#and its translations into Tok Pisin and English.



### nltk -  WordNet
#http://www.nltk.org/howto/wordnet.html
#WordNet is a semantically-oriented dictionary of English, 
#similar to a traditional thesaurus but with a richer structure.

#NLTK includes the English WordNet, with 155,287 words and 117,659 synonym sets

>>> from nltk.corpus import wordnet as wn
>>> wn.synsets('motorcar')
[Synset('car.n.01')]
 
#motorcar has just one possible meaning and it is identified as car.n.01, the first noun sense of car. 
#The entity car.n.01 is called a synset, or "synonym set", a collection of synonymous words (or "lemmas"):
>>> wn.synset('car.n.01').lemma_names()
['car', 'auto', 'automobile', 'machine', 'motorcar']

#OR 
>>> wn.synset('car.n.01').lemmas() 
[Lemma('car.n.01.car'), Lemma('car.n.01.auto'), Lemma('car.n.01.automobile'),
Lemma('car.n.01.machine'), Lemma('car.n.01.motorcar')]
>>> wn.lemma('car.n.01.automobile')
Lemma('car.n.01.automobile')
>>> wn.lemma('car.n.01.automobile').synset() 
Synset('car.n.01')
>>> wn.lemma('car.n.01.automobile').name() 
'automobile'
 
#Each word of a synset can have several meanings, e.g., car can also signify a train carriage, 
#a gondola, or an elevator car. 

#However, we are only interested in the single meaning that is common to all words of the above synset.
#Synsets also come with a prose definition and some example sentences:
>>> wn.synset('car.n.01').definition()
'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
>>> wn.synset('car.n.01').examples()
['he needs a car to get to work']
 
 

#Unlike the word motorcar, which is unambiguous and has one synset, 
#the word car is ambiguous, having five synsets:

>>> wn.synsets('car')
[Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'),
Synset('cable_car.n.01')]
>>> for synset in wn.synsets('car'):
        print(synset.lemma_names())
...
['car', 'auto', 'automobile', 'machine', 'motorcar']
['car', 'railcar', 'railway_car', 'railroad_car']
['car', 'gondola']
['car', 'elevator_car']
['cable_car', 'car']
 
 
#access all the lemmas involving the word car as follows.
>>> wn.lemmas('car')
[Lemma('car.n.01.car'), Lemma('car.n.02.car'), Lemma('car.n.03.car'),
Lemma('car.n.04.car'), Lemma('cable_car.n.01.car')]
 
 
##The WordNet Hierarchy
#Fragment of WordNet Concept Hierarchy: 
#nodes correspond to synsets; 
#edges indicate the hypernym/hyponym relation, i.e. the relation between superordinate and subordinate concepts.

                                          artefacts                             
                                          motorvehicle
               motorcar                   go-kart           truck
       hatchback  compact   gas guzzler 


#WordNet makes it easy to navigate between concepts. 
#For example, given a concept like motorcar, 
#we can look at the concepts that are more specific; the (immediate) .

>>> motorcar = wn.synset('car.n.01')
>>> types_of_motorcar = motorcar.()  #subordinate 
>>> types_of_motorcar[0]
Synset('ambulance.n.01')
>>> sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())
['Model_T', 'S.U.V.', 'SUV', 'Stanley_Steamer', 'ambulance', 'beach_waggon',
'beach_wagon', 'bus', 'cab', 'compact', 'compact_car', 'convertible',
'coupe', 'cruiser', 'electric', 'electric_automobile', 'electric_car',
'estate_car', 'gas_guzzler', 'hack', 'hardtop', 'hatchback', 'heap',
'horseless_carriage', 'hot-rod', 'hot_rod', 'jalopy', 'jeep', 'landrover',
'limo', 'limousine', 'loaner', 'minicar', 'minivan', 'pace_car', 'patrol_car',
'phaeton', 'police_car', 'police_cruiser', 'prowl_car', 'race_car', 'racer',
'racing_car', 'roadster', 'runabout', 'saloon', 'secondhand_car', 'sedan',
'sport_car', 'sport_utility', 'sport_utility_vehicle', 'sports_car', 'squad_car',
'station_waggon', 'station_wagon', 'stock_car', 'subcompact', 'subcompact_car',
'taxi', 'taxicab', 'tourer', 'touring_car', 'two-seater', 'used-car', 'waggon',
'wagon']
 
 
#We can also navigate up the hierarchy by visiting hypernyms. 
#Some words have multiple paths, because they can be classified in more than one way. 

#There are two paths between car.n.01 and entity.n.01 
#because wheeled_vehicle.n.01 can be classified as both a vehicle and a container.
>>> motorcar.hypernyms()  #super-ordinate 
[Synset('motor_vehicle.n.01')]
>>> paths = motorcar.hypernym_paths()
>>> len(paths)
2
>>> [synset.name() for synset in paths[0]]
['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'whole.n.02', 'artifact.n.01',
'instrumentality.n.03', 'container.n.01', 'wheeled_vehicle.n.01',
'self-propelled_vehicle.n.01', 'motor_vehicle.n.01', 'car.n.01']
>>> [synset.name() for synset in paths[1]]
['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'whole.n.02', 'artifact.n.01',
'instrumentality.n.03', 'conveyance.n.03', 'vehicle.n.01', 'wheeled_vehicle.n.01',
'self-propelled_vehicle.n.01', 'motor_vehicle.n.01', 'car.n.01']
 
 
#We can get the most general hypernyms (or root hypernyms) of a synset as follows:
>>> motorcar.root_hypernyms()
[Synset('entity.n.01')]
 
 

##More Lexical Relations
#Hypernyms and  are called lexical relations because they relate one synset to another. 
#These two relations navigate up and down the "is-a" hierarchy. 

#Another important way to navigate the WordNet network is from items to their components (meronyms) 
#or to the things they are contained in (holonyms). 

#For example, the parts of a tree are its trunk, crown, and so on; the part_meronyms(). 
#The substance a tree is made of includes heartwood and sapwood; the substance_meronyms(). 

#A collection of trees forms a forest; the member_holonyms():


 >>> wn.synset('tree.n.01').part_meronyms()
[Synset('burl.n.02'), Synset('crown.n.07'), Synset('limb.n.02'),
Synset('stump.n.01'), Synset('trunk.n.01')]
>>> wn.synset('tree.n.01').substance_meronyms()
[Synset('heartwood.n.01'), Synset('sapwood.n.01')]
>>> wn.synset('tree.n.01').member_holonyms()
[Synset('forest.n.01')]
 
 

#consider the word mint, which has several closely-related senses.
# We can see that mint.n.04 is part of mint.n.02 and the substance from which mint.n.05 is made.

>>> for synset in wn.synsets('mint', wn.NOUN):
        print(synset.name() + ':', synset.definition())
...
batch.n.02: (often followed by `of') a large number or amount or extent
mint.n.02: any north temperate plant of the genus Mentha with aromatic leaves and
           small mauve flowers
mint.n.03: any member of the mint family of plants
mint.n.04: the leaves of a mint plant used fresh or candied
mint.n.05: a candy that is flavored with a mint oil
mint.n.06: a plant where money is coined by authority of the government
>>> wn.synset('mint.n.04').part_holonyms()
[Synset('mint.n.02')]
>>> wn.synset('mint.n.04').substance_holonyms()
[Synset('mint.n.05')]
 
 

#There are also relationships between verbs. 
#For example, the act of walking involves the act of stepping, so walking entails stepping. 
#Some verbs have multiple entailments:

>>> wn.synset('walk.v.01').entailments()
[Synset('step.v.01')]
>>> wn.synset('eat.v.01').entailments()
[Synset('chew.v.01'), Synset('swallow.v.01')]
>>> wn.synset('tease.v.03').entailments()
[Synset('arouse.v.07'), Synset('disappoint.v.01')]
 
 

#Some lexical relationships hold between lemmas, e.g., antonymy:
>>> wn.lemma('supply.n.02.supply').antonyms()
[Lemma('demand.n.02.demand')]
>>> wn.lemma('rush.v.01.rush').antonyms()
[Lemma('linger.v.04.linger')]
>>> wn.lemma('horizontal.a.01.horizontal').antonyms()
[Lemma('inclined.a.02.inclined'), Lemma('vertical.a.01.vertical')]
>>> wn.lemma('staccato.r.01.staccato').antonyms()
[Lemma('legato.r.01.legato')]
 
 

# the lexical relations, and the other methods defined on a synset, using dir(), 
dir(wn.synset('harmony.n.02')).


##Semantic Similarity
 
#Given a particular synset, we can traverse the WordNet network to find synsets with related meanings. 

#Knowing which words are semantically related is useful for indexing a collection of texts, 
#so that a search for a general term like vehicle will match documents containing specific terms like limousine.

#Recall that each synset has one or more hypernym paths that link it to a root hypernym such as entity.n.01. 
#Two synsets linked to the same root may have several hypernyms in common . 
#If two synsets share a very specific hypernym 
#— one that is low down in the hypernym hierarchy — they must be closely related.


>>> right = wn.synset('right_whale.n.01')
>>> orca = wn.synset('orca.n.01')
>>> minke = wn.synset('minke_whale.n.01')
>>> tortoise = wn.synset('tortoise.n.01')
>>> novel = wn.synset('novel.n.01')
>>> right.lowest_common_hypernyms(minke)
[Synset('baleen_whale.n.01')]
>>> right.lowest_common_hypernyms(orca)
[Synset('whale.n.02')]
>>> right.lowest_common_hypernyms(tortoise)
[Synset('vertebrate.n.01')]
>>> right.lowest_common_hypernyms(novel)
[Synset('entity.n.01')]
 
 

#we know that whale is very specific (and baleen whale even more so), 
#while vertebrate is more general and entity is completely general. 

#We can quantify this concept of generality by looking up the depth of each synset:
>>> wn.synset('baleen_whale.n.01').min_depth()
14
>>> wn.synset('whale.n.02').min_depth()
13
>>> wn.synset('vertebrate.n.01').min_depth()
8
>>> wn.synset('entity.n.01').min_depth()
0
 
 
#Similarity measures have been defined over the collection of WordNet synsets 
#which incorporate the above insight. 

#For example, path_similarity assigns a score in the range 0–1 based 
#on the shortest path that connects the concepts in the hypernym hierarchy (-1 is returned in those cases where a path cannot be found). 

#Comparing a synset with itself will return 1. 

#Consider the following similarity scores, relating right whale to minke whale, orca, tortoise, and novel. 

#Although the numbers won't mean much, they decrease as we move away from the semantic space of sea creatures to inanimate objects.
>>> right.path_similarity(minke)
0.25
>>> right.path_similarity(orca)
0.16666666666666666
>>> right.path_similarity(tortoise)
0.07692307692307693
>>> right.path_similarity(novel)
0.043478260869565216
 
##Verbnet
#NLTK also includes VerbNet, a hierarhical verb lexicon linked to WordNet. 
#It can be accessed with nltk.corpus.verbnet.


 
 
 
 
 
 
 
 
 
 
 

###nltk - Processing Raw Text
##http://www.nltk.org/book/ chapter 3 


##nltk-reference 
class nltk.tokenize.api.TokenizerI
        Bases: object
        Base class of all tokenize classes 
    span_tokenize(s)
    span_tokenize_sents(strings)
    tokenize(s)
        Return a tokenized copy of s.
        Return type:	list of str
    tokenize_sents(strings)
        Apply self.tokenize() to each element of strings. I.e.:
            return [self.tokenize(s) for s in strings]
        Return type:	list(list(str))  

nltk.tokenize.sent_tokenize(text, language='english')
    Return a sentence-tokenized copy of text, 
    Uses PunktSentenceTokenizer for the specified language
    Parameters:	
        text – text to split into sentences
        language – the model name in the Punkt corpus

nltk.tokenize.word_tokenize(text, language='english', preserve_line=False)
    Return a tokenized copy of text, 
    uses TreebankWordTokenizer with PunktSentenceTokenizer
    Parameters:	
        text – text to split into words
        language (str) – the model name in the Punkt corpus
        preserve_line – An option to keep the preserve the sentence and not sentence tokenize it.

nltk.tokenize.regexp.regexp_tokenize(text, pattern, gaps=False, 
                        discard_empty=True, flags=56)
    Return a tokenized copy of text
    Uses RegexpTokenizer 

nltk.tokenize.regexp.blankline_tokenize(text)
    Return a tokenized copy of text
    Uses BlanklineTokenizer()
    
nltk.tokenize.regexp.wordpunct_tokenize(text) 
    Return a tokenized copy of text
    Uses WordPunctTokenizer()

  
#nltk.tokenize.treebank module
class nltk.tokenize.treebank.TreebankWordDetokenizer
    Bases: nltk.tokenize.api.TokenizerI
    The Treebank detokenizer uses the reverse regex operations corresponding 
    to the Treebank tokenizer’s regexes.
    
    
class nltk.tokenize.treebank.TreebankWordTokenizer
    Bases: nltk.tokenize.api.TokenizerI
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank. 
    It assumes that the text has already been segmented into sentences, e.g. using sent_tokenize().

    #Example 
        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
        >>> TreebankWordTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
        >>> s = "They'll save and invest more."
        
        >> from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
        >>> d = TreebankWordDetokenizer()
        >>> t = TreebankWordTokenizer()
        >>> toks = t.tokenize(s)
        >>> d.detokenize(toks)
        'Good muffins cost $3.88 in New York. Please buy me two of them. Thanks.'

        #The MXPOST parentheses substitution can be undone using the convert_parentheses parameter:
        >>> s = '''Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'''
        >>> expected_tokens = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
            'New', '-LRB-', 'York', '-RRB-', '.', 'Please', '-LRB-', 'buy',
            '-RRB-', 'me', 'two', 'of', 'them.', '-LRB-', 'Thanks', '-RRB-', '.']
        >>> expected_tokens == t.tokenize(s, convert_parentheses=True)
        True
        >>> expected_detoken = 'Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks).'
        >>> expected_detoken == d.detokenize(t.tokenize(s, convert_parentheses=True), convert_parentheses=True)
        True




        

class nltk.tokenize.regexp.RegexpTokenizer(pattern, gaps=False, discard_empty=True, 
                        flags=56)
    Bases: nltk.tokenize.api.TokenizerI
    A tokenizer that splits a string using a regular expression, 
    which matches either the tokens or the separators between tokens.
    >>> tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    Parameters:	
        pattern (str) – The pattern used to build this tokenizer.
                       (This pattern must not contain capturing parentheses; 
                       Use non-capturing parentheses, e.g. (?:...), instead)
        gaps (bool) – True if this tokenizer’s pattern should be used to find separators between tokens; 
                      False if this tokenizer’s pattern should be used to find the tokens themselves.
        discard_empty (bool) – True if any empty tokens '' generated by the tokenizer should be discarded. 
                               Empty tokens can only be generated if _gaps == True.
        flags (int) – The regexp flags used to compile this tokenizer’s pattern. 
                     By default, the following flags are used: re.UNICODE | re.MULTILINE | re.DOTALL.

    span_tokenize(text)
    tokenize(text)
    unicode_repr()

class nltk.tokenize.regexp.WhitespaceTokenizer
    Bases: nltk.tokenize.regexp.RegexpTokenizer
    Tokenize a string on whitespace (space, tab, newline). 
    In general, users should use the string split() method instead.
    >>> from nltk.tokenize import WhitespaceTokenizer
    >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    >>> WhitespaceTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.',
    'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']

class nltk.tokenize.regexp.WordPunctTokenizer
    Bases: nltk.tokenize.regexp.RegexpTokenizer
    Tokenize a text into a sequence of alphabetic and non-alphabetic characters, 
    using the regexp \w+|[^\w\s]+.
    >>> from nltk.tokenize import WordPunctTokenizer
    >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    >>> WordPunctTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York',
    '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

class nltk.tokenize.regexp.BlanklineTokenizer
    Bases: nltk.tokenize.regexp.RegexpTokenizer
    Tokenize a string, treating any sequence of blank lines as a delimiter. 
    Blank lines are defined as lines containing no characters, 
    except for space or tab characters.



nltk.tokenize.casual.casual_tokenize(text, preserve_case=True, 
                        reduce_len=False, strip_handles=False)
    Tokenizer for tweets, 
    based on class nltk.tokenize.casual.TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)

    >>> from nltk.tokenize import TweetTokenizer
    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    #Examples using strip_handles and reduce_len parameters:

    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']


nltk.tokenize.casual.reduce_lengthening(text)
    Replace repeated character sequences of length 3 
    or greater with sequences of length 3.

nltk.tokenize.casual.remove_handles(text)
    Remove Twitter username handles from text.
     
     

   
class nltk.tokenize.moses.MosesDetokenizer(lang='en')
    Bases: nltk.tokenize.api.TokenizerI
    Moses is a statistical machine translation system that allows 
    to automatically train translation models for any language pair. 
    Using a collection of translated texts (parallel corpus). 
    Once you have a trained model, an efficient search algorithm quickly finds 
    the highest probability translation among the exponential number of choices. 
    This is a Python port of the Moses Detokenizer 
    from https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl

    
class nltk.tokenize.moses.MosesTokenizer(lang='en')
    Bases: nltk.tokenize.api.TokenizerI
    This is a Python port of the Moses Tokenizer 
    from https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
        
    >>> from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
    >>> t, d = MosesTokenizer(), MosesDetokenizer()
    >>> sent = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?"
    >>> expected_tokens = [u'This', u'ain', u'&apos;t', u'funny', u'.', u'It', u'&apos;s', u'actually', u'hillarious', u',', u'yet', u'double', u'Ls', u'.', u'&#124;', u'&#91;', u'&#93;', u'&lt;', u'&gt;', u'&#91;', u'&#93;', u'&amp;', u'You', u'&apos;re', u'gonna', u'shake', u'it', u'off', u'?', u'Don', u'&apos;t', u'?']
    >>> expected_detokens = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [] & You're gonna shake it off? Don't?"
    >>> tokens = t.tokenize(sent)
    >>> tokens == expected_tokens
    True
    >>> detokens = d.detokenize(tokens)
    >>> " ".join(detokens) == expected_detokens
    True    
     
 


class nltk.tokenize.mwe.MWETokenizer(mwes=None, separator='_')
    Bases: nltk.tokenize.api.TokenizerI
    A tokenizer that processes tokenized text 
    and merges multi-word expressions into single tokens.
    
    >>> from nltk.tokenize import MWETokenizer
    >>> tokenizer = MWETokenizer([('a', 'little'), ('a', 'little', 'bit'), ('a', 'lot')])
    >>> tokenizer.add_mwe(('in', 'spite', 'of'))

    >>> tokenizer.tokenize('Testing testing testing one two three'.split())
    ['Testing', 'testing', 'testing', 'one', 'two', 'three']
    >>> tokenizer.tokenize('This is a test in spite'.split())
    ['This', 'is', 'a', 'test', 'in', 'spite']
    >>> tokenizer.tokenize('In a little or a little bit or a lot in spite of'.split())
    ['In', 'a_little', 'or', 'a_little_bit', 'or', 'a_lot', 'in_spite_of']

 
 
 


Punkt Sentence Tokenizer
This tokenizer divides a text into a list of sentences, 
by using an unsupervised algorithm to build a model for abbreviation words, 
collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

The NLTK data package includes a pre-trained Punkt tokenizer for English.
>>> import nltk.data
>>> text = '''
... Punkt knows that the periods in Mr. Smith and Johann S. Bach
... do not mark sentence boundaries.  And sometimes sentences
... can start with non-capitalized words.  i is a good variable
... name.
... '''
>>> sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
>>> print('\n-----\n'.join(sent_detector.tokenize(text.strip())))
Punkt knows that the periods in Mr. Smith and Johann S. Bach
do not mark sentence boundaries.
-----
And sometimes sentences
can start with non-capitalized words.
-----
i is a good variable
name.

#Punctuation following sentences is also included by default (from NLTK 3.0 onwards). It can be excluded with the realign_boundaries flag.

>>> text = '''
... (How does it deal with this parenthesis?)  "It should be part of the
... previous sentence." "(And the same with this one.)" ('And this one!')
... "('(And (this)) '?)" [(and this. )]
... '''
>>> print('\n-----\n'.join(
...     sent_detector.tokenize(text.strip())))
(How does it deal with this parenthesis?)
-----
"It should be part of the
previous sentence."
-----
"(And the same with this one.)"
-----
('And this one!')
-----
"('(And (this)) '?)"
-----
[(and this. )]
>>> print('\n-----\n'.join(
...     sent_detector.tokenize(text.strip(), realign_boundaries=False)))
(How does it deal with this parenthesis?
-----
)  "It should be part of the
previous sentence.
-----
" "(And the same with this one.
-----
)" ('And this one!
-----
')
"('(And (this)) '?
-----
)" [(and this.
-----
)]





class nltk.tokenize.sexpr.SExprTokenizer(parens='()', strict=True)
    Bases: nltk.tokenize.api.TokenizerI
    A tokenizer that divides strings into s-expressions. An s-expresion can be either:
            a parenthesized expression, including any nested parenthesized expressions, or
            a sequence of non-whitespace non-parenthesis characters.
    For example, the string (a (b c)) d e (f) consists of four s-expressions: (a (b c)), d, e, and (f).
    >>> from nltk.tokenize import SExprTokenizer
    >>> SExprTokenizer().tokenize('(a b (c d)) e f (g)')
    ['(a b (c d))', 'e', 'f', '(g)']

 

class nltk.tokenize.stanford.StanfordTokenizer(path_to_jar=None, encoding='utf8', options=None, verbose=False, java_options='-mx1000m')
    Bases: nltk.tokenize.api.TokenizerI
    Interface to the Stanford Tokenizer

    >>> from nltk.tokenize.stanford import StanfordTokenizer
    >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
    >>> StanfordTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
    >>> s = "The colour of the wall is blue."
    >>> StanfordTokenizer(options={"americanize": True}).tokenize(s)
    ['The', 'color', 'of', 'the', 'wall', 'is', 'blue', '.']




class nltk.tokenize.texttiling.TextTilingTokenizer(w=20, k=10, similarity_method=0, stopwords=None, smoothing_method=[0], smoothing_width=2, smoothing_rounds=1, cutoff_policy=1, demo_mode=False)
    Bases: nltk.tokenize.api.TokenizerI
    Tokenize a document into topical sections using the TextTiling algorithm. 
    This algorithm detects subtopic shifts based on the analysis of lexical 
    co-occurrence patterns.
    The process starts by tokenizing the text into pseudosentences of a fixed size w. 
    Then, depending on the method used, similarity scores are assigned at sentence gaps.
    The algorithm proceeds by detecting the peak differences between these scores 
    and marking them as boundaries. 
    The boundaries are normalized to the closest paragraph break 
    and the segmented text is returned.   
        w (int) – Pseudosentence size
        k (int) – Size (in sentences) of the block used in the block comparison method
        similarity_method (constant) – The method used for determining similarity scores: BLOCK_COMPARISON (default) or VOCABULARY_INTRODUCTION.
        stopwords (list(str)) – A list of stopwords that are filtered out (defaults to NLTK’s stopwords corpus)
        smoothing_method (constant) – The method used for smoothing the score plot: DEFAULT_SMOOTHING (default)
        smoothing_width (int) – The width of the window used by the smoothing method
        smoothing_rounds (int) – The number of smoothing passes
        cutoff_policy (constant) – The policy used to determine the number of boundaries: HC (default) or LC
    >>> from nltk.corpus import brown
    >>> tt = TextTilingTokenizer(demo_mode=True)
    >>> text = brown.raw()[:10000]
    >>> s, ss, d, b = tt.tokenize(text)
    >>> b
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]



class nltk.tokenize.toktok.ToktokTokenizer
    Bases: nltk.tokenize.api.TokenizerI
    This is a Python port of the tok-tok.pl from https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl
    The tok-tok tokenizer is a simple, general tokenizer, 
    where the input has one sentence per line; thus only final period is tokenized.

    >>> toktok = ToktokTokenizer()
    >>> text = u'¡This, is a sentence with weird» symbols… appearing everywhere¿'
    >>> expected = u'¡ This , is a sentence with weird » symbols … appearing everywhere ¿'
    >>> assert toktok.tokenize(text, return_str=True) == expected
    >>> toktok.tokenize(text) == [u'¡', u'This', u',', u'is', u'a', u'sentence', u'with', u'weird', u'»', u'symbols', u'…', u'appearing', u'everywhere', u'¿']
    True
   
 
 
nltk.tokenize.util.regexp_span_tokenize(s, regexp)
    Return the offsets of the tokens in s, as a sequence of (start, end) tuples, 
    by splitting the string at each successive match of regexp.
    Return type:    iter(tuple(int, int))
    >>> from nltk.tokenize.util import regexp_span_tokenize
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
    ... two of them.\n\nThanks.'''
    >>> list(regexp_span_tokenize(s, r'\s'))
    [(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36),
    (38, 44), (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]



nltk.tokenize.util.spans_to_relative(spans)
    Return a sequence of relative spans, given a sequence of spans.
        Parameters:	spans (iter(tuple(int, int))) – a sequence of (start, end) offsets of the tokens
        Return type:	iter(tuple(int, int))
    >>> from nltk.tokenize import WhitespaceTokenizer
    >>> from nltk.tokenize.util import spans_to_relative
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
    ... two of them.\n\nThanks.'''
    >>> list(spans_to_relative(WhitespaceTokenizer().span_tokenize(s)))
    [(0, 4), (1, 7), (1, 4), (1, 5), (1, 2), (1, 3), (1, 5), (2, 6),
    (1, 3), (1, 2), (1, 3), (1, 2), (1, 5), (2, 7)]


nltk.tokenize.util.string_span_tokenize(s, sep)
    Return the offsets of the tokens in s, as a sequence of (start, end) tuples, 
    by splitting the string at each occurrence of sep.

    >>> from nltk.tokenize.util import string_span_tokenize
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
    ... two of them.\n\nThanks.'''
    >>> list(string_span_tokenize(s, " "))
    [(0, 4), (5, 12), (13, 17), (18, 26), (27, 30), (31, 36), (37, 37),
    (38, 44), (45, 48), (49, 55), (56, 58), (59, 73)]

 
nltk.tokenize.util.xml_escape(text)
    This function transforms the input text into an “escaped” version suitable 
    for well-formed XML formatting.
    >>> input_str = ''')| & < > ' " ] ['''
    >>> expected_output =  ''')| &amp; &lt; &gt; ' " ] ['''
    >>> escape(input_str) == expected_output
    True
    >>> xml_escape(input_str)
    ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'


nltk.tokenize.util.xml_unescape(text)
    This function transforms the “escaped” version suitable 
    for well-formed XML formatting into humanly-readable string.
    >>> from xml.sax.saxutils import unescape
    >>> s = ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'
    >>> expected = ''')| & < > ' " ] ['''
    >>> xml_unescape(s) == expected
    True

  

 
##Example - Electronic Books

>>> from __future__ import division  # Python 2 users only
>>> import nltk, re, pprint
>>> from nltk import word_tokenize


#http://www.gutenberg.org/catalog/, 
>>> from urllib import request

#to handle proxy 
>>> proxies = {'http': 'http://www.someproxy.com:3128'} #'http://user:pass@server:port'
>>> request.ProxyHandler(proxies)

>>> url = "http://www.gutenberg.org/files/2554/2554.txt"
>>> response = request.urlopen(url)
>>> raw = response.read().decode('utf8')
>>> type(raw)
<class 'str'>
>>> len(raw)
1176893
>>> raw[:75]
'The Project Gutenberg EBook of Crime and Punishment, by Fyodor Dostoevsky\r\n'

#to break up the string into words and punctuation
>>> tokens = word_tokenize(raw)
>>> type(tokens)
<class 'list'>
>>> len(tokens)
254354
>>> tokens[:10]
['The', 'Project', 'Gutenberg', 'EBook', 'of', 'Crime', 'and', 'Punishment', ',', 'by']

#Create a Text 
>>> text = nltk.Text(tokens)
>>> type(text)
<class 'nltk.text.Text'>
>>> text[1024:1062]
['CHAPTER', 'I', 'On', 'an', 'exceptionally', 'hot', 'evening', 'early', 'in',
 'July', 'a', 'young', 'man', 'came', 'out', 'of', 'the', 'garret', 'in',
 'which', 'he', 'lodged', 'in', 'S.', 'Place', 'and', 'walked', 'slowly',
 ',', 'as', 'though', 'in', 'hesitation', ',', 'towards', 'K.', 'bridge', '.']
 
>>> text.collocations()  #sequence of words that occur together unusually often
Katerina Ivanovna; Pyotr Petrovitch; Pulcheria Alexandrovna; Avdotya
Romanovna; Rodion Romanovitch; Marfa Petrovna; Sofya Semyonovna; old


#Notice that Project Gutenberg appears as a collocation
#To remove 
>>> raw.find("PART I")
5338
>>> raw.rfind("End of Project Gutenberg's Crime")
1157743
>>> raw = raw[5338:1157743] 
>>> raw.find("PART I")
0

##Dealing with HTML

>>> url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
>>> html = request.urlopen(url).read().decode('utf8')
>>> html[:60]
'<!doctype html public "-//W3C//DTD HTML 4.0 Transitional//EN'

>>> from bs4 import BeautifulSoup
>>> raw = BeautifulSoup(html, 'html.parser').get_text()
>>> tokens = word_tokenize(raw)
>>> tokens
['BBC', 'NEWS', '|', 'Health', '|', 'Blondes', "'to", 'die', 'out', ...]

#This still contains unwanted material concerning site navigation and related stories

>>> tokens = tokens[110:390]
>>> text = nltk.Text(tokens)
>>> text.concordance('gene')  # every occurrence of a given word
Displaying 5 of 5 matches:
hey say too few people now carry the gene for blondes to last beyond the next
blonde hair is caused by a recessive gene . In order for a child to have blond
have blonde hair , it must have the gene on both sides of the family in the g
ere is a disadvantage of having that gene or by chance . They do n't disappear
des would disappear is if having the gene was a disadvantage and I do not thin


##Processing RSS Feeds

>>> import feedparser
>>> llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
>>> llog['feed']['title']
'Language Log'
>>> len(llog.entries)
15
>>> post = llog.entries[2]
>>> post.title
"He's My BF"
>>> content = post.content[0].value
>>> content[:70]
'<p>Today I was chatting with three of our visiting graduate students f'
>>> raw = BeautifulSoup(content, 'html.parser').get_text()
>>> word_tokenize(raw)
['Today', 'I', 'was', 'chatting', 'with', 'three', 'of', 'our', 'visiting',
'graduate', 'students', 'from', 'the', 'PRC', '.', 'Thinking', 'that', 'I',
'was', 'being', 'au', 'courant', ',', 'I', 'mentioned', 'the', 'expression',
'DUI4XIANG4', '\u5c0d\u8c61', '("', 'boy', '/', 'girl', 'friend', '"', ...]


##Reading Local Files
#to use nltk.data.find() to get the filename for any corpus item.
>>> path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
>>> raw = open(path, 'rU').read()




##Extracting encoded text from files
>>> path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')

>>> f = open(path, encoding='latin2')
>>> for line in f:
        line = line.strip()
        print(line)
Pruska Biblioteka Panstwowa. Jej dawne zbiory znane pod nazwa
"Berlinka" to skarb kultury i sztuki niemieckiej. Przewiezione przez

##if we want to see the underlying numerical values (or "codepoints") 

>>> f = open(path, encoding='latin2')
>>> for line in f:
        line = line.strip()
        print(line.encode('unicode_escape'))
b'Pruska Biblioteka Pa\\u0144stwowa. Jej dawne zbiory znane pod nazw\\u0105'

#NLTK tokenizers allow Unicode strings as input, and correspondingly yield Unicode strings as output.
>>> word_tokenize(line)
['niemców', 'pod', 'koniec', 'ii', 'wojny', 'swiatowej', 'na', 'dolny', 'slask', ',', 'zostaly'
 
 
##Using your local encoding in Python file (.py)
#define a source file code encoding
#include the string '# -*- coding: <coding> -*-' 
#as the first or second line of your file. 
#Note that <coding> has to be a string like 'latin-1', 'big5' or 'utf-8' 


##Pasring raw file based on RE

>>> import re
>>> wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
 
 
#Example - find words ending with ed using the regular expression 


>>> [w for w in wordlist if re.search('ed$', w)]
['abaissed', 'abandoned', 'abased', 'abashed', 'abatised', 'abed', 'aborted', ...]
 
 

#The . wildcard symbol matches any single character
>>> [w for w in wordlist if re.search('^..j..t..$', w)]
['abjectly', 'adjuster', 'dejected', 'dejectly', 'injector', 'majestic', ...]


>>> chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
>>> [w for w in chat_words if re.search('^m+i+n+e+$', w)]
['miiiiiiiiiiiiinnnnnnnnnnneeeeeeeeee', 'miiiiiinnnnnnnnnneeeeeeee', 'mine',
'mmmmmmmmiiiiiiiiinnnnnnnnneeeeeeee']
>>> [w for w in chat_words if re.search('^[ha]+$', w)]
['a', 'aaaaaaaaaaaaaaaaa', 'aaahhhh', 'ah', 'ahah', 'ahahah', 'ahh',
'ahhahahaha', 'ahhh', 'ahhhh', 'ahhhhhh', 'ahhhhhhhhhhhhhh', 'h', 'ha', 'haaa',
'hah', 'haha', 'hahaaa', 'hahah', 'hahaha', 'hahahaa', 'hahahah', 'hahahaha', ...]


>>> wsj = sorted(set(nltk.corpus.treebank.words()))
>>> [w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]
['0.0085', '0.05', '0.1', '0.16', '0.2', '0.25', '0.28', '0.3', '0.4', '0.5',
'0.50', '0.54', '0.56', '0.60', '0.7', '0.82', '0.84', '0.9', '0.95', '0.99',
'1.01', '1.1', '1.125', '1.14', '1.1650', '1.17', '1.18', '1.19', '1.2', ...]
>>> [w for w in wsj if re.search('^[A-Z]+\$$', w)]
['C$', 'US$']
>>> [w for w in wsj if re.search('^[0-9]{4}$', w)]
['1614', '1637', '1787', '1901', '1903', '1917', '1925', '1929', '1933', ...]
>>> [w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]
['10-day', '10-lap', '10-year', '100-share', '12-point', '12-year', ...]
>>> [w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]
['black-and-white', 'bread-and-butter', 'father-in-law', 'machine-gun-toting',
'savings-and-loan']
>>> [w for w in wsj if re.search('(ed|ing)$', w)]
['62%-owned', 'Absorbed', 'According', 'Adopting', 'Advanced', 'Advancing', ...]
 
 
##Extracting Word Pieces

>>> word = 'supercalifragilisticexpialidocious'
>>> re.findall(r'[aeiou]', word)
['u', 'e', 'a', 'i', 'a', 'i', 'i', 'i', 'e', 'i', 'a', 'i', 'o', 'i', 'o', 'u']
>>> len(re.findall(r'[aeiou]', word))
16
 
 
#Example 
>>> wsj = sorted(set(nltk.corpus.treebank.words()))
>>> fd = nltk.FreqDist(vs for word in wsj
                for vs in re.findall(r'[aeiou]{2,}', word)) #given iterator/generator, find how many times it has occured 
>>> fd.most_common(12)
[('io', 549), ('ea', 476), ('ie', 331), ('ou', 329), ('ai', 261), ('ia', 253),
('ee', 217), ('oo', 174), ('ua', 109), ('au', 106), ('ue', 105), ('ui', 95)]
 
 
#Example -  English text is highly redundant, 
#and it is still easy to read when word-internal vowels are left out

>>> regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
>>> def compress(word):
        pieces = re.findall(regexp, word)
        return ''.join(pieces)
    
>>> english_udhr = nltk.corpus.udhr.words('English-Latin1')
>>> print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))
Unvrsl Dclrtn of Hmn Rghts Prmble Whrs rcgntn of the inhrnt dgnty and
of the eql and inlnble rghts of all mmbrs of the hmn fmly is the fndtn
of frdm , jstce and pce in the wrld , Whrs dsrgrd and cntmpt fr hmn
rghts hve rsltd in brbrs acts whch hve outrgd the cnscnce of mnknd ,
and the advnt of a wrld in whch hmn bngs shll enjy frdm of spch and


##Example - regular expressions with conditional frequency distributions. 
#Here we will extract all consonant-vowel sequences from the words of Rotokas, such as ka and si. 

#Since each of these is a pair, it can be used to initialize a conditional frequency distribution. 
#We then tabulate the frequency of each pair:



>>> rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
>>> cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
>>> cfd = nltk.ConditionalFreqDist(cvs)  #
>>> cfd.tabulate()
    a    e    i    o    u
k  418  148   94  420  173
p   83   31  105   34   51
r  187   63   84   89   79
s    0    0  100    2    1
t   47    8    0  148   37
v   93   27  105   48   49
 
#Examining the rows for s and t, we see they are in partial "complementary distribution", 
#which is evidence that they are not distinct phonemes in the language. 

#Thus, we could conceivably drop s from the Rotokas alphabet 
#and simply have a pronunciation rule that the letter t is pronounced s when followed by i

#Exampple - to inspect the words behind the numbers in the above table, 
#it would be helpful to have an index, allowing us 
#to quickly find the list of words that contains a given consonant-vowel pair, 
#e.g. cv_index['su'] should give us all words containing su.



>>> cv_word_pairs = [(cv, w) for w in rotokas_words
                         for cv in re.findall(r'[ptksvr][aeiou]', w)]
#cv_word_pairs list will contain ('ka', 'kasuari'), ('su', 'kasuari') and ('ri', 'kasuari'). 

>>> cv_index = nltk.Index(cv_word_pairs)
>>> cv_index['su']
['kasuari']
>>> cv_index['po']
['kaapo', 'kaapopato', 'kaipori', 'kaiporipie', 'kaiporivira', 'kapo', 'kapoa',
'kapokao', 'kapokapo', 'kapokapo', 'kapokapoa', 'kapokapoa', 'kapokapora', ...]
 
 



##Searching Tokenized Text

#use a special kind of regular expression for searching across multiple words in a text 
#(where a text is a list of tokens). 

#For example, "<a> <man>" finds all instances of a man in the text. 

>>> from nltk.corpus import gutenberg, nps_chat
>>> moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
>>> moby.findall(r"<a> (<.*>) <man>") 
monied; nervous; dangerous; white; white; white; pious; queer; good;
mature; white; Cape; great; wise; wise; butterless; white; fiendish;
pale; furious; better; certain; complete; dismasted; younger; brave;
brave; brave; brave
>>> chat = nltk.Text(nps_chat.words())
>>> chat.findall(r"<.*> <.*> <bro>") 
you rule bro; telling you bro; u twizted bro
>>> chat.findall(r"<l.*>{3,}") 
lol lol lol; lmao lol lol; lol lol lol; la la la la la; la la la; la
la la; lovely lol lol love; lol lol lol.; la la la; la la la

#Example - searching a large text corpus for expressions of the form' x and ys' allows us to discover hypernyms 
>>> from nltk.corpus import brown
>>> hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
>>> hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")
speed and other activities; water and other liquids; tomb and other
landmarks; Statues and other monuments; pearls and other jewels;
charts and other items; roads and other features; figures and other
objects; military and other areas; demands and other factors;
abstracts and other compilations; iron and other metals





###nltk reference - Util module 
class nltk.util.Index(pairs)
    Bases: collections.defaultdict

nltk.util.bigrams(sequence, **kwargs)
    Return the bigrams generated from a sequence of items, 
    as an iterator
    >>> from nltk.util import bigrams
    >>> list(bigrams([1,2,3,4,5]))
    [(1, 2), (2, 3), (3, 4), (4, 5)]


nltk.util.binary_search_file(file, key, cache={}, cacheDepth=-1)
    Return the line from the file with first word key. 
    Searches through a sorted file using the binary search algorithm.


nltk.util.breadth_first(tree, children=<built-in function iter>, maxdepth=-1)
    Traverse the nodes of a tree in breadth-first order. 
    The first argument should be the tree root; 
    children should be a function taking as argument a tree node 
    and returning an iterator of the node’s children.


nltk.util.choose(n, k)
    calculate binomial coefficients
 
nltk.util.clean_html(html)
nltk.util.clean_url(url)
    Both are not implemented in newer versions, Use BeautifulSoup.get_text()


nltk.util.elementtree_indent(elem, level=0)
    Recursive function to indent an ElementTree.

 
nltk.util.everygrams(sequence, min_len=1, max_len=-1, **kwargs)
    Returns all possible ngrams generated from a sequence of items, 
    as an iterator.

    >>> sent = 'a b c'.split()
    >>> list(everygrams(sent))
    [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]
    >>> list(everygrams(sent, max_len=2))
    [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c')]





nltk.util.filestring(f)


nltk.util.flatten(*args)
    Flatten a list.

    >>> from nltk.util import flatten
    >>> flatten(1, 2, ['b', 'a' , ['c', 'd']], 3)
    [1, 2, 'b', 'a', 'c', 'd', 3]




nltk.util.guess_encoding(data)
    Given a byte string, attempt to decode it. 
    Tries the standard ‘UTF8’ and ‘latin-1’ encodings, 
    Plus several gathered from locale information.
    The calling program must first call:
    locale.setlocale(locale.LC_ALL, '')


nltk.util.in_idle()
    Return True if this function is run within idle. 
    Tkinter programs that are run in idle should never call Tk.mainloop; 
    so this function should be used to gate all calls to Tk.mainloop.


nltk.util.invert_dict(d)
    Make value of originals  as key and


nltk.util.invert_graph(graph)
    Inverts a directed graph.
    graph (dict(set)) – the graph, represented as a dictionary of sets 



nltk.util.ngrams(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None)
    Return the ngrams generated from a sequence of items, as an iterator. 
    >>> from nltk.util import ngrams
    >>> list(ngrams([1,2,3,4,5], 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
    >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
    [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]




nltk.util.pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None)
    Returns a padded sequence of items before ngram extraction.
    >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    ['<s>', 1, 2, 3, 4, 5, '</s>']
    >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
    ['<s>', 1, 2, 3, 4, 5]
    >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
    [1, 2, 3, 4, 5, '</s>']




nltk.util.pr(data, start=0, end=None)
    Pretty print a sequence of data items

 
nltk.util.print_string(s, width=70)
    Pretty print a string, breaking lines on whitespace


nltk.util.py25()
nltk.util.py26()
nltk.util.py27()
    Returns True if running in corresponding version 

nltk.util.re_show(regexp, string, left='{', right='}')
    Return a string with markers surrounding the matched substrings. 
    Search str for substrings matching regexp and wrap the matches with braces. 
    This is convenient for learning about regular expressions.



nltk.util.set_proxy(proxy, user=None, password='')
    Set the HTTP proxy for Python to download through.
    If proxy is None then tries to set proxy from environment or system settings.


nltk.util.skipgrams(sequence, n, k, **kwargs)
    Returns all possible skipgrams generated from a sequence of items, as an iterator. 
    Skipgrams are ngrams that allows tokens to be skipped. 
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf
    >>> sent = "Insurgents killed in ongoing fighting".split()
    >>> list(skipgrams(sent, 2, 2))
    [('Insurgents', 'killed'), ('Insurgents', 'in'), ('Insurgents', 'ongoing'), ('killed', 'in'), ('killed', 'ongoing'), ('killed', 'fighting'), ('in', 'ongoing'), ('in', 'fighting'), ('ongoing', 'fighting')]
    >>> list(skipgrams(sent, 3, 2))
    [('Insurgents', 'killed', 'in'), ('Insurgents', 'killed', 'ongoing'), ('Insurgents', 'killed', 'fighting'), ('Insurgents', 'in', 'ongoing'), ('Insurgents', 'in', 'fighting'), ('Insurgents', 'ongoing', 'fighting'), ('killed', 'in', 'ongoing'), ('killed', 'in', 'fighting'), ('killed', 'ongoing', 'fighting'), ('in', 'ongoing', 'fighting')]



 
nltk.util.tokenwrap(tokens, separator=' ', width=70)
    Pretty print a list of text tokens, breaking lines on whitespace

 
nltk.util.transitive_closure(graph, reflexive=False)
    Calculate the transitive closure of a directed graph, optionally the reflexive transitive closure.
    graph (dict(set)) – the initial graph, represented as a dictionary of sets
    reflexive (bool) – if set, also make the closure reflexive
    Return type:dict(set)
 
 
nltk.util.trigrams(sequence, **kwargs)
    Return the trigrams generated from a sequence of items, as an iterator. For example:
    >>> from nltk.util import trigrams
    >>> list(trigrams([1,2,3,4,5]))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]



##Finding Word Stems

# A query for laptops finds documents containing laptop and vice versa. 
#Indeed, laptop and laptops are just two forms of the same dictionary word (or lemma). 

#For some language processing tasks we want to ignore word endings, and just deal with word stems.



##NLTK Stemmers 
#Stemmers remove morphological affixes from words, 
#leaving only the word stem.

##Data
 
>>> raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""
>>> tokens = word_tokenize(raw)
 
#The Porter and Lancaster stemmers follow their own rules for stripping affixes. 
#Observe that the Porter stemmer correctly handles the word lying (mapping it to lie), 
#while the Lancaster stemmer does not.

>>> porter = nltk.PorterStemmer()
>>> lancaster = nltk.LancasterStemmer()
>>> [porter.stem(t) for t in tokens]
['DENNI', ':', 'Listen', ',', 'strang', 'women', 'lie', 'in', 'pond',
'distribut', 'sword', 'is', 'no', 'basi', 'for', 'a', 'system', 'of', 'govern',
'.', 'Suprem', 'execut', 'power', 'deriv', 'from', 'a', 'mandat', 'from',
'the', 'mass', ',', 'not', 'from', 'some', 'farcic', 'aquat', 'ceremoni', '.']
>>> [lancaster.stem(t) for t in tokens]
['den', ':', 'list', ',', 'strange', 'wom', 'lying', 'in', 'pond', 'distribut',
'sword', 'is', 'no', 'bas', 'for', 'a', 'system', 'of', 'govern', '.', 'suprem',
'execut', 'pow', 'der', 'from', 'a', 'mand', 'from', 'the', 'mass', ',', 'not',
'from', 'som', 'farc', 'aqu', 'ceremony', '.']
 
 
#Porter Stemmer is a good choice if you are indexing some texts 
#and want to support search using alternative forms of words 


class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)  #Index is nothing but {key:[values..]} ie dict with values are list 
                                 for (i, word) in enumerate(text)) #same word, list of is

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()
 
 

>>> porter = nltk.PorterStemmer()
>>> grail = nltk.corpus.webtext.words('grail.txt')
>>> text = IndexedText(porter, grail)
>>> text.concordance('lie')
r king ! DENNIS : Listen , strange women lying in ponds distributing swords is no
 beat a very brave retreat . ROBIN : All lies ! MINSTREL : [ singing ] Bravest of
 
 

##Lemmatization
#The WordNet lemmatizer only removes affixes if the resulting word is in its dictionary. 
#This additional checking process makes the lemmatizer slower than the above stemmers. 
#Notice that it doesn't handle lying, but it converts women to woman.

>>> wnl = nltk.WordNetLemmatizer()
>>> [wnl.lemmatize(t) for t in tokens]
['DENNIS', ':', 'Listen', ',', 'strange', 'woman', 'lying', 'in', 'pond',
'distributing', 'sword', 'is', 'no', 'basis', 'for', 'a', 'system', 'of',
'government', '.', 'Supreme', 'executive', 'power', 'derives', 'from', 'a',
'mandate', 'from', 'the', 'mass', ',', 'not', 'from', 'some', 'farcical',
'aquatic', 'ceremony', '.']
 
 
#The WordNet lemmatizer is a good choice if you want to compile the vocabulary of some texts 
#and want a list of valid lemmas (or lexicon headwords).




### Regular Expressions for Tokenizing Text

>>> raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
    though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
    well without--Maybe it's always pepper that makes people hot-tempered,'..."""


>>> print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))
["'", 'When', "I'M", 'a', 'Duchess', ',', "'", 'she', 'said', 'to', 'herself', ',',
'(', 'not', 'in', 'a', 'very', 'hopeful', 'tone', 'though', ')', ',', "'", 'I',
"won't", 'have', 'any', 'pepper', 'in', 'my', 'kitchen', 'AT', 'ALL', '.', 'Soup',
'does', 'very', 'well', 'without', '--', 'Maybe', "it's", 'always', 'pepper',
'that', 'makes', 'people', 'hot-tempered', ',', "'", '...']

#NLTK's Regular Expression Tokenizer
#The function nltk.regexp_tokenize() is similar to re.findall() 
#The special (?x) "verbose flag" tells Python to strip out the embedded whitespace and comments.

>>> text = 'That U.S.A. poster-print costs $12.40...'
>>> pattern = r'''(?x)    # set flag to allow verbose regexps
...     ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
...   | \w+(-\w+)*        # words with optional internal hyphens
...   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
...   | \.\.\.            # ellipsis
...   | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
... '''
>>> nltk.regexp_tokenize(text, pattern)
['That', 'U.S.A.', 'poster-print', 'costs', '$12.40', '...']
 
 

###Segmentation - Sentence Segmentation

>>> len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())
20.250994070456922
 
 
#the text is only available as a stream of characters. 
#Before tokenizing the text into words, we need to segment it into sentences.

#NLTK facilitates this by including the Punkt sentence segmenter (Kiss & Strunk, 2006). 

#Here is an example of its use in segmenting the text of a novel. 


>>> text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
>>> sents = nltk.sent_tokenize(text)
>>> pprint.pprint(sents[79:89])
['"Nonsense!"',
 'said Gregory, who was very rational when anyone else\nattempted paradox.',
 '"Why do all the clerks 
 
 



###Segmentation - Word Segmentation
#For example - for  no visual representation of word boundaries eg chineese 
#or in the processing of spoken language, 

#We can do this by annotating each character with a boolean value to indicate whether or not a word-break appears after the character 

>>> text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
>>> seg1 = "0000000000000001000000000010000000000000000100000000000"
>>> seg2 = "0100100100100001001001000010100100010010000100010010000"



def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words
 
 
>>> text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
>>> seg1 = "0000000000000001000000000010000000000000000100000000000"
>>> seg2 = "0100100100100001001001000010100100010010000100010010000"
>>> segment(text, seg1)
['doyouseethekitty', 'seethedoggy', 'doyoulikethekitty', 'likethedoggy']
>>> segment(text, seg2)
['do', 'you', 'see', 'the', 'kitty', 'see', 'the', 'doggy', 'do', 'you',
'like', 'the', 'kitty', 'like', 'the', 'doggy']
 
 

#Now the segmentation task becomes a search problem: 
#find the bit string that causes the text string to be correctly segmented into words. 

#We assume the learner is acquiring words and storing them in an internal lexicon. 
#Given a suitable lexicon, it is possible to reconstruct the source text as a sequence of lexical items. Following (Brent, 1995), we can define an objective function, a scoring function whose value we will try to optimize, based on the size of the lexicon (number of characters in the words plus an extra delimiter character to mark the end of each word) and the amount of information needed to reconstruct the source text from the lexicon


#Our objective function is to below
def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(word) + 1 for word in set(words))
    return text_size + lexicon_size
 
 

>>> text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
>>> seg1 = "0000000000000001000000000010000000000000000100000000000"
>>> seg2 = "0100100100100001001001000010100100010010000100010010000"
>>> seg3 = "0000100100000011001000000110000100010000001100010000001"
>>> segment(text, seg3)
['doyou', 'see', 'thekitt', 'y', 'see', 'thedogg', 'y', 'doyou', 'like',
 'thekitt', 'y', 'like', 'thedogg', 'y']
>>> evaluate(text, seg3)
47
>>> evaluate(text, seg2)
48
>>> evaluate(text, seg1)
64
 
#The final step is to search for the pattern of zeros and ones that minimizes this objective function, 

#Non-Deterministic Search Using Simulated Annealing: begin searching with phrase segmentations only; 
#randomly perturb the zeros and ones proportional to the "temperature"; 
#with each iteration the temperature is lowered and the perturbation of boundaries is reduced. 

#As this search algorithm is non-deterministic, you may see a slightly different result.
 
 

from random import randint

def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs
 
 

 >>> text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
>>> seg1 = "0000000000000001000000000010000000000000000100000000000"
>>> anneal(text, seg1, 5000, 1.2)
61 ['doyouseetheki', 'tty', 'see', 'thedoggy', 'doyouliketh', 'ekittylike', 'thedoggy']
59 ['doy', 'ouseetheki', 'ttysee', 'thedoggy', 'doy', 'o', 'ulikethekittylike', 'thedoggy']
57 ['doyou', 'seetheki', 'ttysee', 'thedoggy', 'doyou', 'liketh', 'ekittylike', 'thedoggy']
55 ['doyou', 'seethekit', 'tysee', 'thedoggy', 'doyou', 'likethekittylike', 'thedoggy']
54 ['doyou', 'seethekit', 'tysee', 'thedoggy', 'doyou', 'like', 'thekitty', 'like', 'thedoggy']
52 ['doyou', 'seethekittysee', 'thedoggy', 'doyou', 'like', 'thekitty', 'like', 'thedoggy']
43 ['doyou', 'see', 'thekitty', 'see', 'thedoggy', 'doyou', 'like', 'thekitty', 'like', 'thedoggy']
'0000100100000001001000000010000100010000000100010000000'
 
 


###Formatting: From Lists to Strings


def tabulate(cfdist, words, categories):
    print('{:16}'.format('Category'), end=' ')                    # column headings
    for word in words:
        print('{:>6}'.format(word), end=' ')
    print()
    for category in categories:
        print('{:16}'.format(category), end=' ')                  # row heading
        for word in words:                                        # for each word
            print('{:6}'.format(cfdist[category][word]), end=' ') # print table cell
        print()                                                   # end the row

>>> from nltk.corpus import brown
>>> cfd = nltk.ConditionalFreqDist(
            (genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
>>> genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
>>> modals = ['can', 'could', 'may', 'might', 'must', 'will']
>>> tabulate(cfd, modals, genres)
Category            can  could    may  might   must   will
news                 93     86     66     38     50    389
religion             82     59     78     12     54     71
hobbies             268     58    131     22     83    264
science_fiction      16     49      4     12      8     16
romance              74    193     11     51     45     43
humor                16     30      8      8      9     13


#a format string '{:{width}}' and bound a value to the width parameter in format(). 
#This allows us to specify the width of a field using a variable.

>>> '{:{width}}' % ("Monty Python", width=15)
'Monty Python   '
 
 



###NLTK - Structured programming techniques in NLP 

##Example - Trie

#A letter trie is a data structure that can be used for indexing a lexicon, one letter at a time.
#https://en.wikipedia.org/wiki/Trie

#For example, for a letter trie, trie['c'] would be a smaller trie 
#which held all words starting with c. 

#To insert the word chien (French for dog), we split off the c 
#and recursively insert hien into the sub-trie trie['c'].

#The recursion continues until there are no letters remaining in the word, 
#when we store the intended value (in this case, the word dog).

 

#key is string , trie is dict 
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value
 
 

>>> trie = {}
>>> insert(trie, 'chat', 'cat')
>>> insert(trie, 'chien', 'dog')
>>> insert(trie, 'chair', 'flesh')
>>> insert(trie, 'chic', 'stylish')
>>> trie = dict(trie)               # for nicer printing
>>> trie['c']['h']['a']['t']['value']
'cat'
>>> pprint.pprint(trie, width=40)
{'c': {'h': {'a': {'t': {'value': 'cat'}},
                  {'i': {'r': {'value': 'flesh'}}},
             'i': {'e': {'n': {'value': 'dog'}}}
                  {'c': {'value': 'stylish'}}}}}
 
 


 
##Space-Time Tradeoffs- Create Index 
#speed up the execution of a program by building an auxiliary data structure, such as an index. 

#By indexing the document collection it provides much faster lookup.

def raw(file):
    contents = open(file).read() #type str 
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub('\s+', ' ', contents)
    return contents

def snippet(doc, term):
    text = ' '*30 + raw(doc) + ' '*30
    pos = text.index(term) #S.index(sub[, start[, end]]) -> int, Find sub in S and return index 
    return text[pos-30:pos+30]

print("Building Index...")
files = nltk.corpus.movie_reviews.abspaths()
#file line vs list of filenames 
idx = nltk.Index((w, f) for f in files for w in raw(f).split()) #index is a dict(defaultdict) with value is list 

query = ''
while query != "quit":
    query = input("query> ")     # use raw_input() in Python 2
    if query in idx:
        for doc in idx[query]:  #doc is file name 
            print(snippet(doc, query))
    else:
        print("Not found")
 
 

##Dynamic Programming
# Dynamic programming is used when a problem contains overlapping sub-problems. 
#Instead of computing solutions to these sub-problems repeatedly, 
#we simply store them in a lookup table


from nltk import memoize

@memoize
def fun(n):  #this must be recursion, like fib(n) = fib(n-1)+fib(n-2)
 ....
 

##Matplotlib in nltk 
#Example -  a table of numbers showing the frequency of particular modal verbs in the Brown Corpus, classified by genre. 

from numpy import arange
from matplotlib import pyplot

colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black

def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind+c*width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind+width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()
 
 

>>> genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
>>> modals = ['can', 'could', 'may', 'might', 'must', 'will']
>>> cfdist = nltk.ConditionalFreqDist(
                (genre, word)
                for genre in genres
                for word in nltk.corpus.brown.words(categories=genre)
                if word in modals)
    
>>> counts = {}
>>> for genre in genres:
        counts[genre] = [cfdist[genre][word] for word in modals]
>>> bar_chart(genres, modals, counts)
 
 
#It is also possible to generate such data visualizations on the fly. 
#For example, a web page with form input could permit visitors to specify search parameters, 
#submit the form, and see a dynamically generated visualization. 

#To do this we have to specify the Agg backend for matplotlib, 
#which is a library for producing raster (pixel) images 
>>> from matplotlib import use, pyplot
>>> use('Agg') 
>>> pyplot.savefig('modals.png') 
>>> print('Content-Type: text/html')
>>> print()
>>> print('<html><body>')
>>> print('<img src="modals.png"/>')
>>> print('</body></html>')
 

###NetworkX - graph package 
#The NetworkX package is for defining and manipulating structures consisting of nodes and edges, known as graphs


$ pip install networkx

#a Graph is a collection of nodes (vertices) along with identified pairs of nodes 
# nodes can be any hashable object e.g., a text string, an image, an XML object, another Graph, a customized node object, etc.


>>> import networkx as nx
>>> G = nx.Graph()

#add one node at a time,
>>> G.add_node(1)

#add a list of nodes,
>>> G.add_nodes_from([2, 3])


#by adding one edge at a time,
>>> G.add_edge(1, 2) #two nodes given 
>>> e = (2, 3)
>>> G.add_edge(*e)  # unpack edge tuple*

#by adding a list of edges,
>>> G.add_edges_from([(1, 2), (1, 3)])

#NetworkX quietly ignores any that are already present.
>>> G.add_edges_from([(1, 2), (1, 3)])
>>> G.add_node(1)
>>> G.add_edge(1, 2)
>>> G.add_node("spam")        # adds node "spam"
>>> G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
>>> G.add_edge(3, 'm')

#G consists of 8 nodes and 2 edges, as can be seen by:
>>> G.number_of_nodes()
8
>>> G.number_of_edges()
3

#set-like views of the nodes, edges, neighbors (adjacencies), and degrees of nodes
>>> G.nodes, G.edges, G.adj , G.degree. 

>>> list(G.nodes)
['a', 1, 2, 3, 'spam', 'm', 'p', 's']
>>> list(G.edges)
[(1, 2), (1, 3), (3, 'm')]
>>> list(G.adj[1])  # or list(G.neighbors(1))
[2, 3]
>>> G.degree[1]  # the number of edges incident to 1
2
#remove nodes and edges from the graph 
>>> G.remove_node(2)
>>> G.remove_nodes_from("spam")
>>> list(G.nodes)
[1, 3, 'spam']
>>> G.remove_edge(1, 3)


>>> G.add_edge(1, 2)
>>> H = nx.DiGraph(G)   # create a DiGraph using the connections from G
>>> list(H.edges())
[(1, 2), (2, 1)]
>>> edgelist = [(0, 1), (1, 2), (2, 3)]
>>> H = nx.Graph(edgelist)

#access to edges and neighbors is possible using subscript notation.
>>> G[1]  # same as G.adj[1]
AtlasView({2: {}})
>>> G[1][2]
{}
>>> G.edges[1, 2]
{}

#get/set the attributes of an edge using subscript notation 
#if the edge already exists.
>>> G.add_edge(1, 3)
>>> G[1][3]['color'] = "blue"
>>> G.edges[1, 2]['color'] = "red"

#Fast examination of all (node, adjacency) pairs is achieved using G.adjacency(), 
#or G.adj.items(). 
#Note that for undirected graphs, adjacency iteration sees each edge twice.
>>> FG = nx.Graph()
>>> FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
>>> for n, nbrs in FG.adj.items():
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
            if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))
(1, 2, 0.125)
(2, 1, 0.125)
(3, 4, 0.375)
(4, 3, 0.375)

#Convenient access to all edges is achieved with the edges property.
>>> for (u, v, wt) in FG.edges.data('weight'):
        if wt < 0.5: print('(%d, %d, %.3f)' % (u, v, wt))
(1, 2, 0.125)
(3, 4, 0.375)

##Adding attributes to graphs, nodes, and edges
#such as weights, labels, colors, or any Python object 
#Note that adding a node to G.nodes does not add it to the graph, use G.add_node()
#Similarly for edges.
#The special attribute weight should be numeric 

#Graph attributes
>>> G = nx.Graph(day="Friday")
>>> G.graph
{'day': 'Friday'}

#Or you can modify attributes later
>>> G.graph['day'] = "Monday"
>>> G.graph
{'day': 'Monday'}

#Node attributes
#Add node attributes using add_node(), add_nodes_from(), or G.nodes
>>> G.add_node(1, time='5pm')
>>> G.add_nodes_from([3], time='2pm')
>>> G.nodes[1]
{'time': '5pm'}
>>> G.nodes[1]['room'] = 714
>>> G.nodes.data()
NodeDataView({1: {'room': 714, 'time': '5pm'}, 3: {'time': '2pm'}})

#Edge Attributes
#Add/change edge attributes using add_edge(), add_edges_from(), or subscript notation.
>>> G.add_edge(1, 2, weight=4.7 )
>>> G.add_edges_from([(3, 4), (4, 5)], color='red')
>>> G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
>>> G[1][2]['weight'] = 4.7
>>> G.edges[3, 4]['weight'] = 4.2


##Directed graphs
#provides additional properties specific to directed edges, 
DiGraph.out_edges(), DiGraph.in_degree(), DiGraph.predecessors(), DiGraph.successors() etc.
#the directed versions of neighbors() is equivalent to successors() 
#while degree reports the sum of in_degree and out_degree 

>>> DG = nx.DiGraph()
>>> DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
>>> DG.out_degree(1, weight='weight')
0.5
>>> DG.degree(1, weight='weight')
1.25
>>> list(DG.successors(1))
[2]
>>> list(DG.neighbors(1))
[2]

##MultiGraph and MultiDiGraph classes 
#for graphs which allow multiple edges between any pair of nodes. 
>>> MG = nx.MultiGraph()
>>> MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
>>> dict(MG.degree(weight='weight'))
{1: 1.25, 2: 1.75, 3: 0.5}
>>> GG = nx.Graph()
>>> for n, nbrs in MG.adjacency():
        for nbr, edict in nbrs.items():
            minvalue = min([d['weight'] for d in edict.values()])
            GG.add_edge(n, nbr, weight = minvalue)
    
>>> nx.shortest_path(GG, 1, 3)
[1, 2, 3]

##Graph generators and graph operations

#Applying classic graph operations, such as:
# nbunch - A container of nodes (list, dict, set, etc.)
# ebunch - A container of edges as 2-tuples (u,v) or 3-tuples (u,v,d) where d is a dictionary containing edge data.
    subgraph(G, nbunch)      - induced subgraph view of G on nodes in nbunch
    union(G1,G2)             - graph union
    disjoint_union(G1,G2)    - graph union assuming all nodes are different
    cartesian_product(G1,G2) - return Cartesian product graph
    compose(G1,G2)           - combine graphs identifying nodes common to both
    complement(G)            - graph complement
    create_empty_copy(G)     - return an empty copy of the same graph class
    convert_to_undirected(G) - return an undirected representation of G
    convert_to_directed(G)   - return a directed representation of G

#Using a call to one of the classic small graphs, e.g.,
>>> petersen = nx.petersen_graph()
>>> tutte = nx.tutte_graph()
>>> maze = nx.sedgewick_maze_graph()
>>> tet = nx.tetrahedral_graph()

#Using a (constructive) generator for a classic graph, e.g.,
>>> K_5 = nx.complete_graph(5)
>>> K_3_5 = nx.complete_bipartite_graph(3, 5)
>>> barbell = nx.barbell_graph(10, 10)
>>> lollipop = nx.lollipop_graph(10, 20)

#Using a stochastic graph generator, e.g.,
>>> er = nx.erdos_renyi_graph(100, 0.15)
>>> ws = nx.watts_strogatz_graph(30, 3, 0.1)
>>> ba = nx.barabasi_albert_graph(100, 5)
>>> red = nx.random_lobster(100, 0.9, 0.9)

#Reading a graph stored in a file using common graph formats, 
#such as edge lists, adjacency lists, GML, GraphML, pickle, LEDA and others.
>>> nx.write_gml(red, "path.to.file")
>>> mygraph = nx.read_gml("path.to.file")

##Analyzing graphs
#algorithms 
https://networkx.github.io/documentation/latest/reference/algorithms/index.html

#
>>> G = nx.Graph()
>>> G.add_edges_from([(1, 2), (1, 3)])
>>> G.add_node("spam")       # adds node "spam"
>>> list(nx.connected_components(G))
[set([1, 2, 3]), set(['spam'])]
>>> sorted(d for n, d in G.degree())
[0, 1, 1, 2]
>>> nx.clustering(G)
{1: 0, 2: 0, 3: 0, 'spam': 0}

#Some functions with large output iterate over (node, value) 2-tuples. 
#These are easily stored in a dict structure if you desire.
>>> sp = dict(nx.all_pairs_shortest_path(G))
>>> sp[3]
{1: [3, 1], 2: [3, 1, 2], 3: [3]}




##Drawing 
#NetworkX can be used in conjunction with Matplotlib to visualize networks, such as WordNet 

>>> import matplotlib.pyplot as plt

>>> G = nx.petersen_graph()
>>> plt.subplot(121)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw(G, with_labels=True, font_weight='bold')
>>> plt.subplot(122)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
>>> plt.show()



>>> options = {
        'node_color': 'black',
        'node_size': 100,
        'width': 3,
    }
>>> plt.subplot(221)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_random(G, **options)
>>> plt.subplot(222)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_circular(G, **options)
>>> plt.subplot(223)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_spectral(G, **options)
>>> plt.subplot(224)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_shell(G, nlist=[range(5,10), range(5)], **options)


>>> G = nx.dodecahedral_graph()
>>> shells = [[2, 3, 4, 5, 6], [8, 1, 0, 19, 18, 17, 16, 15, 14, 7], [9, 10, 11, 12, 13]]
>>> nx.draw_shell(G, nlist=shells, **options)

#To save drawings to a file
>>> nx.draw(G)
>>> plt.savefig("path.png")

#If Graphviz and PyGraphviz or pydot, are available on your system, 
>>> from networkx.drawing.nx_pydot import write_dot
>>> pos = nx.nx_agraph.graphviz_layout(G)
>>> nx.draw(G, pos=pos)
>>> write_dot(G, 'file.dot')



#NLTK - Example - initializes an empty graph then traverses the WordNet hypernym hierarchy 
#adding edges to the graph [1]. 

 
import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn

def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start) #from wordnet
    for child in node.(): #a word of more specific meaning than a general or superordinate term applicable to it. For example, spoon is a hyponym of cutlery
        graph.add_edge(node.name, child.name) #[1] #from wordnet interface 
        traverse(graph, start, child) #[2]

def hyponym_graph(start):
    G = nx.Graph() #[3]
    G.depth = {}
    traverse(G, start, start)
    return G

def graph_draw(graph):
    nx.draw_graphviz(graph,
         node_size = [16 * graph.degree(n) for n in graph],
         node_color = [graph.depth[n] for n in graph],
         with_labels = False)
    matplotlib.pyplot.show()
 
 

>>> dog = wn.synset('dog.n.01')
>>> graph = hyponym_graph(dog)
>>> graph_draw(graph)
 
 



###NLTK - Collocations Measures 
#Collocations are expressions of multiple words which commonly co-occur. 
#For example, the top ten bigram collocations in Genesis are listed below, 
#as measured using Pointwise Mutual Information.

##Finder class 
#Finding collocations requires first calculating the frequencies of words 
#and their appearance in the context of other words
#contains following methods to get score given score_fn 
score_ngrams(score_fn):
    Returns a sequence of (ngram, score) pairs ordered from highest to
    lowest score, as determined by the scoring function provided.
        
nbest(score_fn, n):
    Returns the top n ngrams when scored by the given function

above_score(score_fn, min_score):
    Returns a sequence of ngrams, ordered by decreasing score, whose
    scores each exceed the given minimum score.
    
apply_freq_filter( min_freq):
    Removes candidate ngrams which have frequency less than min_freq.

apply_ngram_filter(fn):
    Removes candidate ngrams (w1, w2, ...) where fn(w1, w2, ...)    evaluates to True.
       
apply_word_filter(fn)
    Removes candidate ngrams (w1, w2, ...) where any of (fn(w1), fn(w2),       ...) evaluates to True.

@classmethod
from_documents(documents):
    Constructs a collocation finder given a collection of documents,
    each of which is a list (or iterable) of tokens.    
    
@classmethod
from_words(words, window_size=2):
    Construct a BigramCollocationFinder for all bigrams in the given
    sequence.  When window_size > 2, count non-contiguous bigrams, in the
    style of Church and Hanks's (1990) association ratio.

        
##Scoring 
#NgramAssocMeasures, and n-specific BigramAssocMeasures , TrigramAssocMeasures, QuadgramAssocMeasures
#contains many scoring/metrics function - score_fn - to be passed in Finder
#ex-one score_fn is  pmi - Scores ngrams by pointwise mutual information


>>> import nltk
>>> from nltk.collocations import *
>>> bigram_measures = nltk.collocations.BigramAssocMeasures()
>>> trigram_measures = nltk.collocations.TrigramAssocMeasures()
>>> finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
>>> finder.nbest(bigram_measures.pmi, 10)  # doctest: +NORMALIZE_WHITESPACE
[(u'Allon', u'Bacuth'), (u'Ashteroth', u'Karnaim'), (u'Ben', u'Ammi'),
 (u'En', u'Mishpat'), (u'Jegar', u'Sahadutha'), (u'Salt', u'Sea'),
 (u'Whoever', u'sheds'), (u'appoint', u'overseers'), (u'aromatic', u'resin'),
 (u'cutting', u'instrument')]


#While these words are highly collocated, the expressions are also very infrequent. 
#Therefore it is useful to apply filters, 
#such as ignoring all bigrams which occur less than three times in the corpus:

>>> finder.apply_freq_filter(3)
>>> finder.nbest(bigram_measures.pmi, 10)  
[(u'Beer', u'Lahai'), (u'Lahai', u'Roi'), (u'gray', u'hairs'),
 (u'Most', u'High'), (u'ewe', u'lambs'), (u'many', u'colors'),
 (u'burnt', u'offering'), (u'Paddan', u'Aram'), (u'east', u'wind'),
 (u'living', u'creature')]


#We may similarly find collocations among tagged words:

>>> finder = BigramCollocationFinder.from_words(
        nltk.corpus.brown.tagged_words('ca01', tagset='universal'))
>>> finder.nbest(bigram_measures.pmi, 5)  # doctest: +NORMALIZE_WHITESPACE
[(('1,119', 'NUM'), ('votes', 'NOUN')),
 (('1962', 'NUM'), ("governor's", 'NOUN')),
 (('637', 'NUM'), ('E.', 'NOUN')),
 (('Alpharetta', 'NOUN'), ('prison', 'NOUN')),
 (('Bar', 'NOUN'), ('Association', 'NOUN'))]


#Or tags alone:

>>> finder = BigramCollocationFinder.from_words(t for w, t in
        nltk.corpus.brown.tagged_words('ca01', tagset='universal'))
>>> finder.nbest(bigram_measures.pmi, 10)  # doctest: +NORMALIZE_WHITESPACE
[('PRT', 'VERB'), ('PRON', 'VERB'), ('ADP', 'DET'), ('.', 'PRON'), ('DET', 'ADJ'),
 ('CONJ', 'PRON'), ('ADP', 'NUM'), ('NUM', '.'), ('ADV', 'ADV'), ('VERB', 'ADV')]


#Or spanning intervening words:

>>> finder = BigramCollocationFinder.from_words(
        nltk.corpus.genesis.words('english-web.txt'),
        window_size = 20)
>>> finder.apply_freq_filter(2)
>>> ignored_words = nltk.corpus.stopwords.words('english')
>>> finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
>>> finder.nbest(bigram_measures.likelihood_ratio, 10) # doctest: +NORMALIZE_WHITESPACE
[(u'chief', u'chief'), (u'became', u'father'), (u'years', u'became'),
 (u'hundred', u'years'), (u'lived', u'became'), (u'king', u'king'),
 (u'lived', u'years'), (u'became', u'became'), (u'chief', u'chiefs'),
 (u'hundred', u'became')]



##Using Finders
#The collocations package provides collocation finders which by default 
#consider all ngrams in a text as candidate collocations:

>>> text = "I do not like green eggs and ham, I do not like them Sam I am!"
>>> tokens = nltk.wordpunct_tokenize(text)
>>> finder = BigramCollocationFinder.from_words(tokens)
>>> scored = finder.score_ngrams(bigram_measures.raw_freq)
>>> sorted(bigram for bigram, score in scored)  # doctest: +NORMALIZE_WHITESPACE
[(',', 'I'), ('I', 'am'), ('I', 'do'), ('Sam', 'I'), ('am', '!'),
 ('and', 'ham'), ('do', 'not'), ('eggs', 'and'), ('green', 'eggs'),
 ('ham', ','), ('like', 'green'), ('like', 'them'), ('not', 'like'),
 ('them', 'Sam')]


#OR construct the collocation finder from manually-derived FreqDists:

>>> word_fd = nltk.FreqDist(tokens)
>>> bigram_fd = nltk.FreqDist(nltk.bigrams(tokens))
>>> finder = BigramCollocationFinder(word_fd, bigram_fd)
>>> scored == finder.score_ngrams(bigram_measures.raw_freq)
True


#A similar interface is provided for trigrams:

>>> finder = TrigramCollocationFinder.from_words(tokens)
>>> scored = finder.score_ngrams(trigram_measures.raw_freq)
>>> set(trigram for trigram, score in scored) == set(nltk.trigrams(tokens))
True


#We may want to select only the top n results:

>>> sorted(finder.nbest(trigram_measures.raw_freq, 2))
[('I', 'do', 'not'), ('do', 'not', 'like')]


#Alternatively, we can select those above a minimum score value:

>>> sorted(finder.above_score(trigram_measures.raw_freq,
                            1.0 / len(tuple(nltk.trigrams(tokens)))))
[('I', 'do', 'not'), ('do', 'not', 'like')]


#Now spanning intervening words:

>>> finder = TrigramCollocationFinder.from_words(tokens)
>>> finder = TrigramCollocationFinder.from_words(tokens, window_size=4)
>>> sorted(finder.nbest(trigram_measures.raw_freq, 4))
[('I', 'do', 'like'), ('I', 'do', 'not'), ('I', 'not', 'like'), ('do', 'not', 'like')]


#A closer look at the finder's ngram frequencies:

>>> sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10]  # doctest: +NORMALIZE_WHITESPACE
[(('I', 'do', 'like'), 2), (('I', 'do', 'not'), 2), (('I', 'not', 'like'), 2),
 (('do', 'not', 'like'), 2), ((',', 'I', 'do'), 1), ((',', 'I', 'not'), 1),
 ((',', 'do', 'not'), 1), (('I', 'am', '!'), 1), (('Sam', 'I', '!'), 1),
 (('Sam', 'I', 'am'), 1)]



##Filtering candidates
#All the ngrams in a text are often too many to be useful when finding collocations. 
#It is generally useful to remove some words or punctuation, and to require a minimum frequency for candidate collocations.

#Given our sample text above, 
#if we remove all trigrams containing personal pronouns from candidature, 
#score_ngrams should return 6 less results, 
#and 'do not like' will be the only candidate which occurs more than once:

>>> finder = TrigramCollocationFinder.from_words(tokens)
>>> len(finder.score_ngrams(trigram_measures.raw_freq))
14
>>> finder.apply_word_filter(lambda w: w in ('I', 'me'))
>>> len(finder.score_ngrams(trigram_measures.raw_freq))
8
>>> sorted(finder.above_score(trigram_measures.raw_freq,
                                1.0 / len(tuple(nltk.trigrams(tokens)))))
[('do', 'not', 'like')]


#Sometimes a filter is a function on the whole ngram, 
#rather than each word, such as if we may permit 'and' to appear in the middle of a trigram, 
#but not on either edge:

>>> finder.apply_ngram_filter(lambda w1, w2, w3: 'and' in (w1, w3))
>>> len(finder.score_ngrams(trigram_measures.raw_freq))
6


#Finally, it is often important to remove low frequency candidates, 
#as we lack sufficient evidence about their significance as collocations:

>>> finder.apply_freq_filter(2)
>>> len(finder.score_ngrams(trigram_measures.raw_freq))
1



##Association measures
#A number of measures are available to score collocations or other associations. 
#The arguments to measure functions are marginals of a contingency table, 

#in the bigram case , Each association measure
#is provided as a function with three arguments::
        bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)

#The arguments constitute the marginals of a contingency table, counting
#the occurrences of particular events in a corpus. 
#The letter i in the suffix refers to the appearance of the word in question, 
#while x indicates the appearance of any word. 
#for example:
        n_ii counts (w1, w2), i.e. the bigram being scored
        n_ix counts (w1, *)
        n_xi counts (*, w2)
        n_xx counts (*, *), i.e. any bigram

#This may be shown with respect to a contingency table::
                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
             
#Student's t: examples from Manning and Schutze 5.3.2

>>> print('%0.4f' % bigram_measures.student_t(8, (15828, 4675), 14307668))
0.9999
>>> print('%0.4f' % bigram_measures.student_t(20, (42, 20), 14307668))
4.4721


#Chi-square: examples from Manning and Schutze 5.3.3

>>> print('%0.2f' % bigram_measures.chi_sq(8, (15828, 4675), 14307668))
1.55
>>> print('%0.0f' % bigram_measures.chi_sq(59, (67, 65), 571007))
456400


#Likelihood ratios: examples from Dunning, CL, 1993

>>> print('%0.2f' % bigram_measures.likelihood_ratio(110, (2552, 221), 31777))
270.72
>>> print('%0.2f' % bigram_measures.likelihood_ratio(8, (13, 32), 31777))
95.29


#Pointwise Mutual Information: examples from Manning and Schutze 5.4

>>> print('%0.2f' % bigram_measures.pmi(20, (42, 20), 14307668))
18.38
>>> print('%0.2f' % bigram_measures.pmi(20, (15019, 15629), 14307668))
0.29


#Using contingency table values
#While frequency counts make marginals readily available for collocation finding, 
#it is common to find published contingency table values. 
#The collocations package therefore provides a wrapper, ContingencyMeasures, which wraps an association measures class, providing association measures which take contingency values as arguments, (n_ii, n_io, n_oi, n_oo) in the bigram case.

>>> from nltk.metrics import ContingencyMeasures
>>> cont_bigram_measures = ContingencyMeasures(bigram_measures)
>>> print('%0.2f' % cont_bigram_measures.likelihood_ratio(8, 5, 24, 31740))
95.29
>>> print('%0.2f' % cont_bigram_measures.chi_sq(8, 15820, 4667, 14287173))
1.55



#Ranking and correlation
#It is useful to consider the results of finding collocations as a ranking, 
#and the rankings output using different association measures can be compared using the Spearman correlation coefficient.

#Ranks can be assigned to a sorted list of results trivially by assigning strictly increasing ranks to each result:

nltk.metrics.spearman.ranks_from_scores(scores, rank_gap=1e-15)
    Given a sequence of (key, score) tuples, yields each key 
    with an increasing rank, tying with previous key’s rank 
    if the difference between their scores is less than rank_gap. 
    Suitable for use as an argument to spearman_correlation.

nltk.metrics.spearman.ranks_from_sequence(seq)
    Given a sequence, yields each element with an increasing rank, 
    suitable for use as an argument to spearman_correlation.

nltk.metrics.spearman.spearman_correlation(ranks1, ranks2)
    Returns the Spearman correlation coefficient for two rankings, 
    which should be dicts or sequences of (key, rank). 
    The coefficient ranges from -1.0 (ranks are opposite) to 1.0 (ranks are identical), 
    and is only calculated for keys in both rankings 
    (for meaningful results, remove keys present in only one list before ranking).


#Example 
>>> from nltk.metrics.spearman import *
>>> results_list = ['item1', 'item2', 'item3', 'item4', 'item5']
>>> print(list(ranks_from_sequence(results_list)))
[('item1', 0), ('item2', 1), ('item3', 2), ('item4', 3), ('item5', 4)]


#If scores are available for each result, 
#we may allow sufficiently similar results (differing by no more than rank_gap) to be assigned the same rank:

>>> results_scored = [('item1', 50.0), ('item2', 40.0), ('item3', 38.0),
                    ('item4', 35.0), ('item5', 14.0)]
>>> print(list(ranks_from_scores(results_scored, rank_gap=5)))
[('item1', 0), ('item2', 1), ('item3', 1), ('item4', 1), ('item5', 4)]


#The Spearman correlation coefficient gives a number from -1.0 to 1.0 comparing two rankings. 
#A coefficient of 1.0 indicates identical rankings; -1.0 indicates exact opposite rankings.

>>> print('%0.1f' % spearman_correlation(
            ranks_from_sequence(results_list),
            ranks_from_sequence(results_list)))
1.0
>>> print('%0.1f' % spearman_correlation(
            ranks_from_sequence(reversed(results_list)),
            ranks_from_sequence(results_list)))
-1.0
>>> results_list2 = ['item2', 'item3', 'item1', 'item5', 'item4']
>>> print('%0.1f' % spearman_correlation(
        ranks_from_sequence(results_list),
        ranks_from_sequence(results_list2)))
0.6
>>> print('%0.1f' % spearman_correlation(
        ranks_from_sequence(reversed(results_list)),
        ranks_from_sequence(results_list2)))
-0.6

                                                     
 










###chap-5

###Categorizing and Tagging Words
#word classes - nouns, verbs, adjectives, and adverbs




nltk.tag.pos_tag(tokens, tagset=None, lang='eng')
    Use NLTK’s currently recommended part of speech tagger to tag the given list of tokens.
        tokens (list(str)) – Sequence of tokens to be tagged
        tagset (str) – the tagset to be used, e.g. universal, wsj, brown
        lang (str) – the ISO 639 code of the language, e.g. ‘eng’ for English, ‘rus’ for Russian
    Returns:	list(tuple(str, str))

nltk.tag.pos_tag_sents(sentences, tagset=None, lang='eng')
    Use NLTK’s currently recommended part of speech tagger to tag the given list of sentences, 
    each consisting of a list of tokens.
        tokens (list(list(str))) – List of sentences to be tagged
        tagset (str) – the tagset to be used, e.g. universal, wsj, brown
        lang (str) – the ISO 639 code of the language, e.g. ‘eng’ for English, ‘rus’ for Russian
    Returns:	The list of tagged sentences, list(list(tuple(str, str)))

#A part-of-speech tagger, or POS-tagger, processes a sequence of words, 
#and attaches a part of speech tag to each word 
#Tagged tokens are encoded as tuples (tag, token), tag is such as its part of speech

CC, a coordinating conjunction; 
RB, or adverbs; 
IN, a preposition; 
NN, a noun; 
JJ, an adjective.
VBP, present tense verb 
DT: determiner
PRP: pronoun, personal
TO: "to" as preposition or infinitive marker


#Check help 
nltk.help.upenn_tagset('RB')
nltk.help.upenn_tagset('NN.*')
#for all 
>>> nltk.help.upenn_tagset()


#Example 
>>> text = word_tokenize("And now for something completely different")
>>> nltk.pos_tag(text)
[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'),
('completely', 'RB'), ('different', 'JJ')]
 
 

>>> text = nlyk.word_tokenize("They refuse to permit us to obtain the refuse permit")
>>> nltk.pos_tag(text)
[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]


#Example - Consider the following analysis involving woman (a noun), bought (a verb), 
#over (a preposition),  and the (a determiner). 

#The text.similar() method takes a word w, finds all contexts w1w w2, 
#then finds all words w' that appear in the same context, i.e. w1w'w2.

>>> text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
>>> text.similar('woman')
Building word-context index...
man day time year car moment world family house boy child country job
state girl place war way case question
>>> text.similar('bought')
made done put said found had seen given left heard been brought got
set was called felt in that told
>>> text.similar('over')
in on to of and for with from at by that into as up out down through
about all is
>>> text.similar('the')
a his this their its her an that our any all one these my in your no
some other and
 
 
#Observe that searching for woman finds nouns; 
#searching for bought mostly finds verbs; searching for over generally finds prepositions; 
#searching for the finds several determiners. 

#A tagger can correctly identify the tags on these words in the context of a sentence

##Tagged Corpora
#By convention in NLTK, a tagged token is represented using a tuple consisting of the token and the tag. 

>>> tagged_token = nltk.tag.str2tuple('fly/NN')
>>> tagged_token
('fly', 'NN')
>>> tagged_token[0]
'fly'
>>> tagged_token[1]
'NN'
 
 
#construct a list of tagged tokens directly from a string. 

>>> sent = '''
    The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
    other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
    Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
    said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
    accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
    interest/NN of/IN both/ABX governments/NNS ''/'' ./.
    '''
>>> [nltk.tag.str2tuple(t) for t in sent.split()]
[('The', 'AT'), ('grand', 'JJ'), ('jury', 'NN'), ('commented', 'VBD'),
('on', 'IN'), ('a', 'AT'), ('number', 'NN'), ... ('.', '.')]
 
 

 
##Reading Tagged Corpora
#Several of the corpora included with NLTK have been tagged for their part-of-speech. 

#for example - Brown Corpus with a text editor:

The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd / no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.

>>> nltk.corpus.brown.tagged_words()
[('The', 'AT'), ('Fulton', 'NP-TL'), ...]
>>> nltk.corpus.brown.tagged_words(tagset='universal')
[('The', 'DET'), ('Fulton', 'NOUN'), ...]
 
 
#Whenever a corpus contains tagged text, 
#the NLTK corpus interface will have a tagged_words() method. 

>>> print(nltk.corpus.nps_chat.tagged_words())
[('now', 'RB'), ('im', 'PRP'), ('left', 'VBD'), ...]
>>> nltk.corpus.conll2000.tagged_words()
[('Confidence', 'NN'), ('in', 'IN'), ('the', 'DT'), ...]
>>> nltk.corpus.treebank.tagged_words()
[('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ...]
 
 
 
 
#Tagged corpora for several other languages are distributed with NLTK, 
#including Chinese, Hindi, Portuguese, Spanish, Dutch and Catalan. 

#These usually contain non-ASCII text, and Python always displays this in hexadecimal 
#when printing a larger structure such as a list.

>>> nltk.corpus.sinica_treebank.tagged_words()
[('ä', 'Neu'), ('åæ', 'Nad'), ('åç', 'Nba'), ...]
>>> nltk.corpus.indian.tagged_words()
[('??????', 'NN'), ('??????', 'NN'), (':', 'SYM'), ...]
>>> nltk.corpus.mac_morpho.tagged_words()
[('Jersei', 'N'), ('atinge', 'V'), ('m\xe9dia', 'N'), ...]
>>> nltk.corpus.conll2002.tagged_words()
[('Sao', 'NC'), ('Paulo', 'VMI'), ('(', 'Fpa'), ...]
>>> nltk.corpus.cess_cat.tagged_words()
[('El', 'da0ms0'), ('Tribunal_Suprem', 'np0000o'), ...]
 
 
##A Universal Part-of-Speech Tagset

#"Universal Tagset": Simplified and streamlined tags 
>>> nltk.corpus.brown.tagged_words(tagset='universal')
[('The', 'DET'), ('Fulton', 'NOUN'), ...]
>>> nltk.corpus.treebank.tagged_words(tagset='universal')
[('Pierre', 'NOUN'), ('Vinken', 'NOUN'), (',', '.'), ...]

#Tagged corpora use many different conventions for tagging words. 
#To help us get started, we will be looking at a simplified tagset 

#Tag        Meaning             English Examples
ADJ         adjective           new, good, high, special, big, local 
ADP         adposition          on, of, at, with, by, into, under 
ADV         adverb              really, already, still, early, now 
CONJ        conjunction         and, or, but, if, while, although 
DET         determiner          , article the, a, some, most, every, no, which 
NOUN        noun                year, home, costs, time, Africa 
NUM         numeral             twenty-four, fourth, 1991, 14:24 
PRT         particle            at, on, out, over per, that, up, with 
PRON        pronoun             he, their, her, its, my, I, us 
VERB        verb                is, say, told, given, playing, would 
.           punctuation marks   . , ; ! 
X           other               ersatz, esprit, dunno, gr8, univeristy 

#Exmaple - which of these tags are the most common in the news category of the Brown corpus:
>>> from nltk.corpus import brown
>>> brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
>>> tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
>>> tag_fd.most_common()
[('NOUN', 30640), ('VERB', 14399), ('ADP', 12355), ('.', 11928), ('DET', 11389),
 ('ADJ', 6706), ('ADV', 3349), ('CONJ', 2717), ('PRON', 2535), ('PRT', 2264),
 ('NUM', 2166), ('X', 106)]
 
 
 
##Nouns
#Nouns generally refer to people, places, things, or concepts

#Example - inspect some tagged text to see what parts of speech occur before a noun, 
#with the most frequent ones first. 

>>> word_tag_pairs = nltk.bigrams(brown_news_tagged)
>>> noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
>>> fdist = nltk.FreqDist(noun_preceders)
>>> [tag for (tag, _) in fdist.most_common()]
['NOUN', 'DET', 'ADJ', 'ADP', '.', 'VERB', 'CONJ', 'NUM', 'ADV', 'PRT', 'PRON', 'X']
 
 
##Verbs
#Verbs are words that describe events and actions
#Example - What are the most common verbs in news text? Let's sort all the verbs by frequency:



>>> wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
>>> word_tag_fd = nltk.FreqDist(wsj)
>>> [wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB']
['is', 'said', 'are', 'was', 'be', 'has', 'have', 'will', 'says', 'would',
 'were', 'had', 'been', 'could', "'s", 'can', 'do', 'say', 'make', 'may',
 'did', 'rose', 'made', 'does', 'expected', 'buy', 'take', 'get', 'might',
 'sell', 'added', 'sold', 'help', 'including', 'should', 'reported', ...]
 
 
#Note that the items being counted in the frequency distribution are word-tag pairs. 
#Since words and tags are paired, we can treat the word as a condition and the tag as an event, 


>>> cfd1 = nltk.ConditionalFreqDist(wsj)
>>> cfd1['yield'].most_common()
[('VERB', 28), ('NOUN', 20)]
>>> cfd1['cut'].most_common()
[('VERB', 25), ('NOUN', 3)]
 
 
#We can reverse the order of the pairs, so that the tags are the conditions, 
#and the words are the events


>>> wsj = nltk.corpus.treebank.tagged_words()
>>> cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
>>> list(cfd2['VBN'])
['been', 'expected', 'made', 'compared', 'based', 'priced', 'used', 'sold',
'named', 'designed', 'held', 'fined', 'taken', 'paid', 'traded', 'said', ...]
 
 
#To clarify the distinction between VBD (past tense) and VBN (past participle), 
#let's find words which can be both VBD and VBN, and see some surrounding text:

>>> [w for w in cfd1.conditions() if 'VBD' in cfd1[w] and 'VBN' in cfd1[w]]
['Asked', 'accelerated', 'accepted', 'accused', 'acquired', 'added', 'adopted', ...]
>>> idx1 = wsj.index(('kicked', 'VBD'))
>>> wsj[idx1-4:idx1+1]
[('While', 'IN'), ('program', 'NN'), ('trades', 'NNS'), ('swiftly', 'RB'),
 ('kicked', 'VBD')]
>>> idx2 = wsj.index(('kicked', 'VBN'))
>>> wsj[idx2-4:idx2+1]
[('head', 'NN'), ('of', 'IN'), ('state', 'NN'), ('has', 'VBZ'), ('kicked', 'VBN')]
 
 
##Unsimplified Tags

#Example - finds all tags starting with NN
#there are many variants of NN; 
#the most important contain $ for possessive nouns, 
#S for plural nouns (since plural nouns typically end in s) 
#and P for proper nouns. 

#In addition, most of the tags have suffix modifiers: 
#-NC for citations, -HL for words in headlines and -TL for titles (a feature of Brown tabs).

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())

>>> tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
>>> for tag in sorted(tagdict):
        print(tag, tagdict[tag])
...
NN [('year', 137), ('time', 97), ('state', 88), ('week', 85), ('man', 72)]
NN$ [("year's", 13), ("world's", 8), ("state's", 7), ("nation's", 6), ("company's", 6)]
NN$-HL [("Golf's", 1), ("Navy's", 1)]
NN$-TL [("President's", 11), ("Army's", 3), ("Gallery's", 3), ("University's", 3), ("League's", 3)]
NN-HL [('sp.', 2), ('problem', 2), ('Question', 2), ('business', 2), ('Salary', 2)]
NN-NC [('eva', 1), ('aya', 1), ('ova', 1)]
NN-TL [('President', 88), ('House', 68), ('State', 59), ('University', 42), ('City', 41)]
NN-TL-HL [('Fort', 2), ('Dr.', 1), ('Oak', 1), ('Street', 1), ('Basin', 1)]
NNS [('years', 101), ('members', 69), ('people', 52), ('sales', 51), ('men', 46)]
NNS$ [("children's", 7), ("women's", 5), ("janitors'", 3), ("men's", 3), ("taxpayers'", 2)]
NNS$-HL [("Dealers'", 1), ("Idols'", 1)]
NNS$-TL [("Women's", 4), ("States'", 3), ("Giants'", 2), ("Bros.'", 1), ("Writers'", 1)]
NNS-HL [('comments', 1), ('Offenses', 1), ('Sacrifices', 1), ('funds', 1), ('Results', 1)]
NNS-TL [('States', 38), ('Nations', 11), ('Masters', 10), ('Rules', 9), ('Communists', 9)]
NNS-TL-HL [('Nations', 1)]

##Exploring Tagged Corpora
#Example - Suppose we're studying the word 'often' and want to see how it is used in text. 

>>> brown_learned_text = brown.words(categories='learned')
>>> sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))
[',', '.', 'accomplished', 'analytically', 'appear', 'apt', 'associated', 'assuming',
'became', 'become', 'been', 'began', 'call', 'called', 'carefully', 'chose', ...]
 
 
#OR 
>>> brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
>>> tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
>>> fd = nltk.FreqDist(tags)
>>> fd.tabulate()
 PRT  ADV  ADP    . VERB  ADJ
   2    8    7    4   37    6
 
#Notice that the most high-frequency parts of speech following 'often' are verbs. 
#Nouns never appear in this position (in this particular corpus).

#Example - find words involving particular sequences of tags and words 
#(in this case "<Verb> to <Verb>")
 
from nltk.corpus import brown
def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence): 
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')): 
            print(w1, w2, w3) [3]

>>> for tagged_sent in brown.tagged_sents():
        process(tagged_sent)
...
combined to achieve
continue to place
serve to protect
wanted to wait
allowed to place
expected to become
...
 
#Example - let's look for words that are highly ambiguous as to their part of speech tag. 
#Understanding why such words are tagged as they are in each context 
#can help us clarify the distinctions between the tags.


>>> brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
>>> data = nltk.ConditionalFreqDist((word.lower(), tag)
                for (word, tag) in brown_news_tagged)
>>> for word in sorted(data.conditions()):
     if len(data[word]) > 3:
         tags = [tag for (tag, _) in data[word].most_common()]
         print(word, ' '.join(tags))

best ADJ ADV NP V
better ADJ ADV V DET
close ADV ADJ V N
cut V N VN VD
even ADV DET ADJ V
grant NP N V -
hit V VD VN N
lay ADJ V NP VD
left VD ADJ N VN
like CNJ V ADJ P -
near P ADV ADJ DET
open ADJ V N ADV
past N ADJ DET P
present ADJ ADV V N
read V VN VD NP
right ADJ N DET ADV
second NUM ADV DET N
set VN V VD N -
that CNJ V WH DET
 
 
#Example - Incrementally Updating a Dictionary, and Sorting by Value
>>> from collections import defaultdict
#defaultdict take a function() which returns defult value 
>>> counts = defaultdict(int)  #default value is int=0, 
>>> from nltk.corpus import brown
>>> for (word, tag) in brown.tagged_words(categories='news', tagset='universal'):
        counts[tag] += 1
...
>>> counts['NOUN']
30640
>>> sorted(counts)
['ADJ', 'PRT', 'ADV', 'X', 'CONJ', 'PRON', 'VERB', '.', 'NUM', 'NOUN', 'ADP', 'DET']

>>> from operator import itemgetter
>>> sorted(counts.items(), key=itemgetter(1), reverse=True)
[('NOUN', 30640), ('VERB', 14399), ('ADP', 12355), ('.', 11928), ...]
>>> [t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)]
['NOUN', 'VERB', 'ADP', '.', 'DET', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PRT', 'NUM', 'X']
 
#Item getter 
>>> pair = ('NP', 8336)
>>> pair[1]
8336
>>> itemgetter(1)(pair)
8336
 
#Example - index words according to their last two letters:

>>> last_letters = defaultdict(list)  #default value is list=[]
>>> words = nltk.corpus.words.words('en')
>>> for word in words:
        key = word[-2:]
        last_letters[key].append(word)
...
>>> last_letters['ly']
['abactinally', 'abandonedly', 'abasedly', 'abashedly', 'abashlessly', 'abbreviately',
'abdominally', 'abhorrently', 'abidingly', 'abiogenetically', 'abiologically', ...]
>>> last_letters['zy']
['blazy', 'bleezy', 'blowzy', 'boozy', 'breezy', 'bronzy', 'buzzy', 'Chazy', ...]
 
 

#Example -  create an anagram dictionary. 
#An anagram is word or phrase formed by rearranging the letters of a different word or phrase, 
#typically using all the original letters exactly once.
>>> from collections import defaultdict
>>> words = nltk.corpus.words.words('en')
>>> anagrams = defaultdict(list)
>>> for word in words:
        key = ''.join(sorted(word))
        anagrams[key].append(word)

>>> anagrams['aeilnrt']
['entrail', 'latrine', 'ratline', 'reliant', 'retinal', 'trenail']
 
 
#Since accumulating words like this is such a common task, 
#NLTK provides a more convenient way of creating a defaultdict(list),  
#called - nltk.Index(pairs), pairs = (key,value)
#Index is nothing but {key:[values..]} ie dict with values are list 
>>> anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
>>> anagrams['aeilnrt']
['entrail', 'latrine', 'ratline', 'reliant', 'retinal', 'trenail']
 
 
#Example - Let's study the range of possible tags for a word, given the word itself, 
#and the tag of the previous word. 
>>> from collections import defaultdict
>>> pos = defaultdict(lambda: defaultdict(int)) #dict of dict with values int
#[(word,tag),...] 
>>> brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news', tagset='universal')
>>> for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged): 
        pos[(t1, w2)][t2] += 1 
...
>>> pos[('DET', 'right')] 
defaultdict(<class 'int'>, {'ADJ': 11, 'NOUN': 5})
 
 
##Automatic Tagging of a text 
>>> from nltk.corpus import brown
>>> brown_tagged_sents = brown.tagged_sents(categories='news')
>>> brown_sents = brown.sents(categories='news')
 
#reference - Tagger 
#many others in www.nltk.org/api/nltk.tag.htm
class nltk.tag.api.TaggerI
    evaluate(gold)
        Score the accuracy of the tagger against the gold standard. 
        Strip the tags from the gold standard text, retag it using the tagger, 
        then compute the accuracy score.
        Parameters:	gold (list(list(tuple(str, str)))) – 
            The list of tagged sentences to score the tagger on.
        Return type:	float
    tag(tokens)
        Determine the most appropriate tag sequence for the given token sequence, 
        A tagged token is encoded as a tuple (token, tag).
        Return type:	list(tuple(str, str))
    tag_sents(sentences)
        Apply self.tag() to each element of sentences. I.e.:
            return [self.tag(sent) for sent in sentences]
            
            
class nltk.tag.sequential.SequentialBackoffTagger(backoff=None)
    Bases: nltk.tag.api.TaggerI
    taggers that tags words sequentially, left to right. 
    If a tagger is unable to determine a tag for the specified token, 
    then its backoff tagger is consulted.
        backoff        The backoff tagger for this tagger.
            
class nltk.tag.sequential.ContextTagger(context_to_tag, backoff=None)
    Bases: nltk.tag.sequential.SequentialBackoffTagger
    base class for sequential backoff taggers that choose a tag for a token 
    based on the value of its “context”. 
    Different subclasses are used to define different contexts.
    A ContextTagger chooses the tag for a token by calculating the token’s context, 
    and looking up the corresponding tag in a table. 
    This table can be constructed manually; 
    or it can be automatically constructed based on a training corpus,

class nltk.tag.sequential.NgramTagger(n, train=None, model=None, backoff=None, cutoff=0, verbose=False)
    Bases: nltk.tag.sequential.ContextTagger
    A tagger that chooses a token’s tag based on its word string 
    and on the preceding n word’s tags. 
    In particular, a tuple (tags[i-n:i-1], words[i]) is looked up in a table, 
    and the corresponding tag is returned. 
    Train a new NgramTagger using the given training data or the supplied model. 
    In particular, construct a new tagger whose table maps 
    from each context (tag[i-n:i-1], word[i]) to the most frequent tag for that context. 
    But exclude any contexts that are already tagged perfectly by the backoff tagger.
        train – A tagged corpus consisting of a list of tagged sentences, 
                where each sentence is a list of (word, tag) tuples.
        backoff – A backoff tagger, to be used by the new tagger 
                 if it encounters an unknown context.
        cutoff – If the most likely tag for a context occurs fewer 
                 than cutoff times, then exclude it from the context-to-tag table for the new tagger.

class nltk.tag.sequential.BigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False)
    Bases: nltk.tag.sequential.NgramTagger
    A tagger that chooses a token’s tag based its word string 
    and on the preceding words’ tag. 
    In particular, a tuple consisting of the previous tag 
    and the word is looked up in a table, and the corresponding tag is returned.
        train (list(list(tuple(str, str)))) – The corpus of training data, a list of tagged sentences
        model (dict) – The tagger model
        backoff (TaggerI) – Another tagger which this tagger will consult when it is unable to tag a word
        cutoff (int) – The number of instances of training data the tagger must see in order not to use the backoff tagger
        
            
class nltk.tag.sequential.UnigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False)
    Bases: nltk.tag.sequential.NgramTagger
    The UnigramTagger finds the most likely tag 
    for each word in a training corpus, 
    and then uses that information to assign tags to new tokens.
    
class nltk.tag.sequential.RegexpTagger(regexps, backoff=None)
    Bases: nltk.tag.sequential.SequentialBackoffTagger
    The RegexpTagger assigns tags to tokens by comparing their word strings 
    to a series of regular expressions.

class nltk.tag.sequential.TrigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False)
    Bases: nltk.tag.sequential.NgramTagger
    A tagger that chooses a token’s tag based its word string 
    and on the preceding two words’ tags

class nltk.tag.sequential.DefaultTagger(tag)
    Bases: nltk.tag.sequential.SequentialBackoffTagger
    A tagger that assigns the same tag to every token.
    
#The Default Tagger
#the simplest possible tagger assigns the same tag(highest probable) to each token

>>> tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
>>> nltk.FreqDist(tags).max()
'NN'
 
 

#create a tagger that tags everything as NN.
>>> raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
>>> tokens = word_tokenize(raw)
>>> default_tagger = nltk.DefaultTagger('NN')
>>> default_tagger.tag(tokens)
[('I', 'NN'), ('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('green', 'NN'),
('eggs', 'NN'), ('and', 'NN'), ('ham', 'NN'), (',', 'NN'), ('I', 'NN'),
('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('them', 'NN'), ('Sam', 'NN'),
('I', 'NN'), ('am', 'NN'), ('!', 'NN')]
 
#Poor 
>>> default_tagger.evaluate(brown_tagged_sents)
0.13089484257215028
 
 
##The Regular Expression Tagger
 
>>> patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
 ]
 
 
>>> regexp_tagger = nltk.RegexpTagger(patterns)
>>> regexp_tagger.tag(brown_sents[3])
[('``', 'NN'), ('Only', 'NN'), ('a', 'NN'), ('relative', 'NN'), ('handful', 'NN'),
('of', 'NN'), ('such', 'NN'), ('reports', 'NNS'), ('was', 'NNS'), ('received', 'VBD'),
("''", 'NN'), (',', 'NN'), ('the', 'NN'), ('jury', 'NN'), ('said', 'NN'), (',', 'NN'),
('``', 'NN'), ('considering', 'VBG'), ('the', 'NN'), ('widespread', 'NN'), ...]
>>> regexp_tagger.evaluate(brown_tagged_sents)
0.20326391789486245
 
 
##The Lookup Tagger
#A lot of high-frequency words do not have the NN tag. 
#Let's find the hundred most frequent words and store their most likely tag. 
#We can then use this information as the model for a "lookup tagger" (an NLTK UnigramTagger):

>>> fd = nltk.FreqDist(brown.words(categories='news'))
>>> cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news')) #condition=word, count=tag 
>>> most_freq_words = fd.most_common(100)
>>> likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
>>> baseline_tagger = nltk.UnigramTagger(model=likely_tags)
>>> baseline_tagger.evaluate(brown_tagged_sents)
0.45578495136941344
 
 
#Evaluete on untagged text 
>>> sent = brown.sents(categories='news')[3]
>>> baseline_tagger.tag(sent)
[('``', '``'), ('Only', None), ('a', 'AT'), ('relative', None),
('handful', None), ('of', 'IN'), ('such', None), ('reports', None),
('was', 'BEDZ'), ('received', None), ("''", "''"), (',', ','),
('the', 'AT'), ('jury', None), ('said', 'VBD'), (',', ','),
('``', '``'), ('considering', None), ('the', 'AT'), ('widespread', None),
('interest', None), ('in', 'IN'), ('the', 'AT'), ('election', None),
(',', ','), ('the', 'AT'), ('number', None), ('of', 'IN'),
('voters', None), ('and', 'CC'), ('the', 'AT'), ('size', None),
('of', 'IN'), ('this', 'DT'), ('city', None), ("''", "''"), ('.', '.')]
 
 
#Many words have been assigned a tag of None, 
#because they were not among the 100 most frequent words. 

#In these cases we would like to assign the default tag of NN. 
#In other words, we want to use the lookup table first, and if it is unable to assign a tag, 
#then use the default tagger, a process known as backoff 

>>> baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
 
 


##Unigram Tagging
#Unigram taggers are based on a simple statistical algorithm: 
#for each token, assign the tag that is most likely for that particular token. 

#For example, it will assign the tag JJ to any occurrence of the word frequent, 
#since frequent is used as an adjective (e.g. a frequent word) more often than it is used as a verb (e.g. I frequent this cafe). 

#A unigram tagger behaves just like a lookup tagger except there is a more convenient technique for setting it up, called training. 

>>> from nltk.corpus import brown
>>> brown_tagged_sents = brown.tagged_sents(categories='news')
>>> brown_sents = brown.sents(categories='news')
>>> unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
>>> unigram_tagger.tag(brown_sents[2007])
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'),
('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'),
(',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'),
('floor', 'NN'), ('so', 'QL'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'),
('direct', 'JJ'), ('.', '.')]
>>> unigram_tagger.evaluate(brown_tagged_sents)
0.9349006503968017
 
#Separating the Training and Testing Data and train 
 
 >>> size = int(len(brown_tagged_sents) * 0.9)
>>> size
4160
>>> train_sents = brown_tagged_sents[:size]
>>> test_sents = brown_tagged_sents[size:]
>>> unigram_tagger = nltk.UnigramTagger(train_sents)
>>> unigram_tagger.evaluate(test_sents)
0.811721...
 
 
##General N-Gram Tagging 
#The NgramTagger class uses a tagged training corpus to determine which part-of-speech tag is most likely for each context. 

#Here we see a special case of an n-gram tagger, namely a bigram tagger. 
 
>>> bigram_tagger = nltk.BigramTagger(train_sents)
>>> bigram_tagger.tag(brown_sents[2007])
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'),
('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'),
('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'),
('ground', 'NN'), ('floor', 'NN'), ('so', 'CS'), ('that', 'CS'),
('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
>>> unseen_sent = brown_sents[4203]
>>> bigram_tagger.tag(unseen_sent)
[('The', 'AT'), ('population', 'NN'), ('of', 'IN'), ('the', 'AT'), ('Congo', 'NP'),
('is', 'BEZ'), ('13.5', None), ('million', None), (',', None), ('divided', None),
('into', None), ('at', None), ('least', None), ('seven', None), ('major', None),
('``', None), ('culture', None), ('clusters', None), ("''", None), ('and', None),
('innumerable', None), ('tribes', None), ('speaking', None), ('400', None),
('separate', None), ('dialects', None), ('.', None)]
 
>>> bigram_tagger.evaluate(test_sents)
0.102063...
 
 
##Combining Taggers
1.Try tagging the token with the bigram tagger.
2.If the bigram tagger is unable to find a tag for the token, try the unigram tagger.
3.If the unigram tagger is also unable to find a tag, use a default tagger.

>>> t0 = nltk.DefaultTagger('NN')
>>> t1 = nltk.UnigramTagger(train_sents, backoff=t0)
>>> t2 = nltk.BigramTagger(train_sents, backoff=t1)
>>> t2.evaluate(test_sents)
0.844513...
 
 

##Storing Taggers

>>> from pickle import dump
>>> output = open('t2.pkl', 'wb')
>>> dump(t2, output, -1)
>>> output.close()
#load 
>> from pickle import load
>>> input = open('t2.pkl', 'rb')
>>> tagger = load(input)
>>> input.close()
 
#Usage  
>>> text = """The board's action shows what free enterprise
        is up against in our complex maze of regulatory laws ."""
>>> tokens = text.split()
>>> tagger.tag(tokens)
[('The', 'AT'), ("board's", 'NN$'), ('action', 'NN'), ('shows', 'NNS'),
('what', 'WDT'), ('free', 'JJ'), ('enterprise', 'NN'), ('is', 'BEZ'),
('up', 'RP'), ('against', 'IN'), ('in', 'IN'), ('our', 'PP$'), ('complex', 'JJ'),
('maze', 'NN'), ('of', 'IN'), ('regulatory', 'NN'), ('laws', 'NNS'), ('.', '.')]
 
 
##Performance Limitations
#What is the upper limit to the performance of an n-gram tagger? 

#Consider the case of a trigram tagger. 
#How many cases of part-of-speech ambiguity does it encounter? 

>>> cfd = nltk.ConditionalFreqDist(
            ((x[1], y[1], z[0]), z[1])
            for sent in brown_tagged_sents
            for x, y, z in nltk.trigrams(sent))
>>> ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
>>> sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()
0.049297702068029296
 
#Thus, one out of twenty trigrams is ambiguous 

#Another way to investigate the performance of a tagger is to study its mistakes. 
#Some tags may be harder than others to assign, 
#and it might be possible to treat them specially by pre- or post-processing the data. 

#A convenient way to look at tagging errors is the confusion matrix. 
#It charts expected tags (the gold standard) against actual tags generated by a tagger
#offdiagonal should be zero for best result 


class nltk.metrics.confusionmatrix.ConfusionMatrix(reference, test, sort_by_count=False)
    Bases: object
    The confusion matrix between a list of reference values 
    and a corresponding list of test values. 
     Note that the diagonal entries Ri=Tj of this matrix 
    corresponds to correct values; 
    and the off-diagonal entries correspond to incorrect values.
    
    Entry [r,t] of this matrix is a count of the number of times 
    that the reference value r corresponds to the test value t. E.g.:
    >>> from nltk.metrics import ConfusionMatrix
    >>> ref  = 'DET NN VB DET JJ NN NN IN DET NN'.split()
    >>> test = 'DET VB VB DET NN NN NN IN DET NN'.split()
    >>> cm = ConfusionMatrix(ref, test)
    >>> print(cm['NN', 'NN'])
    3
    key()
    pretty_format(show_percents=False, values_in_chart=True, truncate=None, sort_by_count=False)
        Returns:A multi-line string representation of this confusion matrix.
            truncate (int) – If specified, then only show the specified number of values. Any sorting (e.g., sort_by_count) will be performed before truncation.
            sort_by_count – If true, then sort by the count of each label in the reference data. I.e., labels that occur more frequently in the reference label will be towards the left edge of the matrix, and labels that occur less frequently will be towards the right edge.

  
#Example 
>>> test_tags = [tag for sent in brown.sents(categories='editorial')
                    for (word, tag) in t2.tag(sent)]
>>> gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
>>> print(nltk.ConfusionMatrix(gold_tags, test_tags))           
 
 
 
###chap-6
###NLTK -Learning to Classify Text 
 
##Supervised Classification
#Classification is the task of choosing the correct class label for a given input
#in multi-class classification, each instance may be assigned multiple labels; 

#in open-class classification, the set of labels is not defined in advance; 
#in sequence classification, a list of inputs are jointly classified.

#A classifier is called supervised if it is built based on training corpora 
#containing the correct label for each input

 
#Before training, extract Feature 

##Reference 

#Utility functions and classes for classifiers.


nltk.classify.util.accuracy(classifier, gold)

nltk.classify.util.apply_features(feature_func, toks, labeled=None)
    if labeled=False, then the returned list-like object’s values are equal to:
    [feature_func(tok) for tok in toks]
    If labeled=True, then the returned list-like object’s values are equal to:
    [(feature_func(tok), label) for (tok, label) in toks]

    these featuresets are constructed lazily, as-needed
        feature_func – The function that will be applied to each token. 
                        It should return a featureset – 
                        i.e., a dict mapping feature names to feature values.
        toks – The list of tokens to which feature_func should be applied. 
               If labeled=True, then the list elements will be passed directly to feature_func(). 
               If labeled=False, then the list elements should be tuples (tok,label), 
               and tok will be passed to feature_func().
        labeled – If true, then toks contains labeled tokens – 
                  i.e., tuples of the form (tok, label). 
                  (Default: auto-detect based on types.)

nltk.classify.util.attested_labels(tokens)
    Returns:	A list of all labels that are attested in the given list of tokens.
    Return type:	list of (immutable)
    Parameters:	tokens (list) – The list of classified tokens from which to extract labels. 
    A classified token has the form (token, label).

nltk.classify.util.log_likelihood(classifier, gold)

##Classifier Interface 
class nltk.classify.api.ClassifierI
    Bases: object
    A processing interface for labeling tokens with a single category label (or “class”). 
    Labels are typically strs or ints, but can be any immutable type. 
    The set of labels that the classifier chooses from must be fixed and finite.
    
    classify(featureset)
        featureset :a dict mapping feature names to feature values. 
        Returns:	the most appropriate label for the given featureset.
        Return type:	label

    classify_many(featuresets)
        Apply self.classify() to each element of featuresets. I.e.:
            return [self.classify(fs) for fs in featuresets]
        Return type:	list(label)

    labels()
        Returns:	the list of category labels used by this classifier.
        Return type:	list of (immutable)

    prob_classify(featureset)
        featureset :a dict mapping feature names to feature values. 
        Returns:	a probability distribution over labels for the given featureset.
        Return type:	ProbDistI

    prob_classify_many(featuresets)
        Apply self.prob_classify() to each element of featuresets. I.e.:
            return [self.prob_classify(fs) for fs in featuresets]
        Return type:	list(ProbDistI)

class nltk.classify.api.MultiClassifierI
    Bases: object
    A processing interface for labeling tokens with zero or more category labels (or “labels”). 

    classify(featureset)
        Returns:	the most appropriate set of labels for the given featureset.
        Return type:	set(label)

    classify_many(featuresets)
        Apply self.classify() to each element of featuresets. I.e.:
            return [self.classify(fs) for fs in featuresets]
        Return type:	list(set(label))

    labels()
        Returns:	the list of category labels used by this classifier.
        Return type:	list of (immutable)

    prob_classify(featureset)
        Returns:	a probability distribution over sets of labels for the given featureset.
        Return type:	ProbDistI
    prob_classify_many(featuresets)
        Apply self.prob_classify() to each element of featuresets. I.e.:
            return [self.prob_classify(fs) for fs in featuresets]
        Return type:	list(ProbDistI)

        
##NaiveBayesClassifier

class nltk.classify.naivebayes.NaiveBayesClassifier(label_probdist, feature_probdist)
    Bases: nltk.classify.api.ClassifierI
    
    A classifier based on the Naive Bayes algorithm.  In order to find the
    probability for a label, this algorithm first uses the Bayes rule to
    express P(label|features) in terms of P(label) and P(features|label):
                           P(label) * P(features|label)
      P(label|features) = ------------------------------
                                  P(features)
    The algorithm then makes the 'naive' assumption that all features are
    independent, given the label:
                           P(label) * P(f1|label) * ... * P(fn|label)
      P(label|features) = --------------------------------------------
                                             P(features)
    Rather than computing P(features) explicitly, the algorithm just
    calculates the numerator of P(label|features) for each label, 
    and normalizes them so they  sum to one:
                           P(label) * P(f1|label) * ... * P(fn|label)
      P(label|features) = --------------------------------------------
                            SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )

    
    label_probdist: P(label), the probability distribution over labels.  
    It is expressed as a ProbDistI whose samples are labels.  
    I.e., P(label) =   label_probdist.prob(label).

    feature_probdist: P(fname=fval|label), 
    the probability distribution for feature values, given labels.  
    It is  expressed as a dictionary whose keys are (label, fname) pairs 
    and whose values are ProbDistI objects over feature values.  
    I.e., P(fname=fval|label) =  feature_probdist[label,fname].prob(fval).  
    If a given (label,fname) is not a key in feature_probdist, 
    then it is assumed that the corresponding P(fname=fval|label) is 0 for all values of fval.
        
    classify(featureset)
        featureset :a dict mapping feature names to feature values. 
        
    labels()
    
    most_informative_features(n=100)
        Return a list of the ‘most informative’ features used by this classifier. 
        the informativeness of a feature (fname,fval) is equal 
        to the highest value of P(fname=fval|label), for any label, 
        divided by the lowest value of P(fname=fval|label), for any label:      
            max[ P(fname=fval|label1) / P(fname=fval|label2) ]

    prob_classify(featureset)
    
    show_most_informative_features(n=10)

    classmethod train(labeled_featuresets, estimator=<class 'nltk.probability.ELEProbDist'>)
        Parameters:	labeled_featuresets – A list of classified featuresets, 
        i.e., a list of tuples (featureset, label).
        
        
##Example - Names ending in a, e and i are likely to be female, 
#while names ending in k, o, r, s and t are likely to be male. 

#Let's build a classifier to model these differences more precisely.

#Exatract feature ie last alpha 
>>> def gender_features(word):
    return {'last_letter': word[-1]}
>>> gender_features('Shrek')
{'last_letter': 'k'}
 
 
#Get Training data 
#featuresets = list of tuple(features  , label)
#features in nltk is a dict { feature_name1: value, feature_name2:value }
#all tuples should contain the same feature_names

>>> from nltk.corpus import names
>>> labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
            [(name, 'female') for name in names.words('female.txt')])
>>> import random
>>> random.shuffle(labeled_names)
#Split train and test  
>>> featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
>>> train_set, test_set = featuresets[500:], featuresets[:500]
>>> classifier = nltk.NaiveBayesClassifier.train(train_set)
 
#let's just test it out on some names that did not appear in its training data:
>>> classifier.classify(gender_features('Neo'))
'male'
>>> classifier.classify(gender_features('Trinity'))
'female'
 
#Evaluate 
>>> print(nltk.classify.accuracy(classifier, test_set))
0.77
 
 
#Finally, we can examine the classifier to determine 
#which features it found most effective for distinguishing the names' genders:

 >>> classifier.show_most_informative_features(5)
Most Informative Features
             last_letter = 'a'            female : male   =     33.2 : 1.0
             last_letter = 'k'              male : female =     32.6 : 1.0
             last_letter = 'p'              male : female =     19.7 : 1.0
             last_letter = 'v'              male : female =     18.6 : 1.0
             last_letter = 'f'              male : female =     17.3 : 1.0
 
 
#This listing shows that the names in the training set that end in "a" are female 33 times more often than they are male, 
#but names that end in "k" are male 32 times more often than they are female. 
#These ratios are known as likelihood ratios

 
 
##When working with large corpora, 
#constructing a single list that contains the features of every instance 
#can use up a large amount of memory. 

#In these cases, use the function nltk.classify.apply_features, 
#which returns an object that acts like a list 
#but does not store all the feature sets in memory:

>>> from nltk.classify import apply_features
>>> train_set = apply_features(gender_features, labeled_names[500:],True)
>>> test_set = apply_features(gender_features, labeled_names[:500],True)
 
 
 

##Document Classification

#In NLTK, we have examples of corpora where documents have been labeled with categories. 
#Using these corpora, we can build classifiers that will automatically tag 
#new documents with appropriate category labels

#For example - Movie Reviews Corpus, which categorizes each review as positive or negative(['neg', 'pos'])

#document is list of tuples, [(words_list, category),...]
#Note .words(fileid) gives words from fileid, .words() gives all worlds in corpus 
#movie_reviews.categories() => ['neg', 'pos']
>>> from nltk.corpus import movie_reviews
>>> documents = [(list(movie_reviews.words(fileid)), category)
                    for category in movie_reviews.categories()
                    for fileid in movie_reviews.fileids(category)]
>>> random.shuffle(documents)
 
 
#Create smaller subset as features 
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words()) #all_words= FreqDist
word_features = list(all_words)[:2000] #forcing list on FreqDist returns sorted list with most frequent to least frequent 

#Feature extract
#feature is wheather document contains each word of word_features or not 
def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
 
 

>>> print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 
{'contains(waste)': False, 'contains(lot)': False, ...}
 
 
#Get featureset and train 
#featuresets = list of tuple(features  , label)
#features in nltk is a dict { feature_name1: value, feature_name2:value }
#all tuples should contain the same feature_names
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
 
 

>>> print(nltk.classify.accuracy(classifier, test_set)) 
0.81
>>> classifier.show_most_informative_features(5) 
Most Informative Features
   contains(outstanding) = True              pos : neg    =     11.1 : 1.0
        contains(seagal) = True              neg : pos    =      7.7 : 1.0
   contains(wonderfully) = True              pos : neg    =      6.8 : 1.0
         contains(damon) = True              pos : neg    =      5.9 : 1.0
        contains(wasted) = True              neg : pos    =      5.8 : 1.0
 
 
# a review that mentions "Seagal" is almost 8 times more likely to be negative than positive, 
#while a review that mentions "Damon" is about 6 times more likely to be positive.



##Part-of-Speech Tagging with help of classifiers

class nltk.classify.decisiontree.DecisionTreeClassifier(label, feature_name=None, decisions=None, default=None)
    Bases: nltk.classify.api.ClassifierI
    A classifier model that decides which label to assign to a token 
    on the basis of a tree structure, where branches correspond to conditions 
    on feature values, and leaves correspond to label assignments.

    static best_binary_stump(feature_names, labeled_featuresets, feature_values, verbose=False)

    static best_stump(feature_names, labeled_featuresets, verbose=False)

    static binary_stump(feature_name, feature_value, labeled_featuresets)

    classify(featureset)

    error(labeled_featuresets)

    labels()

    static leaf(labeled_featuresets)

    pretty_format(width=70, prefix='', depth=4)
        Return a string containing a pretty-printed version of this decision tree. Each line in this string corresponds to a single decision tree node or leaf, and indentation is used to display the structure of the decision tree.

    pseudocode(prefix='', depth=4)
        Return a string representation of this decision tree 
        that expresses the decisions it makes as a nested set of pseudocode 
        if statements.

    refine(labeled_featuresets, entropy_cutoff, depth_cutoff, support_cutoff, binary=False, feature_values=None, verbose=False)

    static stump(feature_name, labeled_featuresets)

    static train(labeled_featuresets, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10, binary=False, feature_values=None, verbose=False)
        binary – If true, then treat all feature/value pairs as individual binary features, 
        rather than using a single n-way branch for each feature.

    unicode_repr
        Return repr(self).


        
 
#the most common suffixes are:
>>> from nltk.corpus import brown
>>> suffix_fdist = nltk.FreqDist()
>>> for word in brown.words():
        word = word.lower()
        suffix_fdist[word[-1:]] += 1
        suffix_fdist[word[-2:]] += 1
        suffix_fdist[word[-3:]] += 1
 
 
>>> common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
>>> print(common_suffixes)
['e', ',', '.', 's', 'd', 't', 'he', 'n', 'a', 'of', 'the',
 'y', 'r', 'to', 'in', 'f', 'o', 'ed', 'nd', 'is', 'on', 'l',
 'g', 'and', 'ng', 'er', 'as', 'ing', 'h', 'at', 'es', 'or',
 're', 'it', '', 'an', "''", 'm', ';', 'i', 'ly', 'ion', ...]
 
 
 
#feature extractor function which checks a given word for these suffixes:
#featuresets = list of tuple(features  , label)
#features in nltk is a dict { feature_name1: value, feature_name2:value }
#all tuples should contain the same feature_names
>>> def pos_features(word):
        features = {}
        for suffix in common_suffixes:
            features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
        return features
 
 
#Train - label is part of speech  
>>> tagged_words = brown.tagged_words(categories='news') #list of (word, tag)
>>> featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
 
>>> size = int(len(featuresets) * 0.1)
>>> train_set, test_set = featuresets[size:], featuresets[:size]
 
>>> classifier = nltk.DecisionTreeClassifier.train(train_set)
>>> nltk.classify.accuracy(classifier, test_set)
0.62705121829935351
 
>>> classifier.classify(pos_features('cats'))
'NNS'
 
 
 
#One nice feature of decision tree models is that they are often fairly easy to interpret 
#we can even instruct NLTK to print them out as pseudocode:

>>> print(classifier.pseudocode(depth=4))
if endswith(,) == True: return ','
if endswith(,) == False:
  if endswith(the) == True: return 'AT'
  if endswith(the) == False:
    if endswith(s) == True:
      if endswith(is) == True: return 'BEZ'
      if endswith(is) == False: return 'VBZ'
    if endswith(s) == False:
      if endswith(.) == True: return '.'
      if endswith(.) == False: return 'NN'
 
 
##Creating The Test Set in classification 
>>> import random
>>> from nltk.corpus import brown
>>> tagged_sents = list(brown.tagged_sents(categories='news'))
>>> random.shuffle(tagged_sents)
>>> size = int(len(tagged_sents) * 0.1)
>>> train_set, test_set = tagged_sents[size:], tagged_sents[:size]

#Or better - training set and test set are taken from different documents:
>>> file_ids = brown.fileids(categories='news')
>>> size = int(len(file_ids) * 0.1)
>>> train_set = brown.tagged_sents(file_ids[size:])
>>> test_set = brown.tagged_sents(file_ids[:size])
 
 
 
 
##Evaluation in classification
##Accuracy
#measures the percentage of inputs in the test set that the classifier correctly labeled. 

#For example, a name gender classifier that predicts the correct name 60 times 
#in a test set containing 80 names would have an accuracy of 60/80 = 75%. 

#The function nltk.classify.accuracy() will calculate the accuracy of a classifier model on a given test set:

>>> classifier = nltk.NaiveBayesClassifier.train(train_set) 
>>> print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 
0.75
 
##Precision and Recall
•True positives are relevant items that we correctly identified as relevant.
•True negatives are irrelevant items that we correctly identified as irrelevant.
•False positives (or Type I errors) are irrelevant items that we incorrectly identified as relevant.
•False negatives (or Type II errors) are relevant items that we incorrectly identified as irrelevant.

#Given these four numbers, we can define the following metrics:
•Precision, which indicates how many of the items that we identified were relevant, is TP/(TP+FP).
•Recall, which indicates how many of the relevant items that we identified, is TP/(TP+FN).
•The F-Measure (or F-Score), which combines the precision and recall to give a single score, is defined to be the harmonic mean of the precision and recall: (2 × Precision × Recall) / (Precision + Recall).


#A confusion matrix is a table where each cell [i,j] indicates 
#how often label j was predicted when the correct label was i. 
#Thus, the diagonal entries (i.e., cells |ii|) indicate labels that were correctly predicted, 
#and the off-diagonal entries indicate errors. 

class nltk.metrics.confusionmatrix.ConfusionMatrix(reference, test, sort_by_count=False)
    Bases: object
    The confusion matrix between a list of reference values 
    and a corresponding list of test values. 
     Note that the diagonal entries Ri=Tj of this matrix 
    corresponds to correct values; 
    and the off-diagonal entries correspond to incorrect values.
    
    Entry [r,t] of this matrix is a count of the number of times 
    that the reference value r corresponds to the test value t. E.g.:
    >>> from nltk.metrics import ConfusionMatrix
    >>> ref  = 'DET NN VB DET JJ NN NN IN DET NN'.split()
    >>> test = 'DET VB VB DET NN NN NN IN DET NN'.split()
    >>> cm = ConfusionMatrix(ref, test)
    >>> print(cm['NN', 'NN'])
    3
    key()
    pretty_format(show_percents=False, values_in_chart=True, truncate=None, sort_by_count=False)
        Returns:A multi-line string representation of this confusion matrix.
            truncate (int) – If specified, then only show the specified number of values. Any sorting (e.g., sort_by_count) will be performed before truncation.
            sort_by_count – If true, then sort by the count of each label in the reference data. I.e., labels that occur more frequently in the reference label will be towards the left edge of the matrix, and labels that occur less frequently will be towards the right edge.


#Example - we generate a confusion matrix for the bigram tagger developed earlier 

>>> def tag_list(tagged_sents):
        return [tag for sent in tagged_sents for (word, tag) in sent]
>>> def apply_tagger(tagger, corpus):
.       return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
>>> gold = tag_list(brown.tagged_sents(categories='editorial'))
>>> test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
>>> cm = nltk.ConfusionMatrix(gold, test)
>>> print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    |                                         N                      |
    |      N      I      A      J             N             V      N |
    |      N      N      T      J      .      S      ,      B      P |
----+----------------------------------------------------------------+
 NN | <11.8%>  0.0%      .   0.2%      .   0.0%      .   0.3%   0.0% |
 IN |   0.0%  <9.0%>     .      .      .   0.0%      .      .      . |
 AT |      .      .  <8.6%>     .      .      .      .      .      . |
 JJ |   1.7%      .      .  <3.9%>     .      .      .   0.0%   0.0% |
  . |      .      .      .      .  <4.8%>     .      .      .      . |
NNS |   1.5%      .      .      .      .  <3.2%>     .      .   0.0% |
  , |      .      .      .      .      .      .  <4.4%>     .      . |
 VB |   0.9%      .      .   0.0%      .      .      .  <2.4%>     . |
 NP |   1.0%      .      .   0.0%      .      .      .      .  <1.8%>|
----+----------------------------------------------------------------+
(row = reference; col = test)

 
###NLTK - Various Classifications 
1.decision trees, 
2.naive Bayes classifiers, 
3.Maximum Entropy classifiers.


##Maximum Entropy
class nltk.classify.maxent.MaxentClassifier(encoding, weights, logarithmic=True)
    Bases: nltk.classify.api.ClassifierI
    
    A maximum entropy classifier (also known as a “conditional exponential classifier”). 
    This classifier is parameterized by a set of “weights”, 
    which are used to combine the joint-features that are generated 
    from a featureset by an “encoding”. 
    
    In particular, the encoding maps each (featureset, label) pair to a vector. 
    The probability of each label is then computed using the following equation:
                              dotprod(weights, encode(fs,label))
    prob(fs|label) = ---------------------------------------------------
                     sum(dotprod(weights, encode(fs,l)) for l in labels)
    Where dotprod is the dot product:
    dotprod(a,b) = sum(x*y for (x,y) in zip(a,b))

    ALGORITHMS = ['GIS', 'IIS', 'MEGAM', 'TADM']
        A list of the algorithm names that are accepted for the train() method’s 
        algorithm parameter.

    classify(featureset)

    explain(featureset, columns=4)
        Print a table showing the effect of each of the features 
        in the given feature set, 
        and how they combine to determine the probabilities of each label for that featureset.

    labels()

    prob_classify(featureset)

    set_weights(new_weights)
        Set the feature weight vector for this classifier. :param new_weights: The new feature weight vector. :type new_weights: list of float

    show_most_informative_features(n=10, show='all')
        Parameters:	show – all, neg, or pos (for negative-only or positive-only)

    classmethod train(train_toks, algorithm=None, trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, **cutoffs)
        Train a new maxent classifier based on the given corpus of training samples. 
        This classifier will have its weights chosen to maximize entropy 
        while remaining empirically consistent with the training corpus.
            train_toks (list) – Training data, represented as a list of pairs, 
                                the first member of which is a featureset, 
                                and the second of which is a classification label.
            algorithm (str) – string from ALGORITHMS
                              The default algorithm is 'IIS'.
            trace (int) – The level of diagnostic tracing output to produce. 
                          Higher values produce more verbose output.
            encoding (MaxentFeatureEncodingI) – A feature encoding, used to convert featuresets into feature vectors. 
                                                If none is specified, then a BinaryMaxentFeatureEncoding will be built based on the features that are attested in the training corpus.
            labels (list(str)) – The set of possible labels. If none is given, then the set of all labels attested in the training data will be used instead.
            gaussian_prior_sigma – The sigma value for a gaussian prior on model weights. Currently, this is supported by megam. For other algorithms, its value is ignored.
            cutoffs – Arguments specifying various conditions under which the training should be halted. (Some of the cutoff conditions are not supported by some algorithms.)
                max_iter=v: Terminate after v iterations.
                min_ll=v: Terminate after the negative average log-likelihood drops under v.
                min_lldelta=v: Terminate if a single iteration improves log likelihood by less than v.

    unicode_repr()

    weights()
        Returns:	The feature weight vector for this classifier.
        Return type:	list of float
        
        
class nltk.classify.maxent.MaxentFeatureEncodingI[source]
    Bases: object
    A mapping that converts a set of input feature values (list of tuple(features  , label))
    to a vector of joint-feature values, given a label, that can be used by maximum entropy models.       
        
        
        
class nltk.classify.maxent.TypedMaxentFeatureEncoding(labels, mapping, unseen_features=False, alwayson_features=False)[source]
    Bases: nltk.classify.maxent.MaxentFeatureEncodingI
    
class nltk.classify.maxent.BinaryMaxentFeatureEncoding(labels, mapping, unseen_features=False, alwayson_features=False)[source]¶
    Bases: nltk.classify.maxent.MaxentFeatureEncodingI
    
class nltk.classify.maxent.FunctionBackedMaxentFeatureEncoding(func, length, labels)[source]
    Bases: nltk.classify.maxent.MaxentFeatureEncodingI    
    
    
    
##Calculation of  entropy 
import math
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)
 
 

>>> print(entropy(['male', 'male', 'male', 'male'])) 
0.0
>>> print(entropy(['male', 'female', 'male', 'male']))
0.811...
>>> print(entropy(['female', 'male', 'female', 'male']))
1.0
>>> print(entropy(['female', 'female', 'male', 'female']))
0.811...
>>> print(entropy(['female', 'female', 'female', 'female'])) 
0.0
 
 


##Diff between Generative(naive Bayes classifier) vs Conditional Classifiers(Maximum Entropy classifier)
 
#The naive Bayes classifier is an example of a generative classifier, 
#which builds a model that predicts P(input, label), 
#the joint probability of a (input, label) pair. 

#As a result, generative models can be used to answer the following questions:
1.What is the most likely label for a given input?
2.How likely is a given label for a given input?
3.What is the most likely input value?
4.How likely is a given input value?
5.How likely is a given input value with a given label?
6.What is the most likely label for an input that might have one of two values (but we don't know which)?
 
#The Maximum Entropy classifier, on the other hand, is an example of a conditional classifier. 
#Conditional classifiers build models that predict P(label|input) 
#— the probability of a label given the input value. 

#Thus, conditional models can still be used to answer questions 1 and 2. 
#However, conditional models can not be used to answer the remaining questions 3-6.

#In general, generative models are strictly more powerful than conditional models, 
#since we can calculate the conditional probability P(label|input) 
#from the joint probability P(input, label), but not vice versa. 

#However, this additional power comes at a price. 
#Because the model is more powerful, it has more "free parameters" which need to be learned. 

#However, the size of the training set is fixed. 
#Thus, when using a more powerful model, 
#we end up with less data that can be used to train each parameter's value, 
#making it harder to find the best parameter values. 

#As a result, a generative model may not do as good a job at answering questions 1 and 2 as a conditional model, 
#since the conditional model can focus its efforts on those two questions. 




###Details of classifications 

#labels are represented with strings (such as "health" or "sports".)
#A processing interface for labeling tokens with zero or more category labels (or “labels”). 
#Labels are typically strs or ints, but can be any immutable type


#featuresets = list of tuple(features  , label)
#features in nltk is a dict { feature_name1: value, feature_name2:value }
#all tuples should contain the same feature_names


#In NLTK, classifiers are defined using classes that implement the ClassifyI interface:

>>> import nltk
>>> nltk.usage(nltk.classify.ClassifierI)
ClassifierI supports the following operations:
  - self.classify(featureset)   #predict 
  - self.classify_many(featuresets)
  - self.labels()
  - self.prob_classify(featureset)
  - self.prob_classify_many(featuresets)


#NLTK defines several classifier classes:
•ConditionalExponentialClassifier
•DecisionTreeClassifier
•MaxentClassifier   #maximum entropy 
•NaiveBayesClassifier
•WekaClassifier

#Classifiers are typically created by training them on a training corpus.


#Example - We define a very simple training corpus with 3 binary features: ['a', 'b', 'c'], 
#and are two labels: ['x', 'y']. 

>>> train = [
        (dict(a=1,b=1,c=1), 'y'),
        (dict(a=1,b=1,c=1), 'x'),
        (dict(a=1,b=1,c=0), 'y'),
        (dict(a=0,b=1,c=1), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        (dict(a=0,b=0,c=1), 'y'),
        (dict(a=0,b=1,c=0), 'x'),
        (dict(a=0,b=0,c=0), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        ]
>>> test = [
        (dict(a=1,b=0,c=1)), # unseen
        (dict(a=1,b=0,c=0)), # unseen
        (dict(a=0,b=1,c=1)), # seen 3 times, labels=y,y,x
        (dict(a=0,b=1,c=0)), # seen 1 time, label=x
        ]


##Test the Naive Bayes classifier(actually a multiclass)

>>> classifier = nltk.classify.NaiveBayesClassifier.train(train)
>>> sorted(classifier.labels())
['x', 'y']
>>> classifier.classify_many(test)
['y', 'x', 'y', 'x']
>>> for pdist in classifier.prob_classify_many(test):
    print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))
0.3203 0.6797
0.5857 0.4143
0.3792 0.6208
0.6470 0.3530
>>> classifier.show_most_informative_features()
Most Informative Features
                       c = 0                   x : y      =      2.0 : 1.0
                       c = 1                   y : x      =      1.5 : 1.0
                       a = 1                   y : x      =      1.4 : 1.0
                       b = 0                   x : y      =      1.2 : 1.0
                       a = 0                   x : y      =      1.2 : 1.0
                       b = 1                   y : x      =      1.1 : 1.0


##For quick binary class, can use nltk.classify.positivenaivebayes module

>>> from nltk.classify import PositiveNaiveBayesClassifier


#Some sentences about sports:


>>> sports_sentences = [ 'The team dominated the game',
                      'They lost the ball',
                      'The game was intense',
                      'The goalkeeper catched the ball',
                      'The other team controlled the ball' ]


#Mixed topics, including sports:


>>> various_sentences = [ 'The President did not comment',
                       'I lost the keys',
                       'The team won the game',
                       'Sara has two kids',
                       'The ball went off the court',
                       'They had the ball for the whole game',
                       'The show is over' ]


#The features of a sentence are simply the words it contains:


>>> def features(sentence):
        words = sentence.lower().split()
        return dict(('contains(%s)' % w, True) for w in words)


#We use the sports sentences as positive examples, 
#the mixed ones ad unlabeled examples:


>>> positive_featuresets = list(map(features, sports_sentences))
>>> unlabeled_featuresets = list(map(features, various_sentences))
>>> classifier = PositiveNaiveBayesClassifier.train(positive_featuresets,
                                unlabeled_featuresets)


#Is the following sentence about sports?
>>> classifier.classify(features('The cat is on the table'))
False

#What about this one?
>>> classifier.classify(features('My team lost the game'))
True

                     
                       
                       
##Test the Decision Tree classifier:

>>> classifier = nltk.classify.DecisionTreeClassifier.train(
        train, entropy_cutoff=0, support_cutoff=0)
>>> sorted(classifier.labels())
['x', 'y']
>>> print(classifier)
c=0? .................................................. x
  a=0? ................................................ x
  a=1? ................................................ y
c=1? .................................................. y
<BLANKLINE>
>>> classifier.classify_many(test)
['y', 'y', 'y', 'x']
>>> for pdist in classifier.prob_classify_many(test):
...     print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))
Traceback (most recent call last):
  . . .
NotImplementedError


##Test SklearnClassifier, which requires the scikit-learn package.

>>> from nltk.classify import SklearnClassifier
>>> from sklearn.naive_bayes import BernoulliNB
>>> from sklearn.svm import SVC
>>> train_data = [({"a": 4, "b": 1, "c": 0}, "ham"),
                    ({"a": 5, "b": 2, "c": 1}, "ham"),
                    ({"a": 0, "b": 3, "c": 4}, "spam"),
                    ({"a": 5, "b": 1, "c": 1}, "ham"),
                    ({"a": 1, "b": 4, "c": 3}, "spam")]
>>> classif = SklearnClassifier(BernoulliNB()).train(train_data)
>>> test_data = [{"a": 3, "b": 2, "c": 1},
                {"a": 0, "b": 3, "c": 7}]
>>> classif.classify_many(test_data)
['ham', 'spam']
>>> classif = SklearnClassifier(SVC(), sparse=False).train(train_data)
>>> classif.classify_many(test_data)
['ham', 'spam']

##Other exmaples 
>>> from sklearn.svm import LinearSVC
>>> from nltk.classify.scikitlearn import SklearnClassifier
>>> classif = SklearnClassifier(LinearSVC())

#or with pipeline 

>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.pipeline import Pipeline
>>> pipeline = Pipeline([('tfidf', TfidfTransformer()),
                      ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
>>> classif = SklearnClassifier(pipeline)



##Test the Maximum Entropy classifier training algorithms; they should all generate the same results.

>>> def print_maxent_test_header():
            print(' '*11+''.join(['      test[%s]  ' % i
                                for i in range(len(test))]))
            print(' '*11+'     p(x)  p(y)'*len(test))
            print('-'*(11+15*len(test)))

>>> def test_maxent(algorithm):
            print('%11s' % algorithm, end=' ')
            try:
                classifier = nltk.classify.MaxentClassifier.train(
                                train, algorithm, trace=0, max_iter=1000)
            except Exception as e:
                print('Error: %r' % e)
                return
        
            for featureset in test:
                pdist = classifier.prob_classify(featureset)
                print('%8.2f%6.2f' % (pdist.prob('x'), pdist.prob('y')), end=' ')
            print()

>>> print_maxent_test_header(); test_maxent('GIS'); test_maxent('IIS')
                 test[0]        test[1]        test[2]        test[3]
                p(x)  p(y)     p(x)  p(y)     p(x)  p(y)     p(x)  p(y)
-----------------------------------------------------------------------
        GIS     0.16  0.84     0.46  0.54     0.41  0.59     0.76  0.24
        IIS     0.16  0.84     0.46  0.54     0.41  0.59     0.76  0.24

>>> test_maxent('MEGAM'); test_maxent('TADM') # doctest: +SKIP
        MEGAM   0.16  0.84     0.46  0.54     0.41  0.59     0.76  0.24
        TADM    0.16  0.84     0.46  0.54     0.41  0.59     0.76  0.24



##tests for TypedMaxentFeatureEncoding

>>> from nltk.classify import maxent
>>> train = [
            ({'a': 1, 'b': 1, 'c': 1}, 'y'),
            ({'a': 5, 'b': 5, 'c': 5}, 'x'),
            ({'a': 0.9, 'b': 0.9, 'c': 0.9}, 'y'),
            ({'a': 5.5, 'b': 5.4, 'c': 5.3}, 'x'),
            ({'a': 0.8, 'b': 1.2, 'c': 1}, 'y'),
            ({'a': 5.1, 'b': 4.9, 'c': 5.2}, 'x')
        ]

>>> test = [
        {'a': 1, 'b': 0.8, 'c': 1.2},
        {'a': 5.2, 'b': 5.1, 'c': 5}
    ]

>>> encoding = maxent.TypedMaxentFeatureEncoding.train(
        train, count_cutoff=3, alwayson_features=True)

>>> classifier = maxent.MaxentClassifier.train(
        train, bernoulli=False, encoding=encoding, trace=0)

>>> classifier.classify_many(test)
['y', 'x']



##MultiClassifierI is a standard interface for “multi-category classification”, 
#which is like single-category classification 
#except that each text belongs to zero or more categories

#A processing interface for labeling tokens with zero or more category labels (or “labels”). 
#Labels are typically strs or ints, but can be any immutable type
#The set of labels that the multi-classifier chooses from must be fixed and finite.

>>> nltk.usage(nltk.classify.MultiClassifierI)
MultiClassifierI supports the following operations:
  - self.classify(featureset)
  - self.classify_many(featuresets)
  - self.labels()
  - self.prob_classify(featureset)
  - self.prob_classify_many(featuresets)
  
#There is no derived class of MultiClassifierI
# But nltk.NaiveBayesClassifier() is a out-of-box multi-class classifier

#Note binary class  - label x, y
>>> train = [
        (dict(a=1,b=1,c=1), 'y'),
        (dict(a=1,b=1,c=1), 'x'),
        (dict(a=1,b=1,c=0), 'y'),
        (dict(a=0,b=1,c=1), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        (dict(a=0,b=0,c=1), 'y'),
        (dict(a=0,b=1,c=0), 'x'),
        (dict(a=0,b=0,c=0), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        ]
        
#multiclass, label, x,y,z 
>>> train = [
        (dict(a=1,b=1,c=1), 'y'),
        (dict(a=1,b=1,c=1), 'x'),
        (dict(a=1,b=1,c=0), 'z'),
        (dict(a=0,b=1,c=1), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        (dict(a=0,b=0,c=1), 'z'),
        (dict(a=0,b=1,c=0), 'x'),
        (dict(a=0,b=0,c=0), 'x'),
        (dict(a=0,b=1,c=1), 'z'),
        ]
        
#multilabel multiclass, label - ('x',y','z') 
##not possible with nltk?? check again , Use scikit 
>>> train = [
        (dict(a=1,b=1,c=1), 'y'),
        (dict(a=1,b=1,c=1), 'x'),
        (dict(a=1,b=1,c=0), 'y'),
        (dict(a=0,b=1,c=1), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        (dict(a=0,b=0,c=1), 'y'),
        (dict(a=0,b=1,c=0), 'x'),
        (dict(a=0,b=0,c=0), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        ]







###NLTK -  Metrics
  
>>> from __future__ import print_function
>>> from nltk.metrics import *


##Accuracy
#measures the percentage of inputs in the test set that the classifier correctly labeled. 

#For example, a name gender classifier that predicts the correct name 60 times 
#in a test set containing 80 names would have an accuracy of 60/80 = 75%. 

#The function nltk.classify.accuracy() will calculate the accuracy of a classifier model on a given test set:

>>> classifier = nltk.NaiveBayesClassifier.train(train_set) 
>>> print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 
0.75
 
##Precision and Recall
•True positives are relevant items that we correctly identified as relevant.
•True negatives are irrelevant items that we correctly identified as irrelevant.
•False positives (or Type I errors) are irrelevant items that we incorrectly identified as relevant.
•False negatives (or Type II errors) are relevant items that we incorrectly identified as irrelevant.

#Given these four numbers, we can define the following metrics:
•Precision, which indicates how many of the items that we identified were relevant, is TP/(TP+FP).
•Recall, which indicates how many of the relevant items that we identified, is TP/(TP+FN).
•The F-Measure (or F-Score), which combines the precision and recall to give a single score, is defined to be the harmonic mean of the precision and recall: (2 × Precision × Recall) / (Precision + Recall).

#When performing classification tasks with three or more labels, 
#it can be informative to subdivide the errors made by the model based 
#on which types of mistake it made. 

#A confusion matrix is a table where each cell [i,j] indicates 
#how often label j was predicted when the correct label was i. 
#Thus, the diagonal entries (i.e., cells |ii|) indicate labels that were correctly predicted, 
#and the off-diagonal entries indicate errors. 


#Standard IR Scores

# to test the performance of taggers, chunkers

>>> reference = 'DET NN VB DET JJ NN NN IN DET NN'.split()
>>> test    = 'DET VB VB DET NN NN NN IN DET NN'.split()
>>> print(accuracy(reference, test))
0.8


#The following measures apply to sets:

>>> reference_set = set(reference)
>>> test_set = set(test)
>>> precision(reference_set, test_set)
1.0
>>> print(recall(reference_set, test_set))
0.8
>>> print(f_measure(reference_set, test_set))
0.88888888888...


#Measuring the likelihood of the data, given probability distributions:

>>> from nltk import FreqDist, MLEProbDist
>>> pdist1 = MLEProbDist(FreqDist("aldjfalskfjaldsf"))
>>> pdist2 = MLEProbDist(FreqDist("aldjfalssjjlldss"))
>>> print(log_likelihood(['a', 'd'], [pdist1, pdist2]))
-2.7075187496...



#Distance Metrics
#String edit distance (Levenshtein):

>>> edit_distance("rain", "shine")
3


#Other distance measures:

>>> s1 = set([1,2,3,4])
>>> s2 = set([3,4,5])
>>> binary_distance(s1, s2)
1.0
>>> print(jaccard_distance(s1, s2))
0.6
>>> print(masi_distance(s1, s2))
0.868...



#Miscellaneous Measures
#Rank Correlation works with two dictionaries mapping keys to ranks. 
#The dictionaries should have the same set of keys.

>>> spearman_correlation({'e':1, 't':2, 'a':3}, {'e':1, 'a':2, 't':3})
0.5

#Windowdiff uses a sliding window in comparing two segmentations of the same input (e.g. tokenizations, chunkings). Segmentations are represented using strings of zeros and ones.

>>> s1 = "000100000010"
>>> s2 = "000010000100"
>>> s3 = "100000010000"
>>> s4 = "000000000000"
>>> s5 = "111111111111"
>>> windowdiff(s1, s1, 3)
0.0
>>> abs(windowdiff(s1, s2, 3) - 0.3)  < 1e-6  # windowdiff(s1, s2, 3) == 0.3
True
>>> abs(windowdiff(s2, s3, 3) - 0.8)  < 1e-6  # windowdiff(s2, s3, 3) == 0.8
True
>>> windowdiff(s1, s4, 3)
0.5
>>> windowdiff(s1, s5, 3)
1.0



#Confusion Matrix

>>> reference = 'This is the reference data.  Testing 123.  aoaeoeoe'
>>> test =      'Thos iz_the rifirenci data.  Testeng 123.  aoaeoeoe'
>>> print(ConfusionMatrix(reference, test))
  |   . 1 2 3 T _ a c d e f g h i n o r s t z |
--+-------------------------------------------+
  |<8>. . . . . 1 . . . . . . . . . . . . . . |
. | .<2>. . . . . . . . . . . . . . . . . . . |
1 | . .<1>. . . . . . . . . . . . . . . . . . |
2 | . . .<1>. . . . . . . . . . . . . . . . . |
3 | . . . .<1>. . . . . . . . . . . . . . . . |
T | . . . . .<2>. . . . . . . . . . . . . . . |
_ | . . . . . .<.>. . . . . . . . . . . . . . |
a | . . . . . . .<4>. . . . . . . . . . . . . |
c | . . . . . . . .<1>. . . . . . . . . . . . |
d | . . . . . . . . .<1>. . . . . . . . . . . |
e | . . . . . . . . . .<6>. . . 3 . . . . . . |
f | . . . . . . . . . . .<1>. . . . . . . . . |
g | . . . . . . . . . . . .<1>. . . . . . . . |
h | . . . . . . . . . . . . .<2>. . . . . . . |
i | . . . . . . . . . . 1 . . .<1>. 1 . . . . |
n | . . . . . . . . . . . . . . .<2>. . . . . |
o | . . . . . . . . . . . . . . . .<3>. . . . |
r | . . . . . . . . . . . . . . . . .<2>. . . |
s | . . . . . . . . . . . . . . . . . .<2>. 1 |
t | . . . . . . . . . . . . . . . . . . .<3>. |
z | . . . . . . . . . . . . . . . . . . . .<.>|
--+-------------------------------------------+
(row = reference; col = test)
<BLANKLINE>

>>> cm = ConfusionMatrix(reference, test)
>>> print(cm.pretty_format(sort_by_count=True))
  |   e a i o s t . T h n r 1 2 3 c d f g _ z |
--+-------------------------------------------+
  |<8>. . . . . . . . . . . . . . . . . . 1 . |
e | .<6>. 3 . . . . . . . . . . . . . . . . . |
a | . .<4>. . . . . . . . . . . . . . . . . . |
i | . 1 .<1>1 . . . . . . . . . . . . . . . . |
o | . . . .<3>. . . . . . . . . . . . . . . . |
s | . . . . .<2>. . . . . . . . . . . . . . 1 |
t | . . . . . .<3>. . . . . . . . . . . . . . |
. | . . . . . . .<2>. . . . . . . . . . . . . |
T | . . . . . . . .<2>. . . . . . . . . . . . |
h | . . . . . . . . .<2>. . . . . . . . . . . |
n | . . . . . . . . . .<2>. . . . . . . . . . |
r | . . . . . . . . . . .<2>. . . . . . . . . |
1 | . . . . . . . . . . . .<1>. . . . . . . . |
2 | . . . . . . . . . . . . .<1>. . . . . . . |
3 | . . . . . . . . . . . . . .<1>. . . . . . |
c | . . . . . . . . . . . . . . .<1>. . . . . |
d | . . . . . . . . . . . . . . . .<1>. . . . |
f | . . . . . . . . . . . . . . . . .<1>. . . |
g | . . . . . . . . . . . . . . . . . .<1>. . |
_ | . . . . . . . . . . . . . . . . . . .<.>. |
z | . . . . . . . . . . . . . . . . . . . .<.>|
--+-------------------------------------------+
(row = reference; col = test)
<BLANKLINE>

>>> print(cm.pretty_format(sort_by_count=True, truncate=10))
  |   e a i o s t . T h |
--+---------------------+
  |<8>. . . . . . . . . |
e | .<6>. 3 . . . . . . |
a | . .<4>. . . . . . . |
i | . 1 .<1>1 . . . . . |
o | . . . .<3>. . . . . |
s | . . . . .<2>. . . . |
t | . . . . . .<3>. . . |
. | . . . . . . .<2>. . |
T | . . . . . . . .<2>. |
h | . . . . . . . . .<2>|
--+---------------------+
(row = reference; col = test)
<BLANKLINE>

>>> print(cm.pretty_format(sort_by_count=True, truncate=10, values_in_chart=False))
   |                   1 |
   | 1 2 3 4 5 6 7 8 9 0 |
---+---------------------+
 1 |<8>. . . . . . . . . |
 2 | .<6>. 3 . . . . . . |
 3 | . .<4>. . . . . . . |
 4 | . 1 .<1>1 . . . . . |
 5 | . . . .<3>. . . . . |
 6 | . . . . .<2>. . . . |
 7 | . . . . . .<3>. . . |
 8 | . . . . . . .<2>. . |
 9 | . . . . . . . .<2>. |
10 | . . . . . . . . .<2>|
---+---------------------+
(row = reference; col = test)
Value key:
     1:
     2: e
     3: a
     4: i
     5: o
     6: s
     7: t
     8: .
     9: T
    10: h
<BLANKLINE>



##Association measures
#These measures are useful to determine whether the coocurrence of two random events is meaningful. 
#They are used, for instance, to distinguish collocations from other pairs of adjacent words.


>>> n_new_companies, n_new, n_companies, N = 8, 15828, 4675, 14307668
>>> bam = BigramAssocMeasures
>>> bam.raw_freq(20, (42, 20), N) == 20. / N
True
>>> bam.student_t(n_new_companies, (n_new, n_companies), N)
0.999...
>>> bam.chi_sq(n_new_companies, (n_new, n_companies), N)
1.54...
>>> bam.likelihood_ratio(150, (12593, 932), N)
1291...


#For other associations, we ensure the ordering of the measures:

>>> bam.mi_like(20, (42, 20), N) > bam.mi_like(20, (41, 27), N)
True
>>> bam.pmi(20, (42, 20), N) > bam.pmi(20, (41, 27), N)
True
>>> bam.phi_sq(20, (42, 20), N) > bam.phi_sq(20, (41, 27), N)
True
>>> bam.poisson_stirling(20, (42, 20), N) > bam.poisson_stirling(20, (41, 27), N)
True
>>> bam.jaccard(20, (42, 20), N) > bam.jaccard(20, (41, 27), N)
True
>>> bam.dice(20, (42, 20), N) > bam.dice(20, (41, 27), N)
True
>>> bam.fisher(20, (42, 20), N) > bam.fisher(20, (41, 27), N)
False


#For trigrams, we have to provide more count information:

>>> n_w1_w2_w3 = 20
>>> n_w1_w2, n_w1_w3, n_w2_w3 = 35, 60, 40
>>> pair_counts = (n_w1_w2, n_w1_w3, n_w2_w3)
>>> n_w1, n_w2, n_w3 = 100, 200, 300
>>> uni_counts = (n_w1, n_w2, n_w3)
>>> N = 14307668
>>> tam = TrigramAssocMeasures
>>> tam.raw_freq(n_w1_w2_w3, pair_counts, uni_counts, N) == 1. * n_w1_w2_w3 / N
True
>>> uni_counts2 = (n_w1, n_w2, 100)
>>> tam.student_t(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.student_t(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.chi_sq(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.chi_sq(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.mi_like(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.mi_like(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.pmi(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.pmi(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.likelihood_ratio(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.likelihood_ratio(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.poisson_stirling(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.poisson_stirling(n_w1_w2_w3, pair_counts, uni_counts, N)
True
>>> tam.jaccard(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.jaccard(n_w1_w2_w3, pair_counts, uni_counts, N)
True





###chap-7
###NLTK - Information Extraction
 
##Information Extraction Architecture
  
raw text (string)
as Input
 |
STEP - sentence segmentation
OUTPUT - sentences - list of strings 
 |
STEP - tokenization 
OUTPUT - tokenized sentences - list of list of strings 
 |
STEP - partof speech tagging 
OUTPUT - pos-tagged sentences - list of lists of tuples 
 |
STEP - entity detection
OUTPUT - chunked sentences (list of trees )
 |
relation detection 
OUTPUT - relations (list of tuples)

 

#first three tasks as given earlier 
import nltk 
def ie_preprocess(document):
        sentences = nltk.sent_tokenize(document) #sentence segmentation
        sentences = [nltk.word_tokenize(sent) for sent in sentences] #tokenization 
        sentences = [nltk.pos_tag(sent) for sent in sentences] #partof speech tagging 
 
 
##Chunking for entity detection
#which segments and labels multi-token sequences 

##Noun Phrase Chunking - NP-chunking
#where we search for chunks corresponding to individual noun phrases. 

#For example, here is some Wall Street Journal text with NP-chunks marked using brackets:
[ The/DT market/NN ] for/IN [ system-management/NN software/NN ] for/IN [ Digital/NNP ] [ 's/POS hardware/NN ] is/VBZ fragmented/JJ enough/RB that/IN [ a/DT giant/NN ] such/JJ as/IN [ Computer/NNP Associates/NNPS ] should/MD do/VB well/RB there/RB ./. 


#In order to create an NP-chunker, 
#we will first define a chunk grammar, 
#consisting of rules that indicate how sentences should be chunked. 

##Reference 

class nltk.tree.Tree(node, children=None)
    Bases: list
    A Tree represents a hierarchical grouping of leaves and subtrees. 
    A tree’s children are encoded as a list of leaves and subtrees, 
    where a leaf is a basic (non-tree) value; and a subtree is a nested Tree.
    >>> from nltk.tree import Tree
    >>> print(Tree(1, [2, Tree(3, [4]), 5]))
    (1 2 (3 4) 5)
    >>> vp = Tree('VP', [Tree('V', ['saw']),Tree('NP', ['him'])])
    >>> s = Tree('S', [Tree('NP', ['I']), vp])
    >>> print(s)
    (S (NP I) (VP (V saw) (NP him)))
    >>> print(s[1])
    (VP (V saw) (NP him))
    >>> print(s[1,1])
    (NP him)
    >>> t = Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
    >>> s == t
    True
    >>> t[1][1].set_label('X')
    >>> t[1][1].label()
    'X'
    >>> print(t)
    (S (NP I) (VP (V saw) (X him)))
    >>> t[0], t[1,1] = t[1,1], t[0]
    >>> print(t)
    (S (X him) (VP (V saw) (NP I)))
    #Tree Traversal
    def traverse(t):
        try:
            t.label()
        except AttributeError:
            print(t, end=" ")
        else:
            # Now we know that t.node is defined
            print('(', t.label(), end=" ")
            for child in t:
                traverse(child)
            print(')', end=" ")

    >>> t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit)))')
    >>> traverse(t)
     ( S ( NP Alice ) ( VP chased ( NP the rabbit ) ) )

    The length of a tree is the number of children it has.
    >>> len(t)
    2

    The set_label() and label() methods allow individual constituents 
    to be labeled. 
    For example, syntax trees use this label to specify phrase tags, 
    such as “NP” and “VP”.

    Several Tree methods use “tree positions” to specify children 
    or descendants of a tree. 
    Tree positions are defined as follows:
            The tree position i specifies a Tree’s ith child.
            The tree position () specifies the Tree itself.
            If p is the tree position of descendant d, 
            then p+i specifies the ith child of d.

    I.e., every tree position is either a single index i, specifying tree[i]; 
    or a sequence i1, i2, ..., iN, specifying tree[i1][i2]...[iN].

    Constructor can be called in one of two ways:
        Tree(label, children) constructs a new tree with the
            specified label and list of children.
        Tree.fromstring(s) constructs a new tree by parsing the string s.

    chomsky_normal_form(factor='right', horzMarkov=None, vertMarkov=0, childChar='|', parentChar='^')
        This method can modify a tree in three ways:
                Convert a tree into its Chomsky Normal Form (CNF) equivalent – 
                    Every subtree has either two non-terminals 
                    or one terminal as its children. 
                    This process requires the creation of more”artificial” non-terminal nodes.
                Markov (vertical) smoothing of children in new artificial nodes
                Horizontal (parent) annotation of nodes
        factor (str = [left|right]) – Right or left factoring method (default = “right”)
        horzMarkov (int | None) – Markov order for sibling smoothing in artificial nodes (None (default) = include all siblings)
        vertMarkov (int | None) – Markov order for parent smoothing (0 (default) = no vertical annotation)
        childChar (str) – A string used in construction of the artificial nodes, separating the head of the original subtree from the child nodes that have yet to be expanded (default = “|”)
        parentChar (str) – A string used to separate the node representation from its vertical annotation

    collapse_unary(collapsePOS=False, collapseRoot=False, joinChar='+')
        Collapse subtrees with a single child (ie. unary productions) 
        into a new non-terminal (Tree node) joined by ‘joinChar’. 
        This is useful when working with algorithms 
        that do not allow unary productions, 
        and completely removing the unary productions would require loss of useful information. 
        The Tree is modified directly (since it is passed by reference) 
        and no value is returned.
            collapsePOS (bool) – ‘False’ (default) will not collapse the parent of leaf nodes (ie. Part-of-Speech tags) since they are always unary productions
            collapseRoot (bool) – ‘False’ (default) will not modify the root production if it is unary. For the Penn WSJ treebank corpus, this corresponds to the TOP -> productions.
            joinChar (str) – A string used to connect collapsed node values (default = “+”)

    classmethod convert(tree)
        Convert a tree between different subtypes of Tree. 
        cls determines which class will be used to encode the new tree.
        Parameters:	tree (Tree) – The tree that should be converted.
        Returns:	The new Tree.

    copy(deep=False)

    draw()
        Open a new window containing a graphical diagram of this tree.

    flatten()
        Return a flat version of the tree, 
        with all non-root non-terminals removed.
        Returns:	a tree consisting of this tree’s root connected directly to its leaves, omitting all intervening non-terminal nodes.
        Return type:	Tree
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> print(t.flatten())
        (S the dog chased the cat)

    freeze(leaf_freezer=None)

    classmethod fromstring(s, brackets='()', read_node=None, read_leaf=None, node_pattern=None, leaf_pattern=None, remove_empty_top_bracketing=False)
        Read a bracketed tree string and return the resulting tree. 
        Trees are represented as nested brackettings, such as:
        (S (NP (NNP John)) (VP (V runs)))        

    height()
        Return the height of the tree.
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.height()
        5
        >>> print(t[0,0])
        (D the)
        >>> t[0,0].height()
        2 

    label()
        Return the node label of the tree.
        >>> t = Tree.fromstring('(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))')
        >>> t.label()
        'S'
        
    leaf_treeposition(index)
        Returns:The tree position of the index-th leaf in this tree. 
                I.e., if tp=self.leaf_treeposition(i), then self[tp]==self.leaves()[i].
        Raises:	IndexError – If this tree contains fewer than index+1 leaves, or if index<0.

    leaves()
        Return the leaves of the tree.
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.leaves()
        ['the', 'dog', 'chased', 'the', 'cat']        

    node
        Outdated method to access the node value; 
        use the label() method instead.

    pformat(margin=70, indent=0, nodesep='', parens='()', quotes=False)
        Returns:A pretty-printed string representation of this tree.
        Return type:  str
 
    pformat_latex_qtree()
        Returns a representation of the tree compatible 
        with the LaTeX qtree package. 
        This consists of the string \Tree followed by the tree represented 
        in bracketed notation.
        Returns:	A latex qtree representation of this tree.
        Return type:	str

    pos()
        Return a sequence of pos-tagged words extracted from the tree.
                Return type:	list(tuple)
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.pos()
        [('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]  

    pprint(**kwargs)
        Print a string representation of this Tree to ‘stream’

    pretty_print(sentence=None, highlight=(), stream=None, **kwargs)
        Pretty-print this tree as ASCII or Unicode art. 

    productions()
        Generate the productions that correspond to the non-terminal nodes of the tree. 
        For each subtree of the form (P: C1 C2 ... Cn) 
        this produces a production of the form P -> C1 C2 ... Cn.
            Return type:	list(Production)
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.productions()
        [S -> NP VP, NP -> D N, D -> 'the', N -> 'dog', VP -> V NP, V -> 'chased',
        NP -> D N, D -> 'the', N -> 'cat']     

    set_label(label)
        Set the node label of the tree.
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.set_label("T")
        >>> print(t)
        (T (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))

    subtrees(filter=None)
        Generate all the subtrees of this tree, 
        optionally restricted to trees matching the filter function.
            filter (function) – the function to filter all local trees
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> for s in t.subtrees(lambda t: t.height() == 2):
                print(s)
        (D the)
        (N dog)
        (V chased)
        (D the)
        (N cat)

    treeposition_spanning_leaves(start, end)
        Returns:The tree position of the lowest descendant of this tree 
                that dominates self.leaves()[start:end].
        Raises:	ValueError – if end <= start

    treepositions(order='preorder')
            order – One of: preorder, postorder, bothorder, leaves.
        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.treepositions() 
        [(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
        >>> for pos in t.treepositions('leaves'):
                t[pos] = t[pos][::-1].upper()
        >>> print(t)
        (S (NP (D EHT) (N GOD)) (VP (V DESAHC) (NP (D EHT) (N TAC))))

    un_chomsky_normal_form(expandUnary=True, childChar='|', parentChar='^', unaryChar='+')
        This method modifies the tree in three ways:
                Transforms a tree in Chomsky Normal Form back to its original structure (branching greater than two)
                Removes any parent annotation (if it exists)
                (optional) expands unary subtrees (if previously collapsed with collapseUnary(...) )
            expandUnary (bool) – Flag to expand unary or not (default = True)
            childChar (str) – A string separating the head node from its children in an artificial node (default = “|”)
            parentChar (str) – A sting separating the node label from its parent annotation (default = “^”)
            unaryChar (str) – A string joining two non-terminals in a unary production (default = “+”)

            
#Other interesting class 

class nltk.tree.ProbabilisticMixIn(**kwargs)
    Bases: object
    A mix-in class to associate probabilities with other classes (trees, rules, etc.). 
    
    logprob()
        Return log(p), where p is the probability associated with this object.
        Return type:	float

    prob()
        Return the probability associated with this object.
        Return type:	float

    set_logprob(logprob)
        Set the log probability associated with this object to logprob. I.e., set the probability associated with this object to 2**(logprob).
        Parameters:	logprob (float) – The new log probability

    set_prob(prob)
        Set the probability associated with this object to prob.
        Parameters:	prob (float) – The new probability

class nltk.tree.ProbabilisticTree(node, children=None, **prob_kwargs)
    Bases: nltk.tree.Tree, nltk.probability.ProbabilisticMixIn
    
    classmethod convert(val)

    copy(deep=False)

    unicode_repr()    


class nltk.tree.ImmutableTree(node, children=None)
    Bases: nltk.tree.Tree

    append(v)

    extend(v)

    pop(v=None)

    remove(v)

    reverse()

    set_label(value)
        Set the node label. 
        This will only succeed the first time the node label is set, which should occur in ImmutableTree.__init__().

    sort()
    
class nltk.tree.ImmutableProbabilisticTree(node, children=None, **prob_kwargs)
    Bases: nltk.tree.ImmutableTree, nltk.probability.ProbabilisticMixIn

    classmethod convert(val)

    copy(deep=False)

    unicode_repr()
    
    
#Example 
from nltk import Tree, ProbabilisticTree

# Demonstrate tree parsing.
s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
t = Tree.fromstring(s)
print("Convert bracketed string into tree:")
print(t)
print(t.__repr__())

print("Display tree properties:")
print(t.label())         # tree's constituent type
print(t[0])             # tree's first child
print(t[1])             # tree's second child
print(t.height())
print(t.leaves())
print(t[1])
print(t[1,1])
print(t[1,1,0])

# Demonstrate tree modification.
the_cat = t[0]
the_cat.insert(1, Tree.fromstring('(JJ big)'))
print("Tree modification:")
print(t)
t[1,1,1] = Tree.fromstring('(NN cake)')
print(t)
print()

# Tree transforms
print("Collapse unary:")
t.collapse_unary()
print(t)
print("Chomsky normal form:")
t.chomsky_normal_form()
print(t)
print()

# Demonstrate probabilistic trees.
pt = ProbabilisticTree('x', ['y', 'z'], prob=0.5)
print("Probabilistic Tree:")
print(pt)
print()

# Demonstrate parsing of treebank output format.
t = Tree.fromstring(t.pformat())
print("Convert tree to bracketed string and back again:")
print(t)
print()

# Demonstrate LaTeX output
print("LaTeX output:")
print(t.pformat_latex_qtree())
print()

# Demonstrate Productions
print("Production output:")
print(t.productions())
print()

# Demonstrate tree nodes containing objects other than strings
t.set_label(('test', 3))
print(t)



            
class nltk.grammar.Production(lhs, rhs)
    Bases: object
    A grammar production. 
    Each production maps a single symbol on the “left-hand side” 
    to a sequence of symbols on the “right-hand side”. 
    (In the case of context-free productions, 
    the left-hand side must be a Nonterminal, 
    and the right-hand side is a sequence of terminals and Nonterminals.) 
    “terminals” can be any immutable hashable object that is not a Nonterminal. 
    Typically, terminals are strings representing words, such as "dog" or "under".

    is_lexical()
        Return True if the right-hand contain at least one terminal token.
        Return type:	bool

    is_nonlexical()
        Return True if the right-hand side only contains Nonterminals
        Return type:	bool

    lhs()
        Return the left-hand side of this Production.
        Return type:	Nonterminal

    rhs()
        Return the right-hand side of this Production.
        Return type:	sequence(Nonterminal and terminal)

    unicode_repr()
        Return a concise string representation of the Production.
        Return type:	str

class nltk.parse.api.ParserI
    Bases: object
    grammar()
        Returns:	The grammar used by this parser.

    parse(sent, *args, **kwargs)
        Returns:	An iterator that generates parse trees for the sentence.
        When possible this list is sorted from most likely to least likely.
        Parameters:	sent (list(str)) – The sentence to be parsed
        Return type:	iter(Tree)

    parse_all(sent, *args, **kwargs)
        Return type:	list(Tree)

    parse_one(sent, *args, **kwargs)
        Return type:	Tree or None

    parse_sents(sents, *args, **kwargs)
        Apply self.parse() to each element of sents. :rtype: iter(iter(Tree))


        
class nltk.chunk.api.ChunkParserI
    Bases: nltk.parse.api.ParserI
    A processing interface 
    evaluate(gold)
        Score the accuracy of the chunker against the gold standard. 
        Remove the chunking the gold standard text, rechunk it using the chunker, 
        and return a ChunkScore object reflecting the performance of this chunk peraser.
        Parameters:	gold (list(Tree)) – The list of chunked sentences to score the chunker on.
        Return type:	ChunkScore

    parse(tokens)
        Return the best chunk structure for the given tokens and return a tree.
        Parameters:	tokens (list(tuple)) – The list of (word, tag) tokens to be chunked.
        Return type:	Tree

##RegexpChunkParser


class nltk.chunk.regexp.RegexpParser(grammar, root_label='S', loop=1, trace=0)
    Bases: nltk.chunk.api.ChunkParserI
    A grammar based chunk parser.    
    The maximum depth of a parse tree created by this chunk parser 
    is the same as the number of clauses in the grammar.
    A grammar contains one or more clauses in the following form:
    NP:
      {<DT|JJ>}          # chunk determiners and adjectives
      }<[\.VI].*>+{      # chink any tag beginning with V, I, or .
      <.*>}{<DT>         # split a chunk at a determiner
      <DT|JJ>{}<NN.*>    # merge chunk ending with det/adj
                         # with one starting with a noun
    #Note 
        {regexp}         # chunk rule
        }regexp{         # chink rule
        regexp}{regexp   # split rule
        regexp{}regexp   # merge rule   
        
    parse(chunk_struct, trace=None)
        Apply the chunk parser to this input.	
            chunk_struct (Tree) – the chunk structure to be (further) chunked (this tree is modified, and is also returned)
            trace (int) – The level of tracing that should be used when parsing a text. 0 will generate no tracing output; 1 will generate normal tracing output; and 2 or highter will generate verbose tracing output. This value overrides the trace level value that was given to the constructor.
        Return type:Tree

    unicode_repr()
        Returns:	a concise string representation of this chunk.RegexpParser.
        Return type:	str
     
class nltk.chunk.regexp.RegexpChunkParser(rules, chunk_label='NP', root_label='S', trace=0)
    Bases: nltk.chunk.api.ChunkParserI    
    A chunk is a non-overlapping linguistic group, such as a noun phrase. 
    The set of chunks identified in the chunk structure depends 
    on the rules used to define this RegexpChunkParser.
        
    Initially, nothing is chunked. 
    RegexpChunkParser.parse() then applies a sequence of RegexpChunkRule rules 
    to the ChunkString, each of which modifies the chunking that it encodes. 
    
    RegexpChunkParser can only be used to chunk a single kind of phrase. 
    For example, you can use an RegexpChunkParser to chunk the noun phrases in a text,
    or the verb phrases in a text; 
       
    parse(chunk_struct, trace=None)
            chunk_struct (Tree) – the chunk structure to be (further) chunked
            trace (int) – The level of tracing that should be used when parsing a text. 0 will generate no tracing output; 1 will generate normal tracing output; and 2 or highter will generate verbose tracing output. This value overrides the trace level value that was given to the constructor.
        Return type: Tree       

    rules()
        Returns:	the sequence of rules used by RegexpChunkParser.
        Return type:	list(RegexpChunkRule)

    unicode_repr()
        Returns:	a concise string representation of this RegexpChunkParser.
        Return type:	str
        
        
class nltk.chunk.regexp.ChunkString(chunk_struct, debug_level=1)
    Bases: object
    A string-based encoding of a particular chunking of a text. 
    Internally, the ChunkString class uses a single string 
    to encode the chunking of the input text. 
    This string contains a sequence of angle-bracket delimited tags, 
    with chunking indicated by braces. An example of this encoding is:
    {<DT><JJ><NN>}<VBN><IN>{<DT><NN>}<.>{<DT><NN>}<VBD><.>

    ChunkString are created from tagged texts 
    (i.e., lists of tokens whose type is TaggedType). 
    Initially, nothing is chunked.

    The chunking of a ChunkString can be modified with the xform() method, 
    which uses a regular expression to transform the string representation. 
    These transformations should only add and remove braces; 
    they should not modify the sequence of angle-bracket delimited tags.
    
    to_chunkstruct(chunk_label='CHUNK')
        Return the chunk structure encoded by this ChunkString.
        Return type:	Tree
        Raises:	ValueError – If a transformation has generated an invalid chunkstring.

    unicode_repr()
        Return a string representation of this ChunkString. It has the form:

    xform(regexp, repl)
            regexp (str or regexp) – A regular expression matching the substring that should be replaced. This will typically include a named group, which can be used by repl.
            repl (str) – An expression specifying what should replace the matched substring. Typically, this will include a named replacement group, specified by regexp.

##Tag Patterns
#tag_pattern2re_pattern() can be used to transform a tag pattern to regular expression pattern.

#A RegexpChunkRule uses a modified version of regular expression patterns, called “tag patterns”. 
#Tag patterns are used to match sequences of tags. Examples of tag patterns are:
r'(<DT>|<JJ>|<NN>)+'
r'<NN>+'
r'<NN.*>'

#The differences between regular expression patterns and tag patterns are:
1.In tag patterns, '<' and '>' act as parentheses; 
  so '<NN>+' matches one or more repetitions of '<NN>', 
  not '<NN' followed by one or more repetitions of '>'.
2.Whitespace in tag patterns is ignored. 
  So '<DT> | <NN>' is equivalant to '<DT>|<NN>'
3.In tag patterns, '.' is equivalant to '[^{}<>]'; 
so '<NN.*>' matches any single tag starting with 'NN'.

        
  
##RegexpChunkRules
#A RegexpChunkRule is a transformational rule (via apply() method)
#that updates the chunking of a text by modifying its ChunkString. 
    ChunkRule chunks anything that matches a given regular expression.
    ChinkRule chinks/removes anything that matches a given regular expression.
    UnChunkRule will un-chunk any chunk that matches a given regular expression.
    MergeRule can be used to merge two contiguous chunks.
    SplitRule can be used to split a single chunk into two smaller chunks.
    ExpandLeftRule will expand a chunk to incorporate new unchunked material on the left.
    ExpandRightRule will expand a chunk to incorporate new unchunked material on the right.

class nltk.chunk.regexp.RegexpChunkRule(regexp, repl, descr)
    Bases: object
 
    apply(chunkstr)
        Applyies Rule to chunkstr 

    descr()
        Return a short description of the purpose and/or effect of this rule.
        Return type:	str
        
    static fromstring(s)
        Create a RegexpChunkRule from a string description. 
        following formats are supported:
        {regexp}         # chunk rule
        }regexp{         # chink rule
        regexp}{regexp   # split rule
        regexp{}regexp   # merge rule

        Where regexp is a regular expression for the rule. 
        Any text following the comment marker 
        (#) will be used as the rule’s description:
        >>> from nltk.chunk.regexp import RegexpChunkRule
        >>> RegexpChunkRule.fromstring('{<DT>?<NN.*>+}')
        <ChunkRule: '<DT>?<NN.*>+'>

    unicode_repr()
        Return a string representation of this rule. It has the form:
        <RegexpChunkRule: '{<IN|VB.*>}'->'<IN>'>

        

      
class nltk.chunk.regexp.ChinkRule(tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to remove chinks to a ChunkString, 
    using a matching tag pattern. 
    When applied to a ChunkString, it will find any substring 
    that matches this tag pattern and that is contained in a chunk, 
    and remove it from that chunk, thus creating two new chunks.

 
class nltk.chunk.regexp.ChunkRule(tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to add chunks to a ChunkString, 
    using a matching tag pattern. 
    When applied to a ChunkString, it will find any substring 
    that matches this tag pattern and that is not already part of a chunk, 
    and create a new chunk containing that substring.

class nltk.chunk.regexp.ChunkRuleWithContext(left_context_tag_pattern, chunk_tag_pattern, right_context_tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to add chunks to a ChunkString, 
    using three matching tag patterns: 
    one for the left context, one for the chunk, and one for the right context. 
    When applied to a ChunkString, it will find any substring that matches 
    the chunk tag pattern, is surrounded by substrings that match the two context 
    patterns, and is not already part of a chunk; 
    and create a new chunk containing the substring that matched 
    the chunk tag pattern.
    Caveat: Both the left and right context are consumed when this rule matches; 
    therefore, if you need to find overlapping matches, 
    you will need to apply your rule more than once.



        
class nltk.chunk.regexp.ExpandLeftRule(left_tag_pattern, right_tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to expand chunks in a ChunkString to the left, 
    using two matching tag patterns: a left pattern, and a right pattern. 
    When applied to a ChunkString, it will find any chunk 
    whose beginning matches right pattern, 
    and immediately preceded by a chink whose end matches left pattern. 
    It will then expand the chunk to incorporate the new material on the left.

class nltk.chunk.regexp.ExpandRightRule(left_tag_pattern, right_tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to expand chunks in a ChunkString to the right, 
    using two matching tag patterns: a left pattern, and a right pattern. 
    When applied to a ChunkString, it will find any chunk whose end 
    matches left pattern, and immediately followed by a chink whose beginning 
    matches right pattern. 
    It will then expand the chunk to incorporate the new material on the right.

 
class nltk.chunk.regexp.MergeRule(left_tag_pattern, right_tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to merge chunks in a ChunkString, 
    using two matching tag patterns: a left pattern, and a right pattern. 
    When applied to a ChunkString, 
    it will find any chunk whose end matches left pattern, 
    and immediately followed by a chunk whose beginning matches right pattern. 
    It will then merge those two chunks into a single chunk.

class nltk.chunk.regexp.SplitRule(left_tag_pattern, right_tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to split chunks in a ChunkString, 
    using two matching tag patterns: a left pattern, and a right pattern. 
    When applied to a ChunkString, it will find any chunk that matches 
    the left pattern followed by the right pattern. 
    It will then split the chunk into two new chunks, 
    at the point between the two pattern matches.


class nltk.chunk.regexp.UnChunkRule(tag_pattern, descr)
    Bases: nltk.chunk.regexp.RegexpChunkRule
    A rule specifying how to remove chunks to a ChunkString, 
    using a matching tag pattern. 
    When applied to a ChunkString, it will find any complete chunk 
    that matches this tag pattern, and un-chunk it.

##Chunk Utlities 
nltk.chunk.regexp.tag_pattern2re_pattern(tag_pattern)
    Convert a tag pattern to a regular expression pattern. 

class nltk.chunk.util.ChunkScore(**kwargs)
    Bases: object
    A utility class for scoring chunk parsers. 
    ChunkScore can evaluate a chunk parser’s output, 
    based on a number of statistics (precision, recall, f-measure, misssed chunks, incorrect chunks). 
    It can also combine the scores from the parsing of multiple texts; 
    this makes it significantly easier to evaluate a chunk parser 
    that operates one sentence at a time.

    Texts are evaluated with the score method. 
    The results of evaluation can be accessed via a number of accessor methods, 
    such as precision and f_measure. 
    >>> chunkscore = ChunkScore()           
    >>> for correct in correct_sentences:   
            guess = chunkparser.parse(correct.leaves())   
            chunkscore.score(correct, guess)              
    >>> print('F Measure:', chunkscore.f_measure())       
    F Measure: 0.823
 
    accuracy()
        Return the overall tag-based accuracy for all text that have been scored by this ChunkScore, using the IOB (conll2000) tag encoding.
        Return type:	float

    correct()
        Return the chunks which were included in the correct chunk structures, listed in input order.
        Return type:	list of chunks

    f_measure(alpha=0.5)
        Return the overall F measure for all texts that have been scored by this ChunkScore.
        Parameters:	alpha (float) – the relative weighting of precision and recall. Larger alpha biases the score towards the precision value, while smaller alpha biases the score towards the recall value. alpha should have a value in the range [0,1].
        Return type:	float

    guessed()
        Return the chunks which were included in the guessed chunk structures, listed in input order.
        Return type:	list of chunks

    incorrect()
        Return the chunks which were included in the guessed chunk structures, but not in the correct chunk structures, listed in input order.
        Return type:	list of chunks

    missed()
        Return the chunks which were included in the correct chunk structures, but not in the guessed chunk structures, listed in input order.
        Return type:	list of chunks

    precision()
        Return the overall precision for all texts that have been scored by this ChunkScore.
        Return type:	float

    recall()
        Return the overall recall for all texts that have been scored by this ChunkScore.
        Return type:	float

    score(correct, guessed)
        Given a correctly chunked sentence, score another chunked version of the same sentence.
        Parameters:	
            correct (chunk structure) – The known-correct (“gold standard”) chunked sentence.
            guessed (chunk structure) – The chunked sentence to be scored.

nltk.chunk.util.accuracy(chunker, gold)
    Score the accuracy of the chunker against the gold standard. 
    Strip the chunk information from the gold standard 
    and rechunk it using the chunker, then compute the accuracy score.
    Parameters:	
        chunker (ChunkParserI) – The chunker being evaluated.
        gold (tree) – The chunk structures to score the chunker on.
    Return type:float

nltk.chunk.util.conllstr2tree(s, chunk_types=('NP', 'PP', 'VP'), root_label='S')
    Return a chunk structure for a single sentence 
    encoded in the given CONLL 2000 style string. 
    This function converts a CoNLL IOB string into a tree. 
    It uses the specified chunk types (defaults to NP, PP and VP), 
    and creates a tree rooted at a node labeled S (by default).
    Parameters:	
        s (str) – The CoNLL string to be converted.
        chunk_types (tuple) – The chunk types to be converted.
        root_label (str) – The node label to use for the root.
    Return type:Tree

nltk.chunk.util.conlltags2tree(sentence, chunk_types=('NP', 'PP', 'VP'), root_label='S', strict=False)
    Convert the CoNLL IOB format to a tree.

nltk.chunk.util.ieerstr2tree(s, chunk_types=['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'], root_label='S')
    Return a chunk structure containing the chunked tagged text 
    that is encoded in the given IEER style string. 
    Convert a string of chunked tagged text in the IEER named entity format into a chunk structure. 
    Chunks are of several types, LOCATION, ORGANIZATION, PERSON, DURATION, DATE, CARDINAL, PERCENT, MONEY, and MEASURE.
    Return type:	Tree

nltk.chunk.util.tagstr2tree(s, chunk_label='NP', root_label='S', sep='/', source_tagset=None, target_tagset=None)
    Divide a string of bracketted tagged text into chunks and unchunked tokens, 
    and produce a Tree. 
    Chunks are marked by square brackets ([...]). 
    Words are delimited by whitespace, and each word should have the form text/tag. 
    Words that do not contain a slash are assigned a tag of None.
    Parameters:	
        s (str) – The string to be converted
        chunk_label (str) – The label to use for chunk nodes
        root_label (str) – The label to use for the root of the tree
    Return type:Tree

nltk.chunk.util.tree2conllstr(t)
    Return a multiline string where each line contains a word, tag and IOB tag. 
    Convert a tree to the CoNLL IOB string format
    Parameters:	t (Tree) – The tree to be converted.
    Return type:	str

nltk.chunk.util.tree2conlltags(t)
    Return a list of 3-tuples containing (word, tag, IOB-tag). Convert a tree to the CoNLL IOB tag format.
    Parameters:	t (Tree) – The tree to be converted.
    Return type:	list(tuple)        
   
##Example from unittest 
>>> from nltk.chunk import *
>>> from nltk.chunk.util import *
>>> from nltk.chunk.regexp import *
>>> from nltk import Tree

>>> tagged_text = "[ The/DT cat/NN ] sat/VBD on/IN [ the/DT mat/NN ] [ the/DT dog/NN ] chewed/VBD ./."
>>> gold_chunked_text = tagstr2tree(tagged_text)
>>> gold_chunked_text
Tree('S', [Tree('NP', [('The', 'DT'), ('cat', 'NN')]), ('sat', 'VBD'),
 ('on', 'IN'), Tree('NP', [('the', 'DT'), ('mat', 'NN')]), Tree('NP',
[('the', 'DT'), ('dog', 'NN')]), ('chewed', 'VBD'), ('.', '.')])
>>> gold_chunked_text.flatten()
Tree('S', [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD'), ('on', 'IN'),
 ('the', 'DT'), ('mat', 'NN'), ('the', 'DT'), ('dog', 'NN'), ('chewed'
, 'VBD'), ('.', '.')])
>>> unchunked_text = gold_chunked_text.flatten()


#Chunking uses a special regexp syntax for rules that delimit the chunks. 
>>> tag_pattern = "<DT>?<JJ>*<NN.*>"
>>> regexp_pattern = tag_pattern2re_pattern(tag_pattern)
>>> regexp_pattern
'(<(DT)>)?(<(JJ)>)*(<(NN[^\\{\\}<>]*)>)'


#Construct some new chunking rules.

>>> chunk_rule = ChunkRule("<.*>+", "Chunk everything")
>>> chink_rule = ChinkRule("<VBD|IN|\.>", "Chink on verbs/prepositions")
>>> split_rule = SplitRule("<DT><NN>", "<DT><NN>","Split successive determiner/noun pairs")


#Create and score a series of chunk parsers, successively more complex.

>>> chunk_parser = RegexpChunkParser([chunk_rule], chunk_label='NP')
>>> chunked_text = chunk_parser.parse(unchunked_text)
>>> print(chunked_text)
(S
  (NP
    The/DT
    cat/NN
    sat/VBD
    on/IN
    the/DT
    mat/NN
    the/DT
    dog/NN
    chewed/VBD
    ./.))

>>> chunkscore = ChunkScore()
>>> chunkscore.score(gold_chunked_text, chunked_text)
>>> print(chunkscore.precision())
0.0

>>> print(chunkscore.recall())
0.0

>>> print(chunkscore.f_measure())
0

>>> for chunk in sorted(chunkscore.missed()): print(chunk)
(NP The/DT cat/NN)
(NP the/DT dog/NN)
(NP the/DT mat/NN)

>>> for chunk in chunkscore.incorrect(): print(chunk)
(NP
  The/DT
  cat/NN
  sat/VBD
  on/IN
  the/DT
  mat/NN
  the/DT
  dog/NN
  chewed/VBD
  ./.)

>>> chunk_parser = RegexpChunkParser([chunk_rule, chink_rule], chunk_label='NP')
>>> chunked_text = chunk_parser.parse(unchunked_text)
>>> print(chunked_text)
(S
  (NP The/DT cat/NN)
  sat/VBD
  on/IN
  (NP the/DT mat/NN the/DT dog/NN)
  chewed/VBD
  ./.)
>>> assert chunked_text == chunk_parser.parse(list(unchunked_text))

>>> chunkscore = ChunkScore()
>>> chunkscore.score(gold_chunked_text, chunked_text)
>>> chunkscore.precision()
0.5

>>> print(chunkscore.recall())
0.33333333...

>>> print(chunkscore.f_measure())
0.4

>>> for chunk in sorted(chunkscore.missed()): print(chunk)
(NP the/DT dog/NN)
(NP the/DT mat/NN)

>>> for chunk in chunkscore.incorrect(): print(chunk)
(NP the/DT mat/NN the/DT dog/NN)

>>> chunk_parser = RegexpChunkParser([chunk_rule, chink_rule, split_rule],chunk_label='NP')
>>> chunked_text = chunk_parser.parse(unchunked_text, trace=True)
# Input:
 <DT>  <NN>  <VBD>  <IN>  <DT>  <NN>  <DT>  <NN>  <VBD>  <.>
# Chunk everything:
{<DT>  <NN>  <VBD>  <IN>  <DT>  <NN>  <DT>  <NN>  <VBD>  <.>}
# Chink on verbs/prepositions:
{<DT>  <NN>} <VBD>  <IN> {<DT>  <NN>  <DT>  <NN>} <VBD>  <.>
# Split successive determiner/noun pairs:
{<DT>  <NN>} <VBD>  <IN> {<DT>  <NN>}{<DT>  <NN>} <VBD>  <.>
>>> print(chunked_text)
(S
  (NP The/DT cat/NN)
  sat/VBD
  on/IN
  (NP the/DT mat/NN)
  (NP the/DT dog/NN)
  chewed/VBD
  ./.)

>>> chunkscore = ChunkScore()
>>> chunkscore.score(gold_chunked_text, chunked_text)
>>> chunkscore.precision()
1.0

>>> chunkscore.recall()
1.0

>>> chunkscore.f_measure()
1.0

>>> chunkscore.missed()
[]

>>> chunkscore.incorrect()
[]

>>> chunk_parser.rules() # doctest: +NORMALIZE_WHITESPACE
[<ChunkRule: '<.*>+'>, <ChinkRule: '<VBD|IN|\\.>'>,
 <SplitRule: '<DT><NN>', '<DT><NN>'>]


##Printing parsers:

>>> print(repr(chunk_parser))
<RegexpChunkParser with 3 rules>
>>> print(chunk_parser)
RegexpChunkParser with 3 rules:
    Chunk everything
      <ChunkRule: '<.*>+'>
    Chink on verbs/prepositions
      <ChinkRule: '<VBD|IN|\\.>'>
    Split successive determiner/noun pairs
      <SplitRule: '<DT><NN>', '<DT><NN>'>


#ChunkString can be built from a tree of tagged tuples, a tree of trees, or a mixed list of both:
>>> t1 = Tree('S', [('w%d' % i, 't%d' % i) for i in range(10)])
>>> t2 = Tree('S', [Tree('t0', []), Tree('t1', ['c1'])])
>>> t3 = Tree('S', [('w0', 't0'), Tree('t1', ['c1'])])
>>> ChunkString(t1)
<ChunkString: '<t0><t1><t2><t3><t4><t5><t6><t7><t8><t9>'>
>>> ChunkString(t2)
<ChunkString: '<t0><t1>'>
>>> ChunkString(t3)
<ChunkString: '<t0><t1>'>

#Other values generate an error:
>>> ChunkString(Tree('S', ['x']))
Traceback (most recent call last):
  . . .
ValueError: chunk structures must contain tagged tokens or trees


#The str() for a chunk string adds spaces to it, 
#which makes it line up with str() output for other chunk strings over the same underlying input.

>>> cs = ChunkString(t1)
>>> print(cs)
 <t0>  <t1>  <t2>  <t3>  <t4>  <t5>  <t6>  <t7>  <t8>  <t9>
>>> cs.xform('<t3>', '{<t3>}')
>>> print(cs)
 <t0>  <t1>  <t2> {<t3>} <t4>  <t5>  <t6>  <t7>  <t8>  <t9>


#The _verify() method makes sure that our transforms don't corrupt the chunk string. 
#By setting debug_level=2, _verify() will be called at the end of every call to xform.
>>> cs = ChunkString(t1, debug_level=3)

>>> # tag not marked with <...>:
>>> cs.xform('<t3>', 't3')
Traceback (most recent call last):
  . . .
ValueError: Transformation generated invalid chunkstring:
  <t0><t1><t2>t3<t4><t5><t6><t7><t8><t9>

>>> # brackets not balanced:
>>> cs.xform('<t3>', '{<t3>')
Traceback (most recent call last):
  . . .
ValueError: Transformation generated invalid chunkstring:
  <t0><t1><t2>{<t3><t4><t5><t6><t7><t8><t9>


#Chunking Rules
>>> r1 = RegexpChunkRule('<a|b>'+ChunkString.IN_CHINK_PATTERN,'{<a|b>}', 'chunk <a> and <b>')
>>> r2 = RegexpChunkRule(re.compile('<a|b>'+ChunkString.IN_CHINK_PATTERN), '{<a|b>}', 'chunk <a> and <b>')
>>> r3 = ChunkRule('<a|b>', 'chunk <a> and <b>')
>>> r4 = ChinkRule('<a|b>', 'chink <a> and <b>')
>>> r5 = UnChunkRule('<a|b>', 'unchunk <a> and <b>')
>>> r6 = MergeRule('<a>', '<b>', 'merge <a> w/ <b>')
>>> r7 = SplitRule('<a>', '<b>', 'split <a> from <b>')
>>> r8 = ExpandLeftRule('<a>', '<b>', 'expand left <a> <b>')
>>> r9 = ExpandRightRule('<a>', '<b>', 'expand right <a> <b>')
>>> for rule in r1, r2, r3, r4, r5, r6, r7, r8, r9:
        print(rule)
<RegexpChunkRule: '<a|b>(?=[^\\}]*(\\{|$))'->'{<a|b>}'>
<RegexpChunkRule: '<a|b>(?=[^\\}]*(\\{|$))'->'{<a|b>}'>
<ChunkRule: '<a|b>'>
<ChinkRule: '<a|b>'>
<UnChunkRule: '<a|b>'>
<MergeRule: '<a>', '<b>'>
<SplitRule: '<a>', '<b>'>
<ExpandLeftRule: '<a>', '<b>'>
<ExpandRightRule: '<a>', '<b>'>


#tag_pattern2re_pattern() complains if the tag pattern looks problematic:
>>> tag_pattern2re_pattern('{}')
Traceback (most recent call last):
  . . .
ValueError: Bad tag pattern: '{}'



#RegexpChunkParser
#A warning is printed when parsing an empty sentence:
>>> parser = RegexpChunkParser([ChunkRule('<a>', '')])
>>> parser.parse(Tree('S', []))
Warning: parsing empty text
Tree('S', [])


#RegexpParser
>>> parser = RegexpParser('''
        NP: {<DT>? <JJ>* <NN>*} # NP
        P: {<IN>}           # Preposition
        V: {<V.*>}          # Verb
        PP: {<P> <NP>}      # PP -> P NP
        VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
        ''')
>>> print(repr(parser))
<chunk.RegexpParser with 5 stages>
>>> print(parser)
chunk.RegexpParser with 5 stages:
RegexpChunkParser with 1 rules:
    NP   <ChunkRule: '<DT>? <JJ>* <NN>*'>
RegexpChunkParser with 1 rules:
    Preposition   <ChunkRule: '<IN>'>
RegexpChunkParser with 1 rules:
    Verb   <ChunkRule: '<V.*>'>
RegexpChunkParser with 1 rules:
    PP -> P NP   <ChunkRule: '<P> <NP>'>
RegexpChunkParser with 1 rules:
    VP -> V (NP|PP)*   <ChunkRule: '<V> <NP|PP>*'>
>>> print(parser.parse(unchunked_text, trace=True))
# Input:
 <DT>  <NN>  <VBD>  <IN>  <DT>  <NN>  <DT>  <NN>  <VBD>  <.>
# NP:
{<DT>  <NN>} <VBD>  <IN> {<DT>  <NN>}{<DT>  <NN>} <VBD>  <.>
# Input:
 <NP>  <VBD>  <IN>  <NP>  <NP>  <VBD>  <.>
# Preposition:
 <NP>  <VBD> {<IN>} <NP>  <NP>  <VBD>  <.>
# Input:
 <NP>  <VBD>  <P>  <NP>  <NP>  <VBD>  <.>
# Verb:
 <NP> {<VBD>} <P>  <NP>  <NP> {<VBD>} <.>
# Input:
 <NP>  <V>  <P>  <NP>  <NP>  <V>  <.>
# PP -> P NP:
 <NP>  <V> {<P>  <NP>} <NP>  <V>  <.>
# Input:
 <NP>  <V>  <PP>  <NP>  <V>  <.>
# VP -> V (NP|PP)*:
 <NP> {<V>  <PP>  <NP>}{<V>} <.>
(S
  (NP The/DT cat/NN)
  (VP
    (V sat/VBD)
    (PP (P on/IN) (NP the/DT mat/NN))
    (NP the/DT dog/NN))
  (VP (V chewed/VBD))
  ./.)



   
##Chunk - Example - regular Expression NP chunking : rule says that an NP chunk should be formed 
#whenever the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) 
#and then a noun (NN). 

#Rule syntax 
    {regexp}         # chunk rule
    }regexp{         # chink rule
    regexp}{regexp   # split rule
    regexp{}regexp   # merge rule
    Where regexp is a regular expression for the rule. 
    Any text following the comment marker (#) will be used as the rule’s description:
#The differences between regular expression patterns and tag patterns are:
1.In tag patterns, '<' and '>' act as parentheses; 
  so '<NN>+' matches one or more repetitions of '<NN>', 
  not '<NN' followed by one or more repetitions of '>'.
2.Whitespace in tag patterns is ignored. 
  So '<DT> | <NN>' is equivalant to '<DT>|<NN>'
3.In tag patterns, '.' is equivalant to '[^{}<>]'; 
so '<NN.*>' matches any single tag starting with 'NN'.


>>> sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), 
        ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

>>> grammar = "NP: {<DT>?<JJ>*<NN>}" 

>>> cp = nltk.RegexpParser(grammar) 
>>> result = cp.parse(sentence) 
>>> print(result) 
(S
  (NP the/DT little/JJ yellow/JJ dog/NN)
  barked/VBD
  at/IN
  (NP the/DT cat/NN))
>>> result.draw() #draw tree diagram 

#Basically the output is 
                                        S
    NP                               barked/VBD   at/IN          NP
the/DT little/JJ yellow/JJ dog/NN                           the/DT cat/NN
 
 
 


###Multiple rules for Chunking with Regular Expressions

#The first rule matches an optional determiner or possessive pronoun, zero or more adjectives, then a noun. 
#The second rule matches one or more proper nouns. 

grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"), 
                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
 
 

>>> print(cp.parse(sentence)) 
(S
  (NP Rapunzel/NNP)
  let/VBD
  down/RP
  (NP her/PP$ long/JJ golden/JJ hair/NN))
 
>>> cp.parse(sentence).draw() #draw tree diagram 


#If a tag pattern matches at overlapping locations, 
#the leftmost match takes precedence. 

>>> nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
>>> grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
>>> cp = nltk.RegexpParser(grammar)
>>> print(cp.parse(nouns))
(S (NP money/NN market/NN) fund/NN)
 
 
##Using chunker - Exploring Text Corpora

#For a tagged corpus to extract phrases matching a particular sequence of part-of-speech tags. 
#(this can be done via eg Text.findall(r"<a> (<.*>) <man>") , but chunker is easier)

>>> cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')  #CHUNK is label of TREE 
>>> brown = nltk.corpus.brown
>>> for sent in brown.tagged_sents():
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK': print(subtree)

(CHUNK combined/VBN to/TO achieve/VB)
(CHUNK continue/VB to/TO place/VB)
(CHUNK serve/VB to/TO protect/VB)
(CHUNK wanted/VBD to/TO wait/VB)
(CHUNK allowed/VBN to/TO place/VB)
(CHUNK expected/VBN to/TO become/VB)
...
(CHUNK seems/VBZ to/TO overtake/VB)
(CHUNK want/VB to/TO buy/VB)
 
 
##Chinking
#defines what we want to exclude from a chunk. 
#We can define a chink to be a sequence of tokens that is not included in a chunk. 

#In the following example, barked/VBD at/IN is a chink:
[ the/DT little/JJ yellow/JJ dog/NN ] barked/VBD at/IN [ the/DT cat/NN ]


#Chinking is the process of removing a sequence of tokens from a chunk. 

#If the matching sequence of tokens spans an entire chunk, then the whole chunk is removed; 
#if the sequence of tokens appears in the middle of the chunk, these tokens are removed, leaving two chunks where there was only one before. 

#If the sequence is at the periphery of the chunk, these tokens are removed, and a smaller chunk remains. 


#Example -  we put the entire sentence into a single chunk (denoted by {})
# then excise the chinks(denoted by } {  )

grammar = r"""
  NP:
    {<.*>+}          # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
       ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)
>>> print(cp.parse(sentence))
 (S
   (NP the/DT little/JJ yellow/JJ dog/NN)
   barked/VBD
   at/IN
   (NP the/DT cat/NN))
 
 
##Example - Three chinking rules applied to the same chunk
#                Entire chunk            Middle of a chunk               End of a chunk
Input       [a/DT little/JJ dog/NN]     [a/DT little/JJ dog/NN]         [a/DT little/JJ dog/NN] 
Operation   Chink "DT JJ NN"            Chink "JJ"                      Chink "NN" 
Pattern     }DT JJ NN{                  }JJ{                            }NN{ 
Output      a/DT little/JJ dog/NN       [a/DT] little/JJ [dog/NN]       [a/DT little/JJ] dog/NN 

 
##Representing Chunks: Tags vs Trees
#chunk structures can be represented using either tags or trees. 
#The most widespread file representation uses IOB tags. 

#In this scheme, each token is tagged with one of three special chunk tags, 
#I (inside), O (outside), or B (begin). 

#A token is tagged as B if it marks the beginning of a chunk. 
#Subsequent tokens within the chunk are tagged I. 
#All other tokens are tagged O. 

#The B and I tags are suffixed with the chunk type, e.g. B-NP, I-NP. 

#Example 
We PRP B-NP
saw VBD O
the DT B-NP
yellow JJ I-NP
dog NN I-NP

##Reading IOB Format and the CoNLL 2000 Corpus
#corpus - Wall Street Journal text that has been tagged then chunked using the IOB notation. 
#The chunk categories provided in this corpus are NP, VP and PP. 

#As we have seen, each sentence is represented using multiple lines, as shown below:
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
...

#A conversion function chunk.conllstr2tree() builds a tree representation 
#from one of these multi-line strings. 

#Moreover, it permits us to choose any subset of the three chunk types to use,

#FOr example -  for NP chunks:

>>> text = '''
 he PRP B-NP
 accepted VBD B-VP
 the DT B-NP
 position NN I-NP
 of IN B-PP
 vice NN B-NP
 chairman NN I-NP
 of IN B-PP
 Carlyle NNP B-NP
 Group NNP I-NP
 , , O
 a DT B-NP
 merchant NN I-NP
 banking NN I-NP
 concern NN I-NP
 . . O
 '''
>>> nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()
 
 

#The CoNLL 2000 corpus contains 270k words of Wall Street Journal text, 
#divided into "train" and "test" portions, annotated with part-of-speech tags and chunk tags in the IOB format. 

>>> from nltk.corpus import conll2000
>>> print(conll2000.chunked_sents('train.txt')[99])
(S
  (PP Over/IN)
  (NP a/DT cup/NN)
  (PP of/IN)
  (NP coffee/NN)
  ,/,
  (NP Mr./NNP Stone/NNP)
  (VP told/VBD)
  (NP his/PRP$ story/NN)
  ./.)
 
 

#the CoNLL 2000 corpus contains three chunk types: 
#NP chunks, which we have already seen; 
#VP chunks such as 'has already delivered'; 
#and PP chunks such as 'because of'. 

#For -  only NP chunks 
>>> print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99])
(S
  Over/IN
  (NP a/DT cup/NN)
  of/IN
  (NP coffee/NN)
  ,/,
  (NP Mr./NNP Stone/NNP)
  told/VBD
  (NP his/PRP$ story/NN)
  ./.)
 

  

##chunk - Simple Evaluation and creating Baselines

#Establish a baseline for the trivial chunk parser cp that creates no chunks:
>>> from nltk.corpus import conll2000
>>> cp = nltk.RegexpParser("")
>>> test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
>>> print(cp.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  43.4%
    Precision:      0.0%
    Recall:         0.0%
    F-Measure:      0.0%
 
#The IOB tag accuracy indicates that more than a third of the words are tagged with O, 
#i.e. not in an NP chunk. 
#since our tagger did not find any chunks, its precision, recall, and f-measure are all zero. 


#Example - tags beginning with letters that are characteristic of noun phrase tags (e.g. CD, DT, and JJ).
>>> grammar = r"NP: {<[CDJNP].*>+}"
>>> cp = nltk.RegexpParser(grammar)
>>> print(cp.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  87.7%
    Precision:     70.6%
    Recall:        67.8%
    F-Measure:     69.2%
 
 
#OR Using  training corpus to find the chunk tag (I, O, or B) that is most likely for each part-of-speech tag. 
#ie  build a chunker using a unigram tagger. 

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): #list of training sentences, which will be in the form of chunk trees
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents] #Return a list of 3-tuples containing (word, tag, IOB-tag). Convert a tree to the CoNLL IOB tag format.
        self.tagger = nltk.UnigramTagger(train_data)  #Use BigramTagger for BigramChunker for bigram chunker 

    def parse(self, sentence):  #takes a tagged sentence as its input
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags) #Convert the CoNLL IOB format to a tree.

 

>>> test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])  #Tree
>>> train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP']) #Tree
>>> unigram_chunker = UnigramChunker(train_sents)
>>> print(unigram_chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  92.9%
    Precision:     79.9%
    Recall:        86.8%
    F-Measure:     83.2%
 
 

>>> postags = sorted(set(pos for sent in train_sents for (word,pos) in sent.leaves()))
>>> print(unigram_chunker.tagger.tag(postags))
[('#', 'B-NP'), ('$', 'B-NP'), ("''", 'O'), ('(', 'O'), (')', 'O'),
 (',', 'O'), ('.', 'O'), (':', 'O'), ('CC', 'O'), ('CD', 'I-NP'),
 ('DT', 'B-NP'), ('EX', 'B-NP'), ('FW', 'I-NP'), ('IN', 'O'),
 ('JJ', 'I-NP'), ('JJR', 'B-NP'), ('JJS', 'I-NP'), ('MD', 'O'),
 ('NN', 'I-NP'), ('NNP', 'I-NP'), ('NNPS', 'I-NP'), ('NNS', 'I-NP'),
 ('PDT', 'B-NP'), ('POS', 'B-NP'), ('PRP', 'B-NP'), ('PRP$', 'B-NP'),
 ('RB', 'O'), ('RBR', 'O'), ('RBS', 'B-NP'), ('RP', 'O'), ('SYM', 'O'),
 ('TO', 'O'), ('UH', 'O'), ('VB', 'O'), ('VBD', 'O'), ('VBG', 'O'),
 ('VBN', 'O'), ('VBP', 'O'), ('VBZ', 'O'), ('WDT', 'B-NP'),
 ('WP', 'B-NP'), ('WP$', 'B-NP'), ('WRB', 'O'), ('', 'O')]
 
 
#BigramTagger has more score 
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): #list of training sentences, which will be in the form of chunk trees
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)  #Use BigramTagger for BigramChunker for bigram chunker 

    def parse(self, sentence):  #takes a tagged sentence as its input
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

>>> bigram_chunker = BigramChunker(train_sents)
>>> print(bigram_chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  93.3%
    Precision:     82.3%
    Recall:        86.8%
    F-Measure:     84.5%
 
 
##Chunk - Training Classifier-Based Chunkers
#Both the regular-expression based chunkers and the n-gram chunkers decide 
#what chunks to create entirely based on part-of-speech tags. 
#However, sometimes part-of-speech tags are insufficient to determine how a sentence should be chunked

#classifier-based chunker will work by assigning IOB tags to the words in a sentence, 
#and then converting those tags to chunks
class ConsecutiveNPChunkTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) [2]
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
 
 
 
 
#Define a simple feature extractor which just provides the part-of-speech tag of the current token. 
#Using this feature extractor, our classifier-based chunker is very similar to the unigram chunker, 
>>> def npchunk_features(sentence, i, history):
        word, pos = sentence[i]
        return {"pos": pos}
>>> chunker = ConsecutiveNPChunker(train_sents)
>>> print(chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  92.9%
    Precision:     79.9%
    Recall:        86.7%
    F-Measure:     83.2%
 
 
#version-2 : add a feature for the previous part-of-speech tag
#Define a new feature extractor - 
#Adding this feature allows the classifier to model interactions between adjacent tags, 
#and results in a chunker that is closely related to the bigram chunker.



>>> def npchunk_features(sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i-1]
        return {"pos": pos, "prevpos": prevpos}
>>> chunker = ConsecutiveNPChunker(train_sents)
>>> print(chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  93.6%
    Precision:     81.9%
    Recall:        87.2%
    F-Measure:     84.5%
 
 
#version-3: adding a feature for the current word
#Define a new   feature extractor 
#since we hypothesized that word content should be useful for chunking. 

>>> def npchunk_features(sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i-1]
        return {"pos": pos, "word": word, "prevpos": prevpos}
>>> chunker = ConsecutiveNPChunker(train_sents)
>>> print(chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  94.5%
    Precision:     84.2%
    Recall:        89.4%
    F-Measure:     86.7%
 
 
#version-4: extending the feature extractor with a variety of additional features, such as lookahead features [1], paired features [2], and complex contextual features [3]. 
#Define a new   feature extractor 
#This last feature, called tags-since-dt, creates a string describing the set of all part-of-speech tags 
#that have been encountered since the most recent determiner, or since the beginning of the sentence if there is no determiner before index i. .

>>> def npchunk_features(sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i-1]
        if i == len(sentence)-1:
            nextword, nextpos = "<END>", "<END>"
        else:
            nextword, nextpos = sentence[i+1]
        return {"pos": pos,
                "word": word,
                "prevpos": prevpos,
                "nextpos": nextpos, #[1]
                "prevpos+pos": "%s+%s" % (prevpos, pos),  #[2]
                "pos+nextpos": "%s+%s" % (pos, nextpos),
                "tags-since-dt": tags_since_dt(sentence, i)}  #[3]
 
 



>>> def tags_since_dt(sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))
 
 



>>> chunker = ConsecutiveNPChunker(train_sents)
>>> print(chunker.evaluate(test_sents))
ChunkParse score:
    IOB Accuracy:  96.0%
    Precision:     88.6%
    Recall:        91.0%
    F-Measure:     89.8%
 
 
##Chunk - Building Nested Structure with Cascaded Chunkers
#it is possible to build chunk structures of arbitrary depth, 
#by creating a multi-stage chunk grammar containing recursive rules. 

#Example - patterns for noun phrases, prepositional phrases, verb phrases, and sentences. 

#This is a four-stage chunk grammar, 
#and can be used to create structures having a depth of at most four.
grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
    ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
 
>>> print(cp.parse(sentence))
(S
  (NP Mary/NN)
  saw/VBD
  (CLAUSE
    (NP the/DT cat/NN)
    (VP sit/VB (PP on/IN (NP the/DT mat/NN)))))
 
 
#this result misses the VP headed by saw. 

#Also, apply this chunker to a sentence having deeper nesting. 
#Notice that it fails to identify the VP chunk starting at 
>>> sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
        ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
        ("on", "IN"), ("the", "DT"), ("mat", "NN")]
>>> print(cp.parse(sentence))
(S
  (NP John/NNP)
  thinks/VBZ
  (NP Mary/NN)
  saw/VBD # [_saw-vbd]
  (CLAUSE
    (NP the/DT cat/NN)
    (VP sit/VB (PP on/IN (NP the/DT mat/NN)))))
 
 
#SOLUTION -  get the chunker to loop over its patterns: 
#after trying all of them,  it repeats the process.

#add an optional second argument loop to specify the number of times the set of patterns should be run
>>> cp = nltk.RegexpParser(grammar, loop=2)
>>> print(cp.parse(sentence))
(S
  (NP John/NNP)
  thinks/VBZ
  (CLAUSE
    (NP Mary/NN)
    (VP
      saw/VBD
      (CLAUSE
        (NP the/DT cat/NN)
        (VP sit/VB (PP on/IN (NP the/DT mat/NN)))))))
 
 











##Chunk - Named Entity(NEs) Recognition
#Named entities are definite noun phrases that refer to specific types of individuals, 
#such as organizations, persons, dates, and so on.

##Commonly Used Types of Named Entity
#NE Type                    Examples
ORGANIZATION                Georgia-Pacific Corp., WHO 
PERSON                      Eddy Bonte, President Obama 
LOCATION                    Murray River, Mount Everest 
DATE                        June, 2008-06-29 
TIME                        two fifty a m, 1:30 p.m. 
MONEY                       175 million Canadian Dollars, GBP 10.40 
PERCENT                     twenty pct, 18.75 % 
FACILITY                    Washington Monument, Stonehenge 
GPE                         South East Asia, Midlothian 

#The goal of a named entity recognition (NER) system 
#is to identify all textual mentions of the named entities. 

#This can be broken down into two sub-tasks: 
#identifying the boundaries of the NE, and identifying its type

#NLTK provides a classifier that has already been trained to recognize named entities, 
#accessed with the function nltk.ne_chunk(). 

nltk.chunk.ne_chunk(tagged_tokens, binary=False)
    Use NLTK’s currently recommended named entity chunker 
    to chunk the given list of tagged tokens.

nltk.chunk.ne_chunk_sents(tagged_sentences, binary=False)
    Use NLTK’s currently recommended named entity chunker to chunk 
    the given list of tagged sentences, 
    each consisting of a list of tagged tokens.

class nltk.chunk.named_entity.NEChunkParser(train)
    Bases: nltk.chunk.api.ChunkParserI
    Expected input: list of pos-tagged words
    parse(tokens)
        Each token should be a pos-tagged word

class nltk.chunk.named_entity.NEChunkParserTagger(train)
    Bases: nltk.tag.sequential.ClassifierBasedTagger
    The IOB tagger used by the chunk parser.

nltk.chunk.named_entity.build_model(fmt='binary')

nltk.chunk.named_entity.cmp_chunks(correct, guessed)

nltk.chunk.named_entity.load_ace_data(roots, fmt='binary', skip_bnews=True)

nltk.chunk.named_entity.load_ace_file(textfile, fmt)

nltk.chunk.named_entity.postag_tree(tree)

nltk.chunk.named_entity.shape(word)

nltk.chunk.named_entity.simplify_pos(s)


#If we set the parameter binary=True , then named entities are just tagged as NE; 
#otherwise, the classifier adds category labels such as PERSON, ORGANIZATION, and GPE.

>>> sent = nltk.corpus.treebank.tagged_sents()[22]
>>> print(nltk.ne_chunk(sent, binary=True)) 
(S
  The/DT
  (NE U.S./NNP)
  is/VBZ
  one/CD
  ...
  according/VBG
  to/TO
  (NE Brooke/NNP T./NNP Mossman/NNP)
  ...)
 
 
>>> print(nltk.ne_chunk(sent)) 
(S
  The/DT
  (GPE U.S./NNP)
  is/VBZ
  one/CD
  ...
  according/VBG
  to/TO
  (PERSON Brooke/NNP T./NNP Mossman/NNP)
  ...)
 
 
  

##Chunk - Relation Extraction
#Once named entities have been identified in a text, we then want to extract the relations that exist between them

#Look for relations between specified types of named entity. 

#One way of approaching this task is to initially look for all triples of the form (X, a, Y), 
#where X and Y are named entities of the required types, and a is the string of words that intervenes between X and Y. 

#We can then use regular expressions to pull out just those instances of a that express the relation that we are looking for. 

#Example - searches for strings that contain the word in. 
# (?!\b.+ing\b) is a negative lookahead assertion 


>>> IN = re.compile(r'.*\bin\b(?!\b.+ing)')
>>> for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
            print(nltk.sem.rtuple(rel))
[ORG: 'WHYY'] 'in' [LOC: 'Philadelphia']
[ORG: 'McGlashan &AMP; Sarrail'] 'firm in' [LOC: 'San Mateo']
[ORG: 'Freedom Forum'] 'in' [LOC: 'Arlington']
[ORG: 'Brookings Institution'] ', the research group in' [LOC: 'Washington']
[ORG: 'Idealab'] ', a self-described business incubator based in' [LOC: 'Los Angeles']
[ORG: 'Open Text'] ', based in' [LOC: 'Waterloo']
[ORG: 'WGBH'] 'in' [LOC: 'Boston']
[ORG: 'Bastille Opera'] 'in' [LOC: 'Paris']
[ORG: 'Omnicom'] 'in' [LOC: 'New York']
[ORG: 'DDB Needham'] 'in' [LOC: 'New York']
[ORG: 'Kaplan Thaler Group'] 'in' [LOC: 'New York']
[ORG: 'BBDO South'] 'in' [LOC: 'Atlanta']
[ORG: 'Georgia-Pacific'] 'in' [LOC: 'Atlanta']
 
 
#the conll2002 Dutch corpus contains not just named entity annotation but also part-of-speech tags. 
#This allows us to devise patterns that are sensitive to these tags


>>> from nltk.corpus import conll2002
>>> vnv = """
    (
    is/V|    # 3rd sing present and
    was/V|   # past forms of the verb zijn ('be')
    werd/V|  # and also present
    wordt/V  # past of worden ('become)
    )
    .*       # followed by anything
    van/Prep # followed by van ('of')
    """
>>> VAN = re.compile(vnv, re.VERBOSE)
>>> for doc in conll2002.chunked_sents('ned.train'):
        for r in nltk.sem.extract_rels('PER', 'ORG', doc,corpus='conll2002', pattern=VAN):
            print(nltk.sem.clause(r, relsym="VAN")) #or print(rtuple(rel, lcon=True, rcon=True))
VAN("cornet_d'elzius", 'buitenlandse_handel')
VAN('johan_rottiers', 'kardinaal_van_roey_instituut')
VAN('annie_lennox', 'eurythmics')
 
 
##Reference - nltk.sem.relextract module
Code for extracting relational triples from the ieer and conll2002 corpora.
Relations are stored internally as dictionaries (‘reldicts’).
The two serialization outputs are “rtuple” and “clause”.
    An rtuple is a tuple of the form (subj, filler, obj), 
        where subj and obj are pairs of Named Entity mentions,
        and filler is the string of words occurring between sub and obj 
        (with no intervening NEs). S
        trings are printed via repr() to circumvent locale variations in rendering utf-8 encoded strings.
    A clause is an atom of the form relsym(subjsym, objsym), 
        where the relation, subject and object have been canonicalized 
        to single strings.

    
nltk.sem.relextract.extract_rels(subjclass, objclass, doc, corpus='ace', pattern=None, window=10)
    Filter the output of semi_rel2reldict according to specified NE classes and a filler pattern.
    
    The parameters subjclass and objclass can be used to restrict the Named Entities to particular types 
    (any of ‘LOCATION’, ‘ORGANIZATION’, ‘PERSON’, ‘DURATION’, ‘DATE’, ‘CARDINAL’, ‘PERCENT’, ‘MONEY’, ‘MEASURE’).

        subjclass (str) – the class of the subject Named Entity.
        objclass (str) – the class of the object Named Entity.
        doc (ieer document or a list of chunk trees) – input document
        corpus (str) – name of the corpus to take as input; possible values are ‘ieer’ and ‘conll2002’
        pattern (SRE_Pattern) – a regular expression for filtering the fillers of retrieved triples.
        window (int) – filters out fillers which exceed this threshold  

    Return type:list(defaultdict)
 

nltk.sem.relextract.class_abbrev(type)
    Abbreviate an NE class name. 
    type: str 
    rtype: str

nltk.sem.relextract.clause(reldict, relsym)
    Print the relation in clausal form. 
        reldict: a relation dictionary 
        relsym: a label for the relation , str

nltk.sem.relextract.conllesp()

nltk.sem.relextract.conllned(trace=1)
    Find the copula+’van’ relation (‘of’) in the Dutch tagged training corpus from CoNLL 2002.

nltk.sem.relextract.descape_entity(m, defs={..})
    Translate one entity to its ISO Latin value. 
    Inspired by example from effbot.org

nltk.sem.relextract.ieer_headlines()

nltk.sem.relextract.in_demo(trace=0, sql=True)
    Select pairs of organizations and locations 
    whose mentions occur with an intervening occurrence of the preposition “in”.
    If the sql parameter is set to True, then the entity pairs are loaded 
    into an in-memory database, and subsequently pulled out using an SQL “SELECT” query.

nltk.sem.relextract.list2sym(lst)
    Convert a list of strings into a canonical symbol. 
    lst: list 
    return: a Unicode string without whitespace 
    rtype: unicode

nltk.sem.relextract.ne_chunked()


nltk.sem.relextract.rtuple(reldict, lcon=False, rcon=False)
    Pretty print the reldict as an rtuple. 
    reldict: a relation dictionary 

nltk.sem.relextract.semi_rel2reldict(pairs, window=5, trace=False)
    Converts the pairs generated by tree2semi_rel into a ‘reldict’
    ‘reldict’: a dictionary which stores information about the subject 
    and object NEs plus the filler between them. 
    Additionally, a left and right context of length =< window are captured (within a given input sentence).
         pairs – a pair of list(str) and Tree, as generated by
        window (int) – a threshold for the number of items to include in the left and right context
    Returns: ‘relation’ dictionaries whose keys are ‘lcon’, ‘subjclass’, ‘subjtext’, ‘subjsym’, ‘filler’, objclass’, objtext’, ‘objsym’ and ‘rcon’


nltk.sem.relextract.tree2semi_rel(tree)
    Group a chunk structure into a list of ‘semi-relations’ 
    of the form (list(str), Tree).
    In order to facilitate the construction of (Tree, string, Tree) triples, 
    this identifies pairs whose first member is a list (possibly empty) 
    of terminal strings, and whose second member is a Tree of the form (NE_label, terminals).
        Parameters:	tree – a chunk tree
        Returns:	a list of pairs (list(str), Tree)

 
##chunk - Information Extraction and Relation extraction 
#Information Extraction standardly consists of three subtasks:
1.Named Entity Recognition
2.Relation Extraction
3.Template Filling


##Named Entities (ie name of something ie PERSON, ORGANIZATION )
#The IEER corpus is marked up for a variety of Named Entities. 

>>> from nltk.corpus import ieer
>>> docs = ieer.parsed_docs('NYT_19980315')
>>> tree = docs[1].text
>>> print(tree) # doctest: +ELLIPSIS
(DOCUMENT
...
  It's
  a
  chance
  to
  think
  about
  first-level
  questions,''
  said
  Ms.
  (PERSON Cohn)
  ,
  a
  partner
  in
  the
  (ORGANIZATION McGlashan &AMP; Sarrail)
  firm
  in
  (LOCATION San Mateo)
  ,
  (LOCATION Calif.)
  ...)


#CoNLL2002 Dutch and Spanish data 

>>> from nltk.corpus import conll2002
>>> for doc in conll2002.chunked_sents('ned.train')[27]:
        print(doc)
(u'Het', u'Art')
(ORG Hof/N van/Prep Cassatie/N)
(u'verbrak', u'V')
(u'het', u'Art')
(u'arrest', u'N')
(u'zodat', u'Conj')
(u'het', u'Pron')
(u'moest', u'V')
(u'worden', u'V')
(u'overgedaan', u'V')
(u'door', u'Prep')
(u'het', u'Art')
(u'hof', u'N')
(u'van', u'Prep')
(u'beroep', u'N')
(u'van', u'Prep')
(LOC Antwerpen/N)
(u'.', u'Punc')



##Relation Extraction
#Relation Extraction standardly consists of identifying specified relations between Named Entities. 

#The tree2semi_rel() function splits a chunk document into a list of two-member lists, 
#each of which consists of a (possibly empty) string followed by a Tree (i.e., a Named Entity):

>>> from nltk.sem import relextract
>>> pairs = relextract.tree2semi_rel(tree)
>>> for s, tree in pairs[18:22]:
        print('("...%s", %s)' % (" ".join(s[-5:]),tree))
("...about first-level questions,'' said Ms.", (PERSON Cohn))
("..., a partner in the", (ORGANIZATION McGlashan &AMP; Sarrail))
("...firm in", (LOCATION San Mateo))
("...,", (LOCATION Calif.))


#The function semi_rel2reldict() processes triples of these pairs, 
#i.e., pairs of the form ((string1, Tree1), (string2, Tree2), (string3, Tree3)) 
#and outputs a dictionary (a reldict) in which Tree1 is the subject of the relation, 
#string2 is the filler and Tree3 is the object of the relation. 

#string1 and string3 are stored as left and right context respectively.

>>> reldicts = relextract.semi_rel2reldict(pairs)
>>> for k, v in sorted(reldicts[0].items()):
        print(k, '=>', v) # doctest: +ELLIPSIS
filler => of messages to their own Cyberia'' ...
lcon => transactions.'' Each week, they post
objclass => ORGANIZATION
objsym => white_house
objtext => White House
rcon => for access to its planned
subjclass => CARDINAL
subjsym => hundreds
subjtext => hundreds
untagged_filler => of messages to their own Cyberia'' ...


#Example - some of the values for two reldicts corresponding to the 'NYT_19980315' text extract shown earlier.

>>> for r in reldicts[18:20]:
        print('=' * 20)
        print(r['subjtext'])
        print(r['filler'])
        print(r['objtext'])
====================
Cohn
, a partner in the
McGlashan &AMP; Sarrail
====================
McGlashan &AMP; Sarrail
firm in
San Mateo


#The function relextract() allows us to filter the reldicts according to the classes of the subject 
#and object named entities. 
#In addition, we can specify that the filler text has to match a given regular expression, 

#Here, we are looking for pairs of entities in the IN relation, 
#where IN has signature <ORG, LOC>.

>>> import re
>>> IN = re.compile(r'.*\bin\b(?!\b.+ing\b)')
>>> for fileid in ieer.fileids():
        for doc in ieer.parsed_docs(fileid):
            for rel in relextract.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
                print(relextract.rtuple(rel))  # doctest: +ELLIPSIS
[ORG: 'Christian Democrats'] ', the leading political forces in' [LOC: 'Italy']
[ORG: 'AP'] ') _ Lebanese guerrillas attacked Israeli forces in southern' [LOC: 'Lebanon']
[ORG: 'Security Council'] 'adopted Resolution 425. Huge yellow banners hung across intersections in' [LOC: 'Beirut']
[ORG: 'U.N.'] 'failures in' [LOC: 'Africa']
[ORG: 'U.N.'] 'peacekeeping operation in' [LOC: 'Somalia']
[ORG: 'U.N.'] 'partners on a more effective role in' [LOC: 'Africa']
[ORG: 'AP'] ') _ A bomb exploded in a mosque in central' [LOC: 'San`a']
[ORG: 'Krasnoye Sormovo'] 'shipyard in the Soviet city of' [LOC: 'Gorky']
[ORG: 'Kelab Golf Darul Ridzuan'] 'in' [LOC: 'Perak']
[ORG: 'U.N.'] 'peacekeeping operation in' [LOC: 'Somalia']
[ORG: 'WHYY'] 'in' [LOC: 'Philadelphia']
[ORG: 'McGlashan &AMP; Sarrail'] 'firm in' [LOC: 'San Mateo']
[ORG: 'Freedom Forum'] 'in' [LOC: 'Arlington']
[ORG: 'Brookings Institution'] ', the research group in' [LOC: 'Washington']
[ORG: 'Idealab'] ', a self-described business incubator based in' [LOC: 'Los Angeles']
[ORG: 'Open Text'] ', based in' [LOC: 'Waterloo']
...


#Example -  the patter is a disjunction of roles that a PERSON can occupy in an ORGANIZATION.

>>> roles = """
    (.*(
    analyst|
    chair(wo)?man|
    commissioner|
    counsel|
    director|
    economist|
    editor|
    executive|
    foreman|
    governor|
    head|
    lawyer|
    leader|
    librarian).*)|
    manager|
    partner|
    president|
    producer|
    professor|
    researcher|
    spokes(wo)?man|
    writer|
    ,\sof\sthe?\s*  # "X, of (the) Y"
    """
>>> ROLES = re.compile(roles, re.VERBOSE)
>>> for fileid in ieer.fileids():
        for doc in ieer.parsed_docs(fileid):
            for rel in relextract.extract_rels('PER', 'ORG', doc, corpus='ieer', pattern=ROLES):
                print(relextract.rtuple(rel)) # doctest: +ELLIPSIS
[PER: 'Kivutha Kibwana'] ', of the' [ORG: 'National Convention Assembly']
[PER: 'Boban Boskovic'] ', chief executive of the' [ORG: 'Plastika']
[PER: 'Annan'] ', the first sub-Saharan African to head the' [ORG: 'United Nations']
[PER: 'Kiriyenko'] 'became a foreman at the' [ORG: 'Krasnoye Sormovo']
[PER: 'Annan'] ', the first sub-Saharan African to head the' [ORG: 'United Nations']
[PER: 'Mike Godwin'] ', chief counsel for the' [ORG: 'Electronic Frontier Foundation']
...


#In the case of the CoNLL2002 data, we can include POS tags in the query pattern. 
#Example - how the output can be presented as something that looks more like a clause in a logical language.

>>> de = """
    .*
    (
    de/SP|
    del/SP
    )
    """
>>> DE = re.compile(de, re.VERBOSE)
>>> rels = [rel for doc in conll2002.chunked_sents('esp.train')
            for rel in relextract.extract_rels('ORG', 'LOC', doc, corpus='conll2002', pattern = DE)]
>>> for r in rels[:10]:
        print(relextract.clause(r, relsym='DE'))    # doctest: +NORMALIZE_WHITESPACE
DE(u'tribunal_supremo', u'victoria')
DE(u'museo_de_arte', u'alcorc\xf3n')
DE(u'museo_de_bellas_artes', u'a_coru\xf1a')
DE(u'siria', u'l\xedbano')
DE(u'uni\xf3n_europea', u'pek\xedn')
DE(u'ej\xe9rcito', u'rogberi')
DE(u'juzgado_de_instrucci\xf3n_n\xfamero_1', u'san_sebasti\xe1n')
DE(u'psoe', u'villanueva_de_la_serena')
DE(u'ej\xe9rcito', u'l\xedbano')
DE(u'juzgado_de_lo_penal_n\xfamero_2', u'ceuta')
>>> vnv = """
    (
    is/V|
    was/V|
    werd/V|
    wordt/V
    )
    .*
    van/Prep
    """
>>> VAN = re.compile(vnv, re.VERBOSE)
>>> for doc in conll2002.chunked_sents('ned.train'):
        for r in relextract.extract_rels('PER', 'ORG', doc, corpus='conll2002', pattern=VAN):
            print(relextract.clause(r, relsym="VAN"))
VAN(u"cornet_d'elzius", u'buitenlandse_handel')
VAN(u'johan_rottiers', u'kardinaal_van_roey_instituut')
VAN(u'annie_lennox', u'eurythmics')


 
 
 
 
 
 
 
 
 
 
 





###chap-8
###NLTK - Analyzing Sentence Structure

#Earlier chapters focused on words: how to identify them, analyze their structure, assign them to lexical categories, and access their meanings.
#We need a way to deal with the ambiguity that natural language is famous for. 
#We also need to be able to cope with the fact that there are an unlimited number of possible sentences, 
#and we can only write finite programs to analyze their structures and discover their meanings.

#A “grammar” specifies which trees can represent the structure of a given text. 
#Each of these trees is called a “parse tree” for the text (or simply a “parse”). 

#In a “context free” grammar, the set of parse trees for any piece of a text 
#can depend only on that piece, and not on the rest of the text (i.e., the piece’s context). 
 
#The CFG class is used to encode context free grammars
# CFG consists of a start symbol and a set of productions(Production class)
#Start symbols are encoded using the Nonterminal 


#For example, the production <S> -> <NP> <VP> specifies 
#that an S node can be the parent of an NP node and a VP node.

#The operation of replacing the left hand side (lhs) of a production 
#with the right hand side (rhs) in a tree (tree) is known as “expanding” lhs to rhs in tree.


class nltk.grammar.Nonterminal(symbol)
    Bases: object
    The node value that is wrapped by a Nonterminal is known as its “symbol”. 
    Symbols are typically strings representing phrasal categories (such as "NP" or "VP")
    Since symbols are node values, they must be immutable and hashable
    
    symbol()
        Return the node value corresponding to this Nonterminal

nltk.grammar.nonterminals(symbols)
    Given a string containing a list of symbol names, 
    return a list of Nonterminals constructed from those symbols.


class nltk.grammar.Production(lhs, rhs)
    Bases: object
    A grammar production
    In the case of context-free productions, 
    the left-hand side must be a Nonterminal, 
    and the right-hand side is a sequence of terminals and Nonterminals.
    “terminals” can be any immutable hashable object that is not a Nonterminal. 
    Typically, terminals are strings representing words, such as "dog" or "under".

    is_lexical()
        Return True if the right-hand contain at least one terminal token.

    is_nonlexical()
        Return True if the right-hand side only contains Nonterminals

    lhs()/rhs()
        Return the left-hand side/right-hand side  of this Production.
        Return type:Nonterminal  for lhs 
        Return type:sequence(Nonterminal and terminal) for rhs 



class nltk.grammar.CFG(start, productions, calculate_leftcorners=True)
    Bases: object
    A context-free grammar
    A grammar consists of a start state and a set of productions.

    check_coverage(tokens)
        Check whether the grammar rules cover the given list of tokens. 
        If not, then raise an exception.

    classmethod fromstring(input, encoding=None)
        Return the CFG corresponding to the input string(s).

    is_binarised()
        Return True if all productions are at most binary. 
        Note that there can still be empty and unary productions.

    is_chomsky_normal_form()
        Return True if the grammar is of Chomsky Normal Form, 
        i.e. all productions are of the form A -> B C, or A -> “s”.

    is_flexible_chomsky_normal_form()
        Return True if all productions are of the forms A -> B C, A -> B, or A -> “s”.
    
    is_leftcorner(cat, left)
        True if left is a leftcorner of cat, where left can be a terminal or a nonterminal.
            cat (Nonterminal) – the parent of the leftcorner
            left (Terminal or Nonterminal) – the suggested leftcorner

    is_lexical()
        Return True if all productions are lexicalised.
        if the right-hand contain at least one terminal token.
    
    is_nonempty()
        Return True if there are no empty productions.
    
    is_nonlexical()
        Return True if all lexical rules are “preterminals”, 
        that is, unary rules which can be separated in a preprocessing step.
        This means that all productions are of the forms A -> B1 ... Bn (n>=0), 
        or A -> “s”.
        Note: is_lexical() and is_nonlexical() are not opposites. 
        There are grammars which are neither, and grammars which are both.


    leftcorner_parents(cat)
        Return the set of all nonterminals for which the given category is a left corner. 
        This is the inverse of the leftcorner relation.

    leftcorners(cat)
        Return the set of all nonterminals that the given nonterminal can start with, including itself.

    max_len()
        Return the right-hand side length of the longest grammar production.

    min_len()
        Return the right-hand side length of the shortest grammar production.

    productions(lhs=None, rhs=None, empty=False)
        Return the grammar productions, 
        filtered by the left-hand side or the first item in the right-hand side.
            lhs – Only return productions with the given left-hand side.
            rhs – Only return productions with the given first item in the right-hand side.
            empty – Only return productions with an empty right-hand side.     

    start()
        Return the start symbol of the grammar




  
 
 
#Example 
>>> from nltk import CFG
>>> grammar = CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | NP PP
    VP -> V NP | VP PP
    Det -> 'a' | 'the'
    N -> 'dog' | 'cat'
    V -> 'chased' | 'sat'
    P -> 'on' | 'in'
    """)
>>> grammar
<Grammar with 14 productions>
>>> grammar.start()
S
>>> grammar.productions() 
[S -> NP VP, PP -> P NP, NP -> Det N, NP -> NP PP, VP -> V NP, VP -> VP PP,
Det -> 'a', Det -> 'the', N -> 'dog', N -> 'cat', V -> 'chased', V -> 'sat',
P -> 'on', P -> 'in']


#Chomsky Normal Form grammar 

>>> g = CFG.fromstring("VP^<TOP> -> VBP NP^<VP-TOP>")
>>> g.productions()[0].lhs()
VP^<TOP>


###Parsing Grammer - nltk.parse 
#ParserI, a standard interface for parsing texts; 
#two simple implementations of that interface, ShiftReduceParser and RecursiveDescentParser. 

class nltk.parse.api.ParserI
    Bases: object
    A processing class for deriving trees that represent possible structures for a sequence of tokens. 
    These tree structures are known as “parses”.
    
    grammar()
        Returns:The grammar used by this parser. 
        
    parse(sent, *args, **kwargs)
        Returns:  iter(Tree)  that generates parse trees for the sentence. 
        When possible this list is sorted from most likely to least likely.
            sent (list(str)) – The sentence to be parsed     
    
    parse_all(sent, *args, **kwargs)
        Return type:    list(Tree) 
    
    parse_one(sent, *args, **kwargs)
        Return type:    Tree or None 
    
    parse_sents(sents, *args, **kwargs)
        Apply self.parse() to each element of sents. :rtype: iter(iter(Tree))


class nltk.parse.recursivedescent.RecursiveDescentParser(grammar, trace=0)
    Bases: nltk.parse.api.ParserI
    A simple top-down CFG parser that parses texts 
    by recursively expanding the fringe of a Tree, 
    and matching it against a text.

class nltk.parse.recursivedescent.SteppingRecursiveDescentParser(grammar, trace=0)
    Bases: nltk.parse.recursivedescent.RecursiveDescentParser
    A RecursiveDescentParser that allows you to step through the parsing process, 
    performing a single operation at a time.
    
    step()
        Perform a single parsing operation.
        Return type: Production or String or bool 
    
    tree()
        Returns: A partial structure for the text that is currently being parsed. 


class nltk.parse.shiftreduce.ShiftReduceParser(grammar, trace=0)
    Bases: nltk.parse.api.ParserI
    A simple bottom-up CFG parser that uses two operations, “shift” and “reduce”, 
    to find a single parse for a text.

class nltk.parse.shiftreduce.SteppingShiftReduceParser(grammar, trace=0)
    Bases: nltk.parse.shiftreduce.ShiftReduceParser
    A ShiftReduceParser that allows you to setp through the parsing process, 
    performing a single operation at a time
    
    step()
        Perform a single parsing operation

##Example 

>>> from nltk import Nonterminal, nonterminals, Production, CFG

>>> nt1 = Nonterminal('NP')
>>> nt2 = Nonterminal('VP')

>>> nt1.symbol()
'NP'

>>> nt1 == Nonterminal('NP')
True

>>> nt1 == nt2
False

>>> S, NP, VP, PP = nonterminals('S, NP, VP, PP')
>>> N, V, P, DT = nonterminals('N, V, P, DT')

>>> prod1 = Production(S, [NP, VP])
>>> prod2 = Production(NP, [DT, NP])

>>> prod1.lhs()
S

>>> prod1.rhs()
(NP, VP)

>>> prod1 == Production(S, [NP, VP])
True

>>> prod1 == prod2
False

>>> grammar = CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> 'the' N | N PP | 'the' N PP
    VP -> V NP | V PP | V NP PP
    N -> 'cat'
    N -> 'dog'
    N -> 'rug'
    V -> 'chased'
    V -> 'sat'
    P -> 'in'
    P -> 'on'
    """)

#Recursive Descent Parser) class

#over  syntactically ambiguous and unambiguous sentence.

>>> from nltk.parse import RecursiveDescentParser
>>> rd = RecursiveDescentParser(grammar)

>>> sentence1 = 'the cat chased the dog'.split()
>>> sentence2 = 'the cat chased the dog on the rug'.split()

>>> for t in rd.parse(sentence1):
        print(t)
(S (NP the (N cat)) (VP (V chased) (NP the (N dog))))

>>> for t in rd.parse(sentence2):
        print(t)
(S
  (NP the (N cat))
  (VP (V chased) (NP the (N dog) (PP (P on) (NP the (N rug))))))
(S
  (NP the (N cat))
  (VP (V chased) (NP the (N dog)) (PP (P on) (NP the (N rug)))))


#Shift Reduce Parser) class
#over both a syntactically ambiguous and unambiguous sentence. 
#Note that unlike the recursive descent parser, one and only one parse is ever returned.

    >>> from nltk.parse import ShiftReduceParser
    >>> sr = ShiftReduceParser(grammar)

    >>> sentence1 = 'the cat chased the dog'.split()
    >>> sentence2 = 'the cat chased the dog on the rug'.split()

    >>> for t in sr.parse(sentence1):
    ...     print(t)
    (S (NP the (N cat)) (VP (V chased) (NP the (N dog))))

#The shift reduce parser uses heuristics to decide 
#what to do when there are multiple possible shift or reduce operations available - for the supplied grammar clearly the wrong operation is selected.

    >>> for t in sr.parse(sentence2):
        print(t)

##Context Free Grammar - Details 
 
#Read as 'S' start consists of NP then VP, VP consists of ...
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)
 
#| (or) is an abbreviation for the two productions VP -> V NP and VP -> V NP PP.

##Recursive Descent Parsing

#RecursiveDescentParser() takes an optional parameter trace. 
#If trace is greater than zero, then the parser will report the steps that it takes as it parses a text.


#Recursive descent parsing has three key shortcomings. 
#First, left-recursive productions like NP -> NP PP send it into an infinite loop. 
#Second, the parser wastes a lot of time considering words and structures that do not correspond to the input sentence. 
#Third, the backtracking process may discard parsed constituents that will need to be rebuilt again later. 

#Recursive descent parsing is a kind of top-down parsing. 
#Top-down parsers use a grammar to predict what the input will be, before inspecting the input


>>> sent = "Mary saw Bob".split()
>>> rd_parser = nltk.RecursiveDescentParser(grammar1)
>>> for tree in rd_parser.parse(sent):  #each token(word) is fed into parse 
        print(tree)
(S (NP Mary) (VP (V saw) (NP Bob)))
 
#Meaning of the phrases  
S       sentence                the man walked 
NP      noun phrase             a dog 
VP      verb phrase             saw a park 
PP      prepositional phrase    with a telescope 
Det     determiner              the 
N       noun                    dog 
V       verb                    walked 
P       preposition             in 


#Loading grammer from file 

>> grammar1 = nltk.data.load('file:mygrammar.cfg')
>>> sent = "Mary saw Bob".split()
>>> rd_parser = nltk.RecursiveDescentParser(grammar1)
>>> for tree in rd_parser.parse(sent):
        print(tree)
 
 

##Recursion in Syntactic Structure
#A grammar is said to be recursive if a category occurring on the left hand side of a production 
#also appears on the righthand side of a production, 

#The production Nom -> Adj Nom (where Nom is the category of nominals) involves direct recursion on the category Nom, 
#whereas indirect recursion on S arises from the combination of two productions, 
#namely S -> NP VP and VP -> V S.

 

grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det Nom | PropN
  Nom -> Adj Nom | N
  VP -> V Adj | V NP | V S | V NP PP
  PP -> P NP
  PropN -> 'Buster' | 'Chatterer' | 'Joe'
  Det -> 'the' | 'a'
  N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
  Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
  V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
  P -> 'on'
  """)
 
##The Left-Corner Parser
#One of the problems with the recursive descent parser is that it goes into an infinite loop 
#when it encounters a left-recursive production. 

#A left-corner parser is a hybrid between the bottom-up and top-down approaches 

class nltk.parse.chart.BottomUpLeftCornerChartParser(grammar, **parser_args)
class nltk.parse.chart.LeftCornerChartParser(grammar, **parser_args)


##Shift-Reduce Parsing - bottom-up parsers

# a shift-reduce parser tries to find sequences of words and phrases 
#that correspond to the right hand side of a grammar production, 
#and replace them with the left-hand side, until the whole sentence is reduced to an S.


#The shift-reduce parser repeatedly pushes the next input word onto a stack -  the shift operation.
#reduce - If the top n items on the stack match the n items on the right hand side of some production, 
#then they are all popped off the stack, and the item on the left-hand side of the production is pushed on the stack. 


#NLTK provides ShiftReduceParser()
#This parser does not implement any backtracking, so it is not guaranteed to find a parse for a text, even if one exists. 
#Furthermore, it will only find at most one parse, even if more parses exist. 
#We can provide an optional trace parameter that controls how verbosely the parser reports the steps that it takes as it parses a text:

>>> sr_parser = nltk.ShiftReduceParser(grammar1)
>>> sent = 'Mary saw a dog'.split()
>>> for tree in sr_parser.parse(sent):
        print(tree)
  (S (NP Mary) (VP (V saw) (NP (Det a) (N dog))))
 
 



#A shift-reduce parser can reach a dead end and fail to find any parse, 
#even if the input sentence is well-formed according to the grammar. 
#When this happens, no input remains, and the stack contains items which cannot be reduced to an S. 
#The problem arises because there are choices made earlier that cannot be undone by the parser 

#There are two kinds of choices to be made by the parser: 
#(a) which reduction to do when more than one is possible 
#(b) whether to shift or reduce when either action is possible.






##nltk.parse contains three sub-modules for specialized kinds of parsing:
1. nltk.parser.chart defines chart parsing, 
   which uses dynamic programming to efficiently parse texts.
2. nltk.parser.probabilistic defines probabilistic parsing, 
   which associates a probability with each parse.

#Reference 
class nltk.grammar.ProbabilisticProduction(lhs, rhs, **prob)
    Bases: nltk.grammar.Production, nltk.probability.ImmutableProbabilisticMixIn
    essentially just a Production that has an associated probability, 
    which represents how likely it is that this production will be used.


class nltk.grammar.PCFG(start, productions, calculate_leftcorners=True)
    Bases: nltk.grammar.CFG
    A probabilistic context-free grammar. 
    A PCFG consists of a start state and a set of productions with probabilities
    PCFG productions use the ProbabilisticProduction class. 
    PCFGs impose the constraint that the set of productions 
    with any given left-hand-side must have probabilities that sum to 1 
    
    classmethod fromstring(input, encoding=None)
        Return a probabilistic PCFG corresponding to the input string(s).
        Parameters:	input – a grammar, either in the form of a string 
        or else as a list of strings.


nltk.grammar.induce_pcfg(start, productions)
    Induce a PCFG grammar from a list of productions.
        start (Nonterminal) – The start symbol
        productions (list(Production)) – The list of productions that defines the grammar
     
nltk.grammar.read_grammar(input, nonterm_parser, probabilistic=False, encoding=None)
    Return a pair consisting of a starting category and a list of Productions.
        input – a grammar, either in the form of a string or else as a list of strings.
        nonterm_parser – a function for parsing nonterminals. It should take a (string, position) as argument and return a (nonterminal, position) as result.
        probabilistic (bool) – are the grammar rules probabilistic?
        encoding (str) – the encoding of the grammar, if it is a binary string
 

##Example - Probabilistic CFGs:

>>> from nltk import PCFG
>>> toy_pcfg1 = PCFG.fromstring("""
        S -> NP VP [1.0]
        NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
        Det -> 'the' [0.8] | 'my' [0.2]
        N -> 'man' [0.5] | 'telescope' [0.5]
        VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
        V -> 'ate' [0.35] | 'saw' [0.65]
        PP -> P NP [1.0]
        P -> 'with' [0.61] | 'under' [0.39]
        """)
        
        
>>> from nltk.corpus import treebank
>>> from itertools import islice
>>> from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2

#Create a set of PCFG productions.

>>> grammar = PCFG.fromstring("""
    A -> B B [.3] | C B C [.7]
    B -> B D [.5] | C [.5]
    C -> 'a' [.1] | 'b' [0.9]
    D -> 'b' [1.0]
    """)
>>> prod = grammar.productions()[0]
>>> prod
A -> B B [0.3]

>>> prod.lhs()
A

>>> prod.rhs()
(B, B)

>>> print((prod.prob()))
0.3

>>> grammar.start()
A

>>> grammar.productions()
[A -> B B [0.3], A -> C B C [0.7], B -> B D [0.5], B -> C [0.5], C -> 'a' [0.1], C -> 'b' [0.9], D -> 'b' [1.0]]

#Induce some productions using parsed Treebank data.

>>> productions = []
>>> for fileid in treebank.fileids()[:2]:
        for t in treebank.parsed_sents(fileid):
            productions += t.productions()

>>> grammar = induce_pcfg(S, productions)
>>> grammar
<Grammar with 71 productions>

>>> sorted(grammar.productions(lhs=Nonterminal('PP')))[:2]
[PP -> IN NP [1.0]]
>>> sorted(grammar.productions(lhs=Nonterminal('NNP')))[:2]
[NNP -> 'Agnew' [0.0714286], NNP -> 'Consolidated' [0.0714286]]
>>> sorted(grammar.productions(lhs=Nonterminal('JJ')))[:2]
[JJ -> 'British' [0.142857], JJ -> 'former' [0.142857]]
>>> sorted(grammar.productions(lhs=Nonterminal('NP')))[:2]
[NP -> CD NNS [0.133333], NP -> DT JJ JJ NN [0.0666667]]

     

   
##Parse - Dynamic Programming Persers (memoization)
#The simple parsers discussed above suffer from limitations in both completeness and efficiency. 

#Try out the interactive chart parser application nltk.app.chartparser().

class nltk.parse.chart.ChartParser(grammar, strategy=[<nltk.parse.chart.LeafInitRule object>, <nltk.parse.chart.EmptyPredictRule object>, <nltk.parse.chart.BottomUpPredictCombineRule object>, <nltk.parse.chart.SingleEdgeFundamentalRule object>], trace=0, trace_chart_width=50, use_agenda=True, chart_class=<class 'nltk.parse.chart.Chart'>)



##nltk.parser.chart
#When a chart parser begins parsing a text, it creates a new (empty) chart, spanning the text. 
#It then incrementally adds new edges to the chart. 
#A set of “chart rules” specifies the conditions under which new edges should be added to the chart. 
#Once the chart reaches a stage where none of the chart rules adds any new edges, parsing is complete.

#Charts are encoded with the Chart class, 
#and edges are encoded with the TreeEdge and LeafEdge classes. 


class nltk.parse.chart.ChartParser(grammar, strategy=[<nltk.parse.chart.LeafInitRule object>, <nltk.parse.chart.EmptyPredictRule object>, <nltk.parse.chart.BottomUpPredictCombineRule object>, <nltk.parse.chart.SingleEdgeFundamentalRule object>], trace=0, trace_chart_width=50, use_agenda=True, chart_class=<class 'nltk.parse.chart.Chart'>)
    Bases: nltk.parse.api.ParserI
    A generic chart parser. 
    A “strategy”, or list of ChartRuleI instances, is used to decide what edges to add to the chart. 

    chart_parse(tokens, trace=None)
        Return the final parse Chart from which all possible parse trees can be extracted.
        tokens (list(str)) – The sentence to be parsed 
        Return type:Chart 
    
    grammar()
    
    parse(tokens, tree_class=<class 'nltk.tree.Tree'>)


class nltk.parse.chart.Chart(tokens)
    Bases: object
    A chart contains a set of edges, 
    and each edge encodes a single hypothesis about the structure of some portion of the sentence.

    edges()/ iteredges()
        Return a list of all edges/iterator over the edges in this chart. 
        New edges that are added to the chart after the call to edges() will not be contained in this list.
        Return type:list(EdgeI) or iter(EdgeI) 

    leaf(index)
        Return the leaf value of the word at the given index.
        Return type:str 

    leaves()
        Return a list of the leaf values of each word in the chart’s sentence.
        Return type:list(str) 

    num_edges()
        Return the number of edges contained in this chart.

    num_leaves()
        Return the number of words in this chart’s sentence.

    parses(root, tree_class=<class 'nltk.tree.Tree'>)
        Return an iterator of the complete tree structures 
        that span the entire chart, and whose root node is root.

    pretty_format(width=None)
        Return a pretty-printed string representation of this chart.

    pretty_format_edge(edge, width=None)
        Return a pretty-printed string representation of a given edge in this chart.

    pretty_format_leaves(width=None)
        Return a pretty-printed string representation of this chart’s leaves. 
        This string can be used as a header for calls to pretty_format_edge.
        
    child_pointer_lists(edge)
        Return the set of child pointer lists for the given edge. 
        Each child pointer list is a list of edges that have been used to form this edge.
        Return type:	list(list(EdgeI))

    initialize()
        Clear the chart.

    insert(edge, *child_pointer_lists)
        Add a new edge to the chart, 
        and return True if this operation modified the chart. In particular, return true iff the chart did not already contain edge, or if it did not already associate child_pointer_lists with edge.
            edge (EdgeI) – The new edge
            child_pointer_lists (sequence of tuple(EdgeI)) – A sequence of lists of the edges that were used to form this edge. This list is used to reconstruct the trees (or partial trees) that are associated with edge.
        Return type: bool

    insert_with_backpointer(new_edge, previous_edge, child_edge)
        Add a new edge to the chart, using a pointer to the previous edge.

    select(**restrictions)
        Return an iterator over the edges in this chart. 
        Any new edges that are added to the chart before the iterator is exahusted will also be generated. restrictions can be used to restrict the set of edges that will be generated.
            span – Only generate edges e where e.span()==span
            start – Only generate edges e where e.start()==start
            end – Only generate edges e where e.end()==end
            length – Only generate edges e where e.length()==length
            lhs – Only generate edges e where e.lhs()==lhs
            rhs – Only generate edges e where e.rhs()==rhs
            nextsym – Only generate edges e where e.nextsym()==nextsym
            dot – Only generate edges e where e.dot()==dot
            is_complete – Only generate edges e where e.is_complete()==is_complete
            is_incomplete – Only generate edges e where e.is_incomplete()==is_incomplete
        Return type: iter(EdgeI)

    trees(edge, tree_class=<class 'nltk.tree.Tree'>, complete=False)
        Return an iterator of the tree structures that are associated with edge.
        If edge is incomplete, then the unexpanded children will be encoded as childless subtrees, whose node value is the corresponding terminal or nonterminal.
        Return type:	list(Tree)
        Note:	If two trees share a common subtree, then the same Tree may be used to encode that subtree in both trees. If you need to eliminate this subtree sharing, then create a deep copy of each tree.
        
    
    
class nltk.parse.chart.EdgeI
    Bases: object
    A hypothesis about the structure of part of a sentence. 
    Each edge records the fact that a structure is (partially) consistent with the sentence. An edge contains:
        A span, indicating what part of the sentence is consistent with the hypothesized structure.
        A left-hand side, specifying what kind of structure is hypothesized.
        A right-hand side, specifying the contents of the hypothesized structure.
        A dot position, indicating how much of the hypothesized structure is consistent with the sentence.

    Every edge is either complete or incomplete:
        An edge is complete if its structure is fully consistent with the sentence.
        An edge is incomplete if its structure is partially consistent with the sentence. For every incomplete edge, the span specifies a possible prefix for the edge’s structure.

    There are two kinds of edge:
        A TreeEdge records which trees have been found to be (partially) consistent with the text.
        A LeafEdge records the tokens occurring in the text.
 
    dot()
        Return this edge’s dot position, which indicates how much of the hypothesized structure is consistent with the sentence. In particular, self.rhs[:dot] is consistent with tokens[self.start():self.end()].
        Return type:	int

    end()
        Return the end index of this edge’s span.
        Return type:	int

    is_complete()
        Return True if this edge’s structure is fully consistent with the text.
        Return type:	bool

    is_incomplete()
        Return True if this edge’s structure is partially consistent with the text.
        Return type:	bool

    length()
        Return the length of this edge’s span.
        Return type:	int

    lhs()
        Return this edge’s left-hand side, which specifies what kind of structure is hypothesized by this edge.
        
    nextsym()
        Return the element of this edge’s right-hand side that immediately follows its dot.
        Return type:	Nonterminal or terminal or None

    rhs()
        Return this edge’s right-hand side, 
        which specifies the content of the structure hypothesized by this edge.
        
    span()
        Return a tuple (s, e), 
        where tokens[s:e] is the portion of the sentence that 
        is consistent with this edge’s structure.
        Return type:	tuple(int, int)

    start()
        Return the start index of this edge’s span.
        Return type:	int
        
        
class nltk.parse.chart.TreeEdge(span, lhs, rhs, dot=0)
    Bases: nltk.parse.chart.EdgeI
    An edge that records the fact that a tree is (partially) consistent with the sentence.
    
    static from_production(production, index)
        Return a new TreeEdge formed from the given production. 
        The new edge’s left-hand side and right-hand side will be taken from production; its span will be (index,index); and its dot position will be 0.
        Return type:	TreeEdge
    
##Subclases of Chart Parser - based on various stragey
class nltk.parse.chart.BottomUpChartParser(grammar, **parser_args)
    Bases: nltk.parse.chart.ChartParser
    
class nltk.parse.chart.BottomUpLeftCornerChartParser(grammar, **parser_args)
    Bases: nltk.parse.chart.ChartParser
    
class nltk.parse.chart.LeftCornerChartParser(grammar, **parser_args)
    Bases: nltk.parse.chart.ChartParser
    
class nltk.parse.chart.SteppingChartParser(grammar, strategy=[], trace=0)
    Bases: nltk.parse.chart.ChartParser
    A ChartParser that allows you to step through the parsing process, 
    adding a single edge at a time
    
    step()
        Return a generator that adds edges to the chart, one at a time

class nltk.parse.chart.TopDownChartParser(grammar, **parser_args)
    Bases: nltk.parse.chart.ChartParser

##Example - chart parser 

from nltk.parse.chart import *

def demo_grammar():
    from nltk.grammar import CFG
    return CFG.fromstring("""
S  -> NP VP
PP -> "with" NP
NP -> NP PP
VP -> VP PP
VP -> Verb NP
VP -> Verb
NP -> Det Noun
NP -> "John"
NP -> "I"
Det -> "the"
Det -> "my"
Det -> "a"
Noun -> "dog"
Noun -> "cookie"
Verb -> "ate"
Verb -> "saw"
Prep -> "with"
Prep -> "under"
""")

def demo(choice=None,
         print_times=True, print_grammar=False,
         print_trees=True, trace=2,
         sent='I saw John with a dog with my cookie', numparses=5):
    """
    A demonstration of the chart parsers.
    """
    import sys, time
    from nltk import nonterminals, Production, CFG

    # The grammar for ChartParser and SteppingChartParser:
    grammar = demo_grammar()
    if print_grammar:
        print("* Grammar")
        print(grammar)

    # Tokenize the sample sentence.
    print("* Sentence:")
    print(sent)
    tokens = sent.split()
    print(tokens)
    print()

    # Ask the user which parser to test,
    # if the parser wasn't provided as an argument
    if choice is None:
        print('  1: Top-down chart parser')
        print('  2: Bottom-up chart parser')
        print('  3: Bottom-up left-corner chart parser')
        print('  4: Left-corner chart parser with bottom-up filter')
        print('  5: Stepping chart parser (alternating top-down & bottom-up)')
        print('  6: All parsers')
        print('\nWhich parser (1-6)? ', end=' ')
        choice = sys.stdin.readline().strip()
        print()

    choice = str(choice)
    if choice not in "123456":
        print('Bad parser number')
        return

    # Keep track of how long each parser takes.
    times = {}

    strategies = {'1': ('Top-down', TD_STRATEGY),
                  '2': ('Bottom-up', BU_STRATEGY),
                  '3': ('Bottom-up left-corner', BU_LC_STRATEGY),
                  '4': ('Filtered left-corner', LC_STRATEGY)}
    choices = []
    if choice in strategies: choices = [choice]
    if choice=='6': choices = "1234"

    # Run the requested chart parser(s), except the stepping parser.
    for strategy in choices:
        print("* Strategy: " + strategies[strategy][0])
        print()
        cp = ChartParser(grammar, strategies[strategy][1], trace=trace)
        t = time.time()
        chart = cp.chart_parse(tokens)
        parses = list(chart.parses(grammar.start()))

        times[strategies[strategy][0]] = time.time()-t
        print("Nr edges in chart:", len(chart.edges()))
        if numparses:
            assert len(parses)==numparses, 'Not all parses found'
        if print_trees:
            for tree in parses: print(tree)
        else:
            print("Nr trees:", len(parses))
        print()

    # Run the stepping parser, if requested.
    if choice in "56":
        print("* Strategy: Stepping (top-down vs bottom-up)")
        print()
        t = time.time()
        cp = SteppingChartParser(grammar, trace=trace)
        cp.initialize(tokens)
        for i in range(5):
            print('*** SWITCH TO TOP DOWN')
            cp.set_strategy(TD_STRATEGY)
            for j, e in enumerate(cp.step()):
                if j>20 or e is None: break
            print('*** SWITCH TO BOTTOM UP')
            cp.set_strategy(BU_STRATEGY)
            for j, e in enumerate(cp.step()):
                if j>20 or e is None: break
        times['Stepping'] = time.time()-t
        print("Nr edges in chart:", len(cp.chart().edges()))
        if numparses:
            assert len(list(cp.parses()))==numparses, 'Not all parses found'
        if print_trees:
            for tree in cp.parses(): print(tree)
        else:
            print("Nr trees:", len(list(cp.parses())))
        print()

    # Print the times of all parsers:
    if not (print_times and times): return
    print("* Parsing times")
    print()
    maxlen = max(len(key) for key in times)
    format = '%' + repr(maxlen) + 's parser: %6.3fsec'
    times_items = times.items()
    for (parser, t) in sorted(times_items, key=lambda a:a[1]):
        print(format % (parser, t))
        
        
#usage         
>>> import nltk

>>> nltk.parse.chart.demo(2, print_times=False, trace=1,
...                       sent='I saw a dog', numparses=1)
* Sentence:
I saw a dog
['I', 'saw', 'a', 'dog']
<BLANKLINE>
* Strategy: Bottom-up
<BLANKLINE>
|.    I    .   saw   .    a    .   dog   .|
|[---------]         .         .         .| [0:1] 'I'
|.         [---------]         .         .| [1:2] 'saw'
|.         .         [---------]         .| [2:3] 'a'
|.         .         .         [---------]| [3:4] 'dog'
|>         .         .         .         .| [0:0] NP -> * 'I'
|[---------]         .         .         .| [0:1] NP -> 'I' *
|>         .         .         .         .| [0:0] S  -> * NP VP
|>         .         .         .         .| [0:0] NP -> * NP PP
|[--------->         .         .         .| [0:1] S  -> NP * VP
|[--------->         .         .         .| [0:1] NP -> NP * PP
|.         >         .         .         .| [1:1] Verb -> * 'saw'
|.         [---------]         .         .| [1:2] Verb -> 'saw' *
|.         >         .         .         .| [1:1] VP -> * Verb NP
|.         >         .         .         .| [1:1] VP -> * Verb
|.         [--------->         .         .| [1:2] VP -> Verb * NP
|.         [---------]         .         .| [1:2] VP -> Verb *
|.         >         .         .         .| [1:1] VP -> * VP PP
|[-------------------]         .         .| [0:2] S  -> NP VP *
|.         [--------->         .         .| [1:2] VP -> VP * PP
|.         .         >         .         .| [2:2] Det -> * 'a'
|.         .         [---------]         .| [2:3] Det -> 'a' *
|.         .         >         .         .| [2:2] NP -> * Det Noun
|.         .         [--------->         .| [2:3] NP -> Det * Noun
|.         .         .         >         .| [3:3] Noun -> * 'dog'
|.         .         .         [---------]| [3:4] Noun -> 'dog' *
|.         .         [-------------------]| [2:4] NP -> Det Noun *
|.         .         >         .         .| [2:2] S  -> * NP VP
|.         .         >         .         .| [2:2] NP -> * NP PP
|.         [-----------------------------]| [1:4] VP -> Verb NP *
|.         .         [------------------->| [2:4] S  -> NP * VP
|.         .         [------------------->| [2:4] NP -> NP * PP
|[=======================================]| [0:4] S  -> NP VP *
|.         [----------------------------->| [1:4] VP -> VP * PP
Nr edges in chart: 33
(S (NP I) (VP (Verb saw) (NP (Det a) (Noun dog))))
<BLANKLINE>

#Top-down

>>> nltk.parse.chart.demo(1, print_times=False, trace=0,
            sent='I saw John with a dog', numparses=2)
* Sentence:
I saw John with a dog
['I', 'saw', 'John', 'with', 'a', 'dog']
<BLANKLINE>
* Strategy: Top-down
<BLANKLINE>
Nr edges in chart: 48
(S
  (NP I)
  (VP (Verb saw) (NP (NP John) (PP with (NP (Det a) (Noun dog))))))
(S
  (NP I)
  (VP (VP (Verb saw) (NP John)) (PP with (NP (Det a) (Noun dog)))))
<BLANKLINE>

Bottom-up

>>> nltk.parse.chart.demo(2, print_times=False, trace=0,
                sent='I saw John with a dog', numparses=2)
* Sentence:
I saw John with a dog
['I', 'saw', 'John', 'with', 'a', 'dog']
<BLANKLINE>
* Strategy: Bottom-up
<BLANKLINE>
Nr edges in chart: 53
(S
  (NP I)
  (VP (VP (Verb saw) (NP John)) (PP with (NP (Det a) (Noun dog)))))
(S
  (NP I)
  (VP (Verb saw) (NP (NP John) (PP with (NP (Det a) (Noun dog))))))
<BLANKLINE>

#Bottom-up Left-Corner

>>> nltk.parse.chart.demo(3, print_times=False, trace=0,
            sent='I saw John with a dog', numparses=2)
* Sentence:
I saw John with a dog
['I', 'saw', 'John', 'with', 'a', 'dog']
<BLANKLINE>
* Strategy: Bottom-up left-corner
<BLANKLINE>
Nr edges in chart: 36
(S
  (NP I)
  (VP (VP (Verb saw) (NP John)) (PP with (NP (Det a) (Noun dog)))))
(S
  (NP I)
  (VP (Verb saw) (NP (NP John) (PP with (NP (Det a) (Noun dog))))))
<BLANKLINE>

#Left-Corner with Bottom-Up Filter

>>> nltk.parse.chart.demo(4, print_times=False, trace=0,
            sent='I saw John with a dog', numparses=2)
* Sentence:
I saw John with a dog
['I', 'saw', 'John', 'with', 'a', 'dog']
<BLANKLINE>
* Strategy: Filtered left-corner
<BLANKLINE>
Nr edges in chart: 28
(S
  (NP I)
  (VP (VP (Verb saw) (NP John)) (PP with (NP (Det a) (Noun dog)))))
(S
  (NP I)
  (VP (Verb saw) (NP (NP John) (PP with (NP (Det a) (Noun dog))))))
<BLANKLINE>

#The stepping chart parser

>>> nltk.parse.chart.demo(5, print_times=False, trace=1,
                sent='I saw John with a dog', numparses=2)
* Sentence:
I saw John with a dog
['I', 'saw', 'John', 'with', 'a', 'dog']
<BLANKLINE>
* Strategy: Stepping (top-down vs bottom-up)
<BLANKLINE>
*** SWITCH TO TOP DOWN
|[------]      .      .      .      .      .| [0:1] 'I'
|.      [------]      .      .      .      .| [1:2] 'saw'
|.      .      [------]      .      .      .| [2:3] 'John'
|.      .      .      [------]      .      .| [3:4] 'with'
|.      .      .      .      [------]      .| [4:5] 'a'
|.      .      .      .      .      [------]| [5:6] 'dog'
|>      .      .      .      .      .      .| [0:0] S  -> * NP VP
|>      .      .      .      .      .      .| [0:0] NP -> * NP PP
|>      .      .      .      .      .      .| [0:0] NP -> * Det Noun
|>      .      .      .      .      .      .| [0:0] NP -> * 'I'
|[------]      .      .      .      .      .| [0:1] NP -> 'I' *
|[------>      .      .      .      .      .| [0:1] S  -> NP * VP
|[------>      .      .      .      .      .| [0:1] NP -> NP * PP
|.      >      .      .      .      .      .| [1:1] VP -> * VP PP
|.      >      .      .      .      .      .| [1:1] VP -> * Verb NP
|.      >      .      .      .      .      .| [1:1] VP -> * Verb
|.      >      .      .      .      .      .| [1:1] Verb -> * 'saw'
|.      [------]      .      .      .      .| [1:2] Verb -> 'saw' *
|.      [------>      .      .      .      .| [1:2] VP -> Verb * NP
|.      [------]      .      .      .      .| [1:2] VP -> Verb *
|[-------------]      .      .      .      .| [0:2] S  -> NP VP *
|.      [------>      .      .      .      .| [1:2] VP -> VP * PP
*** SWITCH TO BOTTOM UP
|.      .      >      .      .      .      .| [2:2] NP -> * 'John'
|.      .      .      >      .      .      .| [3:3] PP -> * 'with' NP
|.      .      .      >      .      .      .| [3:3] Prep -> * 'with'
|.      .      .      .      >      .      .| [4:4] Det -> * 'a'
|.      .      .      .      .      >      .| [5:5] Noun -> * 'dog'
|.      .      [------]      .      .      .| [2:3] NP -> 'John' *
|.      .      .      [------>      .      .| [3:4] PP -> 'with' * NP
|.      .      .      [------]      .      .| [3:4] Prep -> 'with' *
|.      .      .      .      [------]      .| [4:5] Det -> 'a' *
|.      .      .      .      .      [------]| [5:6] Noun -> 'dog' *
|.      [-------------]      .      .      .| [1:3] VP -> Verb NP *
|[--------------------]      .      .      .| [0:3] S  -> NP VP *
|.      [------------->      .      .      .| [1:3] VP -> VP * PP
|.      .      >      .      .      .      .| [2:2] S  -> * NP VP
|.      .      >      .      .      .      .| [2:2] NP -> * NP PP
|.      .      .      .      >      .      .| [4:4] NP -> * Det Noun
|.      .      [------>      .      .      .| [2:3] S  -> NP * VP
|.      .      [------>      .      .      .| [2:3] NP -> NP * PP
|.      .      .      .      [------>      .| [4:5] NP -> Det * Noun
|.      .      .      .      [-------------]| [4:6] NP -> Det Noun *
|.      .      .      [--------------------]| [3:6] PP -> 'with' NP *
|.      [----------------------------------]| [1:6] VP -> VP PP *
*** SWITCH TO TOP DOWN
|.      .      >      .      .      .      .| [2:2] NP -> * Det Noun
|.      .      .      .      >      .      .| [4:4] NP -> * NP PP
|.      .      .      >      .      .      .| [3:3] VP -> * VP PP
|.      .      .      >      .      .      .| [3:3] VP -> * Verb NP
|.      .      .      >      .      .      .| [3:3] VP -> * Verb
|[=========================================]| [0:6] S  -> NP VP *
|.      [---------------------------------->| [1:6] VP -> VP * PP
|.      .      [---------------------------]| [2:6] NP -> NP PP *
|.      .      .      .      [------------->| [4:6] NP -> NP * PP
|.      [----------------------------------]| [1:6] VP -> Verb NP *
|.      .      [--------------------------->| [2:6] S  -> NP * VP
|.      .      [--------------------------->| [2:6] NP -> NP * PP
|[=========================================]| [0:6] S  -> NP VP *
|.      [---------------------------------->| [1:6] VP -> VP * PP
|.      .      .      .      .      .      >| [6:6] VP -> * VP PP
|.      .      .      .      .      .      >| [6:6] VP -> * Verb NP
|.      .      .      .      .      .      >| [6:6] VP -> * Verb
*** SWITCH TO BOTTOM UP
|.      .      .      .      >      .      .| [4:4] S  -> * NP VP
|.      .      .      .      [------------->| [4:6] S  -> NP * VP
*** SWITCH TO TOP DOWN
*** SWITCH TO BOTTOM UP
*** SWITCH TO TOP DOWN
*** SWITCH TO BOTTOM UP
*** SWITCH TO TOP DOWN
*** SWITCH TO BOTTOM UP
Nr edges in chart: 61
(S
  (NP I)
  (VP (VP (Verb saw) (NP John)) (PP with (NP (Det a) (Noun dog)))))
(S
  (NP I)
  (VP (Verb saw) (NP (NP John) (PP with (NP (Det a) (Noun dog))))))
<BLANKLINE>

##Chart Parser - for LARGE context-free grammars

#Reference 

nltk.parse.util.extract_test_sentences(string, comment_chars='#%;', encoding=None)¶
    Parses a string with one test sentence per line. 
    Lines can optionally begin with:
            a bool, saying if the sentence is grammatical or not, or
            an int, giving the number of parse trees is should have,
    The result information is followed by a colon, and then the sentence. 
    Empty lines and lines beginning with a comment char are ignored.
    Returns:
    a list of tuple of sentences and expected results, 
    where a sentence is a list of str, and a result is None, or bool, or int
    Parameters:
        comment_chars – str of possible comment characters.
        encoding – the encoding of the string, if it is binary

nltk.parse.util.load_parser(grammar_url, trace=0, parser=None, chart_class=None, beam_size=0, **load_args)
    Load a grammar from a file, and build a parser based on that grammar. 
    The parser depends on the grammar format, 
    and might also depend on properties of the grammar itself.
    The following grammar formats are currently supported:
            'cfg' (CFGs: CFG)
            'pcfg' (probabilistic CFGs: PCFG)
            'fcfg' (feature-based CFGs: FeatureGrammar)

    Parameters:	
        grammar_url (str) – A URL specifying where the grammar is located. 
                            The default protocol is "nltk:", which searches for the file in the the NLTK data package.
        trace (int) – The level of tracing that should be used when parsing a text. 0 will generate no tracing output; and higher numbers will produce more verbose tracing output.
        parser – The class used for parsing; should be ChartParser or a subclass. If None, the class depends on the grammar format.
        chart_class – The class used for storing the chart; should be Chart or a subclass. Only used for CFGs and feature CFGs. If None, the chart class depends on the grammar format.
        beam_size (int) – The maximum length for the parser’s edge queue. Only used for probabilistic CFGs.
        load_args – Keyword parameters used when loading the grammar. See data.load for more information.

nltk.parse.util.taggedsent_to_conll(sentence)
    A module to convert a single POS tagged sentence into CONLL format.
    >>> from nltk import word_tokenize, pos_tag
    >>> text = "This is a foobar sentence."
    >>> for line in taggedsent_to_conll(pos_tag(word_tokenize(text))):
    ...     print(line, end="")
    1       This    _       DT      DT      _       0       a       _       _
    2       is      _       VBZ     VBZ     _       0       a       _       _
    3       a       _       DT      DT      _       0       a       _       _
    4       foobar  _       JJ      JJ      _       0       a       _       _
    5       sentence        _       NN      NN      _       0       a       _       _
    6       .               _       .       .       _       0       a       _       _

    Parameters:	sentence (list(tuple(str, str))) – A single input sentence to parse
    Return type:	iter(str)
    Returns:	a generator yielding a single sentence in CONLL format.

nltk.parse.util.taggedsents_to_conll(sentences)
    A module to convert the a POS tagged document stream (i.e. list of list of tuples, a list of sentences) and yield lines in CONLL format. This module yields one line per word and two newlines for end of sentence.
    >>> from nltk import word_tokenize, sent_tokenize, pos_tag
    >>> text = "This is a foobar sentence. Is that right?"
    >>> sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
    >>> for line in taggedsents_to_conll(sentences):
    ...     if line:
    ...         print(line, end="")
    1       This    _       DT      DT      _       0       a       _       _
    2       is      _       VBZ     VBZ     _       0       a       _       _
    3       a       _       DT      DT      _       0       a       _       _
    4       foobar  _       JJ      JJ      _       0       a       _       _
    5       sentence        _       NN      NN      _       0       a       _       _
    6       .               _       .       .       _       0       a       _       _


    1       Is      _       VBZ     VBZ     _       0       a       _       _
    2       that    _       IN      IN      _       0       a       _       _
    3       right   _       NN      NN      _       0       a       _       _
    4       ?       _       .       .       _       0       a       _       _

    Parameters:	sentences – Input sentences to parse
    Return type:	iter(str)
    Returns:	a generator yielding sentences in CONLL format.


#Reading the ATIS grammar.

>>> grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
>>> grammar
<Grammar with 5517 productions>

#Reading the test sentences.

>>> sentences = nltk.data.load('grammars/large_grammars/atis_sentences.txt')
>>> sentences = nltk.parse.util.extract_test_sentences(sentences)
>>> len(sentences)
98
>>> testsentence = sentences[22]
>>> testsentence[0]
['show', 'me', 'northwest', 'flights', 'to', 'detroit', '.']
>>> testsentence[1]  #number of parse trees is should have
17
>>> sentence = testsentence[0]

#Note that the number of edges differ between the strategies.
#Bottom-up parsing.
>>> parser = nltk.parse.BottomUpChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
7661
>>> print((len(list(chart.parses(grammar.start())))))
17

#Bottom-up Left-corner parsing.
>>> parser = nltk.parse.BottomUpLeftCornerChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
4986
>>> print((len(list(chart.parses(grammar.start())))))
17

#Left-corner parsing with bottom-up filter.
>>> parser = nltk.parse.LeftCornerChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
1342
>>> print((len(list(chart.parses(grammar.start())))))
17

#Top-down parsing.
>>> parser = nltk.parse.TopDownChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
28352
>>> print((len(list(chart.parses(grammar.start())))))
17

#Incremental Bottom-up parsing.
>>> parser = nltk.parse.IncrementalBottomUpChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
7661
>>> print((len(list(chart.parses(grammar.start())))))
17

#Incremental Bottom-up Left-corner parsing.
>>> parser = nltk.parse.IncrementalBottomUpLeftCornerChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
4986
>>> print((len(list(chart.parses(grammar.start())))))
17

#Incremental Left-corner parsing with bottom-up filter.
>>> parser = nltk.parse.IncrementalLeftCornerChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
1342
>>> print((len(list(chart.parses(grammar.start())))))
17

#Incremental Top-down parsing.
>>> parser = nltk.parse.IncrementalTopDownChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
28352
>>> print((len(list(chart.parses(grammar.start())))))
17

#Earley parsing. This is similar to the incremental top-down algorithm.
>>> parser = nltk.parse.EarleyChartParser(grammar)
>>> chart = parser.chart_parse(sentence)
>>> print((chart.num_edges()))
28352
>>> print((len(list(chart.parses(grammar.start())))))
17


##Reference - nltk.parse.pchart module


class nltk.parse.pchart.BottomUpProbabilisticChartParser(grammar, beam_size=0, trace=0)
    Bases: nltk.parse.api.ParserI
    An abstract bottom-up parser for PCFG grammars 
    that uses a Chart to record partial results. 
    BottomUpProbabilisticChartParser maintains a queue of edges 
    that can be added to the chart. 
    This queue is initialized with edges for each token in the text 
    that is being parsed. 
    BottomUpProbabilisticChartParser inserts these edges into the chart one 
    at a time, starting with the most likely edges, 
    and proceeding to less likely edges.
    For each edge that is added to the chart, 
    it may become possible to insert additional edges into the chart; 
    these are added to the queue. 
    This process continues until enough complete parses have been generated, 
    or until the queue is empty.
    The BottomUpProbabilisticChartParser constructor has an optional argument beam_size. 
    If non-zero, this controls the size of the beam (aka the edge queue). 
    This option is most useful with InsideChartParser.
 
    grammar()

    parse(tokens)

    sort_queue(queue, chart)
        Sort the given queue of Edge objects, 
        placing the edge that should be tried first at the beginning of the queue. This method will be called after each Edge is added to the queue.
        Parameters:	
            queue (list(Edge)) – The queue of Edge objects to sort. Each edge in this queue is an edge that could be added to the chart by the fundamental rule; but that has not yet been added.
            chart (Chart) – The chart being used to parse the text. This chart can be used to provide extra information for sorting the queue.
        Return type: None

    trace(trace=2)
        Set the level of tracing output that should be generated when parsing a text.
        Parameters:	trace (int) – The trace level. A trace level of 0 will generate no tracing output; and higher trace levels will produce more verbose tracing output.
        Return type:	None
        
#subclass of BottomUpProbabilisticChartParser
1.InsideChartParser searches edges in decreasing order of their trees’ inside probabilities.
2.RandomChartParser searches edges in random order.
3.LongestChartParser searches edges in decreasing order of their location’s length.
4.UnsortedChartParser A bottom-up parser for PCFG grammars that tries edges in whatever order.

#Example 

>>> tokens = "Jack saw Bob with my cookie".split()
>>> grammar = toy_pcfg2
>>> print(grammar)
Grammar with 23 productions (start state = S)
    S -> NP VP [1.0]
    VP -> V NP [0.59]
    VP -> V [0.4]
    VP -> VP PP [0.01]
    NP -> Det N [0.41]
    NP -> Name [0.28]
    NP -> NP PP [0.31]
    PP -> P NP [1.0]
    V -> 'saw' [0.21]
    V -> 'ate' [0.51]
    V -> 'ran' [0.28]
    N -> 'boy' [0.11]
    N -> 'cookie' [0.12]
    N -> 'table' [0.13]
    N -> 'telescope' [0.14]
    N -> 'hill' [0.5]
    Name -> 'Jack' [0.52]
    Name -> 'Bob' [0.48]
    P -> 'with' [0.61]
    P -> 'under' [0.39]
    Det -> 'the' [0.41]
    Det -> 'a' [0.31]
    Det -> 'my' [0.28]


>>> from nltk.parse import pchart

>>> parser = pchart.InsideChartParser(grammar)
>>> for t in parser.parse(tokens):
        print(t)
(S
  (NP (Name Jack))
  (VP
    (V saw)
    (NP
      (NP (Name Bob))
      (PP (P with) (NP (Det my) (N cookie)))))) (p=6.31607e-06)
(S
  (NP (Name Jack))
  (VP
    (VP (V saw) (NP (Name Bob)))
    (PP (P with) (NP (Det my) (N cookie))))) (p=2.03744e-07)

>>> parser = pchart.RandomChartParser(grammar)
>>> for t in parser.parse(tokens):
        print(t)
(S
  (NP (Name Jack))
  (VP
    (V saw)
    (NP
      (NP (Name Bob))
      (PP (P with) (NP (Det my) (N cookie)))))) (p=6.31607e-06)
(S
  (NP (Name Jack))
  (VP
    (VP (V saw) (NP (Name Bob)))
    (PP (P with) (NP (Det my) (N cookie))))) (p=2.03744e-07)

>>> parser = pchart.UnsortedChartParser(grammar)
>>> for t in parser.parse(tokens):
        print(t)
(S
  (NP (Name Jack))
  (VP
    (V saw)
    (NP
      (NP (Name Bob))
      (PP (P with) (NP (Det my) (N cookie)))))) (p=6.31607e-06)
(S
  (NP (Name Jack))
  (VP
    (VP (V saw) (NP (Name Bob)))
    (PP (P with) (NP (Det my) (N cookie))))) (p=2.03744e-07)

>>> parser = pchart.LongestChartParser(grammar)
>>> for t in parser.parse(tokens):
...     print(t)
(S
  (NP (Name Jack))
  (VP
    (V saw)
    (NP
      (NP (Name Bob))
      (PP (P with) (NP (Det my) (N cookie)))))) (p=6.31607e-06)
(S
  (NP (Name Jack))
  (VP
    (VP (V saw) (NP (Name Bob)))
    (PP (P with) (NP (Det my) (N cookie))))) (p=2.03744e-07)

>>> parser = pchart.InsideChartParser(grammar, beam_size = len(tokens)+1)
>>> for t in parser.parse(tokens):
...     print(t)
  


##Parse - Dependencies and Dependency Grammar
#Phrase structure grammar(eg CFG) is concerned 
#with how words and sequences of words combine to form constituents. 

#A complementary approach, dependency grammar, focusses instead 
#on how words relate to other words. 


#Dependency is a binary asymmetric relation that holds between a head and its dependents. 
#The head of a sentence is usually taken to be the tensed verb, 
#and every other word is either dependent on the sentence head, 
#or connects to it through a path of dependencies


class nltk.grammar.DependencyProduction(lhs, rhs)
    Bases: nltk.grammar.Production
    A dependency grammar production. 
    Each production maps a single head word to an unordered list of one 
    or more modifier words.


class nltk.grammar.DependencyGrammar(productions)
    Bases: object
    A dependency grammar. 
    A DependencyGrammar consists of a set of productions. 
    Each production specifies a head/modifier relationship 
    between a pair of words.

    contains(head, mod)
        head (str) – A head word.
        mod (str) – A mod word, to test as a modifier of ‘head’.
        Returns: true if this DependencyGrammar contains a DependencyProduction mapping ‘head’ to ‘mod’.
         
    classmethod fromstring(input)

class nltk.grammar.ProbabilisticDependencyGrammar(productions, events, tags)
    Bases: object
    
    contains(head, mod)
        Return True if this DependencyGrammar 
        contains a DependencyProduction mapping ‘head’ to ‘mod’.



#for sent - 'I shot an elephant in my pajamas'

>>> groucho_dep_grammar = nltk.DependencyGrammar.fromstring("""
        'shot' -> 'I' | 'elephant' | 'in'
        'elephant' -> 'an' | 'in'
        'in' -> 'pajamas'
        'pajamas' -> 'my'
        """)
>>> print(groucho_dep_grammar)
Dependency grammar with 7 productions
  'shot' -> 'I'
  'shot' -> 'elephant'
  'shot' -> 'in'
  'elephant' -> 'an'
  'elephant' -> 'in'
  'in' -> 'pajamas'
  'pajamas' -> 'my'
 
 
##Dependency Parsers 

class nltk.parse.nonprojectivedependencyparser.NonprojectiveDependencyParser(dependency_grammar)
    Bases: object
    A non-projective, rule-based, dependency parser. 
    This parser will return the set of all possible non-projective parses 
    based on the word-to-word relations defined in the parser’s dependency grammar, 
    and will allow the branches of the parse tree to cross in order 
    to capture a variety of linguistic phenomena that a projective parser will not.

    parse(tokens)
        Parses the input tokens with respect to the parser’s grammar. 
        param tokens: A list of tokens to parse. type tokens: list(str) 
        return: An iterator of non-projective parses. rtype: iter(DependencyGraph)

class nltk.parse.projectivedependencyparser.ProjectiveDependencyParser(dependency_grammar)
    Bases: object
    A projective, rule-based, dependency parser. 
    A ProjectiveDependencyParser is created with a DependencyGrammar, 
    a set of productions specifying word-to-word dependency relations. 
    
    concatenate(span1, span2)
        Concatenates the two spans in whichever way possible. 
        This includes rightward concatenation (from the leftmost word of the leftmost span to the rightmost word of the rightmost span) 
        and leftward concatenation (vice-versa) between adjacent spans. 
        Unlike Eisner’s presentation of span concatenation, 
        these spans do not share or pivot on a particular word/word-index.
        Returns:	A list of new spans formed through concatenation.
        Return type:	list(DependencySpan)

    parse(tokens)
        Performs a projective dependency parse on the list of tokens 
        using a chart-based, span-concatenation algorithm similar to Eisner (1996).
        Parameters:	tokens (list(str)) – The list of input tokens.
        Returns:	An iterator over parse trees.
        Return type:	iter(Tree)


#check data/nonProjectiveDependency.png
#A dependency graph is projective if, when all the words are written in linear order, 
#the edges can be drawn above the words without crossing. 
#compare -  'I shot an elephant in my pajamas'
#with 'john saw a dog yesterday which was a Yorkshire Terrier'

#This is equivalent to saying that a word and all its descendents 
#(dependents and dependents of its dependents, etc.) 
#form a contiguous sequence of words within the sentence. 

>>> pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
>>> sent = 'I shot an elephant in my pajamas'.split()
>>> trees = pdp.parse(sent)
>>> for tree in trees:
        print(tree)
(shot I (elephant an (in (pajamas my))))
(shot I (elephant an) (in (pajamas my)))
 
 
##Understanding Verb Phrase and Dependency grammer 

#Example - for below 
a.  The squirrel was frightened. 
b.  Chatterer saw the bear. 
c.  Chatterer thought Buster was angry. 
d.  Joe put the fish on the log. 

#VP productions and their lexical heads
VP -> V Adj             was 
VP -> V NP              saw 
VP -> V S               thought 
VP -> V NP PP           put 

#'was' can occur with a following Adj, 
#'saw' can occur with a following NP, 
#'thought' can occur with a following S 
#'put' can occur with a following NP and PP. 

#The dependents Adj, NP, PP and S are called complements of the respective verbs 
#In dependency grammar term, the verbs are said to have different 'valencies'

#In a CFG, we need some way of constraining grammar productions 
#which expand VP so that verbs only co-occur with their correct complements. 

#We can do this by dividing the class of verbs into "subcategories", 
#each of which is associated with a different set of complements.


#If we introduce a new category label for transitive verbs, namely TV (for Transitive Verb), 
#then we can use it in the following productions:
VP -> TV NP
TV -> 'chased' | 'saw'

#Symbol     Meaning                 Example
IV          intransitive verb       barked 
TV          transitive verb         saw a man 
DatV        dative verb             gave a dog to a man 
SV          sentential verb         said that a dog barked 


###Scaling up - Grammar Development using Treebank 
#In linguistics, a treebank is a parsed text corpus that annotates syntactic or semantic sentence structure
#Most syntactic treebanks annotate variants of either phrase structure  or dependency structure 
#The Penn Treebank (PTB) project selected 2,499 stories from a three year Wall Street Journal (WSJ) collection of 98,732 stories for syntactic annotation



#The corpus module defines the treebank corpus reader, 
#which contains a 10% sample of the Penn Treebank corpus.

>>> from nltk.corpus import treebank
>>> t = treebank.parsed_sents('wsj_0001.mrg')[0]
>>> print(t)
(S
  (NP-SBJ
    (NP (NNP Pierre) (NNP Vinken))
    (, ,)
    (ADJP (NP (CD 61) (NNS years)) (JJ old))
    (, ,))
  (VP
    (MD will)
    (VP
      (VB join)
      (NP (DT the) (NN board))
      (PP-CLR
        (IN as)
        (NP (DT a) (JJ nonexecutive) (NN director)))
      (NP-TMP (NNP Nov.) (CD 29))))
  (. .))
 
 

#We can use above data to help develop a grammar. 
#For example, a simple filter to find verbs that take sentential complements. 

#Assuming we already have a production of the form VP -> Vs S, 
#this information enables us to identify particular verbs that would be included in the expansion of Vs.


def filter(tree):
    child_nodes = [child.label() for child in tree  if isinstance(child, nltk.Tree)]
    return  (tree.label() == 'VP') and ('S' in child_nodes)
 
 
>>> from nltk.corpus import treebank
>>> [subtree for tree in treebank.parsed_sents() for subtree in tree.subtrees(filter)]
[Tree('VP', [Tree('VBN', ['named']), Tree('S', [Tree('NP-SBJ', ...]), ...]), ...]
 
 

##The Prepositional Phrase Attachment Corpus, nltk.corpus.ppattach 
#another source of information about the valency of particular verbs

#Example - It finds pairs of prepositional phrases where the preposition and noun are fixed, 
#but where the choice of verb determines 
#whether the prepositional phrase is attached to the VP or to the NP.

>>> from collections import defaultdict
>>> entries = nltk.corpus.ppattach.attachments('training')
>>> table = defaultdict(lambda: defaultdict(set))
>>> for entry in entries:
        key = entry.noun1 + '-' + entry.prep + '-' + entry.noun2
        table[key][entry.attachment].add(entry.verb)
    
>>> for key in sorted(table):
        if len(table[key]) > 1:
            print(key, 'N:', sorted(table[key]['N']), 'V:', sorted(table[key]['V']))
 
 
#Amongst the output lines of this program 
#we find offer-from-group N: ['rejected'] V: ['received'], 
#which indicates that received expects a separate PP complement attached to the VP, 
#while rejected does not. 
# we can use this information to help construct the grammar.


##The NLTK corpus includes a sample from the Sinica Treebank Corpus, 
#consisting of 10,000 parsed sentences drawn from the Academia Sinica Balanced Corpus of Modern Chinese. 
#Let's load and display one of the trees in this corpus.

>>> nltk.corpus.sinica_treebank.parsed_sents()[3450].draw()             
 
 
##Pernicious Ambiguity
#As the coverage of the grammar increases and the length of the input sentences grows, 
#the number of parse trees grows rapidly. 
#In fact, it grows at an astronomical rate.

#Example 
>>> grammar = nltk.CFG.fromstring("""
    S -> NP V NP
    NP -> NP Sbar
    Sbar -> NP V
    NP -> 'fish'
    V -> 'fish'
    """)
 
 

#parse 'fish fish fish fish fish'
#for eample - from sentence  'fish that other fish fish are in the habit of fishing fish themselves'. 

>>> tokens = ["fish"] * 5
>>> cp = nltk.ChartParser(grammar)
>>> for tree in cp.parse(tokens):
        print(tree)
(S (NP fish) (V fish) (NP (NP fish) (Sbar (NP fish) (V fish))))
(S (NP (NP fish) (Sbar (NP fish) (V fish))) (V fish) (NP fish))
  

#As the length of this sentence goes up (3, 5, 7, ...) 
#we get the following numbers of parse trees: 1; 2; 5; 14; 42; 132; 429; 1,430; 4,862; 16,796; 58,786; 208,012; ... 
#(These are the Catalan numbers)
 
 
##Solution of above -  Weighted grammars and probabilistic parsing algorithms

#A probabilistic context free grammar (or PCFG) is a context free grammar 
#that associates a probability with each of its productions

#The probability of a parse generated by a PCFG is the product of the probabilities of the productions used to generate it.
 
grammar = nltk.PCFG.fromstring("""
    S    -> NP VP              [1.0]
    VP   -> TV NP              [0.4]
    VP   -> IV                 [0.3]
    VP   -> DatV NP NP         [0.3]
    TV   -> 'saw'              [1.0]
    IV   -> 'ate'              [1.0]
    DatV -> 'gave'             [1.0]
    NP   -> 'telescopes'       [0.8]
    NP   -> 'Jack'             [0.2]
    """)
 
>>> print(grammar)
Grammar with 9 productions (start state = S)
    S -> NP VP [1.0]
    VP -> TV NP [0.4]
    VP -> IV [0.3]
    VP -> DatV NP NP [0.3]
    TV -> 'saw' [1.0]
    IV -> 'ate' [1.0]
    DatV -> 'gave' [1.0]
    NP -> 'telescopes' [0.8]
    NP -> 'Jack' [0.2]
 
 

#can use below combinations as well 
VP -> TV NP [0.4] | IV [0.3] | DatV NP NP [0.3]

#Reference 
class nltk.parse.viterbi.ViterbiParser(grammar, trace=0)
    Bases: nltk.parse.api.ParserI
    A bottom-up PCFG parser that uses dynamic programming to find the single most likely parse for a text.
    
    grammar()

    parse(tokens)

    trace(trace=2)
        Set the level of tracing output that should be generated when parsing a text.
        Parameters:	trace (int) – The trace level. A trace level of 0 will generate no tracing output; and higher trace levels will produce more verbose tracing output.
        Return type:	None

    unicode_repr()   

# A parser will be responsible for finding the most likely parses.
>>> viterbi_parser = nltk.ViterbiParser(grammar)
>>> for tree in viterbi_parser.parse(['Jack', 'saw', 'telescopes']):
        print(tree)
(S (NP Jack) (VP (TV saw) (NP telescopes))) (p=0.064)
 
 

##Further example of Dependency parser   

>>> from nltk.parse.dependencygraph import conll_data2
>>> graphs = [
        DependencyGraph(entry) for entry in conll_data2.split('\n\n') if entry
    ]


>>> ppdp = ProbabilisticProjectiveDependencyParser()
>>> ppdp.train(graphs)
>>> sent = ['Cathy', 'zag', 'hen', 'wild', 'zwaaien', '.']
>>> list(ppdp.parse(sent))
[Tree('zag', ['Cathy', 'hen', Tree('zwaaien', ['wild', '.'])])]


>>> from nltk.grammar import DependencyGrammar
>>> from nltk.parse import (
        DependencyGraph,
        ProjectiveDependencyParser,
        NonprojectiveDependencyParser,
    )



#CoNLL Data

>>> treebank_data = """Pierre  NNP     2       NMOD
    Vinken  NNP     8       SUB
    ,       ,       2       P
    61      CD      5       NMOD
    years   NNS     6       AMOD
    old     JJ      2       NMOD
    ,       ,       2       P
    will    MD      0       ROOT
    join    VB      8       VC
    the     DT      11      NMOD
    board   NN      9       OBJ
    as      IN      9       VMOD
    a       DT      15      NMOD
    nonexecutive    JJ      15      NMOD
    director        NN      12      PMOD
    Nov.    NNP     9       VMOD
    29      CD      16      NMOD
    .       .       9       VMOD
    """

>>> dg = DependencyGraph(treebank_data)
>>> dg.tree().pprint()
(will
  (Vinken Pierre , (old (years 61)) ,)
  (join (board the) (as (director a nonexecutive)) (Nov. 29) .))
>>> for head, rel, dep in dg.triples():
        print(
            '({h[0]}, {h[1]}), {r}, ({d[0]}, {d[1]})'
            .format(h=head, r=rel, d=dep)
        )
(will, MD), SUB, (Vinken, NNP)
(Vinken, NNP), NMOD, (Pierre, NNP)
(Vinken, NNP), P, (,, ,)
(Vinken, NNP), NMOD, (old, JJ)
(old, JJ), AMOD, (years, NNS)
(years, NNS), NMOD, (61, CD)
(Vinken, NNP), P, (,, ,)
(will, MD), VC, (join, VB)
(join, VB), OBJ, (board, NN)
(board, NN), NMOD, (the, DT)
(join, VB), VMOD, (as, IN)
(as, IN), PMOD, (director, NN)
(director, NN), NMOD, (a, DT)
(director, NN), NMOD, (nonexecutive, JJ)
(join, VB), VMOD, (Nov., NNP)
(Nov., NNP), NMOD, (29, CD)
(join, VB), VMOD, (., .)


#Using the dependency-parsed version of the Penn Treebank corpus sample.

>>> from nltk.corpus import dependency_treebank
>>> t = dependency_treebank.parsed_sents()[0]
>>> print(t.to_conll(3))  # doctest: +NORMALIZE_WHITESPACE
Pierre      NNP     2
Vinken      NNP     8
,   ,       2
61  CD      5
years       NNS     6
old JJ      2
,   ,       2
will        MD      0
join        VB      8
the DT      11
board       NN      9
as  IN      9
a   DT      15
nonexecutive        JJ      15
director    NN      12
Nov.        NNP     9
29  CD      16
.   .       8


#Using the output of zpar (like Malt-TAB but with zero-based indexing)

>>> zpar_data = """
    Pierre  NNP     1       NMOD
    Vinken  NNP     7       SUB
    ,       ,       1       P
    61      CD      4       NMOD
    years   NNS     5       AMOD
    old     JJ      1       NMOD
    ,       ,       1       P
    will    MD      -1      ROOT
    join    VB      7       VC
    the     DT      10      NMOD
    board   NN      8       OBJ
    as      IN      8       VMOD
    a       DT      14      NMOD
    nonexecutive    JJ      14      NMOD
    director        NN      11      PMOD
    Nov.    NNP     8       VMOD
    29      CD      15      NMOD
    .       .       7       P
    """

>>> zdg = DependencyGraph(zpar_data, zero_based=True)
>>> print(zdg.tree())
(will
  (Vinken Pierre , (old (years 61)) ,)
  (join (board the) (as (director a nonexecutive)) (Nov. 29))
  .)



#Projective Dependency Parsing

>>> grammar = DependencyGrammar.fromstring("""
    'fell' -> 'price' | 'stock'
    'price' -> 'of' 'the'
    'of' -> 'stock'
    'stock' -> 'the'
    """)
>>> print(grammar)
Dependency grammar with 5 productions
  'fell' -> 'price'
  'fell' -> 'stock'
  'price' -> 'of' 'the'
  'of' -> 'stock'
  'stock' -> 'the'

>>> dp = ProjectiveDependencyParser(grammar)
>>> for t in sorted(dp.parse(['the', 'price', 'of', 'the', 'stock', 'fell'])):
...     print(t)
(fell (price the (of (stock the))))
(fell (price the of) (stock the))
(fell (price the of the) stock)



#Non-Projective Dependency Parsing

>>> grammar = DependencyGrammar.fromstring("""
    'taught' -> 'play' | 'man'
    'man' -> 'the'
    'play' -> 'golf' | 'dog' | 'to'
    'dog' -> 'his'
    """)
>>> print(grammar)
Dependency grammar with 7 productions
  'taught' -> 'play'
  'taught' -> 'man'
  'man' -> 'the'
  'play' -> 'golf'
  'play' -> 'dog'
  'play' -> 'to'
  'dog' -> 'his'

>>> dp = NonprojectiveDependencyParser(grammar)
>>> g, = dp.parse(['the', 'man', 'taught', 'his', 'dog', 'to', 'play', 'golf'])

>>> print(g.root['word'])
taught

>>> for _, node in sorted(g.nodes.items()):
        if node['word'] is not None:
            print('{address} {word}: {d}'.format(d=node['deps'][''], **node))
1 the: []
2 man: [1]
3 taught: [2, 7]
4 his: []
5 dog: [4]
6 to: []
7 play: [5, 6, 8]
8 golf: []

>>> print(g.tree())
(taught (man the) (play (dog his) to golf))










###NLTK - TreeTransformation class

#A collection of methods for tree (grammar) transformations 
#used in parsing natural language

1.Chomsky Normal Form (binarization)
  Any grammar has a Chomsky Normal Form (CNF) equivalent grammar 
  where CNF is defined by every production having either two non-terminals or one terminal on its right hand side
  There are two popular methods to convert a tree into CNF: left factoring and right factoring.
2.Parent Annotation(part of CNF transformation)
  The purpose of parent annotation is to refine the probabilities of productions 
  by adding a small amount of context. 
  With this simple addition, a CYK (inside-outside, dynamic programming chart parse) 
  can improve from 74% to 79% accuracy
3.Markov order-N smoothing(part of CNF transformation)
  Markov smoothing combats data sparcity issues as well as decreasing computational requirements 
  by limiting the number of children included in artificial nodes
4.Unary Collapsing
  Collapse unary productions (ie. subtrees with a single child) into a new non-terminal (Tree node). 
  This is useful when working with algorithms that do not allow unary productions

nltk.treetransforms.chomsky_normal_form(tree, factor='right', horzMarkov=None, vertMarkov=0, childChar='|', parentChar='^')
nltk.treetransforms.un_chomsky_normal_form(tree, expandUnary=True, childChar='|', parentChar='^', unaryChar='+')
nltk.treetransforms.collapse_unary(tree, collapsePOS=False, collapseRoot=False, joinChar='+')

#Example 
>>> from copy import deepcopy
>>> from nltk.tree import *
>>> from nltk.treetransforms import *

>>> tree_string = "(TOP (S (S (VP (VBN Turned) (ADVP (RB loose)) (PP (IN in) (NP (NP (NNP Shane) (NNP Longman) (POS 's)) (NN trading) (NN room))))) (, ,) (NP (DT the) (NN yuppie) (NNS dealers)) (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right)))) (. .)))"

>>> tree = Tree.fromstring(tree_string)
>>> print(tree)
(TOP
  (S
    (S
      (VP
        (VBN Turned)
        (ADVP (RB loose))
        (PP
          (IN in)
          (NP
            (NP (NNP Shane) (NNP Longman) (POS 's))
            (NN trading)
            (NN room)))))
    (, ,)
    (NP (DT the) (NN yuppie) (NNS dealers))
    (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right))))
    (. .)))


#Make a copy of the original tree and collapse the subtrees with only one child

>>> collapsedTree = deepcopy(tree)
>>> collapse_unary(collapsedTree)
>>> print(collapsedTree)
(TOP
  (S
    (S+VP
      (VBN Turned)
      (ADVP (RB loose))
      (PP
        (IN in)
        (NP
          (NP (NNP Shane) (NNP Longman) (POS 's))
          (NN trading)
          (NN room))))
    (, ,)
    (NP (DT the) (NN yuppie) (NNS dealers))
    (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right))))
    (. .)))

>>> collapsedTree2 = deepcopy(tree)
>>> collapse_unary(collapsedTree2, collapsePOS=True, collapseRoot=True)
>>> print(collapsedTree2)
(TOP+S
  (S+VP
    (VBN Turned)
    (ADVP+RB loose)
    (PP
      (IN in)
      (NP
        (NP (NNP Shane) (NNP Longman) (POS 's))
        (NN trading)
        (NN room))))
  (, ,)
  (NP (DT the) (NN yuppie) (NNS dealers))
  (VP (AUX do) (NP (NP+RB little) (ADJP+RB right)))
  (. .))


#Convert the tree to Chomsky Normal Form 
#i.e. each subtree has either two subtree children or a single leaf value. 
#This conversion can be performed using either left- or right-factoring.

>>> cnfTree = deepcopy(collapsedTree)
>>> chomsky_normal_form(cnfTree, factor='left')
>>> print(cnfTree)
(TOP
  (S
    (S|<S+VP-,-NP-VP>
      (S|<S+VP-,-NP>
        (S|<S+VP-,>
          (S+VP
            (S+VP|<VBN-ADVP> (VBN Turned) (ADVP (RB loose)))
            (PP
              (IN in)
              (NP
                (NP|<NP-NN>
                  (NP
                    (NP|<NNP-NNP> (NNP Shane) (NNP Longman))
                    (POS 's))
                  (NN trading))
                (NN room))))
          (, ,))
        (NP (NP|<DT-NN> (DT the) (NN yuppie)) (NNS dealers)))
      (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right)))))
    (. .)))

>>> cnfTree = deepcopy(collapsedTree)
>>> chomsky_normal_form(cnfTree, factor='right')
>>> print(cnfTree)
(TOP
  (S
    (S+VP
      (VBN Turned)
      (S+VP|<ADVP-PP>
        (ADVP (RB loose))
        (PP
          (IN in)
          (NP
            (NP (NNP Shane) (NP|<NNP-POS> (NNP Longman) (POS 's)))
            (NP|<NN-NN> (NN trading) (NN room))))))
    (S|<,-NP-VP-.>
      (, ,)
      (S|<NP-VP-.>
        (NP (DT the) (NP|<NN-NNS> (NN yuppie) (NNS dealers)))
        (S|<VP-.>
          (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right))))
          (. .))))))


#Employ some Markov smoothing to make the artificial node labels a bit more readable. 

>>> markovTree = deepcopy(collapsedTree)
>>> chomsky_normal_form(markovTree, horzMarkov=2, vertMarkov=1)
>>> print(markovTree)
(TOP
  (S^<TOP>
    (S+VP^<S>
      (VBN Turned)
      (S+VP|<ADVP-PP>^<S>
        (ADVP^<S+VP> (RB loose))
        (PP^<S+VP>
          (IN in)
          (NP^<PP>
            (NP^<NP>
              (NNP Shane)
              (NP|<NNP-POS>^<NP> (NNP Longman) (POS 's)))
            (NP|<NN-NN>^<PP> (NN trading) (NN room))))))
    (S|<,-NP>^<TOP>
      (, ,)
      (S|<NP-VP>^<TOP>
        (NP^<S> (DT the) (NP|<NN-NNS>^<S> (NN yuppie) (NNS dealers)))
        (S|<VP-.>^<TOP>
          (VP^<S>
            (AUX do)
            (NP^<VP> (NP^<NP> (RB little)) (ADJP^<NP> (RB right))))
          (. .))))))


#Convert the transformed tree back to its original form

>>> un_chomsky_normal_form(markovTree)
>>> tree == markovTree
True



###NLTK - Generating sentences from context-free grammars
  
nltk.parse.generate.generate(grammar, start=None, depth=None, n=None)
    Generates an iterator of all sentences from a CFG.


>>> from nltk.parse.generate import generate, demo_grammar
>>> from nltk import CFG
>>> grammar = CFG.fromstring(demo_grammar)
>>> print(grammar)
Grammar with 13 productions (start state = S)
    S -> NP VP
    NP -> Det N
    PP -> P NP
    VP -> 'slept'
    VP -> 'saw' NP
    VP -> 'walked' PP
    Det -> 'the'
    Det -> 'a'
    N -> 'man'
    N -> 'park'
    N -> 'dog'
    P -> 'in'
    P -> 'with'


#The first 10 generated sentences:

>>> for sentence in generate(grammar, n=10):
        print(' '.join(sentence))
the man slept
the man saw the man
the man saw the park
the man saw the dog
the man saw a man
the man saw a park
the man saw a dog
the man walked in the man
the man walked in the park
the man walked in the dog


#All sentences of max depth 4:

>>> for sentence in generate(grammar, depth=4):
        print(' '.join(sentence))
the man slept
the park slept
the dog slept
a man slept
a park slept
a dog slept


#The number of sentences of different max depths:

>>> len(list(generate(grammar, depth=3)))
0
>>> len(list(generate(grammar, depth=4)))
6
>>> len(list(generate(grammar, depth=5)))
42
>>> len(list(generate(grammar, depth=6)))
114
>>> len(list(generate(grammar)))
114

###NLTK - Best parser for parsing English text 
#Download from https://github.com/bendavis78/pyStatParser/tree/python3
#unzip and run - python setup.py install 

from stat_parser import Parser, display_tree
parser = Parser()

# http://www.thrivenotes.com/the-last-question/
tree = parser.parse("How can the net amount of entropy of the universe be massively decreased?")

display_tree(tree)
#or 
>>> print(tree)
(SBARQ
  (WHADVP (WRB how))
  (SQ
    (MD can)
    (NP
      (NP (DT the) (JJ net) (NN amount))
      (PP
        (IN of)
        (NP
          (NP (NNS entropy))
          (PP (IN of) (NP (DT the) (NN universe))))))
    (VP (VB be) (ADJP (RB massively) (VBN decreased))))
  (. ?))





###chap-9
###NLTK - Building Feature Based Grammars - Features are extracted 

#Natural languages have an extensive range of grammatical constructions 
#which are hard to handle with the CFG based Parser described in chap-8

#In order to gain more flexibility, 
#grammatical categories like S, NP and V may contain features (a dict with suitable keys) which can be extracted 
#Features could be anything important eg extracting the last letter of a word or pos-tag etc 
 

##Example - (* ungrammatical)- singular or plural
 
(1)  
a.  this dog 
b.  *these dog 
(2)  
a.  these dogs 
b.  *this dogs 

#CFG for 'this dog runs'
S   ->   NP VP
NP  ->   Det N
VP  ->   V

Det  ->  'this'
N    ->  'dog'
V    ->  'runs'

#To include plural- Note every line is replicated for singular and  plural
S -> NP_SG VP_SG
S -> NP_PL VP_PL
NP_SG -> Det_SG N_SG
NP_PL -> Det_PL N_PL
VP_SG -> V_SG
VP_PL -> V_PL

Det_SG -> 'this'
Det_PL -> 'these'
N_SG -> 'dog'
N_PL -> 'dogs'
V_SG -> 'runs'
V_PL -> 'run'

 
##SOLUTION - To represent compactly - Use Attributes and Constraints on Non Terminal 
 
 
#Example - category N has a (grammatical) feature called NUM (short for 'number') 
#and that the value of this feature is pl (short for 'plural')
#NUM and pl are example only, we can use any string there 
N[NUM=pl]



##Variables over feature values, - called constraints
N[NUM=?n]       #where ?n is a variable, either sg or pl
#it can be instantiated(by extracting feature) either to sg or pl
#Note number of constraints can be many , separeted by ',' eg VP[TENSE=?t, NUM=?n]

#S   ->   NP VP
#NP  ->   Det N
#VP  ->   V
#Note S is same for all ?n  
#actual instance for variable  is starting from Det[NUM=sg]
S -> NP[NUM=?n] VP[NUM=?n]
NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
VP[NUM=?n] -> V[NUM=?n]

Det[NUM=sg] -> 'this'
Det[NUM=pl] -> 'these'

N[NUM=sg] -> 'dog'
N[NUM=pl] -> 'dogs'
V[NUM=sg] -> 'runs'
V[NUM=pl] -> 'run'
 

#S -> NP[NUM=?n] VP[NUM=?n] creates below 
#ie NUM value is sg 
NP[NUM=sg]
    Det[NUM=sg] -> 'this'
    N[NUM=sg] -> 'dog'
 
#ie NUM value is pl
NP[NUM=pl]
    Det[NUM=pl] -> 'these'
    N[NUM=pl] -> 'dogs'
 
#Production VP[NUM=?n] -> V[NUM=?n] creates below 
S    
    NP[NUM=pl]
        Det[NUM=pl] -> 'these'
        N[NUM=pl] -> 'dogs'
    VP[NUM=pl]
        V[NUM=pl] -> 'run'

        
#We could increase the production  of determiners    to include more values 
Det[NUM=sg] -> 'the' | 'some' | 'any'
Det[NUM=pl] -> 'the' | 'some' | 'any'

#OR 
Det[NUM=?n] -> 'the' | 'some' | 'any'
#OR 
Det -> 'the' | 'some' | 'several'

##Example feature CFG 
#Note '% start S' denotes the start tag as S
>>> nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')
% start S
# Grammar Productions
# S expansion productions
S -> NP[NUM=?n] VP[NUM=?n]
# NP expansion productions, extract value of NUM 
NP[NUM=?n] -> N[NUM=?n]
NP[NUM=?n] -> PropN[NUM=?n]
NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
NP[NUM=pl] -> N[NUM=pl]
# VP expansion productions, extract value of TENSE and NUM 
VP[TENSE=?t, NUM=?n] -> IV[TENSE=?t, NUM=?n]
VP[TENSE=?t, NUM=?n] -> TV[TENSE=?t, NUM=?n] NP
# Lexical Productions
Det[NUM=sg] -> 'this' | 'every'
Det[NUM=pl] -> 'these' | 'all'
Det -> 'the' | 'some' | 'several'
PropN[NUM=sg]-> 'Kim' | 'Jody'
N[NUM=sg] -> 'dog' | 'girl' | 'car' | 'child'
N[NUM=pl] -> 'dogs' | 'girls' | 'cars' | 'children'
IV[TENSE=pres,  NUM=sg] -> 'disappears' | 'walks'
TV[TENSE=pres, NUM=sg] -> 'sees' | 'likes'
IV[TENSE=pres,  NUM=pl] -> 'disappear' | 'walk'
TV[TENSE=pres, NUM=pl] -> 'see' | 'like'
IV[TENSE=past] -> 'disappeared' | 'walked'
TV[TENSE=past] -> 'saw' | 'liked'
 
#Reference 
nltk.parse.util.load_parser(grammar_url, trace=0, parser=None, chart_class=None, beam_size=0, **load_args)
    Load a grammar from a file, and build a parser based on that grammar. 
    The following grammar formats are currently supported based on extension 
            'cfg' (CFGs: CFG)
            'pcfg' (probabilistic CFGs: PCFG)
            'fcfg' (feature-based CFGs: FeatureGrammar) 
        grammar_url (str) – A URL specifying where the grammar is located. 
                            The default protocol is "nltk:", which searches for the file in the the NLTK data package.
        trace (int) – The level of tracing that should be used when parsing a text. 0 will generate no tracing output; and higher numbers will produce more verbose tracing output.
        parser – The class used for parsing; should be ChartParser or a subclass. If None, the class depends on the grammar format.
        chart_class – The class used for storing the chart; should be Chart or a subclass. Only used for CFGs and feature CFGs. If None, the chart class depends on the grammar format.
        beam_size (int) – The maximum length for the parser’s edge queue. Only used for probabilistic CFGs.
        load_args – Keyword parameters used when loading the grammar. See data.load for more information.

        
#Usage 
>>> tokens = 'Kim likes children'.split()
>>> from nltk import load_parser 
>>> cp = load_parser('grammars/book_grammars/feat0.fcfg', trace=2)  #Loading parser and CFG is created 
>>> for tree in cp.parse(tokens):
        print(tree)

|.Kim .like.chil.|
Leaf Init Rule:
|[----]    .    .| [0:1] 'Kim'
|.    [----]    .| [1:2] 'likes'
|.    .    [----]| [2:3] 'children'
Feature Bottom Up Predict Combine Rule:
|[----]    .    .| [0:1] PropN[NUM='sg'] -> 'Kim' *
Feature Bottom Up Predict Combine Rule:
|[----]    .    .| [0:1] NP[NUM='sg'] -> PropN[NUM='sg'] *
Feature Bottom Up Predict Combine Rule:
|[---->    .    .| [0:1] S[] -> NP[NUM=?n] * VP[NUM=?n] {?n: 'sg'}  ##Feature struct, ?n extracted to be 'sg'
Feature Bottom Up Predict Combine Rule:
|.    [----]    .| [1:2] TV[NUM='sg', TENSE='pres'] -> 'likes' *
Feature Bottom Up Predict Combine Rule:
|.    [---->    .| [1:2] VP[NUM=?n, TENSE=?t] -> TV[NUM=?n, TENSE=?t] * NP[] {?n: 'sg', ?t: 'pres'} ##Feature struct, ?n extracted to be 'sg' and ?t extracted to be 'pres'
Feature Bottom Up Predict Combine Rule:
|.    .    [----]| [2:3] N[NUM='pl'] -> 'children' *
Feature Bottom Up Predict Combine Rule:
|.    .    [----]| [2:3] NP[NUM='pl'] -> N[NUM='pl'] *
Feature Bottom Up Predict Combine Rule:
|.    .    [---->| [2:3] S[] -> NP[NUM=?n] * VP[NUM=?n] {?n: 'pl'}
Feature Single Edge Fundamental Rule:
|.    [---------]| [1:3] VP[NUM='sg', TENSE='pres'] -> TV[NUM='sg', TENSE='pres'] NP[] *
Feature Single Edge Fundamental Rule:
|[==============]| [0:3] S[] -> NP[NUM='sg'] VP[NUM='sg'] *
(S[]
  (NP[NUM='sg'] (PropN[NUM='sg'] Kim))
  (VP[NUM='sg', TENSE='pres']
    (TV[NUM='sg', TENSE='pres'] likes)
    (NP[NUM='pl'] (N[NUM='pl'] children))))
 
#Understanding parsing, User can use value of NUM, TENSE for further logic 
>>> trees = list(cp2.parse(tokens)) 
>>> type(trees)  #list of parse trees, here it is one 
<class 'list'>
>>> type(trees[0])
<class 'nltk.tree.Tree'>
>>> print(trees[0])          #Tree structure 
(S[]
  (NP[NUM='sg'] (PropN[NUM='sg'] Kim))
  (VP[NUM='sg', TENSE='pres']
    (TV[NUM='sg', TENSE='pres'] likes)
    (NP[NUM='pl'] (N[NUM='pl'] children))))
    
>>> type(trees[0].label())   #label of Parse tree is Feature struct, a dict kind object 
<class 'nltk.grammar.FeatStructNonterminal'>
>>> trees[0].label()
S[]
>>> trees[0]
Tree(S[], [Tree(NP[NUM='sg'], [Tree(PropN[NUM='sg'], ['Kim'])]), Tree(
VP[NUM='sg', TENSE='pres'], [Tree(TV[NUM='sg', TENSE='pres'], ['likes'
]), Tree(NP[NUM='pl'], [Tree(N[NUM='pl'], ['children'])])])])
 
#To get the value of NUM which is extracted in NP as a featuredict 
>>> trees[0][0] #first children 
Tree(NP[NUM='sg'], [Tree(PropN[NUM='sg'], ['Kim'])])
>>> type(trees[0][0].label())
<class 'nltk.grammar.FeatStructNonterminal'>
>>> trees[0][0].label().keys()
dict_keys([*type*, 'NUM'])
>>> trees[0][0].label()
NP[NUM='sg']
>>> trees[0][0].label()['NUM']
'sg'
#similarly value of TENSE 
>>> trees[0][1].label()
VP[NUM='sg', TENSE='pres']
>>> trees[0][1].label()['TENSE']
'pres'


##Feature grammer - Terminology

#feature values like sg and pl called atomic(can not be broken)

##boolean feature value eg if AUX is true, then production 
V[TENSE=pres, AUX=+] -> 'can'
#OR, instead of AUX=+ or AUX=-, we use +AUX and -AUX

V[TENSE=pres, +AUX] -> 'can'
V[TENSE=pres, +AUX] -> 'may'

V[TENSE=pres, -AUX] -> 'walks'
V[TENSE=pres, -AUX] -> 'likes'

 

##features may take values that are themselves feature structures. 
#Example - we can group together agreement features (e.g., person, number and gender) 
#as a distinguished part of a category, grouped together as  AGR. 
#AGR has a complex value , called attribute value matrix (AVM) 

[POS = N           ]
[                  ]
[AGR = [PER = 3   ]]
[      [NUM = pl  ]]
[      [GND = fem ]]

 
#Note above is equivalent to 

[AGR = [NUM = pl  ]]
[      [PER = 3   ]]
[      [GND = fem ]]
[                  ]
[POS = N           ]

 
#Grammer example - AGR having complex structure 

S                    -> NP[AGR=?n] VP[AGR=?n]
NP[AGR=?n]           -> PropN[AGR=?n]
VP[TENSE=?t, AGR=?n] -> Cop[TENSE=?t, AGR=?n] Adj

Cop[TENSE=pres,  AGR=[NUM=sg, PER=3]] -> 'is'
PropN[AGR=[NUM=sg, PER=3]]            -> 'Kim'
Adj                                   -> 'happy'

 

##Processing Feature Structures - FeatStruct() - a kind of dictionary

#Atomic feature values can be strings or integers.
>>> fs1 = nltk.FeatStruct(TENSE='past', NUM='sg')
>>> print(fs1)
[ NUM   = 'sg'   ]
[ TENSE = 'past' ]
 
#get/set 
>>> fs1 = nltk.FeatStruct(PER=3, NUM='pl', GND='fem')
>>> print(fs1['GND'])
fem
>>> fs1['CASE'] = 'acc'
 
 
#With complex values 

>>> fs2 = nltk.FeatStruct(POS='N', AGR=fs1)
>>> print(fs2)
[       [ CASE = 'acc' ] ]
[ AGR = [ GND  = 'fem' ] ]
[       [ NUM  = 'pl'  ] ]
[       [ PER  = 3     ] ]
[                        ]
[ POS = 'N'              ]
>>> print(fs2['AGR'])
[ CASE = 'acc' ]
[ GND  = 'fem' ]
[ NUM  = 'pl'  ]
[ PER  = 3     ]
>>> print(fs2['AGR']['PER'])
3
 
 
#An alternative method of specifying feature structures 
#use a bracketed string consisting of feature-value pairs in the format feature=value, 
#where values may themselves be feature structures

>>> print(nltk.FeatStruct("[POS='N', AGR=[PER=3, NUM='pl', GND='fem']]"))
[       [ GND = 'fem' ] ]
[ AGR = [ NUM = 'pl'  ] ]
[       [ PER = 3     ] ]
[                       ]
[ POS = 'N'             ]
 
 
 
 
##It is often helpful to view feature structures as graphs- directed acyclic graphs (DAGs).

#The feature names appear as labels on the directed arcs, 
#and feature values appear as labels on the nodes that are pointed to by the arcs.
#Note feature values can be complex ie node may emit many arcs 

#A feature path is a sequence of arcs that can be followed from the root node. 
#We will represent paths as tuples or arcs (arc=feature name)

#check data/dag_as_feature02.png
#Example -('ADDRESS', 'STREET') is a feature path whose value is the node labeled 'rue Pascal'.


#DAGs can have structure sharing or reentrancy.(ie two path arrow meet at a node  from where another sub DAG is emited)
#When two paths have the same value, they are said to be equivalent.

#check data/dag_as_feature03.png
#Example - the value of the path ('ADDRESS') is identical to the value of the path ('SPOUSE', 'ADDRESS')

#In order to indicate reentrancy,
#prefix the first occurrence of a shared feature structure with an integer in parentheses eg (1)
#Any later reference to that structure will use the notation ->(1)



>>> print(nltk.FeatStruct("""[NAME='Lee', ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
                        SPOUSE=[NAME='Kim', ADDRESS->(1)]]"""))
[ ADDRESS = (1) [ NUMBER = 74           ] ]
[               [ STREET = 'rue Pascal' ] ]
[                                         ]
[ NAME    = 'Lee'                         ]
[                                         ]
[ SPOUSE  = [ ADDRESS -> (1)  ]           ]
[           [ NAME    = 'Kim' ]           ]
 
 

#The bracketed integer is called a tag or a coindex. 
#The choice of integer is not significant. 
#There can be any number of tags within a single feature structure.
>>> print(nltk.FeatStruct("[A='a', B=(1)[C='c'], D->(1), E->(1)]"))
[ A = 'a'             ]
[                     ]
[ B = (1) [ C = 'c' ] ]
[                     ]
[ D -> (1)            ]
[ E -> (1)            ]
 
 

##Feature - Subsumption and Unification

##Note a has less information than b and b has less info on c  
a.  
[NUMBER = 74]

 
b.  
[NUMBER = 74          ]
[STREET = 'rue Pascal']

  
c.  
[NUMBER = 74          ]
[STREET = 'rue Pascal']
[CITY = 'Paris'       ]

 
#This ordering is called subsumption; 
#FS0 subsumes FS1 if all the information contained in FS0 is also contained in FS1. 
#use the symbol 'SUB' (subset notation) to represent subsumption.
#if FS0 'SUB' FS1, then FS1 must have all the paths and reentrancies of FS0
 
#Merging information from two feature structures is called unification 
#FS0 UNI FS1 (UNI- union notation)
#supported by the unify() method.
#Note - If FS0 SUB FS1, then FS0 UNI FS1 = FS1 
>>> fs1 = nltk.FeatStruct(NUMBER=74, STREET='rue Pascal')
>>> fs2 = nltk.FeatStruct(CITY='Paris')
>>> print(fs1.unify(fs2))
[ CITY   = 'Paris'      ]
[ NUMBER = 74           ]
[ STREET = 'rue Pascal' ]
 
 

#Unification is symmetric, so FS0 UNI FS1 = FS1 UNI FS0. 

>>> print(fs2.unify(fs1))
[ CITY   = 'Paris'      ]
[ NUMBER = 74           ]
[ STREET = 'rue Pascal' ]
 
 
#If we unify two feature structures which stand in the subsumption relationship, 
#then the result of unification is the most informative of the two


#Unification between FS0 and FS1 will fail 
#if the two feature structures share a path PHI, 
#but the value of PHI in FS0 is a distinct atom from the value of PHI in FS1. 

#This is implemented by setting the result of unification to be None.
>>> fs0 = nltk.FeatStruct(A='a')
>>> fs1 = nltk.FeatStruct(A='b')
>>> fs2 = fs0.unify(fs1)
>>> print(fs2)
None
 
 

##How unification interacts with structure-sharing

>>> fs0 = nltk.FeatStruct("""[NAME=Lee, ADDRESS=[NUMBER=74,STREET='rue Pascal'],
                              SPOUSE= [NAME=Kim,ADDRESS=[NUMBER=74,STREET='rue Pascal']]]""")
>>> print(fs0)
[ ADDRESS = [ NUMBER = 74           ]               ]
[           [ STREET = 'rue Pascal' ]               ]
[                                                   ]
[ NAME    = 'Lee'                                   ]
[                                                   ]
[           [ ADDRESS = [ NUMBER = 74           ] ] ]
[ SPOUSE  = [           [ STREET = 'rue Pascal' ] ] ]
[           [                                     ] ]
[           [ NAME    = 'Kim'                     ] ]
 
 
#Add CITY to SPOUSE 
>>> fs1 = nltk.FeatStruct("[SPOUSE = [ADDRESS = [CITY = Paris]]]")
>>> print(fs1.unify(fs0))
[ ADDRESS = [ NUMBER = 74           ]               ]
[           [ STREET = 'rue Pascal' ]               ]
[                                                   ]
[ NAME    = 'Lee'                                   ]
[                                                   ]
[           [           [ CITY   = 'Paris'      ] ] ]
[           [ ADDRESS = [ NUMBER = 74           ] ] ]
[ SPOUSE  = [           [ STREET = 'rue Pascal' ] ] ]
[           [                                     ] ]
[           [ NAME    = 'Kim'                     ] ]
 
 
#Whereas with structure sharing version


>>> fs2 = nltk.FeatStruct("""[NAME=Lee, ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
                              SPOUSE=[NAME=Kim, ADDRESS->(1)]]""")
>>> print(fs1.unify(fs2))
[               [ CITY   = 'Paris'      ] ]
[ ADDRESS = (1) [ NUMBER = 74           ] ]
[               [ STREET = 'rue Pascal' ] ]
[                                         ]
[ NAME    = 'Lee'                         ]
[                                         ]
[ SPOUSE  = [ ADDRESS -> (1)  ]           ]
[           [ NAME    = 'Kim' ]           ]
 
 

##structure sharing can also be stated using variables such as ?x.
>>> fs1 = nltk.FeatStruct("[ADDRESS1=[NUMBER=74, STREET='rue Pascal']]")
>>> fs2 = nltk.FeatStruct("[ADDRESS1=?x, ADDRESS2=?x]")
>>> print(fs2)
[ ADDRESS1 = ?x ]
[ ADDRESS2 = ?x ]
>>> print(fs2.unify(fs1))
[ ADDRESS1 = (1) [ NUMBER = 74           ] ]
[                [ STREET = 'rue Pascal' ] ]
[                                          ]
[ ADDRESS2 -> (1)                          ]
 
 

##Feature - Subcategorization

#For example -category labels to represent different kinds of verb, 
#used the labels IV and TV for intransitive and transitive verbs respectively
VP -> IV
VP -> TV NP

 
#Can we replace category labels such as TV and IV by V along with a feature 
#that tells us whether the verb combines with a following NP object 
#or whether it can occur without any complement?

#Use Generalized Phrase Structure Grammar (GPSG)
#where lexical categories to bear a SUBCAT 
#which tells us what subcategorization class the item belongs to. 

#values for SUBCAT can be intrans, trans and clause

VP[TENSE=?t, NUM=?n] -> V[SUBCAT=intrans, TENSE=?t, NUM=?n]
VP[TENSE=?t, NUM=?n] -> V[SUBCAT=trans, TENSE=?t, NUM=?n] NP
VP[TENSE=?t, NUM=?n] -> V[SUBCAT=clause, TENSE=?t, NUM=?n] SBar

V[SUBCAT=intrans, TENSE=pres, NUM=sg] -> 'disappears' | 'walks'
V[SUBCAT=trans, TENSE=pres, NUM=sg] -> 'sees' | 'likes'
V[SUBCAT=clause, TENSE=pres, NUM=sg] -> 'says' | 'claims'

V[SUBCAT=intrans, TENSE=pres, NUM=pl] -> 'disappear' | 'walk'
V[SUBCAT=trans, TENSE=pres, NUM=pl] -> 'see' | 'like'
V[SUBCAT=clause, TENSE=pres, NUM=pl] -> 'say' | 'claim'

V[SUBCAT=intrans, TENSE=past, NUM=?n] -> 'disappeared' | 'walked'
V[SUBCAT=trans, TENSE=past, NUM=?n] -> 'saw' | 'liked'
V[SUBCAT=clause, TENSE=past, NUM=?n] -> 'said' | 'claimed'
#SBar - is a label for subordinate clauses
SBar -> Comp S
Comp -> 'that'


#An alternative treatment of subcategorization - categorial grammar
#is represented in feature based frameworks such as PATR and Head-driven Phrase Structure Grammar. 

#Rather than using SUBCAT values as a way of indexing productions, 
#the SUBCAT value directly encodes the valency of a head 
#(the list of arguments that it can combine with). 

#For example, a verb like 'put' that takes NP and PP complements 
#(eg, put the book on the table) might be represented 
 
V[SUBCAT=<NP, NP, PP>]

#This says that the verb can combine with three arguments. 
#The leftmost element in the list is the subject NP, 
#while everything else — an NP followed by a PP in this case 
#— comprises the subcategorized for complements. 

#When a verb like 'put' is combined with appropriate complements, 
#the requirements which are specified in the SUBCAT are discharged, 
#and only a subject NP is needed. 

#This category, which corresponds to what is traditionally thought of as VP, might be represented 
V[SUBCAT=<NP>]

 

#a sentence is a kind of verbal category that has no requirements for further arguments, 
#and hence has a SUBCAT whose value is the empty list

#Note :
#expressions of category V are heads of phrases of category VP. 
#Ns are heads of NPs, 
#As (i.e., adjectives) are heads of APs, 
#Ps (i.e., prepositions) are heads of PPs. 
#Not all phrases have heads — 
#for example,  coordinate phrases (e.g., the book and the bell) lack heads 



##X-bar Syntax - phrasal level. 

#It is usual to recognize three levels. 
#If N represents the lexical level, 
#then N' represents the next level up, corresponding to category Nom, 
#while N'' represents the phrasal level, corresponding to the category NP. 

#N' and N'' are called (phrasal) projections of N
#N'' is the maximal projection
#N is called the zero projection

N"
    Det
        a
    N'
        N
            Student
        P"
            of French 
            
#equivalent to 
NP
    Det
        a
    Norm
        N
            Student
        PP
            of French 
    

#Using X-bar syntax, we mean that all constituents share a structural similarity. 
#Using X as a variable over N, V, A and P, 
#directly subcategorized complements of a lexical head X are always placed as siblings of the head, 
#whereas adjuncts are placed as siblings of the intermediate category, X'. 

#in NLTK
S -> N[BAR=2] V[BAR=2]
N[BAR=2] -> Det N[BAR=1]
N[BAR=1] -> N[BAR=1] P[BAR=2]
N[BAR=1] -> N[BAR=0] P[BAR=2]
N[BAR=1] -> N[BAR=0]XS

 

##Auxiliary Verbs and Inversion

#Inverted clauses — where the order of subject and verb is switched 
#— occur in English interrogatives and also after 'negative' adverbs

 
a.  Do you like children? 
b.  Can Jody walk? 

 

a.  Rarely do you see Kim. 
b.  Never have I seen this dog. 

 
#we cannot place just any verb in pre-subject position:
a.  *Like you children? 
b.  *Walks Jody? 

a.  *Rarely see you Kim. 
b.  *Never saw I this dog. 

 

#Verbs that can be positioned initially in inverted clauses belong to the class known as auxiliaries, 
#and as well as do, can and have include be, will and shall. 

#One way of capturing such structures is with the following production:
#a clause marked as [+INV] consists of an auxiliary verb followed by a VP.
S[+INV] -> V[+AUX] NP VP

 
##Feature grammer - Slash categories

#Note - there is no upper bound on the distance between filler and gap. 
#This fact can be easily illustrated with constructions involving sentential complements, 
 
a.  Who do you like __? 
b.  Who do you claim that you like __? 
c.  Who do you claim that Jody says that you like __? 

#ie - an unbounded dependency construction 
#a filler-gap dependency where there is no upper bound on the distance between filler and gap.

#to handle  use slash categories - in Generalized Phrase Structure Grammar. 
#A slash category has the form Y/XP; 
#a phrase of category Y that is missing a sub-constituent of category XP. 

#For example, S/NP is an S that is missing an NP.
#S/NP is reducible to S[SLASH=NP]
S
    NP[+WH]
        who
    S[+INV]/NP
        V[+AUX]
            do
        NP[-WH]
            you
        VP/NP
            V[-AUX, SUBCAT=trans]
                like 
            NP/NP
    
#The top part of the tree introduces the filler 'who' 
#(treated as an expression of category NP[+wh]) 
#together with a corresponding gap-containing constituent S/NP. 

#The gap information is then "percolated" down the tree via the VP/NP category, 
#until it reaches the category NP/NP (empty string)


#Example 
>>> nltk.data.show_cfg('grammars/book_grammars/feat1.fcfg')
% start S
# Grammar Productions
S[-INV] -> NP VP
S[-INV]/?x -> NP VP/?x
S[-INV] -> NP S/NP
S[-INV] -> Adv[+NEG] S[+INV]
S[+INV] -> V[+AUX] NP VP
S[+INV]/?x -> V[+AUX] NP VP/?x
SBar -> Comp S[-INV]
SBar/?x -> Comp S[-INV]/?x
VP -> V[SUBCAT=intrans, -AUX]
VP -> V[SUBCAT=trans, -AUX] NP
VP/?x -> V[SUBCAT=trans, -AUX] NP/?x
VP -> V[SUBCAT=clause, -AUX] SBar
VP/?x -> V[SUBCAT=clause, -AUX] SBar/?x
VP -> V[+AUX] VP
VP/?x -> V[+AUX] VP/?x
# Lexical Productions
V[SUBCAT=intrans, -AUX] -> 'walk' | 'sing'
V[SUBCAT=trans, -AUX] -> 'see' | 'like'
V[SUBCAT=clause, -AUX] -> 'say' | 'claim'
V[+AUX] -> 'do' | 'can'
NP[-WH] -> 'you' | 'cats'
NP[+WH] -> 'who'
Adv[+NEG] -> 'rarely' | 'never'
NP/NP ->
Comp -> 'that' 
 
#Above contains one "gap-introduction" production, namely S[-INV] -> NP S/NP. 
#In order to percolate the slash feature correctly, 
#we need to add slashes with variable values to both sides of the arrow in productions 
#that expand S, VP and NP. 

#For example, VP/?x -> V SBar/?x is the slashed version of VP -> V SBar 
#and says that a slash value can be specified on the VP parent of a constituent 
#if the same value is also specified on the SBar child. 

#Finally, NP/NP -> allows the slash information on NP to be discharged as the empty string. 

#Example - Parsing - who do you claim that you like

>>> tokens = 'who do you claim that you like'.split()
>>> from nltk import load_parser
>>> cp = load_parser('grammars/book_grammars/feat1.fcfg')
>>> for tree in cp.parse(tokens):
        print(tree)
(S[-INV]
  (NP[+WH] who)
  (S[+INV]/NP[]
    (V[+AUX] do)
    (NP[-WH] you)
    (VP[]/NP[]
      (V[-AUX, SUBCAT='clause'] claim)
      (SBar[]/NP[]
        (Comp[] that)
        (S[-INV]/NP[]
          (NP[-WH] you)
          (VP[]/NP[] (V[-AUX, SUBCAT='trans'] like) (NP[]/NP[] )))))))
 
 
 
#This grammar  will also allow us to parse sentences without gaps:
>>> tokens = 'you claim that you like cats'.split()
>>> for tree in cp.parse(tokens):
        print(tree)
(S[-INV]
  (NP[-WH] you)
  (VP[]
    (V[-AUX, SUBCAT='clause'] claim)
    (SBar[]
      (Comp[] that)
      (S[-INV]
        (NP[-WH] you)
        (VP[] (V[-AUX, SUBCAT='trans'] like) (NP[-WH] cats))))))
 
 

#In addition, it admits inverted sentences which do not involve wh constructions:
>>> tokens = 'rarely do you sing'.split()
>>> for tree in cp.parse(tokens):
        print(tree)
(S[-INV]
  (Adv[+NEG] rarely)
  (S[+INV]
    (V[+AUX] do)
    (NP[-WH] you)
    (VP[] (V[-AUX, SUBCAT='intrans'] sing))))
 
 
 
 
 

 
 
###Feture Grammer - Feature Structures & Unification- Examples 
  
>>> from __future__ import print_function
>>> from nltk.featstruct import FeatStruct
>>> from nltk.sem.logic import Variable, VariableExpression, Expression


#A feature structure is a mapping from feature identifiers to feature values, 
#where feature values can be simple values (like strings or ints),
#nested feature structures, or variables:

>>> fs1 = FeatStruct(number='singular', person=3)
>>> print(fs1)
[ number = 'singular' ]
[ person = 3          ]



>>> fs2 = FeatStruct(type='NP', agr=fs1)
>>> print(fs2)
[ agr  = [ number = 'singular' ] ]
[        [ person = 3          ] ]
[                                ]
[ type = 'NP'                    ]


#Variables are used to indicate that two features should be assigned the same value. 

>>> fs3 = FeatStruct(agr=FeatStruct(number=Variable('?n')),
            subj=FeatStruct(number=Variable('?n')))
>>> print(fs3)
[ agr  = [ number = ?n ] ]
[                        ]
[ subj = [ number = ?n ] ]


#unification 

>>> print(fs2.unify(fs3))
[ agr  = [ number = 'singular' ] ]
[        [ person = 3          ] ]
[                                ]
[ subj = [ number = 'singular' ] ]
[                                ]
[ type = 'NP'                    ]


#When two inconsistent feature structures are unified, the unification fails and returns None.

>>> fs4 = FeatStruct(agr=FeatStruct(person=1))
>>> print(fs4.unify(fs2))
None
>>> print(fs2.unify(fs4))
None

 

##Feature Structure Types
•feature dictionaries, implemented by FeatDict, act like Python dictionaries. 
 Feature identifiers may be strings or instances of the Feature class.
•feature lists, implemented by FeatList, act like Python lists. 
 Feature identifiers are integers.

>>> type(FeatStruct(number='singular'))
<class 'nltk.featstruct.FeatDict'>
>>> type(FeatStruct([1,2,3]))
<class 'nltk.featstruct.FeatList'>


#Two feature lists will unify with each other only if they have equal lengths, 
#and all of their feature values match. 

#If you wish to write a feature list that contains 'unknown' values, you must use variables:

>>> fs1 = FeatStruct([1,2,Variable('?y')])
>>> fs2 = FeatStruct([1,Variable('?x'),3])
>>> fs1.unify(fs2)
[1, 2, 3]

 

##Parsing Feature Structure Strings


>>> FeatStruct('[tense="past", agr=[number="sing", person=3]]')
[agr=[number='sing', person=3], tense='past']

#OR 
>>> FeatStruct('[tense=past, agr=[number=sing, person=3]]')
[agr=[number='sing', person=3], tense='past']

>>> FeatStruct('[1, 2, 3]')
[1, 2, 3]


#The expression [] is treated as an empty feature dictionary, not an empty feature list:

>>> type(FeatStruct('[]'))
<class 'nltk.featstruct.FeatDict'>



##Feature Paths
#Features can be specified using feature paths, 
#or tuples of feature identifiers that specify path through the nested feature structures to a value.

>>> fs1 = FeatStruct('[x=1, y=[1,2,[z=3]]]')
>>> fs1['y']
[1, 2, [z=3]]
>>> fs1['y', 2]
[z=3]
>>> fs1['y', 2, 'z']
3

 

#Reentrance
#Feature structures may contain reentrant feature values. 
#A reentrant feature value is a single feature structure that can be accessed via multiple feature paths.

>>> fs1 = FeatStruct(x='val')
>>> fs2 = FeatStruct(a=fs1, b=fs1)
>>> print(fs2)
[ a = (1) [ x = 'val' ] ]
[                       ]
[ b -> (1)              ]
>>> fs2
[a=(1)[x='val'], b->(1)]


#reentrane is displayed by marking a feature structure with a unique identifier, 

>>> FeatStruct('[a=(1)[], b->(1), c=[d->(1)]]')
[a=(1)[], b->(1), c=[d->(1)]]


#Reentrant feature structures may contain cycles:

>>> fs3 = FeatStruct('(1)[a->(1)]')
>>> fs3['a', 'a', 'a', 'a']
(1)[a->(1)]
>>> fs3['a', 'a', 'a', 'a'] is fs3
True


#Unification preserves the reentrance relations 
#imposed by both of the unified feature structures. 

#In the feature structure resulting from unification, 
#any modifications to a reentrant feature value will be visible using any of its feature paths.

>>> fs3.unify(FeatStruct('[a=[b=12], c=33]'))
(1)[a->(1), b=12, c=33]

 

##Feature Structure Equality
#Two feature structures are considered equal if they assign the same values to all features
#and they contain the same reentrances.

>>> fs1 = FeatStruct('[a=(1)[x=1], b->(1)]')
>>> fs2 = FeatStruct('[a=(1)[x=1], b->(1)]')
>>> fs3 = FeatStruct('[a=[x=1], b=[x=1]]')
>>> fs1 == fs1, fs1 is fs1
(True, True)
>>> fs1 == fs2, fs1 is fs2
(True, False)
>>> fs1 == fs3, fs1 is fs3
(False, False)

#To test two feature structures for equality while ignoring reentrance relations, 
#use the equal_values() method:

>>> fs1.equal_values(fs1)
True
>>> fs1.equal_values(fs2)
True
>>> fs1.equal_values(fs3)
True

 

##Feature Value Sets & Feature Value Tuples
#FeatureValueTuple and FeatureValueSet. 
#Both of these types are considered base values -- i.e., unification does not apply to them. 
#However, variable binding does apply to any values that they contain.

#Feature value tuples are written with parentheses:

>>> fs1 = FeatStruct('[x=(?x, ?y)]')
>>> fs1
[x=(?x, ?y)]
>>> fs1.substitute_bindings({Variable('?x'): 1, Variable('?y'): 2})
[x=(1, 2)]


#Feature sets are written with braces:

>>> fs1 = FeatStruct('[x={?x, ?y}]')
>>> fs1
[x={?x, ?y}]
>>> fs1.substitute_bindings({Variable('?x'): 1, Variable('?y'): 2})
[x={1, 2}]


#nltk defines feature value unions (for sets) and feature value concatenations (for tuples). 
#These are written using '+', and can be used to combine sets & tuples:

>>> fs1 = FeatStruct('[x=((1, 2)+?z), z=?z]')
>>> fs1
[x=((1, 2)+?z), z=?z]
>>> fs1.unify(FeatStruct('[z=(3, 4, 5)]'))
[x=(1, 2, 3, 4, 5), z=(3, 4, 5)]


##Light-weight Feature Structures
#Many of the functions defined by nltk.featstruct can be applied directly to simple Python dictionaries and lists, 
#In other words, Python dicts and lists can be used as "light-weight" feature structures.

>>> # Note: pprint prints dicts sorted
>>> from pprint import pprint
>>> from nltk.featstruct import unify
>>> pprint(unify(dict(x=1, y=dict()), dict(a='a', y=dict(b='b'))))
{'a': 'a', 'x': 1, 'y': {'b': 'b'}}

#diff
•Python dictionaries & lists ignore reentrance 
 when checking for equality between values. 
 But two FeatStructs with different reentrances are considered nonequal, 
 even if all their base values are equal.
•FeatStructs can be easily frozen, allowing them to be used as keys in hash tables. 
 Python dictionaries and lists can not.
•FeatStructs display reentrance in their string representations; 
 Python dictionaries and lists do not.
•FeatStructs may not be mixed with Python dictionaries and lists 
 (e.g., when performing unification).
•FeatStructs provide a number of useful methods, such as walk() and cyclic(), 
 which are not available for Python dicts & lists.

 
## Reference of FeatStruct
class nltk.featstruct.FeatStruct
    copy(deep=True)
    Return a new copy of self. The new copy will not be frozen.

    cyclic()
    Return True if this feature structure contains itself.

    equal_values(other, check_reentrance=False)
    Return True if self and other assign the same value to to every feature. 
    the == is equivalent to equal_values() with check_reentrance=True. 

    freeze()
    Make this feature structure, and any feature structures it contains, immutable. 

    frozen()
    Return True if this feature structure is immutable. 

    remove_variables()
    Return the feature structure that is obtained by deleting any feature whose value is a Variable

    rename_variables(vars=None, used_vars=(), new_vars=None)

    retract_bindings(bindings)

    substitute_bindings(bindings)

    subsumes(other)
    Return True if self subsumes other. I.e., return true If unifying self with other would result in a feature structure equal to other.

    unify(other, bindings=None, trace=False, fail=None, rename_vars=True)variables()

    walk()
    Return an iterator that generates this feature structure, and each feature structure it contains. 



##Dictionary access methods (non-mutating)

>>> fs1 = FeatStruct(a=1, b=2, c=3)
>>> fs2 = FeatStruct(x=fs1, y='x')

#Feature structures support all dictionary methods (excluding the class method dict.fromkeys()). Non-mutating methods:

>>> sorted(fs2.keys())                               # keys()
['x', 'y']
>>> sorted(fs2.values())                             # values()
[[a=1, b=2, c=3], 'x']
>>> sorted(fs2.items())                              # items()
[('x', [a=1, b=2, c=3]), ('y', 'x')]
>>> sorted(fs2)                                      # __iter__()
['x', 'y']
>>> 'a' in fs2, 'x' in fs2                           # __contains__()
(False, True)
>>> fs2.has_key('a'), fs2.has_key('x')               # has_key()
(False, True)
>>> fs2['x'], fs2['y']                               # __getitem__()
([a=1, b=2, c=3], 'x')
>>> fs2['a']                                         # __getitem__()
Traceback (most recent call last):
  . . .
KeyError: 'a'
>>> fs2.get('x'), fs2.get('y'), fs2.get('a')         # get()
([a=1, b=2, c=3], 'x', None)
>>> fs2.get('x', 'hello'), fs2.get('a', 'hello')     # get()
([a=1, b=2, c=3], 'hello')
>>> len(fs1), len(fs2)                               # __len__
(3, 2)
>>> fs2.copy()                                       # copy()
[x=[a=1, b=2, c=3], y='x']
>>> fs2.copy() is fs2                                # copy()
False


##Dictionary access methods (mutating)

>>> fs1 = FeatStruct(a=1, b=2, c=3)
>>> fs2 = FeatStruct(x=fs1, y='x')

#Setting features (__setitem__())

>>> fs1['c'] = 5
>>> fs1
[a=1, b=2, c=5]
>>> fs1['x'] = 12
>>> fs1
[a=1, b=2, c=5, x=12]
>>> fs2['x', 'a'] = 2
>>> fs2
[x=[a=2, b=2, c=5, x=12], y='x']
>>> fs1
[a=2, b=2, c=5, x=12]


#Deleting features (__delitem__())

>>> del fs1['x']
>>> fs1
[a=2, b=2, c=5]
>>> del fs2['x', 'a']
>>> fs1
[b=2, c=5]


#setdefault():

>>> fs1.setdefault('b', 99)
2
>>> fs1
[b=2, c=5]
>>> fs1.setdefault('x', 99)
99
>>> fs1
[b=2, c=5, x=99]


#update():

>>> fs2.update({'a':'A', 'b':'B'}, c='C')
>>> fs2
[a='A', b='B', c='C', x=[b=2, c=5, x=99], y='x']


#pop():

>>> fs2.pop('a')
'A'
>>> fs2
[b='B', c='C', x=[b=2, c=5, x=99], y='x']
>>> fs2.pop('a')
Traceback (most recent call last):
  . . .
KeyError: 'a'
>>> fs2.pop('a', 'foo')
'foo'
>>> fs2
[b='B', c='C', x=[b=2, c=5, x=99], y='x']


#clear():

>>> fs1.clear()
>>> fs1
[]
>>> fs2
[b='B', c='C', x=[], y='x']


#popitem():

>>> sorted([fs2.popitem() for i in range(len(fs2))])
[('b', 'B'), ('c', 'C'), ('x', []), ('y', 'x')]
>>> fs2
[]


#Once a feature structure has been frozen, it may not be mutated.

>>> fs1 = FeatStruct('[x=1, y=2, z=[a=3]]')
>>> fs1.freeze()
>>> fs1.frozen()
True
>>> fs1['z'].frozen()
True

>>> fs1['x'] = 5
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> del fs1['x']
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> fs1.clear()
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> fs1.pop('x')
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> fs1.popitem()
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> fs1.setdefault('x')
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.
>>> fs1.update(z=22)
Traceback (most recent call last):
  . . .
ValueError: Frozen FeatStructs may not be modified.

 

#Feature Paths

>>> fs1 = FeatStruct(a=1, b=2,
                    c=FeatStruct(
                        d=FeatStruct(e=12),
                        f=FeatStruct(g=55, h='hello')))
>>> fs1[()]
[a=1, b=2, c=[d=[e=12], f=[g=55, h='hello']]]
>>> fs1['a'], fs1[('a',)]
(1, 1)
>>> fs1['c','d','e']
12
>>> fs1['c','f','g']
55


#Feature paths that select unknown features raise KeyError:

>>> fs1['c', 'f', 'e']
Traceback (most recent call last):
  . . .
KeyError: ('c', 'f', 'e')
>>> fs1['q', 'p']
Traceback (most recent call last):
  . . .
KeyError: ('q', 'p')


#Feature paths can go through reentrant structures:

>>> fs2 = FeatStruct('(1)[a=[b=[c->(1), d=5], e=11]]')
>>> fs2['a', 'b', 'c', 'a', 'e']
11
>>> fs2['a', 'b', 'c', 'a', 'b', 'd']
5
>>> fs2[tuple('abcabcabcabcabcabcabcabcabcabca')]
(1)[b=[c=[a->(1)], d=5], e=11]


#Indexing requires strings, Features, or tuples; other types raise a TypeError:

>>> fs2[12]
Traceback (most recent call last):
  . . .
TypeError: Expected feature name or path.  Got 12.
>>> fs2[list('abc')]
Traceback (most recent call last):
  . . .
TypeError: Expected feature name or path.  Got ['a', 'b', 'c'].


#Feature paths can also be used with get(), has_key(), and __contains__().

>>> fpath1 = tuple('abcabc')
>>> fpath2 = tuple('abcabz')
>>> fs2.get(fpath1), fs2.get(fpath2)
((1)[a=[b=[c->(1), d=5], e=11]], None)
>>> fpath1 in fs2, fpath2 in fs2
(True, False)
>>> fs2.has_key(fpath1), fs2.has_key(fpath2)
(True, False)

 

#Empty feature struct:

>>> FeatStruct('[]')
[]


#Strings of the form "+name" and "-name" may be used to specify boolean values.

>>> FeatStruct('[-bar, +baz, +foo]')
[-bar, +baz, +foo]


#None, True, and False are recognized as values:

>>> FeatStruct('[bar=True, baz=False, foo=None]')
[+bar, -baz, foo=None]


#Special features:

>>> FeatStruct('NP/VP')
NP[]/VP[]
>>> FeatStruct('?x/?x')
?x[]/?x[]
>>> print(FeatStruct('VP[+fin, agr=?x, tense=past]/NP[+pl, agr=?x]'))
[ *type*  = 'VP'              ]
[                             ]
[           [ *type* = 'NP' ] ]
[ *slash* = [ agr    = ?x   ] ]
[           [ pl     = True ] ]
[                             ]
[ agr     = ?x                ]
[ fin     = True              ]
[ tense   = 'past'            ]

#Here the slash feature gets coerced:
>>> FeatStruct('[*slash*=a, x=b, *type*="NP"]')
NP[x='b']/a[]

>>> FeatStruct('NP[sem=<bob>]/NP')
NP[sem=<bob>]/NP[]
>>> FeatStruct('S[sem=<walk(bob)>]')
S[sem=<walk(bob)>]
>>> print(FeatStruct('NP[sem=<bob>]/NP'))
[ *type*  = 'NP'              ]
[                             ]
[ *slash* = [ *type* = 'NP' ] ]
[                             ]
[ sem     = <bob>             ]


#Playing with ranges:

>>> from nltk.featstruct import RangeFeature, FeatStructReader
>>> width = RangeFeature('width')
>>> reader = FeatStructReader([width])
>>> fs1 = reader.fromstring('[*width*=-5:12]')
>>> fs2 = reader.fromstring('[*width*=2:123]')
>>> fs3 = reader.fromstring('[*width*=-7:-2]')
>>> fs1.unify(fs2)
[*width*=(2, 12)]
>>> fs1.unify(fs3)
[*width*=(-5, -2)]
>>> print(fs2.unify(fs3)) # no overlap in width.
None


#The slash feature has a default value of 'False':

>>> print(FeatStruct('NP[]/VP').unify(FeatStruct('NP[]'), trace=1))
<BLANKLINE>
Unification trace:
   / NP[]/VP[]
  |\ NP[]
  |
  | Unify feature: *type*
  |    / 'NP'
  |   |\ 'NP'
  |   |
  |   +-->'NP'
  |
  | Unify feature: *slash*
  |    / VP[]
  |   |\ False
  |   |
  X   X <-- FAIL
None


>>> FeatStruct(pos='n', agr=FeatStruct(number='pl', gender='f'))
[agr=[gender='f', number='pl'], pos='n']
>>> FeatStruct(r'NP[sem=<bob>]/NP')
NP[sem=<bob>]/NP[]
>>> FeatStruct(r'S[sem=<app(?x, ?y)>]')
S[sem=<?x(?y)>]
>>> FeatStruct('?x/?x')
?x[]/?x[]
>>> FeatStruct('VP[+fin, agr=?x, tense=past]/NP[+pl, agr=?x]')
VP[agr=?x, +fin, tense='past']/NP[agr=?x, +pl]
>>> FeatStruct('S[sem = <app(?subj, ?vp)>]')
S[sem=<?subj(?vp)>]

>>> FeatStruct('S')
S[]


#The parser also includes support for reading sets and tuples.

>>> FeatStruct('[x={1,2,2,2}, y={/}]')
[x={1, 2}, y={/}]
>>> FeatStruct('[x=(1,2,2,2), y=()]')
[x=(1, 2, 2, 2), y=()]
>>> print(FeatStruct('[x=(1,[z=(1,2,?x)],?z,{/})]'))
[ x = (1, [ z = (1, 2, ?x) ], ?z, {/}) ]


#Note that we can't put a featstruct inside a tuple, because doing so would hash it, and it's not frozen yet:
>>> print(FeatStruct('[x={[]}]'))
Traceback (most recent call last):
  . . .
TypeError: FeatStructs must be frozen before they can be hashed.


#There's a special syntax for taking the union of sets: "{...+...}". 
#The elements should only be variables or sets.

>>> FeatStruct('[x={?a+?b+{1,2,3}}]')
[x={?a+?b+{1, 2, 3}}]


#There's a special syntax for taking the concatenation of tuples: "(...+...)". 
#The elements should only be variables or tuples.

>>> FeatStruct('[x=(?a+?b+(1,2,3))]')
[x=(?a+?b+(1, 2, 3))]




#Very simple unifications give the expected results:

>>> FeatStruct().unify(FeatStruct())
[]
>>> FeatStruct(number='singular').unify(FeatStruct())
[number='singular']
>>> FeatStruct().unify(FeatStruct(number='singular'))
[number='singular']
>>> FeatStruct(number='singular').unify(FeatStruct(person=3))
[number='singular', person=3]


#Merging nested structures:

>>> fs1 = FeatStruct('[A=[B=b]]')
>>> fs2 = FeatStruct('[A=[C=c]]')
>>> fs1.unify(fs2)
[A=[B='b', C='c']]
>>> fs2.unify(fs1)
[A=[B='b', C='c']]


#A basic case of reentrant unification

>>> fs4 = FeatStruct('[A=(1)[B=b], E=[F->(1)]]')
>>> fs5 = FeatStruct("[A=[C='c'], E=[F=[D='d']]]")
>>> fs4.unify(fs5)
[A=(1)[B='b', C='c', D='d'], E=[F->(1)]]
>>> fs5.unify(fs4)
[A=(1)[B='b', C='c', D='d'], E=[F->(1)]]


#More than 2 paths to a value

>>> fs1 = FeatStruct("[a=[],b=[],c=[],d=[]]")
>>> fs2 = FeatStruct('[a=(1)[], b->(1), c->(1), d->(1)]')
>>> fs1.unify(fs2)
[a=(1)[], b->(1), c->(1), d->(1)]


#fs1[a] gets unified with itself

>>> fs1 = FeatStruct('[x=(1)[], y->(1)]')
>>> fs2 = FeatStruct('[x=(1)[], y->(1)]')
>>> fs1.unify(fs2)
[x=(1)[], y->(1)]


#Bound variables should get forwarded appropriately

>>> fs1 = FeatStruct('[A=(1)[X=x], B->(1), C=?cvar, D=?dvar]')
>>> fs2 = FeatStruct('[A=(1)[Y=y], B=(2)[Z=z], C->(1), D->(2)]')
>>> fs1.unify(fs2)
[A=(1)[X='x', Y='y', Z='z'], B->(1), C->(1), D->(1)]
>>> fs2.unify(fs1)
[A=(1)[X='x', Y='y', Z='z'], B->(1), C->(1), D->(1)]


#Cyclic structure created by unification.

>>> fs1 = FeatStruct('[F=(1)[], G->(1)]')
>>> fs2 = FeatStruct('[F=[H=(2)[]], G->(2)]')
>>> fs3 = fs1.unify(fs2)
>>> fs3
[F=(1)[H->(1)], G->(1)]
>>> fs3['F'] is fs3['G']
True
>>> fs3['F'] is fs3['G']['H']
True
>>> fs3['F'] is fs3['G']['H']['H']
True
>>> fs3['F'] is fs3['F']['H']['H']['H']['H']['H']['H']['H']['H']
True


#Cyclic structure created w/ variables.

>>> fs1 = FeatStruct('[F=[H=?x]]')
>>> fs2 = FeatStruct('[F=?x]')
>>> fs3 = fs1.unify(fs2, rename_vars=False)
>>> fs3
[F=(1)[H->(1)]]
>>> fs3['F'] is fs3['F']['H']
True
>>> fs3['F'] is fs3['F']['H']['H']
True
>>> fs3['F'] is fs3['F']['H']['H']['H']['H']['H']['H']['H']['H']
True


#Unifying w/ a cyclic feature structure.

>>> fs4 = FeatStruct('[F=[H=[H=[H=(1)[]]]], K->(1)]')
>>> fs3.unify(fs4)
[F=(1)[H->(1)], K->(1)]
>>> fs4.unify(fs3)
[F=(1)[H->(1)], K->(1)]


#Variable bindings should preserve reentrance.

>>> bindings = {}
>>> fs1 = FeatStruct("[a=?x]")
>>> fs2 = fs1.unify(FeatStruct("[a=[]]"), bindings)
>>> fs2['a'] is bindings[Variable('?x')]
True
>>> fs2.unify(FeatStruct("[b=?x]"), bindings)
[a=(1)[], b->(1)]


#Aliased variable tests

>>> fs1 = FeatStruct("[a=?x, b=?x]")
>>> fs2 = FeatStruct("[b=?y, c=?y]")
>>> bindings = {}
>>> fs3 = fs1.unify(fs2, bindings)
>>> fs3
[a=?x, b=?x, c=?x]
>>> bindings
{Variable('?y'): Variable('?x')}
>>> fs3.unify(FeatStruct("[a=1]"))
[a=1, b=1, c=1]


#If we keep track of the bindings, 
#then we can use the same variable over multiple calls to unify.

>>> bindings = {}
>>> fs1 = FeatStruct('[a=?x]')
>>> fs2 = fs1.unify(FeatStruct('[a=[]]'), bindings)
>>> fs2.unify(FeatStruct('[b=?x]'), bindings)
[a=(1)[], b->(1)]
>>> bindings
{Variable('?x'): []}

 

#Unification Bindings

>>> bindings = {}
>>> fs1 = FeatStruct('[a=?x]')
>>> fs2 = FeatStruct('[a=12]')
>>> fs3 = FeatStruct('[b=?x]')
>>> fs1.unify(fs2, bindings)
[a=12]
>>> bindings
{Variable('?x'): 12}
>>> fs3.substitute_bindings(bindings)
[b=12]
>>> fs3 # substitute_bindings didn't mutate fs3.
[b=?x]
>>> fs2.unify(fs3, bindings)
[a=12, b=12]

>>> bindings = {}
>>> fs1 = FeatStruct('[a=?x, b=1]')
>>> fs2 = FeatStruct('[a=5, b=?x]')
>>> fs1.unify(fs2, bindings)
[a=5, b=1]
>>> sorted(bindings.items())
[(Variable('?x'), 5), (Variable('?x2'), 1)]

 

#Expressions

>>> e = Expression.fromstring('\\P y.P(z,y)')
>>> fs1 = FeatStruct(x=e, y=Variable('z'))
>>> fs2 = FeatStruct(y=VariableExpression(Variable('John')))
>>> fs1.unify(fs2)
[x=<\P y.P(John,y)>, y=<John>]



#Remove Variables

>>> FeatStruct('[a=?x, b=12, c=[d=?y]]').remove_variables()
[b=12, c=[]]
>>> FeatStruct('(1)[a=[b=?x,c->(1)]]').remove_variables()
(1)[a=[c->(1)]]



#The equal_values method checks whether two feature structures assign the same value to every feature. 
#If the optional argument check_reentrances is supplied, then it also returns false if there is any difference in the reentrances.

>>> a = FeatStruct('(1)[x->(1)]')
>>> b = FeatStruct('(1)[x->(1)]')
>>> c = FeatStruct('(1)[x=[x->(1)]]')
>>> d = FeatStruct('[x=(1)[x->(1)]]')
>>> e = FeatStruct('(1)[x=[x->(1), y=1], y=1]')
>>> def compare(x,y):
        assert x.equal_values(y, True) == y.equal_values(x, True)
        assert x.equal_values(y, False) == y.equal_values(x, False)
        if x.equal_values(y, True):
            assert x.equal_values(y, False)
            print('equal values, same reentrance')
        elif x.equal_values(y, False):
            print('equal values, different reentrance')
        else:
            print('different values')

>>> compare(a, a)
equal values, same reentrance
>>> compare(a, b)
equal values, same reentrance
>>> compare(a, c)
equal values, different reentrance
>>> compare(a, d)
equal values, different reentrance
>>> compare(c, d)
equal values, different reentrance
>>> compare(a, e)
different values
>>> compare(c, e)
different values
>>> compare(d, e)
different values
>>> compare(e, e)
equal values, same reentrance


#Feature structures may not be hashed until they are frozen:

>>> hash(a)
Traceback (most recent call last):
  . . .
TypeError: FeatStructs must be frozen before they can be hashed.
>>> a.freeze()
>>> v = hash(a)




#Tracing

>>> fs1 = FeatStruct('[a=[b=(1)[], c=?x], d->(1), e=[f=?x]]')
>>> fs2 = FeatStruct('[a=(1)[c="C"], e=[g->(1)]]')
>>> fs1.unify(fs2, trace=True)
<BLANKLINE>
Unification trace:
   / [a=[b=(1)[], c=?x], d->(1), e=[f=?x]]
  |\ [a=(1)[c='C'], e=[g->(1)]]
  |
  | Unify feature: a
  |    / [b=[], c=?x]
  |   |\ [c='C']
  |   |
  |   | Unify feature: a.c
  |   |    / ?x
  |   |   |\ 'C'
  |   |   |
  |   |   +-->Variable('?x')
  |   |
  |   +-->[b=[], c=?x]
  |       Bindings: {?x: 'C'}
  |
  | Unify feature: e
  |    / [f=?x]
  |   |\ [g=[c='C']]
  |   |
  |   +-->[f=?x, g=[b=[], c=?x]]
  |       Bindings: {?x: 'C'}
  |
  +-->[a=(1)[b=(2)[], c='C'], d->(2), e=[f='C', g->(1)]]
      Bindings: {?x: 'C'}
[a=(1)[b=(2)[], c='C'], d->(2), e=[f='C', g->(1)]]
>>>
>>> fs1 = FeatStruct('[a=?x, b=?z, c=?z]')
>>> fs2 = FeatStruct('[a=?y, b=?y, c=?q]')
>>> #fs1.unify(fs2, trace=True)
>>>

 

#Unification on Dicts & Lists
#It's possible to do unification on dictionaries:

>>> from nltk.featstruct import unify
>>> pprint(unify(dict(x=1, y=dict(z=2)), dict(x=1, q=5)), width=1)
{'q': 5, 'x': 1, 'y': {'z': 2}}


#It's possible to do unification on lists as well:

>>> unify([1, 2, 3], [1, Variable('x'), 3])
[1, 2, 3]


#Mixing dicts and lists is fine:

>>> pprint(unify([dict(x=1, y=dict(z=2)),3], [dict(x=1, q=5),3]),
                        width=1)
[{'q': 5, 'x': 1, 'y': {'z': 2}}, 3]


#Mixing dicts and FeatStructs is discouraged:

>>> unify(dict(x=1), FeatStruct(x=1))
Traceback (most recent call last):
  . . .
ValueError: Mixing FeatStruct objects with Python dicts and lists is not supported.


#But you can do it if you really want, by explicitly stating that both dictionaries and FeatStructs should be treated as feature structures:

>>> unify(dict(x=1), FeatStruct(x=1), fs_class=(dict, FeatStruct))
{'x': 1}



#Finding Conflicts

>>> from nltk.featstruct import conflicts
>>> fs1 = FeatStruct('[a=[b=(1)[c=2], d->(1), e=[f->(1)]]]')
>>> fs2 = FeatStruct('[a=[b=[c=[x=5]], d=[c=2], e=[f=[c=3]]]]')
>>> for path in conflicts(fs1, fs2):
...     print('%-8s: %r vs %r' % ('.'.join(path), fs1[path], fs2[path]))
a.b.c   : 2 vs [x=5]
a.e.f.c : 2 vs 3



#Retracting Bindings

>>> from nltk.featstruct import retract_bindings
>>> bindings = {}
>>> fs1 = FeatStruct('[a=?x, b=[c=?y]]')
>>> fs2 = FeatStruct('[a=(1)[c=[d=1]], b->(1)]')
>>> fs3 = fs1.unify(fs2, bindings)
>>> print(fs3)
[ a = (1) [ c = [ d = 1 ] ] ]
[                           ]
[ b -> (1)                  ]
>>> pprint(bindings)
{Variable('?x'): [c=[d=1]], Variable('?y'): [d=1]}
>>> retract_bindings(fs3, bindings)
[a=?x, b=?x]
>>> pprint(bindings)
{Variable('?x'): [c=?y], Variable('?y'): [d=1]}


 
 
 

###Feature grammer - Feature Grammar Parsing - Examples 
                                  
#Grammars can be parsed from strings.

#Example of merging of Feature struct values 

>>> from __future__ import print_function
>>> import nltk
>>> from nltk import grammar, parse
>>> g = """
    % start DP
    DP[AGR=?a] -> D[AGR=?a] N[AGR=?a]
    D[AGR=[NUM='sg', PERS=3]] -> 'this' | 'that'
    D[AGR=[NUM='pl', PERS=3]] -> 'these' | 'those'
    D[AGR=[NUM='pl', PERS=1]] -> 'we'
    D[AGR=[PERS=2]] -> 'you'
    N[AGR=[NUM='sg', GND='m']] -> 'boy'
    N[AGR=[NUM='pl', GND='m']] -> 'boys'
    N[AGR=[NUM='sg', GND='f']] -> 'girl'
    N[AGR=[NUM='pl', GND='f']] -> 'girls'
    N[AGR=[NUM='sg']] -> 'student'
    N[AGR=[NUM='pl']] -> 'students'
    """
>>> grammar = grammar.FeatureGrammar.fromstring(g)
>>> tokens = 'these girls'.split()
>>> parser = parse.FeatureEarleyChartParser(grammar, trace=1)
>>> trees = list(parser.parse(tokens))
|.these.girls.|
|[-----]     .| [0:1] 'these'
|.     [-----]| [1:2] 'girls'
|>     .     .| [0:0] DP[AGR=?a] -> * D[AGR=?a] N[AGR=?a] {}
|>     .     .| [0:0] D[AGR=[NUM='pl', PERS=3]] -> * 'these' {}
|[-----]     .| [0:1] D[AGR=[NUM='pl', PERS=3]] -> 'these' *
|[----->     .| [0:1] DP[AGR=?a] -> D[AGR=?a] * N[AGR=?a] {?a: [NUM='pl', PERS=3]}
|.     >     .| [1:1] N[AGR=[GND='f', NUM='pl']] -> * 'girls' {}
|.     [-----]| [1:2] N[AGR=[GND='f', NUM='pl']] -> 'girls' *
|[===========]| [0:2] DP[AGR=[GND='f', NUM='pl', PERS=3]] -> D[AGR=[GND='f', NUM='pl', PERS=3]] N[AGR=[GND='f', NUM='pl', PERS=3]] *
>>> for tree in trees: print(tree)
(DP[AGR=[GND='f', NUM='pl', PERS=3]]
  (D[AGR=[NUM='pl', PERS=3]] these)
  (N[AGR=[GND='f', NUM='pl']] girls))

>>> print(trees[0])
(DP[AGR=[GND='f', NUM='pl', PERS=3]]
  (D[AGR=[NUM='pl', PERS=3]] these)
  (N[AGR=[GND='f', NUM='pl']] girls))
>>> trees[0].label()
DP[AGR=[GND='f', NUM='pl', PERS=3]]
>>> trees[0].label()['AGR']
[GND='f', NUM='pl', PERS=3]
>>> type(trees[0].label()['AGR'])
<class 'nltk.grammar.FeatStructNonterminal'>
>>> trees[0].label()['AGR']['GND']
'f'
>>> trees[0][1]
Tree(N[AGR=[GND='f', NUM='pl']], ['girls'])
>>> trees[0][1].label()
N[AGR=[GND='f', NUM='pl']]
>>> trees[0][1,0]
'girls'
>>> trees[0].leaves()
['these', 'girls']
>>> trees[0].pos()  #list
[('these', D[AGR=[NUM='pl', PERS=3]]), ('girls', N[AGR=[GND='f', NUM='pl']])]
>>> trees[0].flatten()
Tree(DP[AGR=[GND='f', NUM='pl', PERS=3]], ['these', 'girls'])


#from file 
>>> nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')



>>> cp = parse.load_parser('grammars/book_grammars/feat0.fcfg', trace=1)
>>> sent = 'Kim likes children'
>>> tokens = sent.split()
>>> tokens
['Kim', 'likes', 'children']
>>> trees = cp.parse(tokens)
|.Kim .like.chil.|
|[----]    .    .| [0:1] 'Kim'
|.    [----]    .| [1:2] 'likes'
|.    .    [----]| [2:3] 'children'
|[----]    .    .| [0:1] PropN[NUM='sg'] -> 'Kim' *
|[----]    .    .| [0:1] NP[NUM='sg'] -> PropN[NUM='sg'] *
|[---->    .    .| [0:1] S[] -> NP[NUM=?n] * VP[NUM=?n] {?n: 'sg'}
|.    [----]    .| [1:2] TV[NUM='sg', TENSE='pres'] -> 'likes' *
|.    [---->    .| [1:2] VP[NUM=?n, TENSE=?t] -> TV[NUM=?n, TENSE=?t] * NP[] {?n: 'sg', ?t: 'pres'}
|.    .    [----]| [2:3] N[NUM='pl'] -> 'children' *
|.    .    [----]| [2:3] NP[NUM='pl'] -> N[NUM='pl'] *
|.    .    [---->| [2:3] S[] -> NP[NUM=?n] * VP[NUM=?n] {?n: 'pl'}
|.    [---------]| [1:3] VP[NUM='sg', TENSE='pres'] -> TV[NUM='sg', TENSE='pres'] NP[] *
|[==============]| [0:3] S[] -> NP[NUM='sg'] VP[NUM='sg'] *
>>> for tree in trees: print(tree)
(S[]
  (NP[NUM='sg'] (PropN[NUM='sg'] Kim))
  (VP[NUM='sg', TENSE='pres']
    (TV[NUM='sg', TENSE='pres'] likes)
    (NP[NUM='pl'] (N[NUM='pl'] children))))


##Tracing the parsing 
>>> sent = 'who do you claim that you like'
>>> tokens = sent.split()
>>> nltk.data.show_cfg('grammars/book_grammars/feat1.fcfg')
# -/+ZZ means ZZ is boolean value 
# XX/yy means XX without yy 

% start S
# Grammar Productions 
S[-INV] -> NP VP
S[-INV]/?x -> NP VP/?x
S[-INV] -> NP S/NP
S[-INV] -> Adv[+NEG] S[+INV]
S[+INV] -> V[+AUX] NP VP
S[+INV]/?x -> V[+AUX] NP VP/?x
SBar -> Comp S[-INV]
SBar/?x -> Comp S[-INV]/?x
VP -> V[SUBCAT=intrans, -AUX]
VP -> V[SUBCAT=trans, -AUX] NP
VP/?x -> V[SUBCAT=trans, -AUX] NP/?x
VP -> V[SUBCAT=clause, -AUX] SBar
VP/?x -> V[SUBCAT=clause, -AUX] SBar/?x
VP -> V[+AUX] VP
VP/?x -> V[+AUX] VP/?x
# Lexical Productions
V[SUBCAT=intrans, -AUX] -> 'walk' | 'sing'
V[SUBCAT=trans, -AUX] -> 'see' | 'like'
V[SUBCAT=clause, -AUX] -> 'say' | 'claim'
V[+AUX] -> 'do' | 'can'
NP[-WH] -> 'you' | 'cats'
NP[+WH] -> 'who'
Adv[+NEG] -> 'rarely' | 'never'
NP/NP ->
Comp -> 'that'

>>> cp = parse.load_parser('grammars/book_grammars/feat1.fcfg', trace=1)
>>> trees = cp.parse(tokens)
|.w.d.y.c.t.y.l.|
|[-] . . . . . .| [0:1] 'who'
|. [-] . . . . .| [1:2] 'do'
|. . [-] . . . .| [2:3] 'you'
|. . . [-] . . .| [3:4] 'claim'
|. . . . [-] . .| [4:5] 'that'
|. . . . . [-] .| [5:6] 'you'
|. . . . . . [-]| [6:7] 'like'
|# . . . . . . .| [0:0] NP[]/NP[] -> *
|. # . . . . . .| [1:1] NP[]/NP[] -> *
|. . # . . . . .| [2:2] NP[]/NP[] -> *
|. . . # . . . .| [3:3] NP[]/NP[] -> *
|. . . . # . . .| [4:4] NP[]/NP[] -> *
|. . . . . # . .| [5:5] NP[]/NP[] -> *
|. . . . . . # .| [6:6] NP[]/NP[] -> *
|. . . . . . . #| [7:7] NP[]/NP[] -> *
|[-] . . . . . .| [0:1] NP[+WH] -> 'who' *
|[-> . . . . . .| [0:1] S[-INV] -> NP[] * VP[] {}
|[-> . . . . . .| [0:1] S[-INV]/?x[] -> NP[] * VP[]/?x[] {}
|[-> . . . . . .| [0:1] S[-INV] -> NP[] * S[]/NP[] {}
|. [-] . . . . .| [1:2] V[+AUX] -> 'do' *
|. [-> . . . . .| [1:2] S[+INV] -> V[+AUX] * NP[] VP[] {}
|. [-> . . . . .| [1:2] S[+INV]/?x[] -> V[+AUX] * NP[] VP[]/?x[] {}
|. [-> . . . . .| [1:2] VP[] -> V[+AUX] * VP[] {}
|. [-> . . . . .| [1:2] VP[]/?x[] -> V[+AUX] * VP[]/?x[] {}
|. . [-] . . . .| [2:3] NP[-WH] -> 'you' *
|. . [-> . . . .| [2:3] S[-INV] -> NP[] * VP[] {}
|. . [-> . . . .| [2:3] S[-INV]/?x[] -> NP[] * VP[]/?x[] {}
|. . [-> . . . .| [2:3] S[-INV] -> NP[] * S[]/NP[] {}
|. [---> . . . .| [1:3] S[+INV] -> V[+AUX] NP[] * VP[] {}
|. [---> . . . .| [1:3] S[+INV]/?x[] -> V[+AUX] NP[] * VP[]/?x[] {}
|. . . [-] . . .| [3:4] V[-AUX, SUBCAT='clause'] -> 'claim' *
|. . . [-> . . .| [3:4] VP[] -> V[-AUX, SUBCAT='clause'] * SBar[] {}
|. . . [-> . . .| [3:4] VP[]/?x[] -> V[-AUX, SUBCAT='clause'] * SBar[]/?x[] {}
|. . . . [-] . .| [4:5] Comp[] -> 'that' *
|. . . . [-> . .| [4:5] SBar[] -> Comp[] * S[-INV] {}
|. . . . [-> . .| [4:5] SBar[]/?x[] -> Comp[] * S[-INV]/?x[] {}
|. . . . . [-] .| [5:6] NP[-WH] -> 'you' *
|. . . . . [-> .| [5:6] S[-INV] -> NP[] * VP[] {}
|. . . . . [-> .| [5:6] S[-INV]/?x[] -> NP[] * VP[]/?x[] {}
|. . . . . [-> .| [5:6] S[-INV] -> NP[] * S[]/NP[] {}
|. . . . . . [-]| [6:7] V[-AUX, SUBCAT='trans'] -> 'like' *
|. . . . . . [->| [6:7] VP[] -> V[-AUX, SUBCAT='trans'] * NP[] {}
|. . . . . . [->| [6:7] VP[]/?x[] -> V[-AUX, SUBCAT='trans'] * NP[]/?x[] {}
|. . . . . . [-]| [6:7] VP[]/NP[] -> V[-AUX, SUBCAT='trans'] NP[]/NP[] *
|. . . . . [---]| [5:7] S[-INV]/NP[] -> NP[] VP[]/NP[] *
|. . . . [-----]| [4:7] SBar[]/NP[] -> Comp[] S[-INV]/NP[] *
|. . . [-------]| [3:7] VP[]/NP[] -> V[-AUX, SUBCAT='clause'] SBar[]/NP[] *
|. . [---------]| [2:7] S[-INV]/NP[] -> NP[] VP[]/NP[] *
|. [-----------]| [1:7] S[+INV]/NP[] -> V[+AUX] NP[] VP[]/NP[] *
|[=============]| [0:7] S[-INV] -> NP[] S[]/NP[] *

>>> trees = list(trees)
>>> for tree in trees: print(tree)
(S[-INV]
  (NP[+WH] who)
  (S[+INV]/NP[]
    (V[+AUX] do)
    (NP[-WH] you)
    (VP[]/NP[]
      (V[-AUX, SUBCAT='clause'] claim)
      (SBar[]/NP[]
        (Comp[] that)
        (S[-INV]/NP[]
          (NP[-WH] you)
          (VP[]/NP[] (V[-AUX, SUBCAT='trans'] like) (NP[]/NP[] )))))))


#A different parser should give the same parse trees, but perhaps in a different order:

>>> cp2 = parse.load_parser('grammars/book_grammars/feat1.fcfg', trace=1, parser=parse.FeatureEarleyChartParser)
>>> trees2 = cp2.parse(tokens)
|.w.d.y.c.t.y.l.|
|[-] . . . . . .| [0:1] 'who'
|. [-] . . . . .| [1:2] 'do'
|. . [-] . . . .| [2:3] 'you'
|. . . [-] . . .| [3:4] 'claim'
|. . . . [-] . .| [4:5] 'that'
|. . . . . [-] .| [5:6] 'you'
|. . . . . . [-]| [6:7] 'like'
|> . . . . . . .| [0:0] S[-INV] -> * NP[] VP[] {}
|> . . . . . . .| [0:0] S[-INV]/?x[] -> * NP[] VP[]/?x[] {}
|> . . . . . . .| [0:0] S[-INV] -> * NP[] S[]/NP[] {}
|> . . . . . . .| [0:0] S[-INV] -> * Adv[+NEG] S[+INV] {}
|> . . . . . . .| [0:0] S[+INV] -> * V[+AUX] NP[] VP[] {}
|> . . . . . . .| [0:0] S[+INV]/?x[] -> * V[+AUX] NP[] VP[]/?x[] {}
|> . . . . . . .| [0:0] NP[+WH] -> * 'who' {}
|[-] . . . . . .| [0:1] NP[+WH] -> 'who' *
|[-> . . . . . .| [0:1] S[-INV] -> NP[] * VP[] {}
|[-> . . . . . .| [0:1] S[-INV]/?x[] -> NP[] * VP[]/?x[] {}
|[-> . . . . . .| [0:1] S[-INV] -> NP[] * S[]/NP[] {}
|. > . . . . . .| [1:1] S[-INV]/?x[] -> * NP[] VP[]/?x[] {}
|. > . . . . . .| [1:1] S[+INV]/?x[] -> * V[+AUX] NP[] VP[]/?x[] {}
|. > . . . . . .| [1:1] V[+AUX] -> * 'do' {}
|. > . . . . . .| [1:1] VP[]/?x[] -> * V[-AUX, SUBCAT='trans'] NP[]/?x[] {}
|. > . . . . . .| [1:1] VP[]/?x[] -> * V[-AUX, SUBCAT='clause'] SBar[]/?x[] {}
|. > . . . . . .| [1:1] VP[]/?x[] -> * V[+AUX] VP[]/?x[] {}
|. > . . . . . .| [1:1] VP[] -> * V[-AUX, SUBCAT='intrans'] {}
|. > . . . . . .| [1:1] VP[] -> * V[-AUX, SUBCAT='trans'] NP[] {}
|. > . . . . . .| [1:1] VP[] -> * V[-AUX, SUBCAT='clause'] SBar[] {}
|. > . . . . . .| [1:1] VP[] -> * V[+AUX] VP[] {}
|. [-] . . . . .| [1:2] V[+AUX] -> 'do' *
|. [-> . . . . .| [1:2] S[+INV]/?x[] -> V[+AUX] * NP[] VP[]/?x[] {}
|. [-> . . . . .| [1:2] VP[]/?x[] -> V[+AUX] * VP[]/?x[] {}
|. [-> . . . . .| [1:2] VP[] -> V[+AUX] * VP[] {}
|. . > . . . . .| [2:2] VP[] -> * V[-AUX, SUBCAT='intrans'] {}
|. . > . . . . .| [2:2] VP[] -> * V[-AUX, SUBCAT='trans'] NP[] {}
|. . > . . . . .| [2:2] VP[] -> * V[-AUX, SUBCAT='clause'] SBar[] {}
|. . > . . . . .| [2:2] VP[] -> * V[+AUX] VP[] {}
|. . > . . . . .| [2:2] VP[]/?x[] -> * V[-AUX, SUBCAT='trans'] NP[]/?x[] {}
|. . > . . . . .| [2:2] VP[]/?x[] -> * V[-AUX, SUBCAT='clause'] SBar[]/?x[] {}
|. . > . . . . .| [2:2] VP[]/?x[] -> * V[+AUX] VP[]/?x[] {}
|. . > . . . . .| [2:2] NP[-WH] -> * 'you' {}
|. . [-] . . . .| [2:3] NP[-WH] -> 'you' *
|. [---> . . . .| [1:3] S[+INV]/?x[] -> V[+AUX] NP[] * VP[]/?x[] {}
|. . . > . . . .| [3:3] VP[]/?x[] -> * V[-AUX, SUBCAT='trans'] NP[]/?x[] {}
|. . . > . . . .| [3:3] VP[]/?x[] -> * V[-AUX, SUBCAT='clause'] SBar[]/?x[] {}
|. . . > . . . .| [3:3] VP[]/?x[] -> * V[+AUX] VP[]/?x[] {}
|. . . > . . . .| [3:3] V[-AUX, SUBCAT='clause'] -> * 'claim' {}
|. . . [-] . . .| [3:4] V[-AUX, SUBCAT='clause'] -> 'claim' *
|. . . [-> . . .| [3:4] VP[]/?x[] -> V[-AUX, SUBCAT='clause'] * SBar[]/?x[] {}
|. . . . > . . .| [4:4] SBar[]/?x[] -> * Comp[] S[-INV]/?x[] {}
|. . . . > . . .| [4:4] Comp[] -> * 'that' {}
|. . . . [-] . .| [4:5] Comp[] -> 'that' *
|. . . . [-> . .| [4:5] SBar[]/?x[] -> Comp[] * S[-INV]/?x[] {}
|. . . . . > . .| [5:5] S[-INV]/?x[] -> * NP[] VP[]/?x[] {}
|. . . . . > . .| [5:5] NP[-WH] -> * 'you' {}
|. . . . . [-] .| [5:6] NP[-WH] -> 'you' *
|. . . . . [-> .| [5:6] S[-INV]/?x[] -> NP[] * VP[]/?x[] {}
|. . . . . . > .| [6:6] VP[]/?x[] -> * V[-AUX, SUBCAT='trans'] NP[]/?x[] {}
|. . . . . . > .| [6:6] VP[]/?x[] -> * V[-AUX, SUBCAT='clause'] SBar[]/?x[] {}
|. . . . . . > .| [6:6] VP[]/?x[] -> * V[+AUX] VP[]/?x[] {}
|. . . . . . > .| [6:6] V[-AUX, SUBCAT='trans'] -> * 'like' {}
|. . . . . . [-]| [6:7] V[-AUX, SUBCAT='trans'] -> 'like' *
|. . . . . . [->| [6:7] VP[]/?x[] -> V[-AUX, SUBCAT='trans'] * NP[]/?x[] {}
|. . . . . . . #| [7:7] NP[]/NP[] -> *
|. . . . . . [-]| [6:7] VP[]/NP[] -> V[-AUX, SUBCAT='trans'] NP[]/NP[] *
|. . . . . [---]| [5:7] S[-INV]/NP[] -> NP[] VP[]/NP[] *
|. . . . [-----]| [4:7] SBar[]/NP[] -> Comp[] S[-INV]/NP[] *
|. . . [-------]| [3:7] VP[]/NP[] -> V[-AUX, SUBCAT='clause'] SBar[]/NP[] *
|. [-----------]| [1:7] S[+INV]/NP[] -> V[+AUX] NP[] VP[]/NP[] *
|[=============]| [0:7] S[-INV] -> NP[] S[]/NP[] *

>>> sorted(trees) == sorted(trees2)
True


#load a German grammar:
>>> nltk.data.show_cfg('grammars/book_grammars/german.fcfg')
% start S
 # Grammar Productions
 S -> NP[CASE=nom, AGR=?a] VP[AGR=?a]
 NP[CASE=?c, AGR=?a] -> PRO[CASE=?c, AGR=?a]
 NP[CASE=?c, AGR=?a] -> Det[CASE=?c, AGR=?a] N[CASE=?c, AGR=?a]
 VP[AGR=?a] -> IV[AGR=?a]
 VP[AGR=?a] -> TV[OBJCASE=?c, AGR=?a] NP[CASE=?c]
 # Lexical Productions
 # Singular determiners
 # masc
 Det[CASE=nom, AGR=[GND=masc,PER=3,NUM=sg]] -> 'der'
 Det[CASE=dat, AGR=[GND=masc,PER=3,NUM=sg]] -> 'dem'
 Det[CASE=acc, AGR=[GND=masc,PER=3,NUM=sg]] -> 'den'
 # fem
 Det[CASE=nom, AGR=[GND=fem,PER=3,NUM=sg]] -> 'die'
 Det[CASE=dat, AGR=[GND=fem,PER=3,NUM=sg]] -> 'der'
 Det[CASE=acc, AGR=[GND=fem,PER=3,NUM=sg]] -> 'die'
 # Plural determiners
 Det[CASE=nom, AGR=[PER=3,NUM=pl]] -> 'die'
 Det[CASE=dat, AGR=[PER=3,NUM=pl]] -> 'den'
 Det[CASE=acc, AGR=[PER=3,NUM=pl]] -> 'die'
 # Nouns
 N[AGR=[GND=masc,PER=3,NUM=sg]] -> 'Hund'
 N[CASE=nom, AGR=[GND=masc,PER=3,NUM=pl]] -> 'Hunde'
 N[CASE=dat, AGR=[GND=masc,PER=3,NUM=pl]] -> 'Hunden'
 N[CASE=acc, AGR=[GND=masc,PER=3,NUM=pl]] -> 'Hunde'
 N[AGR=[GND=fem,PER=3,NUM=sg]] -> 'Katze'
 N[AGR=[GND=fem,PER=3,NUM=pl]] -> 'Katzen'
 # Pronouns
 PRO[CASE=nom, AGR=[PER=1,NUM=sg]] -> 'ich'
 PRO[CASE=acc, AGR=[PER=1,NUM=sg]] -> 'mich'
 PRO[CASE=dat, AGR=[PER=1,NUM=sg]] -> 'mir'
 PRO[CASE=nom, AGR=[PER=2,NUM=sg]] -> 'du'
 PRO[CASE=nom, AGR=[PER=3,NUM=sg]] -> 'er' | 'sie' | 'es'
 PRO[CASE=nom, AGR=[PER=1,NUM=pl]] -> 'wir'
 PRO[CASE=acc, AGR=[PER=1,NUM=pl]] -> 'uns'
 PRO[CASE=dat, AGR=[PER=1,NUM=pl]] -> 'uns'
 PRO[CASE=nom, AGR=[PER=2,NUM=pl]] -> 'ihr'
 PRO[CASE=nom, AGR=[PER=3,NUM=pl]] -> 'sie'
 # Verbs
 IV[AGR=[NUM=sg,PER=1]] -> 'komme'
 IV[AGR=[NUM=sg,PER=2]] -> 'kommst'
 IV[AGR=[NUM=sg,PER=3]] -> 'kommt'
 IV[AGR=[NUM=pl, PER=1]] -> 'kommen'
 IV[AGR=[NUM=pl, PER=2]] -> 'kommt'
 IV[AGR=[NUM=pl, PER=3]] -> 'kommen'
 TV[OBJCASE=acc, AGR=[NUM=sg,PER=1]] -> 'sehe' | 'mag'
 TV[OBJCASE=acc, AGR=[NUM=sg,PER=2]] -> 'siehst' | 'magst'
 TV[OBJCASE=acc, AGR=[NUM=sg,PER=3]] -> 'sieht' | 'mag'
 TV[OBJCASE=dat, AGR=[NUM=sg,PER=1]] -> 'folge' | 'helfe'
 TV[OBJCASE=dat, AGR=[NUM=sg,PER=2]] -> 'folgst' | 'hilfst'
 TV[OBJCASE=dat, AGR=[NUM=sg,PER=3]] -> 'folgt' | 'hilft'
 TV[OBJCASE=acc, AGR=[NUM=pl,PER=1]] -> 'sehen' | 'moegen'
 TV[OBJCASE=acc, AGR=[NUM=pl,PER=2]] -> 'sieht' | 'moegt'
 TV[OBJCASE=acc, AGR=[NUM=pl,PER=3]] -> 'sehen' | 'moegen'
 TV[OBJCASE=dat, AGR=[NUM=pl,PER=1]] -> 'folgen' | 'helfen'
 TV[OBJCASE=dat, AGR=[NUM=pl,PER=2]] -> 'folgt' | 'helft'
 TV[OBJCASE=dat, AGR=[NUM=pl,PER=3]] -> 'folgen' | 'helfen'
 
>>> cp = parse.load_parser('grammars/book_grammars/german.fcfg', trace=0)
>>> sent = 'die Katze sieht den Hund'
>>> tokens = sent.split()
>>> trees = cp.parse(tokens)
>>> for tree in trees: print(tree)
(S[]
  (NP[AGR=[GND='fem', NUM='sg', PER=3], CASE='nom']
    (Det[AGR=[GND='fem', NUM='sg', PER=3], CASE='nom'] die)
    (N[AGR=[GND='fem', NUM='sg', PER=3]] Katze))
  (VP[AGR=[NUM='sg', PER=3]]
    (TV[AGR=[NUM='sg', PER=3], OBJCASE='acc'] sieht)
    (NP[AGR=[GND='masc', NUM='sg', PER=3], CASE='acc']
      (Det[AGR=[GND='masc', NUM='sg', PER=3], CASE='acc'] den)
      (N[AGR=[GND='masc', NUM='sg', PER=3]] Hund))))



##Grammar with Binding Operators
#The bindop.fcfg grammar is a semantic grammar that uses lambda calculus. 
#Each element has a core semantics, which is a single lambda calculus expression - f(arg) 
#and a set of binding operators, which bind variables.

#In order to make the binding operators work right, 
#they need to instantiate their bound variable every time they are added to the chart. 
#To do this, we use a special subclass of Chart, called InstantiateVarsChart.

>>> from nltk.parse.featurechart import InstantiateVarsChart
>>> nltk.data.show_cfg('grammars/sample_grammars/bindop.fcfg')
#<> means Compostion of whole from Parts 
# f(x) means result from function application of f with arg x 
# \x.EXPR -> XXX means convert XXX to lambda EXPR(x) 
#{} = Set and { .. + ..} means set addition 
#SEM=[CORE=.., BO=...] is special syntax to apply binding operators on CORE 
# ie SEM=b2(b1(CORE)), would involve alpha-(change of var name), beta-conversion(substitute)
#Binding operator is of form - bo(lambda, @address_of_core)
%start S
S[SEM=[CORE=<?vp(?subj)>, BO={?b1+?b2}]] -> NP[SEM=[CORE=?subj, BO=?b1]] VP[SEM=[CORE=?vp, BO=?b2]]
VP[SEM=[CORE=<?v(?obj)>, BO={?b1+?b2}]] -> TV[SEM=[CORE=?v, BO=?b1]] N
P[SEM=[CORE=?obj, BO=?b2]]
VP[SEM=?s] -> IV[SEM=?s]
NP[SEM=[CORE=<@x>, BO={{<bo(?det(?n), @x)>}+?b1+?b2}]] -> Det[SEM=[CORE=?det, BO=?b1]] N[SEM=[CORE=?n, BO=?b2]]
# Lexical items:
Det[SEM=[CORE=<\Q P.exists x.(Q(x) & P(x))>, BO={/}]] -> 'a'
N[SEM=[CORE=<dog>, BO={/}]] -> 'dog' | 'cat' | 'mouse'
IV[SEM=[CORE=<\x.bark(x)>, BO={/}]] -> 'barks' | 'eats' | 'walks'
TV[SEM=[CORE=<\x y.feed(y,x)>, BO={/}]] -> 'feeds' | 'walks'
NP[SEM=[CORE=<@x>, BO={<bo(\P. P(John), @x)>}]] -> 'john' | 'alex'

>>> cp = parse.load_parser('grammars/sample_grammars/bindop.fcfg', trace=1,
                chart_class=InstantiateVarsChart)


#A simple intransitive sentence:

>>> from nltk.sem import logic
>>> logic._counter._value = 100

>>> trees = cp.parse('john barks'.split())
|. john.barks.|
|[-----]     .| [0:1] 'john'
|.     [-----]| [1:2] 'barks'
|[-----]     .| [0:1] NP[SEM=[BO={bo(\P.P(John),z101)}, CORE=<z101>]] -> 'john' *
|[----->     .| [0:1] S[SEM=[BO={?b1+?b2}, CORE=<?vp(?subj)>]] -> NP[SEM=[BO=?b1, CORE=?subj]] * VP[SEM=[BO=?b2, CORE=?vp]] {?b1: {bo(\P.P(John),z2)}, ?subj: <IndividualVariableExpression z2>}
|.     [-----]| [1:2] IV[SEM=[BO={/}, CORE=<\x.bark(x)>]] -> 'barks' *
|.     [-----]| [1:2] VP[SEM=[BO={/}, CORE=<\x.bark(x)>]] -> IV[SEM=[BO={/}, CORE=<\x.bark(x)>]] *
|[===========]| [0:2] S[SEM=[BO={bo(\P.P(John),z2)}, CORE=<bark(z2)>]] -> NP[SEM=[BO={bo(\P.P(John),z2)}, CORE=<z2>]] VP[SEM=[BO={/}, CORE=<\x.bark(x)>]] *
>>> for tree in trees: print(tree)
(S[SEM=[BO={bo(\P.P(John),z2)}, CORE=<bark(z2)>]]
  (NP[SEM=[BO={bo(\P.P(John),z101)}, CORE=<z101>]] john)
  (VP[SEM=[BO={/}, CORE=<\x.bark(x)>]]
    (IV[SEM=[BO={/}, CORE=<\x.bark(x)>]] barks)))


#A transitive sentence:

>>> trees = list(cp.parse('john feeds a dog'.split()))
|.joh.fee. a .dog.|
|[---]   .   .   .| [0:1] 'john'
|.   [---]   .   .| [1:2] 'feeds'
|.   .   [---]   .| [2:3] 'a'
|.   .   .   [---]| [3:4] 'dog'
|[---]   .   .   .| [0:1] NP[SEM=[BO={bo(\P.P(John),z102)}, CORE=<z102>]] -> 'john' *
|[--->   .   .   .| [0:1] S[SEM=[BO={?b1+?b2}, CORE=<?vp(?subj)>]] -> NP[SEM=[BO=?b1, CORE=?subj]] * VP[SEM=[BO=?b2, CORE=?vp]] {?b1: {bo(\P.P(John),z2)}, ?subj: <IndividualVariableExpression z2>}
|.   [---]   .   .| [1:2] TV[SEM=[BO={/}, CORE=<\x y.feed(y,x)>]] -> 'feeds' *
|.   [--->   .   .| [1:2] VP[SEM=[BO={?b1+?b2}, CORE=<?v(?obj)>]] -> TV[SEM=[BO=?b1, CORE=?v]] * NP[SEM=[BO=?b2, CORE=?obj]] {?b1: {/}, ?v: <LambdaExpression \x y.feed(y,x)>}
|.   .   [---]   .| [2:3] Det[SEM=[BO={/}, CORE=<\Q P.exists x.(Q(x) & P(x))>]] -> 'a' *
|.   .   [--->   .| [2:3] NP[SEM=[BO={?b1+?b2+{bo(?det(?n),@x)}}, CORE=<@x>]] -> Det[SEM=[BO=?b1, CORE=?det]] * N[SEM=[BO=?b2, CORE=?n]] {?b1: {/}, ?det: <LambdaExpression \Q P.exists x.(Q(x) & P(x))>}
|.   .   .   [---]| [3:4] N[SEM=[BO={/}, CORE=<dog>]] -> 'dog' *
|.   .   [-------]| [2:4] NP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z103)}, CORE=<z103>]] -> Det[SEM=[BO={/}, CORE=<\Q P.exists x.(Q(x) & P(x))>]] N[SEM=[BO={/}, CORE=<dog>]] *
|.   .   [------->| [2:4] S[SEM=[BO={?b1+?b2}, CORE=<?vp(?subj)>]] -> NP[SEM=[BO=?b1, CORE=?subj]] * VP[SEM=[BO=?b2, CORE=?vp]] {?b1: {bo(\P.exists x.(dog(x) & P(x)),z2)}, ?subj: <IndividualVariableExpression z2>}
|.   [-----------]| [1:4] VP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z2)}, CORE=<\y.feed(y,z2)>]] -> TV[SEM=[BO={/}, CORE=<\x y.feed(y,x)>]] NP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z2)}, CORE=<z2>]] *
|[===============]| [0:4] S[SEM=[BO={bo(\P.P(John),z2), bo(\P.exists x.(dog(x) & P(x)),z3)}, CORE=<feed(z2,z3)>]] -> NP[SEM=[BO={bo(\P.P(John),z2)}, CORE=<z2>]] VP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z3)}, CORE=<\y.feed(y,z3)>]] *

>>> print(trees[0])
(S[SEM=[BO={bo(\P.P(John),z2), bo(\P.exists x.(dog(x) & P(x)),z3)}, CORE=<feed(z2,z3)>]]
  (NP[SEM=[BO={bo(\P.P(John),z102)}, CORE=<z102>]] john)
  (VP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z2)}, CORE=<\y.feed(y,z2)>]]
    (TV[SEM=[BO={/}, CORE=<\x y.feed(y,x)>]] feeds)
    (NP[SEM=[BO={bo(\P.exists x.(dog(x) & P(x)),z103)}, CORE=<z103>]]
      (Det[SEM=[BO={/}, CORE=<\Q P.exists x.(Q(x) & P(x))>]] a)
      (N[SEM=[BO={/}, CORE=<dog>]] dog))))
      
      
      
## CooperStore help us in applying BO to core 
#CooperStore expects featurestruct with [CORE=.., STORE=..] with STORE as list 

from nltk.sem import cooper_storage as cs
>>> fx = trees[0].label()['SEM']
>> semrep = nltk.FeatStruct(fx)
>>> semrep['STORE'] = list(semrep['BO']); del semrep['BO']
>>> 
>>> cs_semrep = cs.CooperStore(semrep)
>>> print(cs_semrep.core)  #self.core = featstruct['CORE']
feed(z2,z3)
>>> for bo in cs_semrep.store:  # self.store = featstruct['STORE']
        print(bo)
bo(\P.exists x.(dog(x) & P(x)),z3)
bo(\P.P(John),z2)
 
#Finally we call s_retrieve() and check the readings.
>>> cs_semrep.s_retrieve(trace=True)
Permutation 1
   (\P.exists x.(dog(x) & P(x)))(\z3.feed(z2,z3))
   \P.P(John)(\z2.exists x.(dog(x) & feed(z2,x)))
Permutation 2
   \P.P(John)(\z2.feed(z2,z3))
   (\P.exists x.(dog(x) & P(x)))(\z3.feed(John,z3))
 
 
>>> for reading in cs_semrep.readings:  #'john feeds a dog'
        print(reading)
exists x.(dog(x) & feed(John,x))
exists x.(dog(x) & feed(John,x))     
      
      
      
      

 
 
 
 
 
 
 
 
 
 
 

###chap-10
###NLTK - Analyzing the Meaning of Sentences
 
#Example - Querying a Database
 
#If following NL question is asked 
Q.  Which country is Athens in? 
A.  Greece. 

 
#Given table , city_table: A table of cities, countries and populations
City        Country     Population
athens      greece      1368 
bangkok     thailand    1178 
barcelona   spain       1280 
berlin      east_germany 3481 
birmingham united_kingdom 1112 

#In SQL,  
SELECT Country FROM city_table WHERE City = 'athens' 

#In FCFG, convert natural languange(NL) question into QL query
#Each phrase structure rule is supplemented 
#with a recipe for constructing a value for the feature sem. 
#in each case, we use the string concatenation operation + to splice 
#the values for the child constituents to make a value for the parent constituent.

>>> nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')
#In one line, same variable(or LHS,RHS) means 'same meaning'
#in different line, same variable of earlier lines means different variables/meaning
% start S
S[SEM=(?np + WHERE + ?vp)] -> NP[SEM=?np] VP[SEM=?vp]     #1 
VP[SEM=(?v + ?pp)] -> IV[SEM=?v] PP[SEM=?pp]              #2 
VP[SEM=(?v + ?ap)] -> IV[SEM=?v] AP[SEM=?ap]              #3 
NP[SEM=(?det + ?n)] -> Det[SEM=?det] N[SEM=?n]            #4 
PP[SEM=(?p + ?np)] -> P[SEM=?p] NP[SEM=?np]               #5 
AP[SEM=?pp] -> A[SEM=?a] PP[SEM=?pp]                      #6 
NP[SEM='Country="greece"'] -> 'Greece'                    #7 
NP[SEM='Country="china"'] -> 'China'                      #8 
Det[SEM='SELECT'] -> 'Which' | 'What'                     #9 
N[SEM='City FROM city_table'] -> 'cities'                 #10
IV[SEM=''] -> 'are'                                       #11
A[SEM=''] -> 'located'                                    #12
P[SEM=''] -> 'in'                                         #13
#Check 
>>> cp = load_parser('grammars/book_grammars/sql0.fcfg',trace=1)
>>> cp.parse(query.split())
|.W.c.a.l.i.C.|
|[-] . . . . .| [0:1] 'What'
|. [-] . . . .| [1:2] 'cities'
|. . [-] . . .| [2:3] 'are'
|. . . [-] . .| [3:4] 'located'
|. . . . [-] .| [4:5] 'in'
|. . . . . [-]| [5:6] 'China'
|[-] . . . . .| [0:1] Det[SEM='SELECT'] -> 'What' *  #9, then check which RHS contains Det, ie 4
|[-> . . . . .| [0:1] NP[SEM=(?det+?n)] -> Det[SEM=?det] * N[SEM=?n] {?det: 'SELECT'} #4, ?det is determined , undetermined is N[SEM=?n], parse more 
|. [-] . . . .| [1:2] N[SEM='City FROM city_table'] -> 'cities' * #10, got 'cities', ?n = 'City FROM city_table'
|[---] . . . .| [0:2] NP[SEM=(SELECT, City FROM city_table)] -> Det[SEM='SELECT'] N[SEM='City FROM city_table'] * #Fully resolved now #4 , check which RHS got NP[..]
|[---> . . . .| [0:2] S[SEM=(?np+WHERE+?vp)] -> NP[SEM=?np] * VP[SEM=?vp] {?np: (SELECT, City FROM city_table)} #1, above line ?n is now ?np, hence ?np determined , determine ?vp 
|. . [-] . . .| [2:3] IV[SEM=''] -> 'are' * #11, process next work and proceed similarly to get value of ?vp of above line 
|. . [-> . . .| [2:3] VP[SEM=(?v+?pp)] -> IV[SEM=?v] * PP[SEM=?pp] {?v: ''} #2, two VP rules, check each rule seperately
|. . [-> . . .| [2:3] VP[SEM=(?v+?ap)] -> IV[SEM=?v] * AP[SEM=?ap] {?v: ''} #3
|. . . [-] . .| [3:4] A[SEM=''] -> 'located' *
|. . . [-> . .| [3:4] AP[SEM=?pp] -> A[SEM=?a] * PP[SEM=?pp] {?a: ''}
|. . . . [-] .| [4:5] P[SEM=''] -> 'in' *
|. . . . [-> .| [4:5] PP[SEM=(?p+?np)] -> P[SEM=?p] * NP[SEM=?np] {?p: ''}
|. . . . . [-]| [5:6] NP[SEM='Country="china"'] -> 'China' *
|. . . . . [->| [5:6] S[SEM=(?np+WHERE+?vp)] -> NP[SEM=?np] * VP[SEM=?vp] {?np: 'Country="china"'}
|. . . . [---]| [4:6] PP[SEM=(, Country="china")] -> P[SEM=''] NP[SEM='Country="china"'] *
|. . . [-----]| [3:6] AP[SEM=(, Country="china")] -> A[SEM=''] PP[SEM=(, Country="china")] *
|. . [-------]| [2:6] VP[SEM=(, , Country="china")] -> IV[SEM=''] AP[SEM=(, Country="china")] *
|[===========]| [0:6] S[SEM=(SELECT, City FROM city_table, WHERE, , ,Country="china")] -> NP[SEM=(SELECT, City FROM city_table)] VP[SEM=(,, Country="china")] *
 
#Full code 
>>> from nltk import load_parser
>>> cp = load_parser('grammars/book_grammars/sql0.fcfg',trace=1)
>>> query = 'What cities are located in China'
>>> trees = list(cp.parse(query.split()))
>>> print(trees[0])
(S[SEM=(SELECT, City FROM city_table, WHERE, , , Country="china")]  #(?np + WHERE + ?vp) = (?det + ?n) + WHERE + (?v +( (?p + ?np) ))
  (NP[SEM=(SELECT, City FROM city_table)]    #?np = ?det + ?n 
    (Det[SEM='SELECT'] What)                 #?det = SELECT 
    (N[SEM='City FROM city_table'] cities))  #?n = City FROM city_table'
  (VP[SEM=(, , Country="china")]   #?vp = ?v + ?pp or ?v + ?ap, final tree is ?v + ?ap
    (IV[SEM=''] are)               #?v = '' 
    (AP[SEM=(, Country="china")]   #?ap = ?pp = ?pp, Note ?a is discarded
      (A[SEM=''] located)          #?a = ''
      (PP[SEM=(, Country="china")] #?pp = ?p + ?np 
        (P[SEM=''] in)             #?p = ''
        (NP[SEM='Country="china"'] China)))))  #?np = Country="china"'
>>> answer = trees[0].label()['SEM']
>>> answer = [s for s in answer if s]
>>> q = ' '.join(answer)
>>> print(q)
SELECT City FROM city_table WHERE Country="china"
#get answers 
>>> from nltk.sem import chat80
>>> rows = chat80.sql_query('corpora/city_database/city.db', q)
>>> for r in rows: print(r[0], end=" ") 
canton chungking dairen harbin kowloon mukden peking shanghai sian tientsin
 
#Chat-80 was a natural language system which allowed the user 
#to interrogate a Prolog knowledge base in the domain of world geography
#Chat-80 relations are like tables in a relational database
#http://www.nltk.org/howto/chat80.html

>>> from nltk.sem import chat80
>>> print(chat80.items) 
('borders', 'circle_of_lat', 'circle_of_long', 'city', ...)



##Propositional Logic

#A logical language is designed to make reasoning formally explicit.
#For example 
 
[Klaus chased Evi] and [Evi ran away]. 

#can be translated into 
f & ?

#whereas 
f = Klaus chased Evi
? = Evi ran away

>>> nltk.boolean_ops()
negation            -
conjunction         &
disjunction         |
implication         ->
equivalence         <->
 

#Boolean Operator                           Truth Conditions
negation (it is not the case that ...)      -f is true in s iff f is false in s 
conjunction (and)                           (f & ?) is true in s iff f is true in s and ? is true in s 
disjunction (or)                            (f | ?) is true in s iff f is true in s or ? is true in s 
implication (if ..., then ...)              (f -> ?) is true in s iff f is false in s or ? is true in s 
equivalence (if and only if)                (f <-> ?) is true in s iff f and ? are both true in s or both false in s 

#For Example 
#A formula of the form (P -> Q) is only false when P is true and Q is false. 
#If P is false  and Q is true  then P-> Q will come out true
# p -> Q is same as -P|Q 


>>> read_expr = nltk.sem.Expression.fromstring
>>> read_expr('-(P & Q)')
<NegatedExpression -(P & Q)>
>>> read_expr('P & Q')
<AndExpression (P & Q)>
>>> read_expr('P | (R -> Q)')
<OrExpression (P | (R -> Q))>
>>> read_expr('P <-> -- P')
<IffExpression (P <-> --P)>
 
 
##[A1, ..., An] / C represents the argument 
#that conclusion C follows from assumptions [A1, ...,An]. 

#Example 
Sylvania is to the north of Freedonia (FnS)
Therefore, Freedonia is not to the north of Sylvania (-SnF)
 
#Lets 
FnS - Freedonia is to the north of Sylvania
SnF - Sylvania is to the north of Freedonia

#Then example can be written as 
SnF -> -FnS 

#Then the argument can be written as 
[SnF, SnF -> -FnS] / -FnS 


#Arguments can be tested for "syntactic validity" by using a proof system. 
#Logical proofs can be carried out with NLTK's inference module, 
#for example via an interface to the third-party theorem prover Prover9. 
#download from and install to c:\nltk_data, http://www.cs.unm.edu/~mccune/mace4/gui/v05.html
#The inputs to the inference mechanism first have to be converted into logical expressions.



>>> read_expr = nltk.sem.Expression.fromstring
>>> SnF = read_expr('SnF')
>>> NotFnS = read_expr('-FnS')
>>> R = read_expr('SnF -> -FnS')
>>> prover = nltk.Prover9()
>>> prover.config_prover9(r'c:/nltk_data/prover9/bin-win32')
>>> prover.prove(NotFnS, [SnF, R])  #(goal, assumptions)
True
 
 

##A Valuation is a mapping from basic expressions of the logic to their values. 
>>> val = nltk.Valuation([('P', True), ('Q', True), ('R', False)]) #Bases: dict
>>> val['P']
True

##Assignment 
#An assigment can only assign values from its domain
#A dictionary which represents an assignment of values to variables.

#A variable Assignment is a mapping from individual variables to entities in the domain. 
#Individual variables are usually indicated with the letters 'x', 'y', 'w' and 'z', optionally followed by an integer (e.g., 'x0', 'y332').
>>> from nltk.sem.evaluate import Assignment
>>> dom = set(['u1', 'u2', 'u3', 'u4']) #set of entities/values
>>> g3 = Assignment(dom, [('x', 'u1'), ('y', 'u2')]) #(domain, assign=None),
                                                     #domain (set) – the domain of discourse
                                                     #assign (list) – a list of (varname, value) associations

>>> g3 == {'x': 'u1', 'y': 'u2'}
True
>>> print(g3)
g[u1/x][u2/y]

#to update an assignment using the add method:
>>> dom = set(['u1', 'u2', 'u3', 'u4'])
>>> g4 = Assignment(dom)
>>> g4.add('x', 'u1')
{'x': 'u1'}

#With no arguments, purge() is equivalent to clear() on a dictionary:
>>> g4.purge()
>>> g4
{}



#example 
>>> val = nltk.Valuation([('P', True), ('Q', True), ('R', False)]) #Bases: dict

>>> dom = set()
>>> g = nltk.Assignment(dom) #(domain, assign=None) Bases=dict 

#A first order model is a domain D of discourse(ie Universal set) and a valuation V.
#A domain D is a set of entities, and a valuation V is a map 
#that associates expressions with values in the model

#If an unknown expression a is passed to a model M‘s interpretation function i, 
#i will first check whether M‘s valuation assigns an interpretation to a as a constant, 
#and if this fails, i will delegate the interpretation of a to g(ie Assignment)
#g only assigns values to individual variables

>>> m = nltk.Model(dom, val) #(domain, valuation), 
 
>>> print(m.evaluate('(P & Q)', g) )  #evaluate(expr, g, trace=None)
True
>>> print(m.evaluate('-(P & Q)', g))
False
>>> print(m.evaluate('(P & R)', g))
False
>>> print(m.evaluate('(P | R)', g))
True
 
 
##First-Order Logic - also known as first-order predicate calculus and predicate logic

#First-order logic quantifies variables that can have single value
#second-order logic, in addition, also quantifies over sets; 
#third-order logic also quantifies over sets of sets, and so on. 

#Higher-order logic is the union of first-, second-, third-, …, nth-order logic; 
#i.e., higher-order logic admits quantification over sets that are nested arbitrarily deeply.


#First-order logic keeps all the boolean operators of Propositional Logic. 
#Also introduce predicates(function taking entity from domain/unversal set returning true/false)
#predicates take differing numbers of arguments. 

#For example, 'Angus walks' might be formalized as 'walk(angus)' 
#'Angus sees Bertie' as 'see(angus, bertie'). 
#Note how predicate is written, walk is something(fn) with variable, hence walk(x), where x = angus 
#generally, verb, 2nd Noun(object) etc can be predicate and 1st Noun(subject) is value 
#if subject is Pronoun, then it is walk(x) and x is unbound 


#ie  walk a unary predicate, and see a binary predicate. 
#The symbols used as predicates do not have intrinsic meaning, 

#Whether an atomic predication like see(angus, bertie) is true or false 
#in a situation is not a matter of logic, 
#but depends on the particular valuation that we have chosen for the constants see, angus and bertie. 
#For this reason, such expressions are called non-logical constants. 


#By contrast, logical constants (such as the boolean operators) 
#always receive the same interpretation in every model for first-order logic.

#one binary predicate has special status, namely equality
#Equality is regarded as a logical constant, 
#since for individual terms t1 and t2, the formula t1 = t2 is true if and only if t1 and t2 refer to one and the same entity.
 
#Following the tradition of Montague grammar, 
#two basic types are used: 
#e is the type of entities(type of values in domain/universal set ) 
#while t is the type of formulas, i.e., expressions which have truth values. 

#given any types s and t, <s, t> is a complex type 
#corresponding to functions from 's things' to 't things'. 
#unary predicates : <e, t> (read f(e) returning t )is the type of expressions from entities to truth values
#Binary predicates:  <e, <e, t>> (read f(e)(e) returning t) from entities to unary predicate

>>> read_expr = nltk.sem.Expression.fromstring
>>> expr = read_expr('walk(angus)', type_check=True)
>>> expr.argument
<ConstantExpression angus>
>>> expr.argument.type
e
>>> expr.function
<ConstantExpression walk>
>>> expr.function.type
<e,?>
 
#<e,?> : Although the type-checker will try to infer as many types as possible, 
#in this case it has not managed to fully specify the type of walk, 
#since its result type is unknown. 

#Although we are intending walk to receive type <e, t>, 
#as far as the type-checker knows, in this context it could be of some other type such as <e, e> or <e, <e, t>. 

#To help the type-checker, we need to specify a signature, 
#implemented as a dictionary that explicitly associates types with non-logical constants:

>>> sig = {'walk': '<e, t>'}
>>> expr = read_expr('walk(angus)', signature=sig)
>>> expr.function.type
e
 
 
#In NLTK, variables of type e are all lowercase
#In first-order logic, arguments of predicates can also be individual variables such as x, y and z

#Individual variables are similar to personal pronouns like he, she and it


##open formula - no binding(ie uses only Pronoun subject) - Binding means using Noun(subject) 
#eg  with two occurrences of the variable x. (x is not bound ie x is mentioned as Pronoun/without any exact Noun)
#AND is conjuction

a. He is a dog and he disappeared. 
#Note how predicate is reversed  as here He is variable 
b. (Open formula of a) dog(x) AND disappear(x) 

##Existential quantifier EXISTSx ('for some x'/'there exists'/atleast one/Some )
##By placing an EXISTSx in front of (b), 
#we can bind these variables by quantifiers(ie x to some Noun)
 
#all are equivalent 
a.  EXISTSx.(dog(x) AND disappear(x)) 
b.  At least one entity is a dog and disappeared. 
c.  A dog disappeared. 
#NLTK equivalent 
exists x.(dog(x) & disappear(x)) 


##universal quantifier ALLx ('for all x'/every including zero)

#all are equivalent 
a.  ALLx.(dog(x) -> disappear(x))  
b.  Everything has the property that if it is a dog, it disappears. 
    if some x is a dog, then x disappears 
    but it doesn't say that there are any dogs. 
    So in a situation where there are no dogs, it will still come out true. 
    (Remember that (P -> Q) is true when P is false.)
c.  Every(including zero dog) dog disappeared. 
#NLTK
all x.(dog(x) -> disappear(x)) 

##Note, in general , EXISTSx is written with AND, conjuction. &
#ALLx is written with  if ... then .. ( ie ->)

#Some notation , Note how predicate is reversed 
"Some x are P" is EXISTSx(P(x))  #in NLTK some x.P(x) or exists x.P(x)
"Not all x are P" is EXISTSx(~P(x)), or equivalently, ~(ALLx P(x))
If P(x) is never true, EXISTSx(P(x)) is false but EXISTSx(~P(x)) is true.







##Open vs Closed formula 
#Example 
((exists x. dog(x)) -> bark(x))

#The scope of the exists x quantifier is dog(x), 
#so the occurrence of x in bark(x) is unbound. 
#Consequently it can become bound by some other quantifier
all x.((exists x. dog(x)) -> bark(x))

#an occurrence of a variable x in a formula f is free in f 
#if that occurrence doesn't fall within the scope of all x or some x in f. 

#Conversely, if x is free in formula f, then it is bound in 'all x.f and exists x.f.' 
#If all variable occurrences in a formula are bound, the formula is said to be closed.

#class Expression has a method free() which returns the set of variables that are free in expr.

>>> read_expr = nltk.sem.Expression.fromstring
>>> read_expr('dog(cyril)').free()
set()
>>> read_expr('dog(x)').free()
{Variable('x')}
>>> read_expr('own(angus, cyril)').free()
set()
>>> read_expr('exists x.dog(x)').free()
set()
>>> read_expr('((some x. walk(x)) -> sing(x))').free()
{Variable('x')}
>>> read_expr('exists x.own(y, x)').free()
{Variable('y')}
 
 

 
 
 
##First Order Theorem Proving

#Example 
a. if x is to the north of y then y is not to the north of x. 
b. all x. all y.(north_of(x, y) -> -north_of(y, x))

#The general case in theorem proving is to determine 
#whether a formula that we want to prove (a proof goal) 
#can be derived by a finite sequence of inference steps 
#from a list of assumed formulas. 

#We write this as S PROV g, 
#where S is a (possibly empty) list of assumptions, and g is a proof goal

>>> read_expr = nltk.sem.Expression.fromstring
>>> NotFnS = read_expr('-north_of(f, s)')  
>>> SnF = read_expr('north_of(s, f)')    
>>> R = read_expr('all x. all y. (north_of(x, y) -> -north_of(y, x))')  
>>> prover = nltk.Prover9()   
>>> prover.config_prover9(r'c:/nltk_data/prover9/bin-win32')
>>> prover.prove(NotFnS, [SnF, R])  #goal, assumptions
True
 
 
#but below is false 
#all x. all y.(north_of(x, y) -> north_of(y, x))
>>> FnS = read_expr('north_of(f, s)')
>>> prover.prove(FnS, [SnF, R])
False
 
 
#Summary of new logical relations and operators required for First Order Logic, together with two useful methods of the Expression class.
#Example        Description
=               equality 
!=              inequality 
exists          existential quantifier 
all             universal quantifier 
e.free()        show free variables of e 
e.simplify()    carry out ß-reduction on e 





##Truth in Model
#The general process of determining truth or falsity of a formula in a model is called model checking.


#Given a first-order logic language L, 
#a model M for L is a pair <D, Val>, 
#where D is an nonempty set called the domain of the model/universal set 
#and Val is valuation function which assigns values from D to expressions of L as follows:
1.For every individual constant c in L, Val(c) is an element of D.
2.For every predicate symbol P of arity n , Val(P) is a function from Dn to {True, False}. 
#Dn is a tuple of n size , each element is taken from D 
(If the arity of P is 0, then Val(P) is simply a truth value, the P is regarded as a propositional symbol.)
(if P is of arity 2, then Val(P) will be a function f from pairs of elements of D to {True, False})

#Relations are represented semantically in NLTK : as sets of tuples. 
#NLTK: for arity 2, Val(P) is a set S of pairs, defined as follows
#Such an f is called the characteristic function of S 
S = {s | f(s) = True} , s is pairs of D 



#For example, A domain of discourse(universal set) consisting of the individuals Bertie, Olive and Cyril, 
#Bertie is a boy, Olive is a girl and Cyril is a dog. 
#olive walks , cyril walks 
#bertie sees olive, cyril sees bertie , olive sees cyril

 
#Use Valuation.fromstring() 
#to convert a list of strings of the form symbol => value into a Valuation object.

#Example - the value of 'see' is a set of tuples such that Bertie sees Olive, Cyril sees Bertie, and Olive sees Cyril.
#variables are b,o,c
#unary predicates (i.e, boy, girl, dog, walk) is sets of singleton tuples, 
#binary predicate is see
>>> v = """
    bertie => b
    olive => o
    cyril => c
    boy => {b}
    girl => {o}
    dog => {c}
    walk => {o, c}
    see => {(b, o), (c, b), (o, c)} 
    """
>>> val = nltk.Valuation.fromstring(v)
>>> print(val)
{'bertie': 'b',
 'boy': {('b',)},
 'cyril': 'c',
 'dog': {('c',)},
 'girl': {('o',)},
 'olive': 'o',
 'see': {('o', 'c'), ('c', 'b'), ('b', 'o')},
 'walk': {('c',), ('o',)}}
 
 


#A predication of the form P(t1, ... tn), where P is of arity n, comes out true 
#when tuple of values corresponding to (t1, ... tn) belongs to the set of tuples in the value of P.
>>> ('o', 'c') in val['see']
True
>>> ('b',) in val['boy']
True
 
 


##Individual Variables and Assignments
#variable assignment - a mapping from individual variables to entities in the domain. 
#Assignments are created using the Assignment constructor, 
#which also takes the model's domain of discourse as a parameter. 

#We are not required to actually enter any bindings, 
#but if we do, they are in a (variable, value) format 

>>> dom = {'b', 'o', 'c'}
>>> g = nltk.Assignment(dom, [('x', 'o'), ('y', 'c')]) #(dom, assgin)
>>> g
{'y': 'c', 'x': 'o'}
 
>>> print(g)  #similar to Logic textbook syntax 
g[c/y][o/x]
 
 

##evaluate an atomic formula of first-order logic. 
#First, we create a model, then we call the evaluate() method to compute the truth value.
>>> m = nltk.Model(dom, val)
>>> m.evaluate('see(olive, y)', g)  #see(olive, cyril), y comes from Assigment, g
True

>>> g['y']
'c'
 
>>> m.evaluate('see(y, x)', g) #x,y comes from Assigment, g, see(cyril, olive)
False
 
#Complex , Note a Predicate is true if it is part of Valuation 
>>> m.evaluate('see(bertie, olive) & boy(bertie) & -walk(bertie)', g)
True


#The method purge() clears all bindings from an assignment.
>>> g.purge()
>>> g
{}
 
>>> m.evaluate('see(olive, y)', g)
'Undefined'
 
 


 
 
##Quantification

#When is below true?
exists x.(girl(x) & walk(x)) 


#we want to know if there is some u in dom such that g[u/x] satisfies the open formula 
girl(x) & walk(x) 


>>> m.evaluate('exists x.(girl(x) & walk(x))', g)
True
 
 
#Infact x is o 
>>> m.evaluate('girl(x) & walk(x)', g.add('x', 'o'))
True
 
 

#satisfiers() method -returns a set of all the individuals that satisfy an open formula. 
>>> read_expr = nltk.sem.Expression.fromstring
>>> fmla1 = read_expr('girl(x) | boy(x)')
>>> m.satisfiers(fmla1, 'x', g)
{'b', 'o'}
>>> fmla2 = read_expr('girl(x) -> walk(x)')
>>> m.satisfiers(fmla2, 'x', g)
{'c', 'b', 'o'}
>>> fmla3 = read_expr('walk(x) -> girl(x)')
>>> m.satisfiers(fmla3, 'x', g)
{'b', 'o'}
 
#The truth conditions for -> mean that fmla2 is equivalent to -girl(x) | walk(x)
#which is satisfied by something which either isn't a girl or walks. 

# a universally quantified formula ALLx.f is true with respect to g 
#in case for every u, f is true with respect to g[u/x]

#Since neither b (Bertie) nor c (Cyril) are girls, they both satisfy the whole formula. 
#And  o satisfies the formula because o satisfies both disjuncts. 
#Now, since every member of the domain of discourse satisfies fmla2, 
>>> m.evaluate('all x.(girl(x) -> walk(x))', g)
True
 


##Quantifier Scope Ambiguity

Everybody admires someone. 

#There are (at least) two ways of expressing above  in first-order logic:
a.  all x.(person(x) -> exists y.(person(y) & admire(x,y))) 
b.  exists y.(person(y) & all x.(person(x) -> admire(x,y))) 

#(b) is logically stronger than (a): 
#it claims that there is a unique person, say Bruce, who is admired by everyone. 
#(a), requires that for every person u, we can find some person u' whom u admires; 
#but this could be a different person u' in each case. 

#We distinguish between (a) and (b) in terms of the scope of the quantifiers. 
#In the first, ALL has wider scope than EXISTS, 
#while in (b), the scope ordering is reversed. 

#So now we have two ways of representing the meaning of 'Everybody admires someone'
#they are both quite legitimate. 
#In other words, it is ambiguous with respect to quantifier scope

>>> v2 = """
    bruce => b
    elspeth => e
    julia => j
    matthew => m
    person => {b, e, j, m}
    admire => {(j, b), (b, b), (m, e), (e, m)}
    """
>>> val2 = nltk.Valuation.fromstring(v2)
 
#a.  all x.(person(x) -> exists y.(person(y) & admire(x,y))) -- True 
#b.  exists y.(person(y) & all x.(person(x) -> admire(x,y))) -- False 

>>> dom2 = val2.domain
>>> m2 = nltk.Model(dom2, val2)
>>> g2 = nltk.Assignment(dom2)
>>> read_expr = nltk.sem.Expression.fromstring
>>> fmla4 = read_expr('(person(x) -> exists y.(person(y) & admire(x, y)))')
>>> m2.satisfiers(fmla4, 'x', g2)
{'e', 'b', 'm', 'j'}
 
>>> fmla5 = read_expr('(person(y) & all x.(person(x) -> admire(x, y)))')
>>> m2.satisfiers(fmla5, 'y', g2)
set()
#That is, there is no person that is admired by everybody. 

#Taking a different open formula, fmla6, 
#we can verify that there is a person, namely Bruce, who is admired by both Julia and Bruce.

>>> fmla6 = read_expr('(person(y) & all x.((x = bruce | x = julia) -> admire(x, y)))')
>>> m2.satisfiers(fmla6, 'y', g2)
{'b'}
 
 
##Model Building
#model building tries to create a new model, given some set of sentences. 
#If it succeeds, then we know that the set is consistent, 
#since we have an existence proof of the model.

#One option is to treat our candidate set of sentences as assumptions, 
#while leaving the goal unspecified. 

#The following interaction shows how both [a, c1] and [a, c2] are consistent lists, 
#since Mace succeeds in building a model for each of them, while [c1, c2] is inconsistent

>>> read_expr = nltk.sem.Expression.fromstring
>>> a3 = read_expr('exists x.(man(x) & walks(x))')
>>> c1 = read_expr('mortal(socrates)')
>>> c2 = read_expr('-mortal(socrates)')
>>> mb = nltk.Mace(5)  #end_size=5

##below configuration does not work 
>>> mb.config_prover9(r'c:/nltk_data/prover9/bin-win32')
#hack , intsllation dir of prover9 
>>> mb._mace4_bin = r"c:/nltk_data/prover9/bin-win32/mace4.exe"

>>> print(mb.build_model(None, [a3, c1])) #goal, assumptions
True
>>> print(mb.build_model(None, [a3, c2]))
True
>>> print(mb.build_model(None, [c1, c2]))
False
 
 

#We can also use the model builder as an adjunct(supplementary) to the theorem prover. 

# to prove S PROV g, 
#i.e. that g is logically derivable from assumptions S = [s1, s2, ..., sn]. 

#We can feed this same input to Mace4, 
#and the model builder will try to find a counterexample, 
#that is, to show that g does not follow from S. 

#So, given this input, Mace4 will try to find a model 
#for the set S together with the negation of g, S' =[s1, s2, ..., sn, -g]. 
#If g fails to follow from S, 
#then Mace4 may well return with a counterexample faster than Prover9 concludes that it cannot find the required proof. 

#Conversely, if g is provable from S, Mace4 may take a long time unsuccessfully trying to find a countermodel, 
#and will eventually give up.

#Example - 
#Our assumptions are the list 
#[There is a woman that every man loves, Adam is a man, Eve is a woman]. 
#Our conclusion is Adam loves Eve. 

#Can Mace4 find a model in which the premises are true but the conclusion is false? 

#use MaceCommand() which will let us inspect the model that has been built.
class nltk.inference.mace.MaceCommand(goal=None, assumptions=None, max_models=500, model_builder=None)

>>> read_expr = nltk.sem.Expression.fromstring
>>> a4 = read_expr('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
>>> a5 = read_expr('man(adam)')
>>> a6 = read_expr('woman(eve)')
>>> g = read_expr('love(adam,eve)')
>>> mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6])
>>> mc._modelbuilder._mace4_bin = r"c:/nltk_data/prover9/bin-win32/mace4.exe"
>>> mc._interpformat_bin = r"c:/nltk_data/prover9/bin-win32/interpformat.exe"
>>> mc.build_model()
True
>>> print(mc.valuation)
{'C1': 'b',
 'adam': 'a',
 'eve': 'a',
 'love': {('a', 'b')},
 'man': {('a',)},
 'woman': {('a',), ('b',)}}
 
>>> mc.print_assumptions()
exists y.(woman(y) & all x.(man(x) -> love(x,y)))
man(adam)
woman(eve)

#C1 - "skolem constant" 
#that the model builder introduces as a representative of the existential quantifier. 

#That is, when the model builder encountered the exists y part of a4 above, 
#it knew that there is some individual b in the domain which satisfies the open formula in the body of a4. 

#However, it doesn't know whether b is also the denotation of an individual constant anywhere else in its input, 
#so it makes up a new name for b on the fly, namely C1. 

#Note we didn't specify that man and woman denote disjoint sets, 
#so the model builder lets their denotations overlap. 

#So let's add a new assumption which makes the sets of men and women disjoint. 

>>> a7 = read_expr('all x. (man(x) -> -woman(x))')
>>> g = read_expr('love(adam,eve)')
>>> mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6, a7])
>>> mc._modelbuilder._mace4_bin = r"c:/nltk_data/prover9/bin-win32/mace4.exe"
>>> mc._interpformat_bin = r"c:/nltk_data/prover9/bin-win32/interpformat.exe"
>>> mc.build_model()
True
>>> print(mc.valuation)
{'C1': 'c',
 'adam': 'a',
 'eve': 'b',
 'love': {('a', 'c')},
 'man': {('a',)},
 'woman': {('c',), ('b',)}}
#there is nothing in our premises which says that Eve is the only woman in the domain of discourse
#to ensure that there is only one woman in the model, add 
exists y. all x. (woman(x) -> (x = y))


 
##Compositional Semantics in Feature-Based Grammar

#Principle of Compositionality
1.The meaning of a whole is a function of the meanings of the parts 
  and of the way they are syntactically combined., Hence use function application 

##first approximation to the kind of analyses we would like to build  

#the sem value at the root node shows a semantic representation for the whole sentence, 
#while the sem values at lower nodes show semantic representations for constituents of the sentence. 

#Since the values of sem have to be treated in special manner(ie Whole and  parts)
#they are distinguished from other feature values by being enclosed in angle brackets.

S[SEM=<bark(cyril)>]
    NP[SEM=<cyril>]
        cyril
    VP[SEM=<bark>]
        IV[SEM=<\x.bark(x)>]   #Denotes 'bark' is Lambda function 
            barks

#To build above , use function application  
#suppose we have a NP and VP constituents with appropriate values for their sem nodes. 
#Then the sem value of an S is handled by a rule like below. 
#(Observe that in the case where the value of sem is a variable, we omit the angle brackets.)
S[SEM=<?vp(?np)>] -> NP[SEM=?np] VP[SEM=?vp] 
#means that given some sem value ?np for the subject NP and some sem value ?vp for the VP, 
#the sem value of the S parent is constructed by applying ?vp as a function expression to ?np.
#ie, ?vp has to denote a function which has the denotation of ?np in its domain

#To complete the grammar 
VP[SEM=?v] -> IV[SEM=?v]
NP[SEM=<cyril>] -> 'Cyril'
IV[SEM=<\x.bark(x)>] -> 'barks'
#The VP rule says that the parent's semantics is the same as the head child's semantics. 
#The two lexical rules provide non-logical constants to serve as the semantic values of Cyril and barks respectively. 
#last rule also contains lambda-Calculus


##The lambda Calculus - \x. EXPR can be treated as function(x) ie taking arg name = x and returning EXPR

#the set of all w such that w is an element of V (the vocabulary) and w has property P".
{w | w BELONGSTO V & P(w)} 
#OR using lambda operator 
LAMBDAw. (V(w) & P(w)) 


#lambda is a binding operator, just as the first-order logic quantifiers are. 

#For open formula (a), we can bind with  lambda-operator 
a.  (walk(x) AND chew_gum(x)) 
b.  LAMBDAx.(walk(x) AND chew_gum(x)) 
#NLTK
c.  \x.(walk(x) & chew_gum(x)) 
#Means 
#"be an x such that x walks and x chews gum" or "have the property of walking and chewing gum".

>>> read_expr = nltk.sem.Expression.fromstring
>>> expr = read_expr(r'\x.(walk(x) & chew_gum(x))')
>>> expr
<LambdaExpression \x.(walk(x) & chew_gum(x))>
>>> expr.free()
set()
>>> print(read_expr(r'\x.(walk(x) & chew_gum(y))'))
\x.(walk(x) & chew_gum(y))
 
 
 
#LAMBDA-operator is useful for below semantics 
a.  To walk and chew-gum is hard 
b.  hard(\x.(walk(x) & chew_gum(x))) 

#if f is an open formula, then the abstract LAMBDAx.f can be used as a unary predicate 
\x.(walk(x) & chew_gum(x)) (gerald) 
#says that Gerald has the property of walking and chewing gum, 
#which has the same meaning as 
(walk(gerald) & chew_gum(gerald)) 

#OR 
#use a[ß/x] as notation for the operation of replacing 
#all free occurrences of x in a by the expression ß(called ß-reduction)
(walk(x) & chew_gum(x))[gerald/x]  #ie value of x is gerald
#LAMBDAx. a(ß) has the same semantic values as a[ß/x]

#in NLTK, we can call the simplify() method 
>>> expr = read_expr(r'\x.(walk(x) & chew_gum(x))(gerald)')
>>> print(expr)
\x.(walk(x) & chew_gum(x))(gerald)
>>> print(expr.simplify()) 
(walk(gerald) & chew_gum(gerald))
 
 
#Here's an example with two LAMBDAs: works like a binary predicate
\x.\y.(dog(x) & own(y, x)) 
# \x.\y. to be written in the abbreviated form \x y. 
>>> print(read_expr(r'\x.\y.(dog(x) & own(y, x))(cyril)').simplify())
\y.(dog(cyril) & own(y,cyril))
>>> print(read_expr(r'\x y.(dog(x) & own(y, x))(cyril, angus)').simplify()) 
(dog(cyril) & own(angus,cyril))
 
 

#The process of relabeling bound variables is known as alpha-conversion
# Using ==, we are  testing for alpha-equivalence:
>>> expr1 = read_expr('exists x.P(x)')
>>> print(expr1)
exists x.P(x)
>>> expr2 = expr1.alpha_convert(nltk.sem.Variable('z'))
>>> print(expr2)
exists z.P(z)
>>> expr1 == expr2
True
 
#This relabeling is carried out automatically by the ß-reduction code in logic

>>> expr3 = read_expr('\P.(exists x.P(x))(\y.see(y, x))') #P is arg name, call LAMBDA with another lambda P = \y.see(y, x)
>>> print(expr3)
(\P.exists x.P(x))(\y.see(y,x))
>>> print(expr3.simplify())
exists z1.see(z1,x)
 
 
##Quantified NPs
 
a.  A dog barks. 
b1.  exists x.(dog(x) & bark(x)) 

#how do we give a semantic representation to the quantified NPs 'a dog' 
#so that it can be combined with bark to give the result in b1
 
#ie way of instantiating ?np so that 
[SEM=<?np(\x.bark(x))>] 
#is equivalent to 
[SEM=<exists x.(dog(x) & bark(x))>].

#b1 is equivalent to below 
#create Lambda operator P which is used in place of P (lambda = function(P))
\P.exists x.(dog(x) & P(x))  #***REF-1***

#OR with another level of abstraction (ie lambda(Q,P) function )
\Q P.exists x.(Q(x) & P(x))

#Note -Applying above as a function expression to 'dog' yields earlier one
#, and applying that to 'bark' gives us 
\P.exists x.(dog(x) & P(x))(\x.bark(x)). 
#Finally, carrying out ß-reduction yields just what we wanted, namely (b1).


#Note: a universally quantified NP will look like 
\P.all x.(dog(x) -> P(x)) 



##Transitive Verbs

a. Angus chases a dog. 
b2. exists x.(dog(x) & chase(angus, x))
 
#1st  level of abstraction with lambda - \y. EXPR 
\y.exists x.(dog(x) & chase(y, x)) 

#equivalent to \P.EXPR and call P(\z.chase(y, z))
\P.exists x.(dog(x) & P(x))(\z.chase(y, z)) 

#replace the above function expression by a variable X = \P.exists x.(dog(x) & P(x))
X(\z.chase(y, z)) 
#or in lambda Notation , \X.EXPR , EXPR =  \P.exists x.(dog(x) & P(x)) = \x.X(..)
#and call X with value (\P.exists x.(dog(x) & P(x)))
\X. \x.X(\y.chase(x, y)) 
#OR 
\X x.X(\y.chase(x, y)) 

>>> read_expr = nltk.sem.Expression.fromstring
>>> tvp = read_expr(r'\X x.X(\y.chase(x,y))')
>>> np = read_expr(r'(\P.exists x.(dog(x) & P(x)))')
>>> vp = nltk.sem.ApplicationExpression(tvp, np) #tvp(np)
>>> print(vp)
(\X x.X(\y.chase(x,y)))(\P.exists x.(dog(x) & P(x)))
>>> print(vp.simplify())
\x.exists z2.(dog(z2) & chase(x,z2))
 
 

#Note  lambda expression for single value , Angus.
\P. P(angus) 

#simple-sem.fcfg contains a small set of rules for parsing and translating simple examples of the kind 

>>> from nltk import load_parser
>>> nltk.data.show_cfg('grammars/book_grammars/simple-sem.fcfg')
#<> means Compostion of whole from Parts 
# f(x) means result from function application of f with arg x 
# \x.EXPR -> XXX means convert XXX to lambda EXPR(x) 
% start S
# Grammar Rules
S[SEM = <?subj(?vp)>] -> NP[NUM=?n,SEM=?subj] VP[NUM=?n,SEM=?vp]
NP[NUM=?n,SEM=<?det(?nom)> ] -> Det[NUM=?n,SEM=?det]  Nom[NUM=?n,SEM=?nom]
NP[LOC=?l,NUM=?n,SEM=?np] -> PropN[LOC=?l,NUM=?n,SEM=?np]
Nom[NUM=?n,SEM=?nom] -> N[NUM=?n,SEM=?nom]
VP[NUM=?n,SEM=?v] -> IV[NUM=?n,SEM=?v]
VP[NUM=?n,SEM=<?v(?obj)>] -> TV[NUM=?n,SEM=?v] NP[SEM=?obj]
VP[NUM=?n,SEM=<?v(?obj,?pp)>] -> DTV[NUM=?n,SEM=?v] NP[SEM=?obj] PP[+TO,SEM=?pp]
PP[+TO, SEM=?np] -> P[+TO] NP[SEM=?np]
# Lexical Rules
PropN[-LOC,NUM=sg,SEM=<\P.P(angus)>] -> 'Angus'
PropN[-LOC,NUM=sg,SEM=<\P.P(cyril)>] -> 'Cyril'
PropN[-LOC,NUM=sg,SEM=<\P.P(irene)>] -> 'Irene'
Det[NUM=sg,SEM=<\P Q.all x.(P(x) -> Q(x))>] -> 'every'
Det[NUM=pl,SEM=<\P Q.all x.(P(x) -> Q(x))>] -> 'all'
Det[SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'some'
Det[NUM=sg,SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'a'
Det[NUM=sg,SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'an'
N[NUM=sg,SEM=<\x.man(x)>] -> 'man'
N[NUM=sg,SEM=<\x.girl(x)>] -> 'girl'
N[NUM=sg,SEM=<\x.boy(x)>] -> 'boy'
N[NUM=sg,SEM=<\x.bone(x)>] -> 'bone'
N[NUM=sg,SEM=<\x.ankle(x)>] -> 'ankle'
N[NUM=sg,SEM=<\x.dog(x)>] -> 'dog'
N[NUM=pl,SEM=<\x.dog(x)>] -> 'dogs'
IV[NUM=sg,SEM=<\x.bark(x)>,TNS=pres] -> 'barks'
IV[NUM=pl,SEM=<\x.bark(x)>,TNS=pres] -> 'bark'
IV[NUM=sg,SEM=<\x.walk(x)>,TNS=pres] -> 'walks'
IV[NUM=pl,SEM=<\x.walk(x)>,TNS=pres] -> 'walk'
TV[NUM=sg,SEM=<\X x.X(\y.chase(x,y))>,TNS=pres] -> 'chases'
TV[NUM=pl,SEM=<\X x.X(\y.chase(x,y))>,TNS=pres] -> 'chase'
TV[NUM=sg,SEM=<\X x.X(\y.see(x,y))>,TNS=pres] -> 'sees'
TV[NUM=pl,SEM=<\X x.X(\y.see(x,y))>,TNS=pres] -> 'see'
TV[NUM=sg,SEM=<\X x.X(\y.bite(x,y))>,TNS=pres] -> 'bites'
TV[NUM=pl,SEM=<\X x.X(\y.bite(x,y))>,TNS=pres] -> 'bite'
DTV[NUM=sg,SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>,TNS=pres] -> 'gives'
DTV[NUM=pl,SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>,TNS=pres] -> 'give'
P[+to] -> 'to'


>>> parser = load_parser('grammars/book_grammars/simple-sem.fcfg', trace=1)
>>> sentence = 'Angus gives a bone to every dog'
>>> tokens = sentence.split()
>>> for tree in parser.parse(tokens):
        print(tree.label()['SEM'])
all z2.(dog(z2) -> exists z1.(bone(z1) & give(angus,z1,z2)))

>>> sen = list(parser.parse(tokens))[0].label()['SEM']
#Above can be tested via Model builder , given assumptions 
>>> read_expr = nltk.sem.Expression.fromstring
>>> a4 = sen 
>>> a5 = read_expr('dog(tom)')
>>> a6 = read_expr('bone(haddi)')
>>> g = read_expr('give(angus,haddi,tom)')
>>> mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6])
>>> mc._modelbuilder._mace4_bin = r"c:/nltk_data/prover9/bin-win32/mace4.exe"
>>> mc._interpformat_bin = r"c:/nltk_data/prover9/bin-win32/interpformat.exe"
>>> mc.build_model()
True
>>> print(mc.valuation)
{'angus': 'a',
 'bone': {('b',), ('a',)},
 'dog': {('a',)},
 'give': {('a', 'b', 'a')},
 'haddi': 'a',
 'tom': 'a'}
 
>>> mc.print_assumptions()
all z10.(dog(z10) -> exists z9.(bone(z9) & give(angus,z9,z10)))
dog(tom)
bone(haddi)


#details 
>>> trees = list(parser.parse(tokens))
|.A.g.a.b.t.e.d.|
|[-] . . . . . .| [0:1] 'Angus'
|. [-] . . . . .| [1:2] 'gives'
|. . [-] . . . .| [2:3] 'a'
|. . . [-] . . .| [3:4] 'bone'
|. . . . [-] . .| [4:5] 'to'
|. . . . . [-] .| [5:6] 'every'
|. . . . . . [-]| [6:7] 'dog'
|[-] . . . . . .| [0:1] PropN[-LOC, NUM='sg', SEM=<\P.P(angus)>] -> 'Angus' *
|[-] . . . . . .| [0:1] NP[-LOC, NUM='sg', SEM=<\P.P(angus)>] -> PropN[-LOC, NUM='sg', SEM=<\P.P(angus)>] *
|[-> . . . . . .| [0:1] S[SEM=<?subj(?vp)>] -> NP[NUM=?n, SEM=?subj] * VP[NUM=?n, SEM=?vp] {?n: 'sg', ?subj: <LambdaExpression \P.P(angus)>}
|. [-] . . . . .| [1:2] DTV[NUM='sg', SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>, TNS='pres'] -> 'gives' *
|. [-> . . . . .| [1:2] VP[NUM=?n, SEM=<?v(?obj,?pp)>] -> DTV[NUM=?n,SEM=?v] * NP[SEM=?obj] PP[SEM=?pp, +TO] {?n: 'sg', ?v: <LambdaExpression \Y X x.X(\z.Y(\y.give(x,y,z)))>}
|. . [-] . . . .| [2:3] Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'a' *
|. . [-> . . . .| [2:3] NP[NUM=?n, SEM=<?det(?nom)>] -> Det[NUM=?n, SEM=?det] * Nom[NUM=?n, SEM=?nom] {?det: <LambdaExpression \P Q.exists x.(P(x) & Q(x))>, ?n: 'sg'}
|. . . [-] . . .| [3:4] N[NUM='sg', SEM=<\x.bone(x)>] -> 'bone' *
|. . . [-] . . .| [3:4] Nom[NUM='sg', SEM=<\x.bone(x)>] -> N[NUM='sg', SEM=<\x.bone(x)>] *
|. . [---] . . .| [2:4] NP[NUM='sg', SEM=<\Q.exists x.(bone(x) & Q(x))>] -> Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] Nom[NUM='sg', SEM=<\x.bone(x)>] *
|. . [---> . . .| [2:4] S[SEM=<?subj(?vp)>] -> NP[NUM=?n, SEM=?subj] * VP[NUM=?n, SEM=?vp] {?n: 'sg', ?subj: <LambdaExpression \Q.exists x.(bone(x) & Q(x))>}
|. [-----> . . .| [1:4] VP[NUM=?n, SEM=<?v(?obj,?pp)>] -> DTV[NUM=?n,SEM=?v] NP[SEM=?obj] * PP[SEM=?pp, +TO] {?n: 'sg', ?obj: <LambdaExpression \Q.exists x.(bone(x) & Q(x))>, ?v: <LambdaExpression \Y X x.X(\z.Y(\y.give(x,y,z)))>}
|. . . . [-] . .| [4:5] P[+to] -> 'to' *
|. . . . [-> . .| [4:5] PP[SEM=?np, +TO] -> P[+TO] * NP[SEM=?np] {}
|. . . . . [-] .| [5:6] Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] -> 'every' *
|. . . . . [-> .| [5:6] NP[NUM=?n, SEM=<?det(?nom)>] -> Det[NUM=?n, SEM=?det] * Nom[NUM=?n, SEM=?nom] {?det: <LambdaExpression \P Q.all x.(P(x) -> Q(x))>, ?n: 'sg'}
|. . . . . . [-]| [6:7] N[NUM='sg', SEM=<\x.dog(x)>] -> 'dog' *
|. . . . . . [-]| [6:7] Nom[NUM='sg', SEM=<\x.dog(x)>] -> N[NUM='sg',SEM=<\x.dog(x)>] *
|. . . . . [---]| [5:7] NP[NUM='sg', SEM=<\Q.all x.(dog(x) -> Q(x))>]-> Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] Nom[NUM='sg', SEM=<\x.dog(x)>] *
|. . . . . [--->| [5:7] S[SEM=<?subj(?vp)>] -> NP[NUM=?n, SEM=?subj] * VP[NUM=?n, SEM=?vp] {?n: 'sg', ?subj: <LambdaExpression \Q.all x.(dog(x) -> Q(x))>}
|. . . . [-----]| [4:7] PP[SEM=<\Q.all x.(dog(x) -> Q(x))>, +TO] -> P[+TO] NP[SEM=<\Q.all x.(dog(x) -> Q(x))>] *
|. [-----------]| [1:7] VP[NUM='sg', SEM=<\x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2)))>] -> DTV[NUM='sg', SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>] NP[SEM=<\Q.exists x.(bone(x) & Q(x))>] PP[SEM=<\Q.all x.(dog(x) -> Q(x))>, +TO] *
|[=============]| [0:7] S[SEM=<all z2.(dog(z2) -> exists z1.(bone(z1)& give(angus,z1,z2)))>] -> NP[NUM='sg', SEM=<\P.P(angus)>] VP[NUM='sg', SEM=<\x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2)))>] *

>>> print(trees[0])
#ie S = ?subj(?vp) = \P.P(angus)(\x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2))))
#= beta reduction, P=?vp ie = \x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2)))(angus)
#?subj from NP  = \P.P(angus)
#?vp from VP = \x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2)))

(S[SEM=<all z2.(dog(z2) -> exists z1.(bone(z1) & give(angus,z1,z2)))>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(angus)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(angus)>] Angus))
  (VP[NUM='sg', SEM=<\x.all z2.(dog(z2) -> exists z1.(bone(z1) & give(x,z1,z2)))>]
    (DTV[NUM='sg', SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>, TNS='pres']    gives)
    (NP[NUM='sg', SEM=<\Q.exists x.(bone(x) & Q(x))>]
      (Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] a)
      (Nom[NUM='sg', SEM=<\x.bone(x)>]
        (N[NUM='sg', SEM=<\x.bone(x)>] bone)))
    (PP[SEM=<\Q.all x.(dog(x) -> Q(x))>, +TO])
      (P[+to] to)
      (NP[NUM='sg', SEM=<\Q.all x.(dog(x) -> Q(x))>]
        (Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] every)
        (Nom[NUM='sg', SEM=<\x.dog(x)>]
          (N[NUM='sg', SEM=<\x.dog(x)>] dog))))))
>>>

##Reference 

nltk.sem.util.evaluate_sents(inputs, grammar, model, assignment, trace=0)
    Add the truth-in-a-model value to each semantic representation 
    for each syntactic parse of each input sentences.
        inputs (list(str)) – a list of sentences
        grammar (nltk.grammar.FeatureGrammar) – FeatureGrammar or name of feature-based grammar
    Returns:a mapping from sentences to lists of triples (parse-tree, semantic-representations, evaluation-in-model)
    Return type: list(list(tuple(nltk.tree.Tree, nltk.sem.logic.ConstantExpression, bool or dict(str): bool)))

nltk.sem.util.interpret_sents(inputs, grammar, semkey='SEM', trace=0)
    Add the semantic representation to each syntactic parse tree of each input sentence.
        inputs (list(str)) – a list of sentences
        grammar (nltk.grammar.FeatureGrammar) – FeatureGrammar or name of feature-based grammar
    Returns: a mapping from sentences to lists of pairs (parse-tree, semantic-representations)
    Return type:list(list(tuple(nltk.tree.Tree, nltk.sem.logic.ConstantExpression)))

nltk.sem.util.parse_sents(inputs, grammar, trace=0)
    Convert input sentences into syntactic trees.
        inputs (list(str)) – sentences to be parsed
        grammar (nltk.grammar.FeatureGrammar) – FeatureGrammar or name of feature-based grammar
    Return type:list(nltk.tree.Tree) or dict(list(str)): list(Tree)
    Returns:a mapping from input sentences to a list of ``Tree``s

nltk.sem.util.read_sents(filename, encoding='utf8')

nltk.sem.util.root_semrep(syntree, semkey='SEM')
    Find the semantic representation at the root of a tree.
        syntree – a parse Tree
        semkey – the feature label to use for the root semantics in the tree
    Returns:  the semantic representation at the root of a Tree
    Return type: sem.Expression
 
 
#The function interpret_sents() is intended for interpretation of a list of input sentences. 
#It builds a dictionary d where for each sentence 'sent' in the input, 
#d[sent] is a list of pairs (synrep, semrep) consisting of trees and semantic representations for sent. 

>>> sents = ['Irene walks', 'Cyril bites an ankle']
>>> grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
>>> for results in nltk.interpret_sents(sents, grammar_file):
        for (synrep, semrep) in results:
            print(synrep)
(S[SEM=<walk(irene)>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(irene)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(irene)>] Irene))
  (VP[NUM='sg', SEM=<\x.walk(x)>]
    (IV[NUM='sg', SEM=<\x.walk(x)>, TNS='pres'] walks)))
(S[SEM=<exists z3.(ankle(z3) & bite(cyril,z3))>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(cyril)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(cyril)>] Cyril))
  (VP[NUM='sg', SEM=<\x.exists z3.(ankle(z3) & bite(x,z3))>]
    (TV[NUM='sg', SEM=<\X x.X(\y.bite(x,y))>, TNS='pres'] bites)
    (NP[NUM='sg', SEM=<\Q.exists x.(ankle(x) & Q(x))>]
      (Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] an)
      (Nom[NUM='sg', SEM=<\x.ankle(x)>]
        (N[NUM='sg', SEM=<\x.ankle(x)>] ankle)))))
 
 
##Truth value of English sentences 
#We have seen now how to convert English sentences into logical forms, 
#and earlier we saw how logical forms could be checked as true or false in a model. 

#Putting these two mappings together, we can check the truth value of English sentences 

>>> v = """
    bertie => b
    olive => o
    cyril => c
    boy => {b}
    girl => {o}
    dog => {c}
    walk => {o, c}
    see => {(b, o), (c, b), (o, c)}
    """
>>> val = nltk.Valuation.fromstring(v)
>>> g = nltk.Assignment(val.domain)
>>> m = nltk.Model(val.domain, val)
>>> sent = 'Cyril sees every boy'
>>> grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
>>> results = nltk.evaluate_sents([sent], grammar_file, m, g)
>>> for (syntree, semrep, value) in results:
        print(semrep)
        print(value)
all z4.(boy(z4) -> see(cyril,z4))
True
 
 
##Dealing with  Quantifier Ambiguity 

#Note above model would translate below to a, not b 
Every girl chases a dog.  
a.  all x.(girl(x) -> exists y.(dog(y) & chase(x,y))) 
b.  exists y.(dog(y) & all x.(girl(x) -> chase(x,y))) 

##NLTK- One SOlution is Cooper storage, 
#a semantic representation is no longer an expression of first-order logic, 
#but instead a pair consisting of a "core" semantic representation plus a list of binding operators. 

# a binding operator as being identical to the semantic representation of a quantified NP 
\P.all x.(girl(x) -> P(x)) 
\P.exists x.(dog(x) & P(x)) 

#let's take our core to be the open formula chase(x,y). 
#Given a list of above binding operators
#we pick a binding operator and combine it with the core=chase(x,y) .
\P.exists y.(dog(y) & P(y))(\z2.chase(z1,z2))

#Then we take the result, and apply the next binding operator  to it.
\P.all x.(girl(x) -> P(x))(\z1.exists x.(dog(x) & chase(z1,x)))

#Once the list is empty, we have a conventional logical form for the sentence.
#Combining binding operators with the core in this way is called S-Retrieval. 

#If we are careful to allow every possible order of binding operators 
#(for example, by taking all permutations of the list), 
#then we will be able to generate every possible scope ordering of quantifiers.

##To build Cooper storage, introduce new features 
#each phrasal and lexical rule in the grammar will have a sem feature, 
#now there will be embedded features core and store. 

#Example -  Cyril smiles. 
#Here's a lexical rule for the verb smiles (taken from the grammar storage.fcfg) 
IV[SEM=[core=<\x.smile(x)>, store=(/)]] -> 'smiles'

#The rule for the proper name Cyril is more complex.
NP[SEM=[core=<@x>, store=(<bo(\P.P(cyril),@x)>)]] -> 'Cyril'

#The bo predicate has two subparts: 
#the standard representation of a proper name, 
#and the expression @x, which is called the address of the binding operator.
 
#@x is a metavariable, 
#that is, a variable that ranges over individual variables of the logic 
#it also provides the value of core. 

#The rule for VP just percolates up the semantics of the IV
VP[SEM=?s] -> IV[SEM=?s]

S[SEM=[core=<?vp(?np)>, store=(?b1+?b2)]] ->
   NP[SEM=[core=?np, store=?b1]] VP[SEM=[core=?vp, store=?b2]]


#The core value at the S node is the result of applying the VP's core value, 
#namely \x.smile(x), to the subject NP's value. 

#The latter will not be @x, but rather an instantiation of @x, say z3. 
#After ß-reduction, <?vp(?np)> will be unified with <smile(z3)>. 

#Now, when @x is instantiated as part of the parsing process, 
#it will be instantiated uniformly. 
#In particular, the occurrence of @x in the subject NP's store will also be mapped to z3, 
#yielding the element bo(\P.P(cyril),z3). 

#These steps can be seen in the following parse tree.
(S[SEM=[core=<smile(z3)>, store=(bo(\P.P(cyril),z3))]]
  (NP[SEM=[core=<z3>, store=(bo(\P.P(cyril),z3))]] Cyril)
  (VP[SEM=[core=<\x.smile(x)>, store=()]]
    (IV[SEM=[core=<\x.smile(x)>, store=()]] smiles)))


#for 'Every girl chases a dog.'
core  = <chase(z1,z2)>
store = (bo(\P.all x.(girl(x) -> P(x)),z1), bo(\P.exists x.(dog(x) & P(x)),z2))

 
#The module nltk.sem.cooper_storage deals with the task of turning storage-style semantic representations into standard logical forms. 

class nltk.sem.cooper_storage.CooperStore(featstruct)
    Bases: object
    A container for handling quantifier ambiguity via Cooper storage.
    s_retrieve(trace=False)
        Carry out S-Retrieval of binding operators in store. 
        If hack=True, serialize the bindop and core as strings and reparse. 
        
        Each permutation of the store (i.e. list of binding operators) 
        is taken to be a possible scoping of quantifiers. 
        We iterate through the binding operators in each permutation, 
        and successively apply them to the current term, 
        starting with the core semantic representation, 
        working from the inside out.
        Binding operators are of the form:
        bo(\P.all x.(man(x) -> P(x)),z1)


nltk.sem.cooper_storage.parse_with_bindops(sentence, grammar=None, trace=0)
    Use a grammar with Binding Operators to parse a sentence.

#Example   
>>> from nltk.sem import cooper_storage as cs
>>> sentence = 'every girl chases a dog'
>>> trees = cs.parse_with_bindops(sentence, grammar='grammars/book_grammars/storage.fcfg')
>>> semrep = trees[0].label()['SEM']
>>> cs_semrep = cs.CooperStore(semrep)
>>> print(cs_semrep.core)  #self.core = featstruct['CORE']
chase(z2,z4)
>>> for bo in cs_semrep.store:  # self.store = featstruct['STORE']
        print(bo)
bo(\P.all x.(girl(x) -> P(x)),z2)
bo(\P.exists x.(dog(x) & P(x)),z4)
 
 
#Finally we call s_retrieve() and check the readings.
>>> cs_semrep.s_retrieve(trace=True)
Permutation 1
   (\P.all x.(girl(x) -> P(x)))(\z2.chase(z2,z4))
   (\P.exists x.(dog(x) & P(x)))(\z4.all x.(girl(x) -> chase(x,z4)))
Permutation 2
   (\P.exists x.(dog(x) & P(x)))(\z4.chase(z2,z4))
   (\P.all x.(girl(x) -> P(x)))(\z2.exists x.(dog(x) & chase(z2,x)))
 
 
>>> for reading in cs_semrep.readings:
        print(reading)
exists x.(dog(x) & all z3.(girl(z3) -> chase(z3,x)))
all x.(girl(x) -> exists z4.(dog(z4) & chase(x,z4)))
 
 
##NLTK- Discourse Semantics

#A discourse is a sequence of sentences. 
#Very often, the interpretation of a sentence in a discourse depends what preceded it. 

#Given discourse such as 'Angus used to have a dog. But he recently disappeared.', 
#you will probably interpret he as referring to Angus's dog. 

#However, in 'Angus used to have a dog. He took him for walks in New Town'., 
#you are more likely to interpret he as referring to Angus himself.

 
##NLTK- Discourse Representation Theory(DRT)
 
a.  Angus owns a dog. It bit Irene. 
b.  EXISTSx.(dog(x) AND own(Angus, x) AND bite(x, Irene)) 

#A discourse representation structure (DRS) presents the meaning of discourse 
#in terms of a list of discourse referents and a list of conditions. 

#The discourse referents are the things under discussion in the discourse, 
#and they correspond to the individual variables of first-order logic. 

#The DRS conditions apply to those discourse referents, 
#and correspond to atomic open formulas of first-order logic


##Building a DRS; 
#the DRS on the left hand side represents the result of processing the first sentence in the discourse, 
#while the DRS on the right hand side shows the effect of processing the second sentence and integrating its content.
 
 Angus owns a dog           Angus owns a dog, it bit Irene  
                                                
 x y                        x y u z                 
 ------                     ------              
 Angus(x)                   Angus(x)            
 dog(y)                     dog(y)              
 own(x,y)                   own(x,y)            
                            u = y
                            Irene(z)
                            bite(u,z)
 
#in NLTK, 
>>> read_dexpr = nltk.sem.DrtExpression.fromstring
>>> drs1 = read_dexpr('([x, y], [angus(x), dog(y), own(x, y)])') 
>>> print(drs1)
([x,y],[angus(x), dog(y), own(x,y)])
 
#to visualize the result
>>> drs1.draw()
 
 
#every DRS can be translated into a formula of first-order logic, 
#the fol() method implements this translation.
>>> print(drs1.fol())
exists x y.(angus(x) & dog(y) & own(x,y))
 
 

#In addition to the functionality available for first-order logic expressions, 
#DRT Expressions have a DRS-concatenation operator, represented as the + symbol. 

#The concatenation of two DRSs is a single DRS containing 
#the merged discourse referents and the conditions from both arguments.

#DRS-concatenation automatically alpha-converts bound variables to avoid name-clashes.

>>> drs2 = read_dexpr('([x], [walk(x)]) + ([y], [run(y)])')
>>> print(drs2)
(([x],[walk(x)]) + ([y],[run(y)]))
>>> print(drs2.simplify())
([x,y],[walk(x), run(y)])
 
 
#it is possible to embed one DRS within another, 
#and this is how universal quantification is handled. 
>>> drs3 = read_dexpr('([], [(([x], [dog(x)]) -> ([y],[ankle(y), bite(x, y)]))])')
>>> print(drs3.fol())
all x.(dog(x) -> exists y.(ankle(y) & bite(x,y)))
 
 

#DRT is designed to allow anaphoric pronouns to be interpreted by linking to existing discourse referents. 
#if the DRS contains a condition of the form PRO(x), 
#the method resolve_anaphora() replaces this 
#with a condition of the form x = [...], where [...] is a list of possible antecedents.

>>> drs4 = read_dexpr('([x, y], [angus(x), dog(y), own(x, y)])')
>>> drs5 = read_dexpr('([u, z], [PRO(u), irene(z), bite(u, z)])')
>>> drs6 = drs4 + drs5
>>> print(drs6.simplify())
([u,x,y,z],[angus(x), dog(y), own(x,y), PRO(u), irene(z), bite(u,z)])
>>> print(drs6.simplify().resolve_anaphora())
([u,x,y,z],[angus(x), dog(y), own(x,y), (u = [x,y,z]), irene(z), bite(u,z)])
 
 

#to build compositional semantic representations which are based on DRT 
#rather than first-order logic
Det[num=sg,SEM=<\P Q.(([x],[]) + P(x) + Q(x))>] -> 'a'
Det[num=sg,SEM=<\P Q.(([x],[]) + P(x) + Q(x))>] -> 'a'
Det[num=sg,SEM=<\P Q. exists x.(P(x) & Q(x))>] -> 'a'


#To get a better idea of how the DRT rule works, 
#look at this subtree for the NP a dog.
(NP[num='sg', SEM=<\Q.(([x],[dog(x)]) + Q(x))>]
  (Det[num='sg', SEM=<\P Q.((([x],[]) + P(x)) + Q(x))>] a)
  (Nom[num='sg', SEM=<\x.([],[dog(x)])>]
    (N[num='sg', SEM=<\x.([],[dog(x)])>] dog)))))



#In order to parse with grammar drt.fcfg

>>> from nltk import load_parser
>>> nltk.data.show_cfg('grammars/book_grammars/drt.fcfg')
% start S
# Grammar Rules
S[SEM = <app(?subj,?vp)>] -> NP[NUM=?n,SEM=?subj] VP[NUM=?n,SEM=?vp]
NP[NUM=?n,SEM=<app(?det,?nom)> ] -> Det[NUM=?n,SEM=?det]  Nom[NUM=?n,SEM=?nom]
NP[LOC=?l,NUM=?n,SEM=?np] -> PropN[LOC=?l,NUM=?n,SEM=?np]
Nom[NUM=?n,SEM=?nom] -> N[NUM=?n,SEM=?nom]
Nom[NUM=?n,SEM=<app(?pp,?nom)>] -> N[NUM=?n,SEM=?nom] PP[SEM=?pp]
VP[NUM=?n,SEM=?v] -> IV[NUM=?n,SEM=?v]
VP[NUM=?n,SEM=<app(?v,?obj)>] -> TV[NUM=?n,SEM=?v] NP[SEM=?obj]
# Lexical Rules
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Angus(x)])+P(x))>] -> 'Angus'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Irene(x)])+P(x))>] -> 'Irene'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[John(x)])+P(x))>] -> 'John'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Mary(x)])+P(x))>] -> 'Mary'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Suzie(x)])+P(x))>] -> 'Suzie'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Vincent(x)])+P(x))>] -> 'Vincent'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Mia(x)])+P(x))>] -> 'Mia'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Marsellus(x)])+P(x))>] -> 'Marsellus'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[Fido(x)])+P(x))>] -> 'Fido'
PropN[+LOC,NUM=sg,SEM=<\P.(DRS([x],[Noosa(x)])+P(x))>] -> 'Noosa'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[PRO(x)])+P(x))>] -> 'he'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[PRO(x)])+P(x))>] -> 'she'
PropN[-LOC,NUM=sg,SEM=<\P.(DRS([x],[PRO(x)])+P(x))>] -> 'it'
Det[NUM=sg,SEM=<\P Q.DRS([],[((DRS([x],[])+P(x)) implies Q(x))])>] ->'every' | 'Every'
Det[NUM=pl,SEM=<\P Q.DRS([],[((DRS([x],[])+P(x)) implies Q(x))])>] ->'all' | 'All'
Det[SEM=<\P Q.((DRS([x],[])+P(x))+Q(x))>] -> 'some' | 'Some'
Det[NUM=sg,SEM=<\P Q.((DRS([x],[])+P(x))+Q(x))>] -> 'a' | 'A'
Det[NUM=sg,SEM=<\P Q.(not ((DRS([x],[])+P(x))+Q(x)))>] -> 'no' | 'No'
N[NUM=sg,SEM=<\x.DRS([],[boy(x)])>] -> 'boy'
N[NUM=pl,SEM=<\x.DRS([],[boy(x)])>] -> 'boys'
N[NUM=sg,SEM=<\x.DRS([],[girl(x)])>] -> 'girl'
N[NUM=pl,SEM=<\x.DRS([],[girl(x)])>] -> 'girls'
N[NUM=sg,SEM=<\x.DRS([],[dog(x)])>] -> 'dog'
N[NUM=pl,SEM=<\x.DRS([],[dog(x)])>] -> 'dogs'
N[NUM=sg,SEM=<\x.DRS([],[student(x)])>] -> 'student'
N[NUM=pl,SEM=<\x.DRS([],[student(x)])>] -> 'students'
N[NUM=sg,SEM=<\x.DRS([],[person(x)])>] -> 'person'
N[NUM=pl,SEM=<\x.DRS([],[person(x)])>] -> 'persons'
N[NUM=sg,SEM=<\x.DRS([],[boxerdog(x)])>] -> 'boxer'
N[NUM=pl,SEM=<\x.DRS([],[boxerdog(x)])>] -> 'boxers'
N[NUM=sg,SEM=<\x.DRS([],[boxer(x)])>] -> 'boxer'
N[NUM=pl,SEM=<\x.DRS([],[boxer(x)])>] -> 'boxers'
N[NUM=sg,SEM=<\x.DRS([],[garden(x)])>] -> 'garden'
N[NUM=sg,SEM=<\x.DRS([],[kitchen(x)])>] -> 'kitchen'
IV[NUM=sg,SEM=<\x.DRS([],[bark(x)])>,tns=pres] -> 'barks'
IV[NUM=pl,SEM=<\x.DRS([],[bark(x)])>,tns=pres] -> 'bark'
IV[NUM=sg,SEM=<\x.DRS([],[walk(x)])>,tns=pres] -> 'walks'
IV[NUM=pl,SEM=<\x.DRS([],[walk(x)])>,tns=pres] -> 'walk'
IV[NUM=pl,SEM=<\x.DRS([],[dance(x)])>,tns=pres] -> 'dance'
IV[NUM=sg,SEM=<\x.DRS([],[dance(x)])>,tns=pres] -> 'dances'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[own(x,y)]))>,tns=pres] -> 'owns'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[own(x,y)]))>,tns=pres] -> 'own'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[bite(x,y)]))>,tns=pres] -> 'bites'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[bite(x,y)]))>,tns=pres] -> 'bite'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[chase(x,y)]))>,tns=pres] -> 'chases'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[chase(x,y)]))>,tns=pres] -> 'chase'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[marry(x,y)]))>,tns=pres] -> 'marries'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[marry(x,y)]))>,tns=pres] -> 'marry'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[know(x,y)]))>,tns=pres] -> 'knows'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[know(x,y)]))>,tns=pres] -> 'know'
TV[NUM=sg,SEM=<\X x.X(\y.DRS([],[see(x,y)]))>,tns=pres] -> 'sees'
TV[NUM=pl,SEM=<\X x.X(\y.DRS([],[see(x,y)]))>,tns=pres] -> 'see'
>>> parser = load_parser('grammars/book_grammars/drt.fcfg', logic_parser=nltk.sem.drt.DrtParser(), trace=1)
>>> trees = list(parser.parse('Angus owns a dog'.split()))
|.Ang.own. a .dog.|
|[---]   .   .   .| [0:1] 'Angus'
|.   [---]   .   .| [1:2] 'owns'
|.   .   [---]   .| [2:3] 'a'
|.   .   .   [---]| [3:4] 'dog'
|[---]   .   .   .| [0:1] PropN[-LOC, NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>] -> 'Angus' *
|[---]   .   .   .| [0:1] NP[-LOC, NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>] -> PropN[-LOC, NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>] *
|[--->   .   .   .| [0:1] S[SEM=<?subj(?vp)>] -> NP[NUM=?n, SEM=?subj] * VP[NUM=?n, SEM=?vp] {?n: 'sg', ?subj: <DrtLambdaExpression \P.(([x],[Angus(x)]) + P(x))>}
|.   [---]   .   .| [1:2] TV[NUM='sg', SEM=<\X x.X(\y.([],[own(x,y)]))>, tns='pres'] -> 'owns' *
|.   [--->   .   .| [1:2] VP[NUM=?n, SEM=<?v(?obj)>] -> TV[NUM=?n, SEM=?v] * NP[SEM=?obj] {?n: 'sg', ?v: <DrtLambdaExpression \X x.X(\y.([],[own(x,y)]))>}
|.   .   [---]   .| [2:3] Det[NUM='sg', SEM=<\P Q.(([x],[]) + P(x) + Q(x))>] -> 'a' *
|.   .   [--->   .| [2:3] NP[NUM=?n, SEM=<?det(?nom)>] -> Det[NUM=?n,
SEM=?det] * Nom[NUM=?n, SEM=?nom] {?det: <DrtLambdaExpression \P Q.(([x],[]) + P(x) + Q(x))>, ?n: 'sg'}
|.   .   .   [---]| [3:4] N[NUM='sg', SEM=<\x.([],[dog(x)])>] -> 'dog' *
|.   .   .   [---]| [3:4] Nom[NUM='sg', SEM=<\x.([],[dog(x)])>] -> N[NUM='sg', SEM=<\x.([],[dog(x)])>] *
|.   .   .   [--->| [3:4] Nom[NUM=?n, SEM=<?pp(?nom)>] -> N[NUM=?n, SEM=?nom] * PP[SEM=?pp] {?n: 'sg', ?nom: <DrtLambdaExpression \x.([],[dog(x)])>}
|.   .   [-------]| [2:4] NP[NUM='sg', SEM=<\Q.(([x],[dog(x)]) + Q(x))>] -> Det[NUM='sg', SEM=<\P Q.(([x],[]) + P(x) + Q(x))>] Nom[NUM='sg', SEM=<\x.([],[dog(x)])>] *
|.   .   [------->| [2:4] S[SEM=<?subj(?vp)>] -> NP[NUM=?n, SEM=?subj] * VP[NUM=?n, SEM=?vp] {?n: 'sg', ?subj: <DrtLambdaExpression \Q.(([x],[dog(x)]) + Q(x))>}
|.   [-----------]| [1:4] VP[NUM='sg', SEM=<\z1.([x],[dog(x), own(z1,x)])>] -> TV[NUM='sg', SEM=<\X x.X(\y.([],[own(x,y)]))>] NP[SEM=<\Q.(([x],[dog(x)]) + Q(x))>] *
|[===============]| [0:4] S[SEM=<([x,z2],[Angus(x), dog(z2), own(x,z2)])>] -> NP[NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>] VP[NUM='sg',SEM=<\z1.([x],[dog(x), own(z1,x)])>] *

>>> print(trees[0])
(S[SEM=<([x,z2],[Angus(x), dog(z2), own(x,z2)])>]
  (NP[-LOC, NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>]
    (PropN[-LOC, NUM='sg', SEM=<\P.(([x],[Angus(x)]) + P(x))>] Angus))
  (VP[NUM='sg', SEM=<\z1.([x],[dog(x), own(z1,x)])>]
    (TV[NUM='sg', SEM=<\X x.X(\y.([],[own(x,y)]))>, tns='pres'] owns)
    (NP[NUM='sg', SEM=<\Q.(([x],[dog(x)]) + Q(x))>]
      (Det[NUM='sg', SEM=<\P Q.(([x],[]) + P(x) + Q(x))>] a)
      (Nom[NUM='sg', SEM=<\x.([],[dog(x)])>]
        (N[NUM='sg', SEM=<\x.([],[dog(x)])>] dog)))))
        
>>> print(trees[0].label()['SEM'].simplify())
([x,z2],[Angus(x), dog(z2), own(x,z2)])
 
 


##NLTK-Discourse Processing

#Whereas a discourse is a sequence s1, ... sn of sentences, 
#a discourse thread is a sequence s1-ri, ... sn-rj of readings, 
#one for each sentence in the discourse. 

#The nltk.inference.discourse processes sentences incrementally, 
#keeping track of all possible threads when there is ambiguity.

>>> dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
>>> dt.readings()
s0 readings:
s0-r0: exists x.(student(x) & dance(x))
s1 readings:
s1-r0: all x.(student(x) -> person(x))
 
 

#When a new sentence is added to the current discourse, 
#setting the parameter consistchk=True causes consistency to be checked by invoking the model checker for each thread, 
#i.e., sequence of admissible readings. 

#In this case, the user has the option of retracting the sentence in question.

>>> dt.add_sentence('No person dances', consistchk=True)
Inconsistent discourse: d0 ['s0-r0', 's1-r0', 's2-r0']:
    s0-r0: exists x.(student(x) & dance(x))
    s1-r0: all x.(student(x) -> person(x))
    s2-r0: -exists x.(person(x) & dance(x))
 
>>> dt.retract_sentence('No person dances', verbose=True)
Current sentences are
s0: A student dances
s1: Every student is a person
 
 

#we use informchk=True to check whether a new sentence f is informative relative to the current discourse. 

#The theorem prover treats existing sentences in the thread as assumptions 
#and attempts to prove f; it is informative if no such proof can be found.

>>> dt.add_sentence('A person dances', informchk=True)
Sentence 'A person dances' under reading 'exists x.(person(x) & dance(x))':
Not informative relative to thread 'd0'
 
 
#The discourse module can accommodate semantic ambiguity 
#and filter out readings that are not admissible. 

#The following example invokes both Glue Semantics as well as DRT. 
#Since the Glue Semantics module is configured to use the wide-coverage Malt dependency parser, 
#the input (Every dog chases a boy. He runs.) needs to be tagged as well as tokenized.


>>> from nltk.tag import RegexpTagger
>>> tagger = RegexpTagger(
        [('^(chases|runs)$', 'VB'),
        ('^(a)$', 'ex_quant'),
        ('^(every)$', 'univ_quant'),
        ('^(dog|boy)$', 'NN'),
        ('^(He)$', 'PRP')
    ])
>>> rc = nltk.DrtGlueReadingCommand(depparser=nltk.MaltParser(tagger=tagger))
>>> dt = nltk.DiscourseTester(['Every dog chases a boy', 'He runs'], rc)
>>> dt.readings()

s0 readings:

s0-r0: ([],[(([x],[dog(x)]) -> ([z3],[boy(z3), chases(x,z3)]))])
s0-r1: ([z4],[boy(z4), (([x],[dog(x)]) -> ([],[chases(x,z4)]))])

s1 readings:

s1-r0: ([x],[PRO(x), runs(x)])
 
 
#The first sentence of the discourse has two possible readings, 
#depending on the quantfier scoping. 
#The unique reading of the second sentence represents the pronoun 
#He via the condition PRO(x)`. Now let's look at the discourse threads that result:

>>> dt.readings(show_thread_readings=True)
d0: ['s0-r0', 's1-r0'] : INVALID: AnaphoraResolutionException
d1: ['s0-r1', 's1-r0'] : ([z6,z10],[boy(z6), (([x],[dog(x)]) ->
([],[chases(x,z6)])), (z10 = z6), runs(z10)])
 
 
#Inadmissible readings can be filtered out by passing the parameter filter=True.
>>> dt.readings(show_thread_readings=True, filter=True)
d1: ['s0-r1', 's1-r0'] : ([z12,z15],[boy(z12), (([x],[dog(x)]) ->
([],[chases(x,z12)])), (z17 = z12), runs(z15)])
 
 



###NLTK - Examples of Semantics
  
>>> import nltk
>>> from nltk.sem import Valuation, Model
>>> v = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'),
    ('girl', set(['g1', 'g2'])), ('boy', set(['b1', 'b2'])),
    ('dog', set(['d1'])),
    ('love', set([('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')]))]
    
    
#Valuation: Keys are strings representing the constants to be interpreted, 
#and values correspond to individuals (represented as strings) and n-ary relations (represented as sets of tuples of strings)
>>> val = Valuation(v)  #Bases: dict 
>>> dom = val.domain #{'g1', 'b1', 'd1', 'b2', 'g2'}

#Model class 
class nltk.sem.evaluate.Model(domain, valuation)
    Bases: object
    A first order model is a domain D of discourse and a valuation V.
    A domain D is a set, and a valuation V is a map that associates expressions 
    with values in the model. The domain of V should be a subset of D.
    domain (set) – A set of entities representing the domain of discourse of the model.
    valuation (Valuation) – the valuation of the model.
    prop – If this is set, then we are building a propositional model and don’t require the domain of V to be subset of D.
 
    evaluate(expr, g, trace=None)
    Returns bool or ‘Undefined’

    i(parsed, g, trace=False)
    An interpretation function.
    Returns:a semantic value
     
    satisfiers(parsed, varex, g, trace=None, nesting=0)
    Generate the entities from the model’s domain that satisfy an open formula.
    Returns:a set of the entities that satisfy parsed.
     
    satisfy(parsed, g, trace=None)
    Recursive interpretation function for a formula of first-order logic.
    Returns: Returns a truth value or Undefined if parsed is complex, and calls the interpretation function i if parsed is atomic.
        
    Parameters:
    expr : string of open formula 
    parsed (Expression) – an open formula , parsed using nltk.sem.Expression.fromstring
    varex (VariableExpression or str) – the relevant free individual variable in parsed.
    g (Assignment) – a variable assignment
    
    
     
>>> m = Model(dom, val)
#Evaluation
>>> dom = val.domain
#Assignment(domain, assign=None), Bases: dict
#A variable Assignment is a mapping from individual variables to entities in the domain
>>> g = nltk.sem.Assignment(dom)
>>> m.evaluate('all x.(boy(x) -> - girl(x))', g)
True
#evaluate() calls a recursive function satisfy(), 
#which in turn calls a function i() to interpret non-logical constants and individual variables. 
#i() delegates the interpretation of these to the the model's Valuation and the variable assignment g respectively. 

>>> m.evaluate('walk(adam)', g, trace=2)
<BLANKLINE>
'walk(adam)' is undefined under M, g
'Undefined'



##Batch Processing

#use interpret_sents() and evaluate_sents() are for  multiple sentences. 

>>> sents = ['Mary walks']
>>> results = nltk.sem.util.interpret_sents(sents, 'grammars/sample_grammars/sem2.fcfg')
>>> for result in results:
        for (synrep, semrep) in result:
            print(synrep)
(S[SEM=<walk(mary)>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(mary)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(mary)>] Mary))
  (VP[NUM='sg', SEM=<\x.walk(x)>]
    (IV[NUM='sg', SEM=<\x.walk(x)>, TNS='pres'] walks)))


#backwards compatibility with 'legacy' grammars 
#where the semantics value is specified with a lowercase sem feature, 
#the relevant feature name can be passed to the function using the semkey parameter

>>> sents = ['raining']
>>> g = nltk.grammar.FeatureGrammar.fromstring("""
    % start S
    S[sem=<raining>] -> 'raining'
    """)
>>> results = nltk.sem.util.interpret_sents(sents, g, semkey='sem')
>>> for result in results:
        for (synrep, semrep) in result:
            print(semrep)
raining



##tests for relations and valuations

>>> from nltk.sem import *

#Relations are sets of tuples, all of the same length.

>>> s1 = set([('d1', 'd2'), ('d1', 'd1'), ('d2', 'd1')])
>>> is_rel(s1)
True
>>> s2 = set([('d1', 'd2'), ('d1', 'd2'), ('d1',)])
>>> is_rel(s2)
Traceback (most recent call last):
  . . .
ValueError: Set set([('d1', 'd2'), ('d1',)]) contains sequences of different lengths
>>> s3 = set(['d1', 'd2'])
>>> is_rel(s3)
Traceback (most recent call last):
  . . .
ValueError: Set set(['d2', 'd1']) contains sequences of different lengths
>>> s4 = set2rel(s3)
>>> is_rel(s4)
True
>>> is_rel(set())
True
>>> null_binary_rel = set([(None, None)])
>>> is_rel(null_binary_rel)
True


#Sets of entities are converted into sets of singleton tuples (containing strings).

>>> sorted(set2rel(s3))
[('d1',), ('d2',)]
>>> sorted(set2rel(set([1,3,5,])))
['1', '3', '5']
>>> set2rel(set()) == set()
True
>>> set2rel(set2rel(s3)) == set2rel(s3)
True


#Predication is evaluated by set membership.

>>> ('d1', 'd2') in s1
True
>>> ('d2', 'd2') in s1
False
>>> ('d1',) in s1
False
>>> 'd2' in s1
False
>>> ('d1',) in s4
True
>>> ('d1',) in set()
False
>>> 'd1' in  null_binary_rel
False
#Valuation is key vs relation or individual
>>> val = Valuation([('Fido', 'd1'), ('dog', set(['d1', 'd2'])), ('walk', set())])
>>> sorted(val['dog'])
[('d1',), ('d2',)]
>>> val.domain == set(['d1', 'd2'])
True
>>> print(val.symbols)
['Fido', 'dog', 'walk']


#Parse a valuation from a string.

>>> v = """
    john => b1
    mary => g1
    suzie => g2
    fido => d1
    tess => d2
    noosa => n
    girl => {g1, g2}
    boy => {b1, b2}
    dog => {d1, d2}
    bark => {d1, d2}
    walk => {b1, g2, d1}
    chase => {(b1, g1), (b2, g1), (g1, d1), (g2, d2)}
    see => {(b1, g1), (b2, d2), (g1, b1),(d2, b1), (g2, n)}
    in => {(b1, n), (b2, n), (d2, n)}
    with => {(b1, g1), (g1, b1), (d1, b1), (b1, d1)}
    """
>>> val = Valuation.fromstring(v)

>>> print(val) # doctest: +SKIP
{'bark': set([('d1',), ('d2',)]),
 'boy': set([('b1',), ('b2',)]),
 'chase': set([('b1', 'g1'), ('g2', 'd2'), ('g1', 'd1'), ('b2', 'g1')]),
 'dog': set([('d1',), ('d2',)]),
 'fido': 'd1',
 'girl': set([('g2',), ('g1',)]),
 'in': set([('d2', 'n'), ('b1', 'n'), ('b2', 'n')]),
 'john': 'b1',
 'mary': 'g1',
 'noosa': 'n',
 'see': set([('b1', 'g1'), ('b2', 'd2'), ('d2', 'b1'), ('g2', 'n'), ('g1', 'b1')]),
 'suzie': 'g2',
 'tess': 'd2',
 'walk': set([('d1',), ('b1',), ('g2',)]),
 'with': set([('b1', 'g1'), ('d1', 'b1'), ('b1', 'd1'), ('g1', 'b1')])}



#Model is (domain, Valuation)
#Assignment is (domain, assignment_as_dict_key_as_variableName)
>>> v = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'),\
        ('girl', set(['g1', 'g2'])), ('boy', set(['b1', 'b2'])), ('dog', set(['d1'])),
        ('love', set([('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')])),
        ('kiss', null_binary_rel)]
>>> val = Valuation(v)
>>> dom = val.domain
>>> m = Model(dom, val)
>>> g = Assignment(dom)
>>> sorted(val['boy'])
[('b1',), ('b2',)]
>>> ('b1',) in val['boy']
True
>>> ('g1',) in val['boy']
False
>>> ('foo',) in val['boy']
False
>>> ('b1', 'g1') in val['love']
True
>>> ('b1', 'b1') in val['kiss']
False
>>> sorted(val.domain)
['b1', 'b2', 'd1', 'g1', 'g2']



#Model Tests

Extension of Lambda expressions

>>> v0 = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'),\
        ('girl', set(['g1', 'g2'])), ('boy', set(['b1', 'b2'])),
        ('dog', set(['d1'])),
        ('love', set([('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')]))]

>>> val0 = Valuation(v0)
>>> dom0 = val0.domain
>>> m0 = Model(dom0, val0)
>>> g0 = Assignment(dom0)

>>> print(m0.evaluate(r'\x. \y. love(x, y)', g0) == {'g2': {'g2': False, 'b2': False, 'b1': True, 'g1': False, 'd1': False}, 'b2': {'g2': True, 'b2': False, 'b1': False, 'g1': False, 'd1': False}, 'b1': {'g2': False, 'b2': False, 'b1': False, 'g1': True, 'd1': False}, 'g1': {'g2': False, 'b2': False, 'b1': True, 'g1': False, 'd1': False}, 'd1': {'g2': False, 'b2': False, 'b1': False, 'g1': False, 'd1': False}})
True
>>> print(m0.evaluate(r'\x. dog(x) (adam)', g0))
False
>>> print(m0.evaluate(r'\x. (dog(x) | boy(x)) (adam)', g0))
True
>>> print(m0.evaluate(r'\x. \y. love(x, y)(fido)', g0) == {'g2': False, 'b2': False, 'b1': False, 'g1': False, 'd1': False})
True
>>> print(m0.evaluate(r'\x. \y. love(x, y)(adam)', g0) == {'g2': False, 'b2': False, 'b1': False, 'g1': True, 'd1': False})
True
>>> print(m0.evaluate(r'\x. \y. love(x, y)(betty)', g0) == {'g2': False, 'b2': False, 'b1': True, 'g1': False, 'd1': False})
True
>>> print(m0.evaluate(r'\x. \y. love(x, y)(betty)(adam)', g0))
True
>>> print(m0.evaluate(r'\x. \y. love(x, y)(betty, adam)', g0))
True
>>> print(m0.evaluate(r'\y. \x. love(x, y)(fido)(adam)', g0))
False
>>> print(m0.evaluate(r'\y. \x. love(x, y)(betty, adam)', g0))
True
>>> print(m0.evaluate(r'\x. exists y. love(x, y)', g0) == {'g2': True, 'b2': True, 'b1': True, 'g1': True, 'd1': False})
True
>>> print(m0.evaluate(r'\z. adam', g0) == {'g2': 'b1', 'b2': 'b1', 'b1': 'b1', 'g1': 'b1', 'd1': 'b1'})
True
>>> print(m0.evaluate(r'\z. love(x, y)', g0) == {'g2': False, 'b2': False, 'b1': False, 'g1': False, 'd1': False})
True



#Propositional Model Test

>>> tests = [
        ('P & Q', True),
        ('P & R', False),
        ('- P', False),
        ('- R', True),
        ('- - P', True),
        ('- (P & R)', True),
        ('P | R', True),
        ('R | P', True),
        ('R | R', False),
        ('- P | R', False),
        ('P | - P', True),
        ('P -> Q', True),
        ('P -> R', False),
        ('R -> P', True),
        ('P <-> P', True),
        ('R <-> R', True),
        ('P <-> R', False),
        ]
>>> val1 = Valuation([('P', True), ('Q', True), ('R', False)])
>>> dom = set([])
>>> m = Model(dom, val1)
>>> g = Assignment(dom)
>>> for (sent, testvalue) in tests:
        semvalue = m.evaluate(sent, g)
        if semvalue == testvalue:
            print('*', end=' ')
* * * * * * * * * * * * * * * * *



#Test of i Function

>>> from nltk.sem import Expression
>>> v = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'),
        ('girl', set(['g1', 'g2'])), ('boy', set(['b1', 'b2'])), ('dog', set(['d1'])),
        ('love', set([('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')]))]
>>> val = Valuation(v)
>>> dom = val.domain
>>> m = Model(dom, val)
>>> g = Assignment(dom, [('x', 'b1'), ('y', 'g2')])
>>> exprs = ['adam', 'girl', 'love', 'walks', 'x', 'y', 'z']
>>> parsed_exprs = [Expression.fromstring(e) for e in exprs]
>>> sorted_set = lambda x: sorted(x) if isinstance(x, set) else x
>>> for parsed in parsed_exprs:
        try:
            print("'%s' gets value %s" % (parsed, sorted_set(m.i(parsed, g))))
        except Undefined:
            print("'%s' is Undefined" % parsed)
'adam' gets value b1
'girl' gets value [('g1',), ('g2',)]
'love' gets value [('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')]
'walks' is Undefined
'x' gets value b1
'y' gets value g2
'z' is Undefined



#Test for formulas in Model

>>> tests = [
        ('love(adam, betty)', True),
        ('love(adam, sue)', 'Undefined'),
        ('dog(fido)', True),
        ('- dog(fido)', False),
        ('- - dog(fido)', True),
        ('- dog(sue)', 'Undefined'),
        ('dog(fido) & boy(adam)', True),
        ('- (dog(fido) & boy(adam))', False),
        ('- dog(fido) & boy(adam)', False),
        ('dog(fido) | boy(adam)', True),
        ('- (dog(fido) | boy(adam))', False),
        ('- dog(fido) | boy(adam)', True),
        ('- dog(fido) | - boy(adam)', False),
        ('dog(fido) -> boy(adam)', True),
        ('- (dog(fido) -> boy(adam))', False),
        ('- dog(fido) -> boy(adam)', True),
        ('exists x . love(adam, x)', True),
        ('all x . love(adam, x)', False),
        ('fido = fido', True),
        ('exists x . all y. love(x, y)', False),
        ('exists x . (x = fido)', True),
        ('all x . (dog(x) | - dog(x))', True),
        ('adam = mia', 'Undefined'),
        ('\\x. (boy(x) | girl(x))', {'g2': True, 'b2': True, 'b1': True, 'g1': True, 'd1': False}),
        ('\\x. exists y. (boy(x) & love(x, y))', {'g2': False, 'b2': True, 'b1': True, 'g1': False, 'd1': False}),
        ('exists z1. boy(z1)', True),
        ('exists x. (boy(x) & - (x = adam))', True),
        ('exists x. (boy(x) & all y. love(y, x))', False),
        ('all x. (boy(x) | girl(x))', False),
        ('all x. (girl(x) -> exists y. boy(y) & love(x, y))', False),
        ('exists x. (boy(x) & all y. (girl(y) -> love(y, x)))', True),
        ('exists x. (boy(x) & all y. (girl(y) -> love(x, y)))', False),
        ('all x. (dog(x) -> - girl(x))', True),
        ('exists x. exists y. (love(x, y) & love(x, y))', True),
        ]
>>> for (sent, testvalue) in tests:
        semvalue = m.evaluate(sent, g)
        if semvalue == testvalue:
            print('*', end=' ')
        else:
            print(sent, semvalue)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



##Satisfier Tests

>>> formulas = [
        'boy(x)',
        '(x = x)',
        '(boy(x) | girl(x))',
        '(boy(x) & girl(x))',
        'love(adam, x)',
        'love(x, adam)',
        '- (x = adam)',
        'exists z22. love(x, z22)',
        'exists y. love(y, x)',
        'all y. (girl(y) -> love(x, y))',
        'all y. (girl(y) -> love(y, x))',
        'all y. (girl(y) -> (boy(x) & love(y, x)))',
        'boy(x) & all y. (girl(y) -> love(x, y))',
        'boy(x) & all y. (girl(y) -> love(y, x))',
        'boy(x) & exists y. (girl(y) & love(y, x))',
        'girl(x) -> dog(x)',
        'all y. (dog(y) -> (x = y))',
        '- exists y. love(y, x)',
        'exists y. (love(adam, y) & love(y, x))'
        ]
>>> g.purge()
>>> g.add('x', 'b1')
{'x': 'b1'}
>>> for f in formulas: # doctest: +NORMALIZE_WHITESPACE
        try:
            print("'%s' gets value: %s" % (f, m.evaluate(f, g)))
        except Undefined:
            print("'%s' is Undefined" % f)
'boy(x)' gets value: True
'(x = x)' gets value: True
'(boy(x) | girl(x))' gets value: True
'(boy(x) & girl(x))' gets value: False
'love(adam, x)' gets value: False
'love(x, adam)' gets value: False
'- (x = adam)' gets value: False
'exists z22. love(x, z22)' gets value: True
'exists y. love(y, x)' gets value: True
'all y. (girl(y) -> love(x, y))' gets value: False
'all y. (girl(y) -> love(y, x))' gets value: True
'all y. (girl(y) -> (boy(x) & love(y, x)))' gets value: True
'boy(x) & all y. (girl(y) -> love(x, y))' gets value: False
'boy(x) & all y. (girl(y) -> love(y, x))' gets value: True
'boy(x) & exists y. (girl(y) & love(y, x))' gets value: True
'girl(x) -> dog(x)' gets value: True
'all y. (dog(y) -> (x = y))' gets value: False
'- exists y. love(y, x)' gets value: False
'exists y. (love(adam, y) & love(y, x))' gets value: True

>>> from nltk.sem import Expression
>>> for fmla in formulas: # doctest: +NORMALIZE_WHITESPACE
        p = Expression.fromstring(fmla)
        g.purge()
        print("Satisfiers of '%s':\n\t%s" % (p, sorted(m.satisfiers(p, 'x', g))))
Satisfiers of 'boy(x)':
['b1', 'b2']
Satisfiers of '(x = x)':
['b1', 'b2', 'd1', 'g1', 'g2']
Satisfiers of '(boy(x) | girl(x))':
['b1', 'b2', 'g1', 'g2']
Satisfiers of '(boy(x) & girl(x))':
[]
Satisfiers of 'love(adam,x)':
['g1']
Satisfiers of 'love(x,adam)':
['g1', 'g2']
Satisfiers of '-(x = adam)':
['b2', 'd1', 'g1', 'g2']
Satisfiers of 'exists z22.love(x,z22)':
['b1', 'b2', 'g1', 'g2']
Satisfiers of 'exists y.love(y,x)':
['b1', 'g1', 'g2']
Satisfiers of 'all y.(girl(y) -> love(x,y))':
[]
Satisfiers of 'all y.(girl(y) -> love(y,x))':
['b1']
Satisfiers of 'all y.(girl(y) -> (boy(x) & love(y,x)))':
['b1']
Satisfiers of '(boy(x) & all y.(girl(y) -> love(x,y)))':
[]
Satisfiers of '(boy(x) & all y.(girl(y) -> love(y,x)))':
['b1']
Satisfiers of '(boy(x) & exists y.(girl(y) & love(y,x)))':
['b1']
Satisfiers of '(girl(x) -> dog(x))':
['b1', 'b2', 'd1']
Satisfiers of 'all y.(dog(y) -> (x = y))':
['d1']
Satisfiers of '-exists y.love(y,x)':
['b2', 'd1']
Satisfiers of 'exists y.(love(adam,y) & love(y,x))':
['b1']



#Tests based on the Blackburn & Bos testsuite

>>> v1 = [('jules', 'd1'), ('vincent', 'd2'), ('pumpkin', 'd3'),
            ('honey_bunny', 'd4'), ('yolanda', 'd5'),
            ('customer', set(['d1', 'd2'])),
            ('robber', set(['d3', 'd4'])),
            ('love', set([('d3', 'd4')]))]
>>> val1 = Valuation(v1)
>>> dom1 = val1.domain
>>> m1 = Model(dom1, val1)
>>> g1 = Assignment(dom1)

>>> v2 = [('jules', 'd1'), ('vincent', 'd2'), ('pumpkin', 'd3'),
            ('honey_bunny', 'd4'), ('yolanda', 'd4'),
            ('customer', set(['d1', 'd2', 'd5', 'd6'])),
            ('robber', set(['d3', 'd4'])),
            ('love', set([(None, None)]))]
>>> val2 = Valuation(v2)
>>> dom2 = set(['d1', 'd2', 'd3', 'd4', 'd5', 'd6'])
>>> m2 = Model(dom2, val2)
>>> g2 = Assignment(dom2)
>>> g21 = Assignment(dom2)
>>> g21.add('y', 'd3')
{'y': 'd3'}

>>> v3 = [('mia', 'd1'), ('jody', 'd2'), ('jules', 'd3'),
            ('vincent', 'd4'),
            ('woman', set(['d1', 'd2'])), ('man', set(['d3', 'd4'])),
            ('joke', set(['d5', 'd6'])), ('episode', set(['d7', 'd8'])),
            ('in', set([('d5', 'd7'), ('d5', 'd8')])),
            ('tell', set([('d1', 'd5'), ('d2', 'd6')]))]
>>> val3 = Valuation(v3)
>>> dom3 = set(['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'])
>>> m3 = Model(dom3, val3)
>>> g3 = Assignment(dom3)

>>> tests = [
        ('exists x. robber(x)', m1, g1, True),
        ('exists x. exists y. love(y, x)', m1, g1, True),
        ('exists x0. exists x1. love(x1, x0)', m2, g2, False),
        ('all x. all y. love(y, x)', m2, g2, False),
        ('- (all x. all y. love(y, x))', m2, g2, True),
        ('all x. all y. - love(y, x)', m2, g2, True),
        ('yolanda = honey_bunny', m2, g2, True),
        ('mia = honey_bunny', m2, g2, 'Undefined'),
        ('- (yolanda = honey_bunny)', m2, g2, False),
        ('- (mia = honey_bunny)', m2, g2, 'Undefined'),
        ('all x. (robber(x) | customer(x))', m2, g2, True),
        ('- (all x. (robber(x) | customer(x)))', m2, g2, False),
        ('(robber(x) | customer(x))', m2, g2, 'Undefined'),
        ('(robber(y) | customer(y))', m2, g21, True),
        ('exists x. (man(x) & exists x. woman(x))', m3, g3, True),
        ('exists x. (man(x) & exists x. woman(x))', m3, g3, True),
        ('- exists x. woman(x)', m3, g3, False),
        ('exists x. (tasty(x) & burger(x))', m3, g3, 'Undefined'),
        ('- exists x. (tasty(x) & burger(x))', m3, g3, 'Undefined'),
        ('exists x. (man(x) & - exists y. woman(y))', m3, g3, False),
        ('exists x. (man(x) & - exists x. woman(x))', m3, g3, False),
        ('exists x. (woman(x) & - exists x. customer(x))', m2, g2, 'Undefined'),
    ]

>>> for item in tests:
        sentence, model, g, testvalue = item
        semvalue = model.evaluate(sentence, g)
        if semvalue == testvalue:
            print('*', end=' ')
        g.purge()
* * * * * * * * * * * * * * * * * * * * * *



#Tests for mapping from syntax to semantics
#Load a valuation from a file.

>>> import nltk.data
>>> from nltk.sem.util import parse_sents
>>> val = nltk.data.load('grammars/sample_grammars/valuation1.val')
>>> dom = val.domain
>>> m = Model(dom, val)
>>> g = Assignment(dom)
>>> gramfile = 'grammars/sample_grammars/sem2.fcfg'
>>> inputs = ['John sees a girl', 'every dog barks']
>>> parses = parse_sents(inputs, gramfile)
>>> for sent, trees in zip(inputs, parses):
        print()
        print("Sentence: %s" % sent)
        for tree in trees:
            print("Parse:\n %s" %tree)
            print("Semantics: %s" %  root_semrep(tree))
<BLANKLINE>
Sentence: John sees a girl
Parse:
 (S[SEM=<exists x.(girl(x) & see(john,x))>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(john)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(john)>] John))
  (VP[NUM='sg', SEM=<\y.exists x.(girl(x) & see(y,x))>]
    (TV[NUM='sg', SEM=<\X y.X(\x.see(y,x))>, TNS='pres'] sees)
    (NP[NUM='sg', SEM=<\Q.exists x.(girl(x) & Q(x))>]
      (Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] a)
      (Nom[NUM='sg', SEM=<\x.girl(x)>]
        (N[NUM='sg', SEM=<\x.girl(x)>] girl)))))
Semantics: exists x.(girl(x) & see(john,x))
<BLANKLINE>
Sentence: every dog barks
Parse:
 (S[SEM=<all x.(dog(x) -> bark(x))>]
  (NP[NUM='sg', SEM=<\Q.all x.(dog(x) -> Q(x))>]
    (Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] every)
    (Nom[NUM='sg', SEM=<\x.dog(x)>]
      (N[NUM='sg', SEM=<\x.dog(x)>] dog)))
  (VP[NUM='sg', SEM=<\x.bark(x)>]
    (IV[NUM='sg', SEM=<\x.bark(x)>, TNS='pres'] barks)))
Semantics: all x.(dog(x) -> bark(x))

>>> sent = "every dog barks"
>>> result = nltk.sem.util.interpret_sents([sent], gramfile)[0]
>>> for (syntree, semrep) in result:
        print(syntree)
        print()
        print(semrep)
(S[SEM=<all x.(dog(x) -> bark(x))>]
  (NP[NUM='sg', SEM=<\Q.all x.(dog(x) -> Q(x))>]
    (Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] every)
    (Nom[NUM='sg', SEM=<\x.dog(x)>]
      (N[NUM='sg', SEM=<\x.dog(x)>] dog)))
  (VP[NUM='sg', SEM=<\x.bark(x)>]
    (IV[NUM='sg', SEM=<\x.bark(x)>, TNS='pres'] barks)))
<BLANKLINE>
all x.(dog(x) -> bark(x))

>>> result = nltk.sem.util.evaluate_sents([sent], gramfile, m, g)[0]
>>> for (syntree, semrel, value) in result:
        print(syntree)
        print()
        print(semrep)
        print()
        print(value)
(S[SEM=<all x.(dog(x) -> bark(x))>]
  (NP[NUM='sg', SEM=<\Q.all x.(dog(x) -> Q(x))>]
    (Det[NUM='sg', SEM=<\P Q.all x.(P(x) -> Q(x))>] every)
    (Nom[NUM='sg', SEM=<\x.dog(x)>]
      (N[NUM='sg', SEM=<\x.dog(x)>] dog)))
  (VP[NUM='sg', SEM=<\x.bark(x)>]
    (IV[NUM='sg', SEM=<\x.bark(x)>, TNS='pres'] barks)))
<BLANKLINE>
all x.(dog(x) -> bark(x))
<BLANKLINE>
True

>>> sents = ['Mary walks', 'John sees a dog']
>>> results = nltk.sem.util.interpret_sents(sents, 'grammars/sample_grammars/sem2.fcfg')
>>> for result in results:
        for (synrep, semrep) in result:
            print(synrep)
(S[SEM=<walk(mary)>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(mary)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(mary)>] Mary))
  (VP[NUM='sg', SEM=<\x.walk(x)>]
    (IV[NUM='sg', SEM=<\x.walk(x)>, TNS='pres'] walks)))
(S[SEM=<exists x.(dog(x) & see(john,x))>]
  (NP[-LOC, NUM='sg', SEM=<\P.P(john)>]
    (PropN[-LOC, NUM='sg', SEM=<\P.P(john)>] John))
  (VP[NUM='sg', SEM=<\y.exists x.(dog(x) & see(y,x))>]
    (TV[NUM='sg', SEM=<\X y.X(\x.see(y,x))>, TNS='pres'] sees)
    (NP[NUM='sg', SEM=<\Q.exists x.(dog(x) & Q(x))>]
      (Det[NUM='sg', SEM=<\P Q.exists x.(P(x) & Q(x))>] a)
      (Nom[NUM='sg', SEM=<\x.dog(x)>]
        (N[NUM='sg', SEM=<\x.dog(x)>] dog)))))



##Cooper Storage

>>> from nltk.sem import cooper_storage as cs
>>> sentence = 'every girl chases a dog'
>>> trees = cs.parse_with_bindops(sentence, grammar='grammars/book_grammars/storage.fcfg')
>>> semrep = trees[0].label()['SEM']
>>> cs_semrep = cs.CooperStore(semrep)
>>> print(cs_semrep.core)
chase(z2,z4)
>>> for bo in cs_semrep.store:
        print(bo)
bo(\P.all x.(girl(x) -> P(x)),z2)
bo(\P.exists x.(dog(x) & P(x)),z4)
>>> cs_semrep.s_retrieve(trace=True)
Permutation 1
   (\P.all x.(girl(x) -> P(x)))(\z2.chase(z2,z4))
   (\P.exists x.(dog(x) & P(x)))(\z4.all x.(girl(x) -> chase(x,z4)))
Permutation 2
   (\P.exists x.(dog(x) & P(x)))(\z4.chase(z2,z4))
   (\P.all x.(girl(x) -> P(x)))(\z2.exists x.(dog(x) & chase(z2,x)))

>>> for reading in cs_semrep.readings:
        print(reading)
exists x.(dog(x) & all z3.(girl(z3) -> chase(z3,x)))
all x.(girl(x) -> exists z4.(dog(z4) & chase(x,z4)))



###Example - Logic & Lambda Calculus
  
#The nltk.logic package allows expressions of First-Order Logic (FOL) to be parsed into Expression objects. 
#In addition to FOL, the parser handles lambda-abstraction with variables of higher order.




>>> from nltk.sem.logic import *


#The default inventory of logical constants is the following:

>>> boolean_ops() # doctest: +NORMALIZE_WHITESPACE
negation           -
conjunction        &
disjunction        |
implication        ->
equivalence        <->
>>> equality_preds() # doctest: +NORMALIZE_WHITESPACE
equality           =
inequality         !=
>>> binding_ops() # doctest: +NORMALIZE_WHITESPACE
existential        exists
universal          all
lambda             \



#Untyped Logic

>>> read_expr = Expression.fromstring



#Test for equality under alpha-conversion

>>> e1 = read_expr('exists x.P(x)')
>>> print(e1)
exists x.P(x)
>>> e2 = e1.alpha_convert(Variable('z'))
>>> print(e2)
exists z.P(z)
>>> e1 == e2
True

>>> l = read_expr(r'\X.\X.X(X)(1)').simplify()
>>> id = read_expr(r'\X.X(X)')
>>> l == id
True



#Test numerals

>>> zero = read_expr(r'\F x.x')
>>> one = read_expr(r'\F x.F(x)')
>>> two = read_expr(r'\F x.F(F(x))')
>>> three = read_expr(r'\F x.F(F(F(x)))')
>>> four = read_expr(r'\F x.F(F(F(F(x))))')
>>> succ = read_expr(r'\N F x.F(N(F,x))')
>>> plus = read_expr(r'\M N F x.M(F,N(F,x))')
>>> mult = read_expr(r'\M N F.M(N(F))')
>>> pred = read_expr(r'\N F x.(N(\G H.H(G(F)))(\u.x)(\u.u))')
>>> v1 = ApplicationExpression(succ, zero).simplify()
>>> v1 == one
True
>>> v2 = ApplicationExpression(succ, v1).simplify()
>>> v2 == two
True
>>> v3 = ApplicationExpression(ApplicationExpression(plus, v1), v2).simplify()
>>> v3 == three
True
>>> v4 = ApplicationExpression(ApplicationExpression(mult, v2), v2).simplify()
>>> v4 == four
True
>>> v5 = ApplicationExpression(pred, ApplicationExpression(pred, v4)).simplify()
>>> v5 == two
True


#Overloaded operators also exist, for convenience.

>>> print(succ(zero).simplify() == one)
True
>>> print(plus(one,two).simplify() == three)
True
>>> print(mult(two,two).simplify() == four)
True
>>> print(pred(pred(four)).simplify() == two)
True

>>> john = read_expr(r'john')
>>> man = read_expr(r'\x.man(x)')
>>> walk = read_expr(r'\x.walk(x)')
>>> man(john).simplify()
<ApplicationExpression man(john)>
>>> print(-walk(john).simplify())
-walk(john)
>>> print((man(john) & walk(john)).simplify())
(man(john) & walk(john))
>>> print((man(john) | walk(john)).simplify())
(man(john) | walk(john))
>>> print((man(john) > walk(john)).simplify())
(man(john) -> walk(john))
>>> print((man(john) < walk(john)).simplify())
(man(john) <-> walk(john))


#Python's built-in lambda operator can also be used with Expressions

>>> john = VariableExpression(Variable('john'))
>>> run_var = VariableExpression(Variable('run'))
>>> run = lambda x: run_var(x)
>>> run(john)
<ApplicationExpression run(john)>



#Tests based on Blackburn & Bos' book, Representation and Inference for Natural Language.

>>> x1 = read_expr(r'\P.P(mia)(\x.walk(x))').simplify()
>>> x2 = read_expr(r'walk(mia)').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'exists x.(man(x) & ((\P.exists x.(woman(x) & P(x)))(\y.love(x,y))))').simplify()
>>> x2 = read_expr(r'exists x.(man(x) & exists y.(woman(y) & love(x,y)))').simplify()
>>> x1 == x2
True
>>> x1 = read_expr(r'\a.sleep(a)(mia)').simplify()
>>> x2 = read_expr(r'sleep(mia)').simplify()
>>> x1 == x2
True
>>> x1 = read_expr(r'\a.\b.like(b,a)(mia)').simplify()
>>> x2 = read_expr(r'\b.like(b,mia)').simplify()
>>> x1 == x2
True
>>> x1 = read_expr(r'\a.(\b.like(b,a)(vincent))').simplify()
>>> x2 = read_expr(r'\a.like(vincent,a)').simplify()
>>> x1 == x2
True
>>> x1 = read_expr(r'\a.((\b.like(b,a)(vincent)) & sleep(a))').simplify()
>>> x2 = read_expr(r'\a.(like(vincent,a) & sleep(a))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\a.\b.like(b,a)(mia)(vincent))').simplify()
>>> x2 = read_expr(r'like(vincent,mia)').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'P((\a.sleep(a)(vincent)))').simplify()
>>> x2 = read_expr(r'P(sleep(vincent))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.A((\b.sleep(b)(vincent)))').simplify()
>>> x2 = read_expr(r'\A.A(sleep(vincent))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.A(sleep(vincent))').simplify()
>>> x2 = read_expr(r'\A.A(sleep(vincent))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\A.A(vincent)(\b.sleep(b)))').simplify()
>>> x2 = read_expr(r'sleep(vincent)').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.believe(mia,A(vincent))(\b.sleep(b))').simplify()
>>> x2 = read_expr(r'believe(mia,sleep(vincent))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\A.(A(vincent) & A(mia)))(\b.sleep(b))').simplify()
>>> x2 = read_expr(r'(sleep(vincent) & sleep(mia))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.\B.(\C.C(A(vincent))(\d.probably(d)) & (\C.C(B(mia))(\d.improbably(d))))(\f.walk(f))(\f.talk(f))').simplify()
>>> x2 = read_expr(r'(probably(walk(vincent)) & improbably(talk(mia)))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\a.\b.(\C.C(a,b)(\d.\f.love(d,f))))(jules)(mia)').simplify()
>>> x2 = read_expr(r'love(jules,mia)').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\A.\B.exists c.(A(c) & B(c)))(\d.boxer(d),\d.sleep(d))').simplify()
>>> x2 = read_expr(r'exists c.(boxer(c) & sleep(c))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.Z(A)(\c.\a.like(a,c))').simplify()
>>> x2 = read_expr(r'Z(\c.\a.like(a,c))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'\A.\b.A(b)(\c.\b.like(b,c))').simplify()
>>> x2 = read_expr(r'\b.(\c.\b.like(b,c)(b))').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\a.\b.(\C.C(a,b)(\b.\a.loves(b,a))))(jules)(mia)').simplify()
>>> x2 = read_expr(r'loves(jules,mia)').simplify()
>>> x1 == x2
True

>>> x1 = read_expr(r'(\A.\b.(exists b.A(b) & A(b)))(\c.boxer(c))(vincent)').simplify()
>>> x2 = read_expr(r'((exists b.boxer(b)) & boxer(vincent))').simplify()
>>> x1 == x2
True



#Test Parser

>>> print(read_expr(r'john'))
john
>>> print(read_expr(r'x'))
x
>>> print(read_expr(r'-man(x)'))
-man(x)
>>> print(read_expr(r'--man(x)'))
--man(x)
>>> print(read_expr(r'(man(x))'))
man(x)
>>> print(read_expr(r'((man(x)))'))
man(x)
>>> print(read_expr(r'man(x) <-> tall(x)'))
(man(x) <-> tall(x))
>>> print(read_expr(r'(man(x) <-> tall(x))'))
(man(x) <-> tall(x))
>>> print(read_expr(r'(man(x) & tall(x) & walks(x))'))
(man(x) & tall(x) & walks(x))
>>> print(read_expr(r'(man(x) & tall(x) & walks(x))').first)
(man(x) & tall(x))
>>> print(read_expr(r'man(x) | tall(x) & walks(x)'))
(man(x) | (tall(x) & walks(x)))
>>> print(read_expr(r'((man(x) & tall(x)) | walks(x))'))
((man(x) & tall(x)) | walks(x))
>>> print(read_expr(r'man(x) & (tall(x) | walks(x))'))
(man(x) & (tall(x) | walks(x)))
>>> print(read_expr(r'(man(x) & (tall(x) | walks(x)))'))
(man(x) & (tall(x) | walks(x)))
>>> print(read_expr(r'P(x) -> Q(x) <-> R(x) | S(x) & T(x)'))
((P(x) -> Q(x)) <-> (R(x) | (S(x) & T(x))))
>>> print(read_expr(r'exists x.man(x)'))
exists x.man(x)
>>> print(read_expr(r'exists x.(man(x) & tall(x))'))
exists x.(man(x) & tall(x))
>>> print(read_expr(r'exists x.(man(x) & tall(x) & walks(x))'))
exists x.(man(x) & tall(x) & walks(x))
>>> print(read_expr(r'-P(x) & Q(x)'))
(-P(x) & Q(x))
>>> read_expr(r'-P(x) & Q(x)') == read_expr(r'(-P(x)) & Q(x)')
True
>>> print(read_expr(r'\x.man(x)'))
\x.man(x)
>>> print(read_expr(r'\x.man(x)(john)'))
\x.man(x)(john)
>>> print(read_expr(r'\x.man(x)(john) & tall(x)'))
(\x.man(x)(john) & tall(x))
>>> print(read_expr(r'\x.\y.sees(x,y)'))
\x y.sees(x,y)
>>> print(read_expr(r'\x  y.sees(x,y)'))
\x y.sees(x,y)
>>> print(read_expr(r'\x.\y.sees(x,y)(a)'))
(\x y.sees(x,y))(a)
>>> print(read_expr(r'\x  y.sees(x,y)(a)'))
(\x y.sees(x,y))(a)
>>> print(read_expr(r'\x.\y.sees(x,y)(a)(b)'))
((\x y.sees(x,y))(a))(b)
>>> print(read_expr(r'\x  y.sees(x,y)(a)(b)'))
((\x y.sees(x,y))(a))(b)
>>> print(read_expr(r'\x.\y.sees(x,y)(a,b)'))
((\x y.sees(x,y))(a))(b)
>>> print(read_expr(r'\x  y.sees(x,y)(a,b)'))
((\x y.sees(x,y))(a))(b)
>>> print(read_expr(r'((\x.\y.sees(x,y))(a))(b)'))
((\x y.sees(x,y))(a))(b)
>>> print(read_expr(r'P(x)(y)(z)'))
P(x,y,z)
>>> print(read_expr(r'P(Q)'))
P(Q)
>>> print(read_expr(r'P(Q(x))'))
P(Q(x))
>>> print(read_expr(r'(\x.exists y.walks(x,y))(x)'))
(\x.exists y.walks(x,y))(x)
>>> print(read_expr(r'exists x.(x = john)'))
exists x.(x = john)
>>> print(read_expr(r'((\P.\Q.exists x.(P(x) & Q(x)))(\x.dog(x)))(\x.bark(x))'))
((\P Q.exists x.(P(x) & Q(x)))(\x.dog(x)))(\x.bark(x))
>>> a = read_expr(r'exists c.exists b.A(b,c) & A(b,c)')
>>> b = read_expr(r'(exists c.(exists b.A(b,c))) & A(b,c)')
>>> print(a == b)
True
>>> a = read_expr(r'exists c.(exists b.A(b,c) & A(b,c))')
>>> b = read_expr(r'exists c.((exists b.A(b,c)) & A(b,c))')
>>> print(a == b)
True
>>> print(read_expr(r'exists x.x = y'))
exists x.(x = y)
>>> print(read_expr('A(B)(C)'))
A(B,C)
>>> print(read_expr('(A(B))(C)'))
A(B,C)
>>> print(read_expr('A((B)(C))'))
A(B(C))
>>> print(read_expr('A(B(C))'))
A(B(C))
>>> print(read_expr('(A)(B(C))'))
A(B(C))
>>> print(read_expr('(((A)))(((B))(((C))))'))
A(B(C))
>>> print(read_expr(r'A != B'))
-(A = B)
>>> print(read_expr('P(x) & x=y & P(y)'))
(P(x) & (x = y) & P(y))
>>> try: print(read_expr(r'\walk.walk(x)'))
... except LogicalExpressionException as e: print(e)
'walk' is an illegal variable name.  Constants may not be abstracted.
\walk.walk(x)
 ^
>>> try: print(read_expr(r'all walk.walk(john)'))
... except LogicalExpressionException as e: print(e)
'walk' is an illegal variable name.  Constants may not be quantified.
all walk.walk(john)
    ^
>>> try: print(read_expr(r'x(john)'))
... except LogicalExpressionException as e: print(e)
'x' is an illegal predicate name.  Individual variables may not be used as predicates.
x(john)
^

>>> from nltk.sem.logic import LogicParser # hack to give access to custom quote chars
>>> lpq = LogicParser()
>>> lpq.quote_chars = [("'", "'", "\\", False)]
>>> print(lpq.parse(r"(man(x) & 'tall\'s,' (x) & walks (x) )"))
(man(x) & tall's,(x) & walks(x))
>>> lpq.quote_chars = [("'", "'", "\\", True)]
>>> print(lpq.parse(r"'tall\'s,'"))
'tall\'s,'
>>> print(lpq.parse(r"'spaced name(x)'"))
'spaced name(x)'
>>> print(lpq.parse(r"-'tall\'s,'(x)"))
-'tall\'s,'(x)
>>> print(lpq.parse(r"(man(x) & 'tall\'s,' (x) & walks (x) )"))
(man(x) & 'tall\'s,'(x) & walks(x))



#Simplify

>>> print(read_expr(r'\x.man(x)(john)').simplify())
man(john)
>>> print(read_expr(r'\x.((man(x)))(john)').simplify())
man(john)
>>> print(read_expr(r'\x.\y.sees(x,y)(john, mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'\x  y.sees(x,y)(john, mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'\x.\y.sees(x,y)(john)(mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'\x  y.sees(x,y)(john)(mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'\x.\y.sees(x,y)(john)').simplify())
\y.sees(john,y)
>>> print(read_expr(r'\x  y.sees(x,y)(john)').simplify())
\y.sees(john,y)
>>> print(read_expr(r'(\x.\y.sees(x,y)(john))(mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'(\x  y.sees(x,y)(john))(mary)').simplify())
sees(john,mary)
>>> print(read_expr(r'exists x.(man(x) & (\x.exists y.walks(x,y))(x))').simplify())
exists x.(man(x) & exists y.walks(x,y))
>>> e1 = read_expr(r'exists x.(man(x) & (\x.exists y.walks(x,y))(y))').simplify()
>>> e2 = read_expr(r'exists x.(man(x) & exists z1.walks(y,z1))')
>>> e1 == e2
True
>>> print(read_expr(r'(\P Q.exists x.(P(x) & Q(x)))(\x.dog(x))').simplify())
\Q.exists x.(dog(x) & Q(x))
>>> print(read_expr(r'((\P.\Q.exists x.(P(x) & Q(x)))(\x.dog(x)))(\x.bark(x))').simplify())
exists x.(dog(x) & bark(x))
>>> print(read_expr(r'\P.(P(x)(y))(\a b.Q(a,b))').simplify())
Q(x,y)



#Replace

>>> a = read_expr(r'a')
>>> x = read_expr(r'x')
>>> y = read_expr(r'y')
>>> z = read_expr(r'z')

>>> print(read_expr(r'man(x)').replace(x.variable, a, False))
man(a)
>>> print(read_expr(r'(man(x) & tall(x))').replace(x.variable, a, False))
(man(a) & tall(a))
>>> print(read_expr(r'exists x.man(x)').replace(x.variable, a, False))
exists x.man(x)
>>> print(read_expr(r'exists x.man(x)').replace(x.variable, a, True))
exists a.man(a)
>>> print(read_expr(r'exists x.give(x,y,z)').replace(y.variable, a, False))
exists x.give(x,a,z)
>>> print(read_expr(r'exists x.give(x,y,z)').replace(y.variable, a, True))
exists x.give(x,a,z)
>>> e1 = read_expr(r'exists x.give(x,y,z)').replace(y.variable, x, False)
>>> e2 = read_expr(r'exists z1.give(z1,x,z)')
>>> e1 == e2
True
>>> e1 = read_expr(r'exists x.give(x,y,z)').replace(y.variable, x, True)
>>> e2 = read_expr(r'exists z1.give(z1,x,z)')
>>> e1 == e2
True
>>> print(read_expr(r'\x y z.give(x,y,z)').replace(y.variable, a, False))
\x y z.give(x,y,z)
>>> print(read_expr(r'\x y z.give(x,y,z)').replace(y.variable, a, True))
\x a z.give(x,a,z)
>>> print(read_expr(r'\x.\y.give(x,y,z)').replace(z.variable, a, False))
\x y.give(x,y,a)
>>> print(read_expr(r'\x.\y.give(x,y,z)').replace(z.variable, a, True))
\x y.give(x,y,a)
>>> e1 = read_expr(r'\x.\y.give(x,y,z)').replace(z.variable, x, False)
>>> e2 = read_expr(r'\z1.\y.give(z1,y,x)')
>>> e1 == e2
True
>>> e1 = read_expr(r'\x.\y.give(x,y,z)').replace(z.variable, x, True)
>>> e2 = read_expr(r'\z1.\y.give(z1,y,x)')
>>> e1 == e2
True
>>> print(read_expr(r'\x.give(x,y,z)').replace(z.variable, y, False))
\x.give(x,y,y)
>>> print(read_expr(r'\x.give(x,y,z)').replace(z.variable, y, True))
\x.give(x,y,y)

>>> from nltk.sem import logic
>>> logic._counter._value = 0
>>> e1 = read_expr('e1')
>>> e2 = read_expr('e2')
>>> print(read_expr('exists e1 e2.(walk(e1) & talk(e2))').replace(e1.variable, e2, True))
exists e2 e01.(walk(e2) & talk(e01))



#Variables / Free

>>> examples = [r'walk(john)',
                r'walk(x)',
                r'?vp(?np)',
                r'see(john,mary)',
                r'exists x.walk(x)',
                r'\x.see(john,x)',
                r'\x.see(john,x)(mary)',
                r'P(x)',
                r'\P.P(x)',
                r'aa(x,bb(y),cc(z),P(w),u)',
                r'bo(?det(?n),@x)']
>>> examples = [read_expr(e) for e in examples]

>>> for e in examples:
        print('%-25s' % e, sorted(e.free()))
walk(john)                []
walk(x)                   [Variable('x')]
?vp(?np)                  []
see(john,mary)            []
exists x.walk(x)          []
\x.see(john,x)            []
(\x.see(john,x))(mary)    []
P(x)                      [Variable('P'), Variable('x')]
\P.P(x)                   [Variable('x')]
aa(x,bb(y),cc(z),P(w),u)  [Variable('P'), Variable('u'), Variable('w'), Variable('x'), Variable('y'), Variable('z')]
bo(?det(?n),@x)           []

>>> for e in examples:
        print('%-25s' % e, sorted(e.constants()))
walk(john)                [Variable('john')]
walk(x)                   []
?vp(?np)                  [Variable('?np')]
see(john,mary)            [Variable('john'), Variable('mary')]
exists x.walk(x)          []
\x.see(john,x)            [Variable('john')]
(\x.see(john,x))(mary)    [Variable('john'), Variable('mary')]
P(x)                      []
\P.P(x)                   []
aa(x,bb(y),cc(z),P(w),u)  []
bo(?det(?n),@x)           [Variable('?n'), Variable('@x')]

>>> for e in examples:
        print('%-25s' % e, sorted(e.predicates()))
walk(john)                [Variable('walk')]
walk(x)                   [Variable('walk')]
?vp(?np)                  [Variable('?vp')]
see(john,mary)            [Variable('see')]
exists x.walk(x)          [Variable('walk')]
\x.see(john,x)            [Variable('see')]
(\x.see(john,x))(mary)    [Variable('see')]
P(x)                      []
\P.P(x)                   []
aa(x,bb(y),cc(z),P(w),u)  [Variable('aa'), Variable('bb'), Variable('cc')]
bo(?det(?n),@x)           [Variable('?det'), Variable('bo')]

>>> for e in examples:
        print('%-25s' % e, sorted(e.variables()))
walk(john)                []
walk(x)                   [Variable('x')]
?vp(?np)                  [Variable('?np'), Variable('?vp')]
see(john,mary)            []
exists x.walk(x)          []
\x.see(john,x)            []
(\x.see(john,x))(mary)    []
P(x)                      [Variable('P'), Variable('x')]
\P.P(x)                   [Variable('x')]
aa(x,bb(y),cc(z),P(w),u)  [Variable('P'), Variable('u'), Variable('w'), Variable('x'), Variable('y'), Variable('z')]
bo(?det(?n),@x)           [Variable('?det'), Variable('?n'), Variable('@x')]

normalize>>> print(read_expr(r'\e083.(walk(e083, z472) & talk(e092, z938))').normalize())
\e01.(walk(e01,z3) & talk(e02,z4))



##Typed Logic

>>> from nltk.sem.logic import LogicParser
>>> tlp = LogicParser(True)
>>> print(tlp.parse(r'man(x)').type)
?
>>> print(tlp.parse(r'walk(angus)').type)
?
>>> print(tlp.parse(r'-man(x)').type)
t
>>> print(tlp.parse(r'(man(x) <-> tall(x))').type)
t
>>> print(tlp.parse(r'exists x.(man(x) & tall(x))').type)
t
>>> print(tlp.parse(r'\x.man(x)').type)
<e,?>
>>> print(tlp.parse(r'john').type)
e
>>> print(tlp.parse(r'\x y.sees(x,y)').type)
<e,<e,?>>
>>> print(tlp.parse(r'\x.man(x)(john)').type)
?
>>> print(tlp.parse(r'\x.\y.sees(x,y)(john)').type)
<e,?>
>>> print(tlp.parse(r'\x.\y.sees(x,y)(john)(mary)').type)
?
>>> print(tlp.parse(r'\P.\Q.exists x.(P(x) & Q(x))').type)
<<e,t>,<<e,t>,t>>
>>> print(tlp.parse(r'\x.y').type)
<?,e>
>>> print(tlp.parse(r'\P.P(x)').type)
<<e,?>,?>

>>> parsed = tlp.parse('see(john,mary)')
>>> print(parsed.type)
?
>>> print(parsed.function)
see(john)
>>> print(parsed.function.type)
<e,?>
>>> print(parsed.function.function)
see
>>> print(parsed.function.function.type)
<e,<e,?>>

>>> parsed = tlp.parse('P(x,y)')
>>> print(parsed)
P(x,y)
>>> print(parsed.type)
?
>>> print(parsed.function)
P(x)
>>> print(parsed.function.type)
<e,?>
>>> print(parsed.function.function)
P
>>> print(parsed.function.function.type)
<e,<e,?>>

>>> print(tlp.parse(r'P').type)
?

>>> print(tlp.parse(r'P', {'P': 't'}).type)
t

>>> a = tlp.parse(r'P(x)')
>>> print(a.type)
?
>>> print(a.function.type)
<e,?>
>>> print(a.argument.type)
e

>>> a = tlp.parse(r'-P(x)')
>>> print(a.type)
t
>>> print(a.term.type)
t
>>> print(a.term.function.type)
<e,t>
>>> print(a.term.argument.type)
e

>>> a = tlp.parse(r'P & Q')
>>> print(a.type)
t
>>> print(a.first.type)
t
>>> print(a.second.type)
t

>>> a = tlp.parse(r'(P(x) & Q(x))')
>>> print(a.type)
t
>>> print(a.first.type)
t
>>> print(a.first.function.type)
<e,t>
>>> print(a.first.argument.type)
e
>>> print(a.second.type)
t
>>> print(a.second.function.type)
<e,t>
>>> print(a.second.argument.type)
e

>>> a = tlp.parse(r'\x.P(x)')
>>> print(a.type)
<e,?>
>>> print(a.term.function.type)
<e,?>
>>> print(a.term.argument.type)
e

>>> a = tlp.parse(r'\P.P(x)')
>>> print(a.type)
<<e,?>,?>
>>> print(a.term.function.type)
<e,?>
>>> print(a.term.argument.type)
e

>>> a = tlp.parse(r'(\x.P(x)(john)) & Q(x)')
>>> print(a.type)
t
>>> print(a.first.type)
t
>>> print(a.first.function.type)
<e,t>
>>> print(a.first.function.term.function.type)
<e,t>
>>> print(a.first.function.term.argument.type)
e
>>> print(a.first.argument.type)
e

>>> a = tlp.parse(r'\x y.P(x,y)(john)(mary) & Q(x)')
>>> print(a.type)
t
>>> print(a.first.type)
t
>>> print(a.first.function.type)
<e,t>
>>> print(a.first.function.function.type)
<e,<e,t>>

>>> a = tlp.parse(r'--P')
>>> print(a.type)
t
>>> print(a.term.type)
t
>>> print(a.term.term.type)
t

>>> tlp.parse(r'\x y.P(x,y)').type
<e,<e,?>>
>>> tlp.parse(r'\x y.P(x,y)', {'P': '<e,<e,t>>'}).type
<e,<e,t>>

>>> a = tlp.parse(r'\P y.P(john,y)(\x y.see(x,y))')
>>> a.type
<e,?>
>>> a.function.type
<<e,<e,?>>,<e,?>>
>>> a.function.term.term.function.function.type
<e,<e,?>>
>>> a.argument.type
<e,<e,?>>

>>> a = tlp.parse(r'exists c f.(father(c) = f)')
>>> a.type
t
>>> a.term.term.type
t
>>> a.term.term.first.type
e
>>> a.term.term.first.function.type
<e,e>
>>> a.term.term.second.type
e


##typecheck()

>>> a = tlp.parse('P(x)')
>>> b = tlp.parse('Q(x)')
>>> a.type
?
>>> c = a & b
>>> c.first.type
?
>>> c.typecheck() # doctest: +ELLIPSIS
{...}
>>> c.first.type
t

>>> a = tlp.parse('P(x)')
>>> b = tlp.parse('P(x) & Q(x)')
>>> a.type
?
>>> typecheck([a,b]) # doctest: +ELLIPSIS
{...}
>>> a.type
t

>>> e = tlp.parse(r'man(x)')
>>> print(dict((k,str(v)) for k,v in e.typecheck().items()) == {'x': 'e', 'man': '<e,?>'})
True
>>> sig = {'man': '<e, t>'}
>>> e = tlp.parse(r'man(x)', sig)
>>> print(e.function.type)
<e,t>
>>> print(dict((k,str(v)) for k,v in e.typecheck().items()) == {'x': 'e', 'man': '<e,t>'})
True
>>> print(e.function.type)
<e,t>
>>> print(dict((k,str(v)) for k,v in e.typecheck(sig).items()) == {'x': 'e', 'man': '<e,t>'})
True


#findtype()

>>> print(tlp.parse(r'man(x)').findtype(Variable('man')))
<e,?>
>>> print(tlp.parse(r'see(x,y)').findtype(Variable('see')))
<e,<e,?>>
>>> print(tlp.parse(r'P(Q(R(x)))').findtype(Variable('Q')))
?


#reading types from strings

>>> Type.fromstring('e')
e
>>> Type.fromstring('<e,t>')
<e,t>
>>> Type.fromstring('<<e,t>,<e,t>>')
<<e,t>,<e,t>>
>>> Type.fromstring('<<e,?>,?>')
<<e,?>,?>


#alternative type format

>>> Type.fromstring('e').str()
'IND'
>>> Type.fromstring('<e,?>').str()
'(IND -> ANY)'
>>> Type.fromstring('<<e,t>,t>').str()
'((IND -> BOOL) -> BOOL)'


#Type.__eq__()

>>> from nltk.sem.logic import *

>>> e = ENTITY_TYPE
>>> t = TRUTH_TYPE
>>> a = ANY_TYPE
>>> et = ComplexType(e,t)
>>> eet = ComplexType(e,ComplexType(e,t))
>>> at = ComplexType(a,t)
>>> ea = ComplexType(e,a)
>>> aa = ComplexType(a,a)

>>> e == e
True
>>> t == t
True
>>> e == t
False
>>> a == t
False
>>> t == a
False
>>> a == a
True
>>> et == et
True
>>> a == et
False
>>> et == a
False
>>> a == ComplexType(a,aa)
True
>>> ComplexType(a,aa) == a
True


#matches()

>>> e.matches(t)
False
>>> a.matches(t)
True
>>> t.matches(a)
True
>>> a.matches(et)
True
>>> et.matches(a)
True
>>> ea.matches(eet)
True
>>> eet.matches(ea)
True
>>> aa.matches(et)
True
>>> aa.matches(t)
True



#Type error during parsing

>>> try: print(tlp.parse(r'exists x y.(P(x) & P(x,y))'))
    except InconsistentTypeHierarchyException as e: print(e)
The variable 'P' was found in multiple places with different types.
>>> try: tlp.parse(r'\x y.see(x,y)(\x.man(x))')
    except TypeException as e: print(e)
The function '\x y.see(x,y)' is of type '<e,<e,?>>' and cannot be applied to '\x.man(x)' of type '<e,?>'.  Its argument must match type 'e'.
>>> try: tlp.parse(r'\P x y.-P(x,y)(\x.-man(x))')
    except TypeException as e: print(e)
The function '\P x y.-P(x,y)' is of type '<<e,<e,t>>,<e,<e,t>>>' and cannot be applied to '\x.-man(x)' of type '<e,t>'.  Its argument must match type '<e,<e,t>>'.

>>> a = tlp.parse(r'-talk(x)')
>>> signature = a.typecheck()
>>> try: print(tlp.parse(r'-talk(x,y)', signature))
    except InconsistentTypeHierarchyException as e: print(e)
The variable 'talk' was found in multiple places with different types.

>>> a = tlp.parse(r'-P(x)')
>>> b = tlp.parse(r'-P(x,y)')
>>> a.typecheck() # doctest: +ELLIPSIS
{...}
>>> b.typecheck() # doctest: +ELLIPSIS
{...}
>>> try: typecheck([a,b])
    except InconsistentTypeHierarchyException as e: print(e)
The variable 'P' was found in multiple places with different types.

>>> a = tlp.parse(r'P(x)')
>>> b = tlp.parse(r'P(x,y)')
>>> signature = {'P': '<e,t>'}
>>> a.typecheck(signature) # doctest: +ELLIPSIS
{...}
>>> try: typecheck([a,b], signature)
    except InconsistentTypeHierarchyException as e: print(e)
The variable 'P' was found in multiple places with different types.


###Examples - Discourse Checking
  
>>> from nltk import *
>>> from nltk.sem import logic
>>> logic._counter._value = 0



#The DiscourseTester constructor takes a list of sentences as a parameter.

>>> dt = DiscourseTester(['a boxer walks', 'every boxer chases a girl'])

>>> dt.sentences()
s0: a boxer walks
s1: every boxer chases a girl

>>> dt.grammar() 
% start S
# Grammar Rules
S[SEM = <app(?subj,?vp)>] -> NP[NUM=?n,SEM=?subj] VP[NUM=?n,SEM=?vp]
NP[NUM=?n,SEM=<app(?det,?nom)> ] -> Det[NUM=?n,SEM=?det]  Nom[NUM=?n,SEM=?nom]
NP[LOC=?l,NUM=?n,SEM=?np] -> PropN[LOC=?l,NUM=?n,SEM=?np]
...


#Given a sentence identifier of the form si, each reading of that sentence is given an identifier si-rj.

>>> dt.readings()
<BLANKLINE>
s0 readings:
<BLANKLINE>
s0-r0: exists z1.(boxer(z1) & walk(z1))
s0-r1: exists z1.(boxerdog(z1) & walk(z1))
<BLANKLINE>
s1 readings:
<BLANKLINE>
s1-r0: all z2.(boxer(z2) -> exists z3.(girl(z3) & chase(z2,z3)))
s1-r1: all z1.(boxerdog(z1) -> exists z2.(girl(z2) & chase(z1,z2)))


#In this case, the only source of ambiguity lies in the word boxer, 
#which receives two translations: boxer and boxerdog. 

#We can also investigate the readings of a specific sentence:

>>> dt.readings('a boxer walks')
The sentence 'a boxer walks' has these readings:
    exists x.(boxer(x) & walk(x))
    exists x.(boxerdog(x) & walk(x))


#Given that each sentence is two-ways ambiguous, 
#we potentially have four different discourse 'threads', 
#taking all combinations of readings. 

#each thread is assigned an identifier of the form di. 
#Following the identifier is a list of the readings that constitute that thread.

>>> dt.readings(threaded=True) 
d0: ['s0-r0', 's1-r0']
d1: ['s0-r0', 's1-r1']
d2: ['s0-r1', 's1-r0']
d3: ['s0-r1', 's1-r1']


#Checking Consistency
#With no parameter, this method will try to find a model for every discourse thread in the current discourse. 
#However, we can also specify just one thread, say d1.

>>> dt.models('d1')
--------------------------------------------------------------------------------
Model for Discourse Thread d1
--------------------------------------------------------------------------------
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
c1 = 0.
<BLANKLINE>
f1(0) = 0.
f1(1) = 0.
<BLANKLINE>
  boxer(0).
- boxer(1).
<BLANKLINE>
- boxerdog(0).
- boxerdog(1).
<BLANKLINE>
- girl(0).
- girl(1).
<BLANKLINE>
  walk(0).
- walk(1).
<BLANKLINE>
- chase(0,0).
- chase(0,1).
- chase(1,0).
- chase(1,1).
<BLANKLINE>
Consistent discourse: d1 ['s0-r0', 's1-r1']:
    s0-r0: exists z1.(boxer(z1) & walk(z1))
    s1-r1: all z1.(boxerdog(z1) -> exists z2.(girl(z2) & chase(z1,z2)))
<BLANKLINE>


#There are various formats for rendering Mace4 models --- 
#here, we have used the 'cooked' format (which is intended to be human-readable). 
#There are a number of points to note.
1.The entities in the domain are all treated as non-negative integers. In this case, there are only two entities, 0 and 1.
2.The - symbol indicates negation. So 0 is the only boxerdog and the only thing that walks. Nothing is a boxer, or a girl or in the chase relation. Thus the universal sentence is vacuously true.
3.c1 is an introduced constant that denotes 0.
4.f1 is a Skolem function, but it plays no significant role in this model.

#We might want to now add another sentence to the discourse, 
#and there is method add_sentence() for doing just this.

>>> dt.add_sentence('John is a boxer')
>>> dt.sentences()
s0: a boxer walks
s1: every boxer chases a girl
s2: John is a boxer


>>> dt.readings()
<BLANKLINE>
s0 readings:
<BLANKLINE>
s0-r0: exists z1.(boxer(z1) & walk(z1))
s0-r1: exists z1.(boxerdog(z1) & walk(z1))
<BLANKLINE>
s1 readings:
<BLANKLINE>
s1-r0: all z1.(boxer(z1) -> exists z2.(girl(z2) & chase(z1,z2)))
s1-r1: all z1.(boxerdog(z1) -> exists z2.(girl(z2) & chase(z1,z2)))
<BLANKLINE>
s2 readings:
<BLANKLINE>
s2-r0: boxer(John)
s2-r1: boxerdog(John)
>>> dt.readings(threaded=True) # doctest: +NORMALIZE_WHITESPACE
d0: ['s0-r0', 's1-r0', 's2-r0']
d1: ['s0-r0', 's1-r0', 's2-r1']
d2: ['s0-r0', 's1-r1', 's2-r0']
d3: ['s0-r0', 's1-r1', 's2-r1']
d4: ['s0-r1', 's1-r0', 's2-r0']
d5: ['s0-r1', 's1-r0', 's2-r1']
d6: ['s0-r1', 's1-r1', 's2-r0']
d7: ['s0-r1', 's1-r1', 's2-r1']



>>> thread = dt.expand_threads('d1')
>>> for rid, reading in thread:
        print(rid, str(reading.normalize()))
s0-r0 exists z1.(boxer(z1) & walk(z1))
s1-r0 all z1.(boxer(z1) -> exists z2.(girl(z2) & chase(z1,z2)))
s2-r1 boxerdog(John)


##Suppose we have already defined a discourse, as follows:

>>> dt = DiscourseTester(['A student dances', 'Every student is a person'])


#consistchk=True parameter of add_sentence() allows us to check:

>>> dt.add_sentence('No person dances', consistchk=True)
Inconsistent discourse: d0 ['s0-r0', 's1-r0', 's2-r0']:
    s0-r0: exists z1.(student(z1) & dance(z1))
    s1-r0: all z1.(student(z1) -> person(z1))
    s2-r0: -exists z1.(person(z1) & dance(z1))
<BLANKLINE>
>>> dt.readings()
<BLANKLINE>
s0 readings:
<BLANKLINE>
s0-r0: exists z1.(student(z1) & dance(z1))
<BLANKLINE>
s1 readings:
<BLANKLINE>
s1-r0: all z1.(student(z1) -> person(z1))
<BLANKLINE>
s2 readings:
<BLANKLINE>
s2-r0: -exists z1.(person(z1) & dance(z1))


#let's retract the inconsistent sentence:

>>> dt.retract_sentence('No person dances', verbose=True) # doctest: +NORMALIZE_WHITESPACE
Current sentences are
s0: A student dances
s1: Every student is a person


#We can now verify that result is consistent.

>>> dt.models()
--------------------------------------------------------------------------------
Model for Discourse Thread d0
--------------------------------------------------------------------------------
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
c1 = 0.
<BLANKLINE>
  dance(0).
- dance(1).
<BLANKLINE>
  person(0).
- person(1).
<BLANKLINE>
  student(0).
- student(1).
<BLANKLINE>
Consistent discourse: d0 ['s0-r0', 's1-r0']:
    s0-r0: exists z1.(student(z1) & dance(z1))
    s1-r0: all z1.(student(z1) -> person(z1))
<BLANKLINE>



##Checking Informativity

#we check whether it is informative with respect to what has gone before.

>>> dt.add_sentence('A person dances', informchk=True)
Sentence 'A person dances' under reading 'exists x.(person(x) & dance(x))':
Not informative relative to thread 'd0'


# we are just checking whether the new sentence is entailed by the preceding discourse.

>>> dt.models()
--------------------------------------------------------------------------------
Model for Discourse Thread d0
--------------------------------------------------------------------------------
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
c1 = 0.
<BLANKLINE>
c2 = 0.
<BLANKLINE>
  dance(0).
- dance(1).
<BLANKLINE>
  person(0).
- person(1).
<BLANKLINE>
  student(0).
- student(1).
<BLANKLINE>
Consistent discourse: d0 ['s0-r0', 's1-r0', 's2-r0']:
    s0-r0: exists z1.(student(z1) & dance(z1))
    s1-r0: all z1.(student(z1) -> person(z1))
    s2-r0: exists z1.(person(z1) & dance(z1))
<BLANKLINE>



##Adding Background Knowledge

#Let's build a new discourse, and look at the readings of the component sentences:

>>> dt = DiscourseTester(['Vincent is a boxer', 'Fido is a boxer', 'Vincent is married', 'Fido barks'])
>>> dt.readings()
<BLANKLINE>
s0 readings:
<BLANKLINE>
s0-r0: boxer(Vincent)
s0-r1: boxerdog(Vincent)
<BLANKLINE>
s1 readings:
<BLANKLINE>
s1-r0: boxer(Fido)
s1-r1: boxerdog(Fido)
<BLANKLINE>
s2 readings:
<BLANKLINE>
s2-r0: married(Vincent)
<BLANKLINE>
s3 readings:
<BLANKLINE>
s3-r0: bark(Fido)


#This gives us a lot of threads:

>>> dt.readings(threaded=True) 
d0: ['s0-r0', 's1-r0', 's2-r0', 's3-r0']
d1: ['s0-r0', 's1-r1', 's2-r0', 's3-r0']
d2: ['s0-r1', 's1-r0', 's2-r0', 's3-r0']
d3: ['s0-r1', 's1-r1', 's2-r0', 's3-r0']


#We can eliminate some of the readings, 
#and hence some of the threads, by adding background information.

>>> import nltk.data
>>> bg = nltk.data.load('grammars/book_grammars/background.fol')
>>> dt.add_background(bg)
>>> dt.background()
all x.(boxerdog(x) -> dog(x))
all x.(boxer(x) -> person(x))
all x.-(dog(x) & person(x))
all x.(married(x) <-> exists y.marry(x,y))
all x.(bark(x) -> dog(x))
all x y.(marry(x,y) -> (person(x) & person(y)))
-(Vincent = Mia)
-(Vincent = Fido)
-(Mia = Fido)


#The background information allows us to reject three of the threads as inconsistent. 
#To see what remains, use the filter=True parameter on readings().

>>> dt.readings(filter=True) 
d1: ['s0-r0', 's1-r1', 's2-r0', 's3-r0']


#The models() method gives us more information about the surviving thread.

>>> dt.models()
--------------------------------------------------------------------------------
Model for Discourse Thread d0
--------------------------------------------------------------------------------
No model found!
<BLANKLINE>
--------------------------------------------------------------------------------
Model for Discourse Thread d1
--------------------------------------------------------------------------------
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 3
<BLANKLINE>
Fido = 0.
<BLANKLINE>
Mia = 1.
<BLANKLINE>
Vincent = 2.
<BLANKLINE>
f1(0) = 0.
f1(1) = 0.
f1(2) = 2.
<BLANKLINE>
  bark(0).
- bark(1).
- bark(2).
<BLANKLINE>
- boxer(0).
- boxer(1).
  boxer(2).
<BLANKLINE>
  boxerdog(0).
- boxerdog(1).
- boxerdog(2).
<BLANKLINE>
  dog(0).
- dog(1).
- dog(2).
<BLANKLINE>
- married(0).
- married(1).
  married(2).
<BLANKLINE>
- person(0).
- person(1).
  person(2).
<BLANKLINE>
- marry(0,0).
- marry(0,1).
- marry(0,2).
- marry(1,0).
- marry(1,1).
- marry(1,2).
- marry(2,0).
- marry(2,1).
  marry(2,2).
<BLANKLINE>
--------------------------------------------------------------------------------
Model for Discourse Thread d2
--------------------------------------------------------------------------------
No model found!
<BLANKLINE>
--------------------------------------------------------------------------------
Model for Discourse Thread d3
--------------------------------------------------------------------------------
No model found!
<BLANKLINE>
Inconsistent discourse: d0 ['s0-r0', 's1-r0', 's2-r0', 's3-r0']:
    s0-r0: boxer(Vincent)
    s1-r0: boxer(Fido)
    s2-r0: married(Vincent)
    s3-r0: bark(Fido)
<BLANKLINE>
Consistent discourse: d1 ['s0-r0', 's1-r1', 's2-r0', 's3-r0']:
    s0-r0: boxer(Vincent)
    s1-r1: boxerdog(Fido)
    s2-r0: married(Vincent)
    s3-r0: bark(Fido)
<BLANKLINE>
Inconsistent discourse: d2 ['s0-r1', 's1-r0', 's2-r0', 's3-r0']:
    s0-r1: boxerdog(Vincent)
    s1-r0: boxer(Fido)
    s2-r0: married(Vincent)
    s3-r0: bark(Fido)
<BLANKLINE>
Inconsistent discourse: d3 ['s0-r1', 's1-r1', 's2-r0', 's3-r0']:
    s0-r1: boxerdog(Vincent)
    s1-r1: boxerdog(Fido)
    s2-r0: married(Vincent)
    s3-r0: bark(Fido)
<BLANKLINE>

 
#In order to play around with your own version of background knowledge, 
#you might want to start off with a local copy of background.fol:

>>> nltk.data.retrieve('grammars/book_grammars/background.fol')
Retrieving 'nltk:grammars/book_grammars/background.fol', saving to 'background.fol'


#After you have modified the file, 
#the load_fol() function will parse the strings in the file into expressions of nltk.sem.logic.

>>> from nltk.inference.discourse import load_fol
>>> mybg = load_fol(open('background.fol').read())


#The result can be loaded as an argument of add_background() in the manner shown earlier.
 

##Regression Testing from book

>>> logic._counter._value = 0

>>> from nltk.tag import RegexpTagger
>>> tagger = RegexpTagger(
            [('^(chases|runs)$', 'VB'),
            ('^(a)$', 'ex_quant'),
            ('^(every)$', 'univ_quant'),
            ('^(dog|boy)$', 'NN'),
            ('^(He)$', 'PRP')
        ])
>>> rc = DrtGlueReadingCommand(depparser=MaltParser(tagger=tagger))
>>> dt = DiscourseTester(map(str.split, ['Every dog chases a boy', 'He runs']), rc)
>>> dt.readings()
<BLANKLINE>
s0 readings:
<BLANKLINE>
s0-r0: ([z2],[boy(z2), (([z5],[dog(z5)]) -> ([],[chases(z5,z2)]))])
s0-r1: ([],[(([z1],[dog(z1)]) -> ([z2],[boy(z2), chases(z1,z2)]))])
<BLANKLINE>
s1 readings:
<BLANKLINE>
s1-r0: ([z1],[PRO(z1), runs(z1)])
>>> dt.readings(show_thread_readings=True)
d0: ['s0-r0', 's1-r0'] : ([z1,z2],[boy(z1), (([z3],[dog(z3)]) -> ([],[chases(z3,z1)])), (z2 = z1), runs(z2)])
d1: ['s0-r1', 's1-r0'] : INVALID: AnaphoraResolutionException
>>> dt.readings(filter=True, show_thread_readings=True)
d0: ['s0-r0', 's1-r0'] : ([z1,z3],[boy(z1), (([z2],[dog(z2)]) -> ([],[chases(z2,z1)])), (z3 = z1), runs(z3)])

>>> logic._counter._value = 0

>>> from nltk.parse import FeatureEarleyChartParser
>>> from nltk.sem.drt import DrtParser
>>> grammar = nltk.data.load('grammars/book_grammars/drt.fcfg', logic_parser=DrtParser())
>>> parser = FeatureEarleyChartParser(grammar, trace=0)
>>> trees = parser.parse('Angus owns a dog'.split())
>>> print(list(trees)[0].label()['SEM'].simplify().normalize())
([z1,z2],[Angus(z1), dog(z2), own(z1,z2)])






###Examples - Discourse Representation Theory
  
>>> from nltk.sem import logic
>>> from nltk.inference import TableauProver

#A DRS can be created with the DRS() constructor. 
#This takes two arguments: a list of discourse referents and list of conditions

>>> from nltk.sem.drt import *
>>> dexpr = DrtExpression.fromstring
>>> man_x = dexpr('man(x)')
>>> walk_x = dexpr('walk(x)')
>>> x = dexpr('x')
>>> print(DRS([x], [man_x, walk_x]))
([x],[man(x), walk(x)])


#The parse() method can also be applied directly to DRS expressions, 
#which allows them to be specified more easily.

>>> drs1 = dexpr('([x],[man(x),walk(x)])')
>>> print(drs1)
([x],[man(x), walk(x)])


#DRSs can be merged using the + operator.

>>> drs2 = dexpr('([y],[woman(y),stop(y)])')
>>> drs3 = drs1 + drs2
>>> print(drs3)
(([x],[man(x), walk(x)]) + ([y],[woman(y), stop(y)]))
>>> print(drs3.simplify())
([x,y],[man(x), walk(x), woman(y), stop(y)])


#We can embed DRSs as components of an implies condition.

>>> s = '([], [(%s -> %s)])' % (drs1, drs2)
>>> print(dexpr(s))
([],[(([x],[man(x), walk(x)]) -> ([y],[woman(y), stop(y)]))])


#The fol() method converts DRSs into FOL formulae.

>>> print(dexpr(r'([x],[man(x), walks(x)])').fol())
exists x.(man(x) & walks(x))
>>> print(dexpr(r'([],[(([x],[man(x)]) -> ([],[walks(x)]))])').fol())
all x.(man(x) -> walks(x))


#In order to visualize a DRS, the pretty_format() method can be used.

>>> print(drs3.pretty_format())
  _________     __________
 | x       |   | y        |
(|---------| + |----------|)
 | man(x)  |   | woman(y) |
 | walk(x) |   | stop(y)  |
 |_________|   |__________|



#DRSs can be used for building compositional semantics in a feature based grammar. 
#To specify that we want to use DRSs, the appropriate logic parser needs be passed as a parameter to load_earley()

>>> from nltk.parse import load_parser
>>> from nltk.sem.drt import DrtParser
>>> parser = load_parser('grammars/book_grammars/drt.fcfg', trace=0, logic_parser=DrtParser())
>>> for tree in parser.parse('a dog barks'.split()):
...     print(tree.label()['SEM'].simplify())
...
([x],[dog(x), bark(x)])


#Alternatively, a FeatStructReader can be passed with the logic_parser set on it

>>> from nltk.featstruct import FeatStructReader
>>> from nltk.grammar import FeatStructNonterminal
>>> parser = load_parser('grammars/book_grammars/drt.fcfg', trace=0, fstruct_reader=FeatStructReader(fdict_class=FeatStructNonterminal, logic_parser=DrtParser()))
>>> for tree in parser.parse('every girl chases a dog'.split()):
...     print(tree.label()['SEM'].simplify().normalize())
...
([],[(([z1],[girl(z1)]) -> ([z2],[dog(z2), chase(z1,z2)]))])



#Parser

>>> print(dexpr(r'([x,y],[sees(x,y)])'))
([x,y],[sees(x,y)])
>>> print(dexpr(r'([x],[man(x), walks(x)])'))
([x],[man(x), walks(x)])
>>> print(dexpr(r'\x.([],[man(x), walks(x)])'))
\x.([],[man(x), walks(x)])
>>> print(dexpr(r'\x.\y.([],[sees(x,y)])'))
\x y.([],[sees(x,y)])

>>> print(dexpr(r'([x,y],[(x = y)])'))
([x,y],[(x = y)])
>>> print(dexpr(r'([x,y],[(x != y)])'))
([x,y],[-(x = y)])





#simplify()

>>> print(dexpr(r'\x.([],[man(x), walks(x)])(john)').simplify())
([],[man(john), walks(john)])
>>> print(dexpr(r'\x.\y.([z],[dog(z),sees(x,y)])(john)(mary)').simplify())
([z],[dog(z), sees(john,mary)])
>>> print(dexpr(r'\R x.([],[big(x,R)])(\y.([],[mouse(y)]))').simplify())
\x.([],[big(x,\y.([],[mouse(y)]))])


#fol()

>>> print(dexpr(r'([x,y],[sees(x,y)])').fol())
exists x y.sees(x,y)
>>> print(dexpr(r'([x],[man(x), walks(x)])').fol())
exists x.(man(x) & walks(x))
>>> print(dexpr(r'\x.([],[man(x), walks(x)])').fol())
\x.(man(x) & walks(x))
>>> print(dexpr(r'\x y.([],[sees(x,y)])').fol())
\x y.sees(x,y)



#resolve_anaphora()

>>> from nltk.sem.drt import AnaphoraResolutionException

>>> print(resolve_anaphora(dexpr(r'([x,y,z],[dog(x), cat(y), walks(z), PRO(z)])')))
([x,y,z],[dog(x), cat(y), walks(z), (z = [x,y])])
>>> print(resolve_anaphora(dexpr(r'([],[(([x],[dog(x)]) -> ([y],[walks(y), PRO(y)]))])')))
([],[(([x],[dog(x)]) -> ([y],[walks(y), (y = x)]))])
>>> print(resolve_anaphora(dexpr(r'(([x,y],[]) + ([],[PRO(x)]))')).simplify())
([x,y],[(x = y)])
>>> try: print(resolve_anaphora(dexpr(r'([x],[walks(x), PRO(x)])')))
    except AnaphoraResolutionException as e: print(e)
Variable 'x' does not resolve to anything.
>>> print(resolve_anaphora(dexpr('([e01,z6,z7],[boy(z6), PRO(z7), run(e01), subj(e01,z7)])')))
([e01,z6,z7],[boy(z6), (z7 = z6), run(e01), subj(e01,z7)])



#equiv():

>>> a = dexpr(r'([x],[man(x), walks(x)])')
>>> b = dexpr(r'([x],[walks(x), man(x)])')
>>> print(a.equiv(b, TableauProver()))
True



#replace():

>>> a = dexpr(r'a')
>>> w = dexpr(r'w')
>>> x = dexpr(r'x')
>>> y = dexpr(r'y')
>>> z = dexpr(r'z')



#replace bound

>>> print(dexpr(r'([x],[give(x,y,z)])').replace(x.variable, a, False))
([x],[give(x,y,z)])
>>> print(dexpr(r'([x],[give(x,y,z)])').replace(x.variable, a, True))
([a],[give(a,y,z)])



#replace unbound

>>> print(dexpr(r'([x],[give(x,y,z)])').replace(y.variable, a, False))
([x],[give(x,a,z)])
>>> print(dexpr(r'([x],[give(x,y,z)])').replace(y.variable, a, True))
([x],[give(x,a,z)])



#replace unbound with bound

>>> dexpr(r'([x],[give(x,y,z)])').replace(y.variable, x, False) == \
... dexpr('([z1],[give(z1,x,z)])')
True
>>> dexpr(r'([x],[give(x,y,z)])').replace(y.variable, x, True) == \
... dexpr('([z1],[give(z1,x,z)])')
True



#replace unbound with unbound

>>> print(dexpr(r'([x],[give(x,y,z)])').replace(y.variable, z, False))
([x],[give(x,z,z)])
>>> print(dexpr(r'([x],[give(x,y,z)])').replace(y.variable, z, True))
([x],[give(x,z,z)])



#replace unbound

>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(z.variable, a, False))
(([x],[P(x,y,a)]) + ([y],[Q(x,y,a)]))
>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(z.variable, a, True))
(([x],[P(x,y,a)]) + ([y],[Q(x,y,a)]))



#replace bound

>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(x.variable, a, False))
(([x],[P(x,y,z)]) + ([y],[Q(x,y,z)]))
>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(x.variable, a, True))
(([a],[P(a,y,z)]) + ([y],[Q(a,y,z)]))



#replace unbound with unbound

>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(z.variable, a, False))
(([x],[P(x,y,a)]) + ([y],[Q(x,y,a)]))
>>> print(dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,z)])').replace(z.variable, a, True))
(([x],[P(x,y,a)]) + ([y],[Q(x,y,a)]))



#replace unbound with bound on same side

>>> dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,w)])').replace(z.variable, x, False) == \
    dexpr(r'(([z1],[P(z1,y,x)]) + ([y],[Q(z1,y,w)]))')
True
>>> dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,w)])').replace(z.variable, x, True) == \
    dexpr(r'(([z1],[P(z1,y,x)]) + ([y],[Q(z1,y,w)]))')
True



#replace unbound with bound on other side

>>> dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,w)])').replace(w.variable, x, False) == \
    dexpr(r'(([z1],[P(z1,y,z)]) + ([y],[Q(z1,y,x)]))')
True
>>> dexpr(r'([x],[P(x,y,z)])+([y],[Q(x,y,w)])').replace(w.variable, x, True) == \
    dexpr(r'(([z1],[P(z1,y,z)]) + ([y],[Q(z1,y,x)]))')
True



#replace unbound with double bound

>>> dexpr(r'([x],[P(x,y,z)])+([x],[Q(x,y,w)])').replace(z.variable, x, False) == \
    dexpr(r'(([z1],[P(z1,y,x)]) + ([z1],[Q(z1,y,w)]))')
True
>>> dexpr(r'([x],[P(x,y,z)])+([x],[Q(x,y,w)])').replace(z.variable, x, True) == \
    dexpr(r'(([z1],[P(z1,y,x)]) + ([z1],[Q(z1,y,w)]))')
True



#regression tests

>>> d = dexpr('([x],[A(c), ([y],[B(x,y,z,a)])->([z],[C(x,y,z,a)])])')
>>> print(d)
([x],[A(c), (([y],[B(x,y,z,a)]) -> ([z],[C(x,y,z,a)]))])
>>> print(d.pretty_format())
 ____________________________________
| x                                  |
|------------------------------------|
| A(c)                               |
|   ____________      ____________   |
|  | y          |    | z          |  |
| (|------------| -> |------------|) |
|  | B(x,y,z,a) |    | C(x,y,z,a) |  |
|  |____________|    |____________|  |
|____________________________________|
>>> print(str(d))
([x],[A(c), (([y],[B(x,y,z,a)]) -> ([z],[C(x,y,z,a)]))])
>>> print(d.fol())
exists x.(A(c) & all y.(B(x,y,z,a) -> exists z.C(x,y,z,a)))
>>> print(d.replace(Variable('a'), DrtVariableExpression(Variable('r'))))
([x],[A(c), (([y],[B(x,y,z,r)]) -> ([z],[C(x,y,z,r)]))])


#Parse errors

>>> def parse_error(drtstring):
        try: dexpr(drtstring)
        except logic.LogicalExpressionException as e: print(e)

>>> parse_error(r'')
End of input found.  Expression expected.
<BLANKLINE>
^

#Pretty Printing

>>> dexpr(r"([],[])").pretty_print()
 __
|  |
|--|
|__|

>>> dexpr(r"([],[([x],[big(x), dog(x)]) -> ([],[bark(x)]) -([x],[walk(x)])])").pretty_print()
 _____________________________
|                             |
|-----------------------------|
|   ________      _________   |
|  | x      |    |         |  |
| (|--------| -> |---------|) |
|  | big(x) |    | bark(x) |  |
|  | dog(x) |    |_________|  |
|  |________|                 |
|      _________              |
|     | x       |             |
| __  |---------|             |
|   | | walk(x) |             |
|     |_________|             |
|_____________________________|

>>> dexpr(r"([x,y],[x=y]) + ([z],[dog(z), walk(z)])").pretty_print()
  _________     _________
 | x y     |   | z       |
(|---------| + |---------|)
 | (x = y) |   | dog(z)  |
 |_________|   | walk(z) |
               |_________|

>>> dexpr(r"([],[([x],[]) | ([y],[]) | ([z],[dog(z), walk(z)])])").pretty_print()
 _______________________________
|                               |
|-------------------------------|
|   ___     ___     _________   |
|  | x |   | y |   | z       |  |
| (|---| | |---| | |---------|) |
|  |___|   |___|   | dog(z)  |  |
|                  | walk(z) |  |
|                  |_________|  |
|_______________________________|

>>> dexpr(r"\P.\Q.(([x],[]) + P(x) + Q(x))(\x.([],[dog(x)]))").pretty_print()
          ___                        ________
 \       | x |                 \    |        |
 /\ P Q.(|---| + P(x) + Q(x))( /\ x.|--------|)
         |___|                      | dog(x) |
                                    |________|


               
        


        
               
###Examples - Glue Semantics
#a linguistic theory of semantic composition and the syntax-semantics interface 
#which assumes that meaning composition is constrained by a set of instructions 
#stated within a formal logic (linear logic). 

#These instructions, called meaning constructors, 
#state how the meanings of the parts of a sentence can be combined to provide 
#the meaning of the sentence.                           

#Linear logic

>>> from nltk.sem import logic
>>> from nltk.sem.glue import *
>>> from nltk.sem.linearlogic import *

>>> from nltk.sem.linearlogic import Expression
>>> read_expr = Expression.fromstring


#Parser

>>> print(read_expr(r'f'))
f
>>> print(read_expr(r'(g -o f)'))
(g -o f)
>>> print(read_expr(r'(g -o (h -o f))'))
(g -o (h -o f))
>>> print(read_expr(r'((g -o G) -o G)'))
((g -o G) -o G)
>>> print(read_expr(r'(g -o f)(g)'))
(g -o f)(g)
>>> print(read_expr(r'((g -o G) -o G)((g -o f))'))
((g -o G) -o G)((g -o f))


#Simplify

>>> print(read_expr(r'f').simplify())
f
>>> print(read_expr(r'(g -o f)').simplify())
(g -o f)
>>> print(read_expr(r'((g -o G) -o G)').simplify())
((g -o G) -o G)
>>> print(read_expr(r'(g -o f)(g)').simplify())
f
>>> try: read_expr(r'(g -o f)(f)').simplify()
    except LinearLogicApplicationException as e: print(e)
...
Cannot apply (g -o f) to f. Cannot unify g with f given {}
>>> print(read_expr(r'(G -o f)(g)').simplify())
f
>>> print(read_expr(r'((g -o G) -o G)((g -o f))').simplify())
f


#Test BindingDict

>>> h = ConstantExpression('h')
>>> g = ConstantExpression('g')
>>> f = ConstantExpression('f')

>>> H = VariableExpression('H')
>>> G = VariableExpression('G')
>>> F = VariableExpression('F')

>>> d1 = BindingDict({H: h})
>>> d2 = BindingDict({F: f, G: F})
>>> d12 = d1 + d2
>>> all12 = ['%s: %s' % (v, d12[v]) for v in d12.d]
>>> all12.sort()
>>> print(all12)
['F: f', 'G: f', 'H: h']

>>> BindingDict([(F,f),(G,g),(H,h)]) == BindingDict({F:f, G:g, H:h})
True

>>> d4 = BindingDict({F: f})
>>> try: d4[F] = g
    except VariableBindingException as e: print(e)
Variable F already bound to another value


#Test Unify

>>> try: f.unify(g, BindingDict())
    except UnificationException as e: print(e)
...
Cannot unify f with g given {}

>>> f.unify(G, BindingDict()) == BindingDict({G: f})
True
>>> try: f.unify(G, BindingDict({G: h}))
    except UnificationException as e: print(e)
...
Cannot unify f with G given {G: h}
>>> f.unify(G, BindingDict({G: f})) == BindingDict({G: f})
True
>>> f.unify(G, BindingDict({H: f})) == BindingDict({G: f, H: f})
True

>>> G.unify(f, BindingDict()) == BindingDict({G: f})
True
>>> try: G.unify(f, BindingDict({G: h}))
    except UnificationException as e: print(e)
...
Cannot unify G with f given {G: h}
>>> G.unify(f, BindingDict({G: f})) == BindingDict({G: f})
True
>>> G.unify(f, BindingDict({H: f})) == BindingDict({G: f, H: f})
True

>>> G.unify(F, BindingDict()) == BindingDict({G: F})
True
>>> try: G.unify(F, BindingDict({G: H}))
    except UnificationException as e: print(e)
...
Cannot unify G with F given {G: H}
>>> G.unify(F, BindingDict({G: F})) == BindingDict({G: F})
True
>>> G.unify(F, BindingDict({H: F})) == BindingDict({G: F, H: F})
True


#Test Compile

>>> print(read_expr('g').compile_pos(Counter(), GlueFormula))
(<ConstantExpression g>, [])
>>> print(read_expr('(g -o f)').compile_pos(Counter(), GlueFormula))
(<ImpExpression (g -o f)>, [])
>>> print(read_expr('(g -o (h -o f))').compile_pos(Counter(), GlueFormula))
(<ImpExpression (g -o (h -o f))>, [])



#Glue
#Demo of "John walks"

>>> john = GlueFormula("John", "g")
>>> print(john)
John : g
>>> walks = GlueFormula(r"\x.walks(x)", "(g -o f)")
>>> print(walks)
\x.walks(x) : (g -o f)
>>> print(walks.applyto(john))
\x.walks(x)(John) : (g -o f)(g)
>>> print(walks.applyto(john).simplify())
walks(John) : f



#Demo of "A dog walks"

>>> a = GlueFormula("\P Q.some x.(P(x) and Q(x))", "((gv -o gr) -o ((g -o G) -o G))")
>>> print(a)
\P Q.exists x.(P(x) & Q(x)) : ((gv -o gr) -o ((g -o G) -o G))
>>> man = GlueFormula(r"\x.man(x)", "(gv -o gr)")
>>> print(man)
\x.man(x) : (gv -o gr)
>>> walks = GlueFormula(r"\x.walks(x)", "(g -o f)")
>>> print(walks)
\x.walks(x) : (g -o f)
>>> a_man = a.applyto(man)
>>> print(a_man.simplify())
\Q.exists x.(man(x) & Q(x)) : ((g -o G) -o G)
>>> a_man_walks = a_man.applyto(walks)
>>> print(a_man_walks.simplify())
exists x.(man(x) & walks(x)) : f



#Demo of 'every girl chases a dog'
#Individual words:

>>> every = GlueFormula("\P Q.all x.(P(x) -> Q(x))", "((gv -o gr) -o ((g -o G) -o G))")
>>> print(every)
\P Q.all x.(P(x) -> Q(x)) : ((gv -o gr) -o ((g -o G) -o G))
>>> girl = GlueFormula(r"\x.girl(x)", "(gv -o gr)")
>>> print(girl)
\x.girl(x) : (gv -o gr)
>>> chases = GlueFormula(r"\x y.chases(x,y)", "(g -o (h -o f))")
>>> print(chases)
\x y.chases(x,y) : (g -o (h -o f))
>>> a = GlueFormula("\P Q.some x.(P(x) and Q(x))", "((hv -o hr) -o ((h -o H) -o H))")
>>> print(a)
\P Q.exists x.(P(x) & Q(x)) : ((hv -o hr) -o ((h -o H) -o H))
>>> dog = GlueFormula(r"\x.dog(x)", "(hv -o hr)")
>>> print(dog)
\x.dog(x) : (hv -o hr)


#Noun Quantification can only be done one way:

>>> every_girl = every.applyto(girl)
>>> print(every_girl.simplify())
\Q.all x.(girl(x) -> Q(x)) : ((g -o G) -o G)
>>> a_dog = a.applyto(dog)
>>> print(a_dog.simplify())
\Q.exists x.(dog(x) & Q(x)) : ((h -o H) -o H)


#The first reading is achieved by combining 'chases' with 'a dog' first. 
#Since 'a girl' requires something of the form '(h -o H)' we must get rid of the 'g' in the glue of 'see'. 
#We will do this with the '-o elimination' rule. So, x1 will be our subject placeholder.

>>> xPrime = GlueFormula("x1", "g")
>>> print(xPrime)
x1 : g
>>> xPrime_chases = chases.applyto(xPrime)
>>> print(xPrime_chases.simplify())
\y.chases(x1,y) : (h -o f)
>>> xPrime_chases_a_dog = a_dog.applyto(xPrime_chases)
>>> print(xPrime_chases_a_dog.simplify())
exists x.(dog(x) & chases(x1,x)) : f


#Now we can retract our subject placeholder using lambda-abstraction 
#and combine with the true subject.

>>> chases_a_dog = xPrime_chases_a_dog.lambda_abstract(xPrime)
>>> print(chases_a_dog.simplify())
\x1.exists x.(dog(x) & chases(x1,x)) : (g -o f)
>>> every_girl_chases_a_dog = every_girl.applyto(chases_a_dog)
>>> r1 = every_girl_chases_a_dog.simplify()
>>> r2 = GlueFormula(r'all x.(girl(x) -> exists z1.(dog(z1) & chases(x,z1)))', 'f')
>>> r1 == r2
True


#The second reading is achieved by combining 'every girl' with 'chases' first.

>>> xPrime = GlueFormula("x1", "g")
>>> print(xPrime)
x1 : g
>>> xPrime_chases = chases.applyto(xPrime)
>>> print(xPrime_chases.simplify())
\y.chases(x1,y) : (h -o f)
>>> yPrime = GlueFormula("x2", "h")
>>> print(yPrime)
x2 : h
>>> xPrime_chases_yPrime = xPrime_chases.applyto(yPrime)
>>> print(xPrime_chases_yPrime.simplify())
chases(x1,x2) : f
>>> chases_yPrime = xPrime_chases_yPrime.lambda_abstract(xPrime)
>>> print(chases_yPrime.simplify())
\x1.chases(x1,x2) : (g -o f)
>>> every_girl_chases_yPrime = every_girl.applyto(chases_yPrime)
>>> print(every_girl_chases_yPrime.simplify())
all x.(girl(x) -> chases(x,x2)) : f
>>> every_girl_chases = every_girl_chases_yPrime.lambda_abstract(yPrime)
>>> print(every_girl_chases.simplify())
\x2.all x.(girl(x) -> chases(x,x2)) : (h -o f)
>>> every_girl_chases_a_dog = a_dog.applyto(every_girl_chases)
>>> r1 = every_girl_chases_a_dog.simplify()
>>> r2 = GlueFormula(r'exists x.(dog(x) & all z2.(girl(z2) -> chases(z2,x)))', 'f')
>>> r1 == r2
True



#Compilation

>>> for cp in GlueFormula('m', '(b -o a)').compile(Counter()): print(cp)
m : (b -o a) : {1}
>>> for cp in GlueFormula('m', '((c -o b) -o a)').compile(Counter()): print(cp)
v1 : c : {1}
m : (b[1] -o a) : {2}
>>> for cp in GlueFormula('m', '((d -o (c -o b)) -o a)').compile(Counter()): print(cp)
v1 : c : {1}
v2 : d : {2}
m : (b[1, 2] -o a) : {3}
>>> for cp in GlueFormula('m', '((d -o e) -o ((c -o b) -o a))').compile(Counter()): print(cp)
v1 : d : {1}
v2 : c : {2}
m : (e[1] -o (b[2] -o a)) : {3}
>>> for cp in GlueFormula('m', '(((d -o c) -o b) -o a)').compile(Counter()): print(cp)
v1 : (d -o c) : {1}
m : (b[1] -o a) : {2}
>>> for cp in GlueFormula('m', '((((e -o d) -o c) -o b) -o a)').compile(Counter()): print(cp)
v1 : e : {1}
v2 : (d[1] -o c) : {2}
m : (b[2] -o a) : {3}



#Demo of 'a man walks' using Compilation
#Premises

>>> a = GlueFormula('\\P Q.some x.(P(x) and Q(x))', '((gv -o gr) -o ((g -o G) -o G))')
>>> print(a)
\P Q.exists x.(P(x) & Q(x)) : ((gv -o gr) -o ((g -o G) -o G))

>>> man = GlueFormula('\\x.man(x)', '(gv -o gr)')
>>> print(man)
\x.man(x) : (gv -o gr)

>>> walks = GlueFormula('\\x.walks(x)', '(g -o f)')
>>> print(walks)
\x.walks(x) : (g -o f)


#Compiled Premises:

>>> counter = Counter()
>>> ahc = a.compile(counter)
>>> g1 = ahc[0]
>>> print(g1)
v1 : gv : {1}
>>> g2 = ahc[1]
>>> print(g2)
v2 : g : {2}
>>> g3 = ahc[2]
>>> print(g3)
\P Q.exists x.(P(x) & Q(x)) : (gr[1] -o (G[2] -o G)) : {3}
>>> g4 = man.compile(counter)[0]
>>> print(g4)
\x.man(x) : (gv -o gr) : {4}
>>> g5 = walks.compile(counter)[0]
>>> print(g5)
\x.walks(x) : (g -o f) : {5}


#Derivation:

>>> g14 = g4.applyto(g1)
>>> print(g14.simplify())
man(v1) : gr : {1, 4}
>>> g134 = g3.applyto(g14)
>>> print(g134.simplify())
\Q.exists x.(man(x) & Q(x)) : (G[2] -o G) : {1, 3, 4}
>>> g25 = g5.applyto(g2)
>>> print(g25.simplify())
walks(v2) : f : {2, 5}
>>> g12345 = g134.applyto(g25)
>>> print(g12345.simplify())
exists x.(man(x) & walks(x)) : f : {1, 2, 3, 4, 5}



#Dependency Graph to Glue Formulas

>>> from nltk.corpus.reader.dependency import DependencyGraph

>>> depgraph = DependencyGraph("""1 John    _       NNP     NNP     _       2       SUBJ    _       _
    2       sees    _       VB      VB      _       0       ROOT    _       _
    3       a       _       ex_quant        ex_quant        _       4       SPEC    _       _
    4       dog     _       NN      NN      _       2       OBJ     _       _
    """)
>>> gfl = GlueDict('nltk:grammars/sample_grammars/glue.semtype').to_glueformula_list(depgraph)
>>> for gf in gfl:
        print(gf)
\x y.sees(x,y) : (f -o (i -o g))
\P Q.exists x.(P(x) & Q(x)) : ((fv -o fr) -o ((f -o F2) -o F2))
\x.John(x) : (fv -o fr)
\x.dog(x) : (iv -o ir)
\P Q.exists x.(P(x) & Q(x)) : ((iv -o ir) -o ((i -o I5) -o I5))
>>> glue = Glue()
>>> for r in sorted([r.simplify().normalize() for r in glue.get_readings(glue.gfl_to_compiled(gfl))], key=str):
        print(r)
exists z1.(John(z1) & exists z2.(dog(z2) & sees(z1,z2)))
exists z1.(dog(z1) & exists z2.(John(z2) & sees(z2,z1)))



#Dependency Graph to LFG f-structure

>>> from nltk.sem.lfg import FStructure

>>> fstruct = FStructure.read_depgraph(depgraph)

>>> print(fstruct)
f:[pred 'sees'
   obj h:[pred 'dog'
          spec 'a']
   subj g:[pred 'John']]

>>> fstruct.to_depgraph().tree().pprint()
(sees (dog a) John)



#LFG f-structure to Glue

>>> for gf in fstruct.to_glueformula_list(GlueDict('nltk:grammars/sample_grammars/glue.semtype')): # doctest: +SKIP
        print(gf)
\x y.sees(x,y) : (i -o (g -o f))
\x.dog(x) : (gv -o gr)
\P Q.exists x.(P(x) & Q(x)) : ((gv -o gr) -o ((g -o G3) -o G3))
\P Q.exists x.(P(x) & Q(x)) : ((iv -o ir) -o ((i -o I4) -o I4))
\x.John(x) : (iv -o ir)






###Examples - Logical Inference and Model Building
  
>>> from nltk import *
>>> from nltk.sem.drt import DrtParser
>>> from nltk.sem import logic
>>> logic._counter._value = 0


#NLTK Interface to Theorem Provers

#There are currently three theorem provers included with NLTK: 
#Prover9, TableauProver, and ResolutionProver. 
#The first is an off-the-shelf prover, 
#while the other two are written in Python and included in the nltk.inference package.

>>> from nltk.sem import Expression
>>> read_expr = Expression.fromstring
>>> p1 = read_expr('man(socrates)')
>>> p2 = read_expr('all x.(man(x) -> mortal(x))')
>>> c  = read_expr('mortal(socrates)')
>>> nltk.Prover9().prove(c, [p1,p2])  #prover.config_prover9(r'c:/nltk_data/prover9/bin-win32') 
True
>>> nltk.inference.TableauProver().prove(c, [p1,p2])
True
>>> nltk.inference.ResolutionProver().prove(c, [p1,p2], verbose=True)
[1] {-mortal(socrates)}     A
[2] {man(socrates)}         A
[3] {-man(z2), mortal(z2)}  A
[4] {-man(socrates)}        (1, 3)
[5] {mortal(socrates)}      (2, 3)
[6] {}                      (1, 5)
<BLANKLINE>
True



#A ProverCommand is a stateful holder for a theorem prover. 
#The command stores a theorem prover instance (of type Prover), 
#a goal, a list of assumptions, the result of the proof, and a string version of the entire proof. 

#there are three ProverCommand implementations: 
#Prover9Command, TableauProverCommand, and ResolutionProverCommand.

#The ProverCommand's constructor takes its goal and assumptions. 
#The prove() command executes the Prover 
#and proof() returns a String form of the proof 


>>> prover = ResolutionProverCommand(c, [p1,p2])
>>> print(prover.proof()) 
Traceback (most recent call last):
  File "...", line 1212, in __run
    compileflags, 1) in test.globs
  File "<doctest nltk/test/inference.doctest[10]>", line 1, in <module>
  File "...", line ..., in proof
    raise LookupError("You have to call prove() first to get a proof!")
LookupError: You have to call prove() first to get a proof!
>>> prover.prove()
True
>>> print(prover.proof())
[1] {-mortal(socrates)}     A
[2] {man(socrates)}         A
[3] {-man(z4), mortal(z4)}  A
[4] {-man(socrates)}        (1, 3)
[5] {mortal(socrates)}      (2, 3)
[6] {}                      (1, 5)
<BLANKLINE>


>>> prover.assumptions()
[<ApplicationExpression man(socrates)>, <Alread_expression all x.(man(x) -> mortal(x))>]
>>> prover.goal()
<ApplicationExpression mortal(socrates)>


#The assumptions list may be modified using the add_assumptions() 
#and retract_assumptions() methods. 
#Both methods take a list of Expression objects. 

>>> prover.retract_assumptions([read_expr('man(socrates)')])
>>> prover.prove()
False
>>> print(prover.proof())
[1] {-mortal(socrates)}     A
[2] {-man(z6), mortal(z6)}  A
[3] {-man(socrates)}        (1, 2)
<BLANKLINE>
>>> prover.add_assumptions([read_expr('man(socrates)')])
>>> prover.prove()
True



##Prover9
#https://www.cs.unm.edu/~mccune/mace4/gui/v05.html

>>> p = Prover9()
>>> p.binary_locations() 
['/usr/local/bin/prover9',
 '/usr/local/bin/prover9/bin',
 '/usr/local/bin',
 '/usr/bin',
 '/usr/local/prover9',
 '/usr/local/share/prover9']


#the environment variable PROVER9HOME may be configured with the binary's location.

#or 
>>> config_prover9(path='/usr/local/bin')
[Found prover9: /usr/local/bin/prover9]


#The general case in theorem proving is to determine whether S |- g holds, 
#where S is a possibly empty set of assumptions, and g is a proof goal.

#NLTK input to Prover9 must be Expressions of nltk.sem.logic. 

>>> goal = read_expr('(man(x) <-> --man(x))')
>>> prover = Prover9Command(goal)
>>> prover.prove()
True


#add_assumptions 
>>> g = read_expr('mortal(socrates)')
>>> a1 = read_expr('all x.(man(x) -> mortal(x))')
>>> prover = Prover9Command(g, assumptions=[a1])
>>> prover.print_assumptions()
all x.(man(x) -> mortal(x))


#the assumptions are not sufficient to derive the goal:
>>> print(prover.prove())
False


#So let's add another assumption:

>>> a2 = read_expr('man(socrates)')
>>> prover.add_assumptions([a2])
>>> prover.print_assumptions()
all x.(man(x) -> mortal(x))
man(socrates)
>>> print(prover.prove())
True

>>> prover.print_assumptions(output_format='Prover9')
all x (man(x) -> mortal(x))
man(socrates)

#Assumptions can be retracted from the list of assumptions.

>>> prover.retract_assumptions([a1])
>>> prover.print_assumptions()
man(socrates)
>>> prover.retract_assumptions([a1])


#Statements can be loaded from a file and parsed. 
#We can then add these statements as new assumptions.

>>> g = read_expr('all x.(boxer(x) -> -boxerdog(x))')
>>> prover = Prover9Command(g)
>>> prover.prove()
False
>>> import nltk.data
>>> new = nltk.data.load('grammars/sample_grammars/background0.fol')
>>> for a in new:
...     print(a)
all x.(boxerdog(x) -> dog(x))
all x.(boxer(x) -> person(x))
all x.-(dog(x) & person(x))
exists x.boxer(x)
exists x.boxerdog(x)
>>> prover.add_assumptions(new)
>>> print(prover.prove())
True
>>> print(prover.proof()) # doctest: +ELLIPSIS
============================== prooftrans ============================
Prover9 (...) version ...
Process ... was started by ... on ...
...
The command was ".../prover9".
============================== end of head ===========================
<BLANKLINE>
============================== end of input ==========================
<BLANKLINE>
============================== PROOF =================================
<BLANKLINE>
% -------- Comments from original proof --------
% Proof 1 at ... seconds.
% Length of proof is 13.
% Level of proof is 4.
% Maximum clause weight is 0.000.
% Given clauses 0.
<BLANKLINE>
<BLANKLINE>
1 (all x (boxerdog(x) -> dog(x))).  [assumption].
2 (all x (boxer(x) -> person(x))).  [assumption].
3 (all x -(dog(x) & person(x))).  [assumption].
6 (all x (boxer(x) -> -boxerdog(x))).  [goal].
8 -boxerdog(x) | dog(x).  [clausify(1)].
9 boxerdog(c3).  [deny(6)].
11 -boxer(x) | person(x).  [clausify(2)].
12 boxer(c3).  [deny(6)].
14 -dog(x) | -person(x).  [clausify(3)].
15 dog(c3).  [resolve(9,a,8,a)].
18 person(c3).  [resolve(12,a,11,a)].
19 -person(c3).  [resolve(15,a,14,a)].
20 $F.  [resolve(19,a,18,a)].
<BLANKLINE>
============================== end of proof ==========================



#The equiv() method
#to check if two Expressions have the same meaning. 

>>> a = read_expr(r'exists x.(man(x) & walks(x))')
>>> b = read_expr(r'exists x.(walks(x) & man(x))')
>>> print(a.equiv(b))
True


#The same method can be used on Discourse Representation Structures (DRSs). 
>>> dp = DrtParser()
>>> a = dp.parse(r'([x],[man(x), walks(x)])')
>>> b = dp.parse(r'([x],[walks(x), man(x)])')
>>> print(a.equiv(b))
True



##NLTK Interface to Model Builders
#The top-level to model builders is parallel to that for theorem-provers. 
#interfaces with the Mace4 model builder.

#Typically we use a model builder to show that some set of formulas has a model, 
#and is therefore consistent. 
#One way of doing this is by treating our candidate set of sentences as assumptions, and leaving the goal unspecified. 

>>> a3 = read_expr('exists x.(man(x) and walks(x))')
>>> c1 = read_expr('mortal(socrates)')
>>> c2 = read_expr('-mortal(socrates)')
>>> mace = Mace()  #_mace4_bin should be set, check earlier 
>>> print(mace.build_model(None, [a3, c1]))
True
>>> print(mace.build_model(None, [a3, c2]))
True

#We can also use the model builder as an adjunct(sumplementary) to theorem prover. 
#Let's suppose we are trying to prove S |- g, i.e. that g is logically entailed by assumptions S = {s1, s2, ..., sn}. 

#We can this same input to Mace4, and the model builder will try to find a counterexample, 
#that is, to show that g does not follow from S. 

#So, given this input, Mace4 will try to find a model for the set S' = {s1, s2, ..., sn, (not g)}. 
#If g fails to follow from S, then Mace4 may well return with a counterexample faster than Prover9 concludes that it cannot find the required proof. 

#Conversely, if g is provable from S, Mace4 may take a long time unsuccessfully trying to find a counter model, and will eventually give up.

>>> a4 = read_expr('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
>>> a5 = read_expr('man(adam)')
>>> a6 = read_expr('woman(eve)')
>>> g = read_expr('love(adam,eve)')
>>> print(mace.build_model(g, [a4, a5, a6]))
True

#The Model Builder will fail to find a model if the assumptions do entail the goal. 
#Mace will continue to look for models of ever-increasing sizes 
#until the end_size number is reached. 

>>> a7 = read_expr('all x.(man(x) -> mortal(x))')
>>> a8 = read_expr('man(socrates)')
>>> g2 = read_expr('mortal(socrates)')
>>> print(Mace(end_size=50).build_model(g2, [a7, a8]))
False


#Show the model in 'tabular' format.

>>> a = read_expr('(see(mary,john) & -(mary = john))')
>>> mb = MaceCommand(assumptions=[a])
>>> mb.build_model()
True

>>> print(mb.model(format='tabular'))
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
 john : 0
<BLANKLINE>
 mary : 1
<BLANKLINE>
 see :
       | 0 1
    ---+----
     0 | 0 0
     1 | 1 0
<BLANKLINE>


Show the model in 'tabular' format.

>>> print(mb.model(format='cooked'))
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
john = 0.
<BLANKLINE>
mary = 1.
<BLANKLINE>
- see(0,0).
- see(0,1).
  see(1,0).
- see(1,1).
<BLANKLINE>


#The property valuation accesses the stored Valuation.

>>> print(mb.valuation)
{'john': 'a', 'mary': 'b', 'see': {('b', 'a')}}


#We can return to our earlier example and inspect the model:

>>> mb = MaceCommand(g, assumptions=[a4, a5, a6])
>>> m = mb.build_model()
>>> print(mb.model(format='cooked'))
% number = 1
% seconds = 0
<BLANKLINE>
% Interpretation of size 2
<BLANKLINE>
adam = 0.
<BLANKLINE>
eve = 0.
<BLANKLINE>
c1 = 1.
<BLANKLINE>
  man(0).
- man(1).
<BLANKLINE>
  woman(0).
  woman(1).
<BLANKLINE>
- love(0,0).
  love(0,1).
- love(1,0).
- love(1,1).
<BLANKLINE>


#Mace can also be used with propositional logic.

>>> p = read_expr('P')
>>> q = read_expr('Q')
>>> mb = MaceCommand(q, [p, p>-q])
>>> mb.build_model()
True
>>> mb.valuation['P']
True
>>> mb.valuation['Q']
False

             
             
         
###Examples - Resolution Theorem Prover
  
>>> from nltk.inference.resolution import *
>>> from nltk.sem import logic
>>> from nltk.sem.logic import *
>>> logic._counter._value = 0
>>> read_expr = logic.Expression.fromstring

>>> P = read_expr('P')
>>> Q = read_expr('Q')
>>> R = read_expr('R')
>>> A = read_expr('A')
>>> B = read_expr('B')
>>> x = read_expr('x')
>>> y = read_expr('y')
>>> z = read_expr('z')



#Test most_general_unification()

>>> print(most_general_unification(x, x))
{}
>>> print(most_general_unification(A, A))
{}
>>> print(most_general_unification(A, x))
{x: A}
>>> print(most_general_unification(x, A))
{x: A}
>>> print(most_general_unification(x, y))
{x: y}
>>> print(most_general_unification(P(x), P(A)))
{x: A}
>>> print(most_general_unification(P(x,B), P(A,y)))
{x: A, y: B}
>>> print(most_general_unification(P(x,B), P(B,x)))
{x: B}
>>> print(most_general_unification(P(x,y), P(A,x)))
{x: A, y: x}
>>> print(most_general_unification(P(Q(x)), P(y)))
{y: Q(x)}



#Test unify()

>>> print(Clause([]).unify(Clause([])))
[]
>>> print(Clause([P(x)]).unify(Clause([-P(A)])))
[{}]
>>> print(Clause([P(A), Q(x)]).unify(Clause([-P(x), R(x)])))
[{R(A), Q(A)}]
>>> print(Clause([P(A), Q(x), R(x,y)]).unify(Clause([-P(x), Q(y)])))
[{Q(y), Q(A), R(A,y)}]
>>> print(Clause([P(A), -Q(y)]).unify(Clause([-P(x), Q(B)])))
[{}]
>>> print(Clause([P(x), Q(x)]).unify(Clause([-P(A), -Q(B)])))
[{-Q(B), Q(A)}, {-P(A), P(B)}]
>>> print(Clause([P(x,x), Q(x), R(x)]).unify(Clause([-P(A,z), -Q(B)])))
[{-Q(B), Q(A), R(A)}, {-P(A,z), R(B), P(B,B)}]

>>> a = clausify(read_expr('P(A)'))
>>> b = clausify(read_expr('A=B'))
>>> print(a[0].unify(b[0]))
[{P(B)}]



#Test is_tautology()

>>> print(Clause([P(A), -P(A)]).is_tautology())
True
>>> print(Clause([-P(A), P(A)]).is_tautology())
True
>>> print(Clause([P(x), -P(A)]).is_tautology())
False
>>> print(Clause([Q(B), -P(A), P(A)]).is_tautology())
True
>>> print(Clause([-Q(A), P(R(A)), -P(R(A)), Q(x), -R(y)]).is_tautology())
True
>>> print(Clause([P(x), -Q(A)]).is_tautology())
False



#Test subsumes()

>>> print(Clause([P(A), Q(B)]).subsumes(Clause([P(A), Q(B)])))
True
>>> print(Clause([-P(A)]).subsumes(Clause([P(A)])))
False
>>> print(Clause([P(A), Q(B)]).subsumes(Clause([Q(B), P(A)])))
True
>>> print(Clause([P(A), Q(B)]).subsumes(Clause([Q(B), R(A), P(A)])))
True
>>> print(Clause([P(A), R(A), Q(B)]).subsumes(Clause([Q(B), P(A)])))
False
>>> print(Clause([P(x)]).subsumes(Clause([P(A)])))
True
>>> print(Clause([P(A)]).subsumes(Clause([P(x)])))
True



#Test prove()

>>> print(ResolutionProverCommand(read_expr('man(x)')).prove())
False
>>> print(ResolutionProverCommand(read_expr('(man(x) -> man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) -> --man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('-(man(x) & -man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) | -man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) -> man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('-(man(x) & -man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) | -man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) -> man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('(man(x) <-> man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('-(man(x) <-> -man(x))')).prove())
True
>>> print(ResolutionProverCommand(read_expr('all x.man(x)')).prove())
False
>>> print(ResolutionProverCommand(read_expr('-all x.some y.F(x,y) & some x.all y.(-F(x,y))')).prove())
False
>>> print(ResolutionProverCommand(read_expr('some x.all y.sees(x,y)')).prove())
False

>>> p1 = read_expr('all x.(man(x) -> mortal(x))')
>>> p2 = read_expr('man(Socrates)')
>>> c = read_expr('mortal(Socrates)')
>>> ResolutionProverCommand(c, [p1,p2]).prove()
True

>>> p1 = read_expr('all x.(man(x) -> walks(x))')
>>> p2 = read_expr('man(John)')
>>> c = read_expr('some y.walks(y)')
>>> ResolutionProverCommand(c, [p1,p2]).prove()
True

>>> p = read_expr('some e1.some e2.(believe(e1,john,e2) & walk(e2,mary))')
>>> c = read_expr('some e0.walk(e0,mary)')
>>> ResolutionProverCommand(c, [p]).prove()
True



#Test proof()

>>> p1 = read_expr('all x.(man(x) -> mortal(x))')
>>> p2 = read_expr('man(Socrates)')
>>> c = read_expr('mortal(Socrates)')
>>> logic._counter._value = 0
>>> tp = ResolutionProverCommand(c, [p1,p2])
>>> tp.prove()
True
>>> print(tp.proof())
[1] {-mortal(Socrates)}     A
[2] {-man(z2), mortal(z2)}  A
[3] {man(Socrates)}         A
[4] {-man(Socrates)}        (1, 2)
[5] {mortal(Socrates)}      (2, 3)
[6] {}                      (1, 5)
<BLANKLINE>



#Question Answering
#One answer
>>> p1 = read_expr('father_of(art,john)')
>>> p2 = read_expr('father_of(bob,kim)')
>>> p3 = read_expr('all x.all y.(father_of(x,y) -> parent_of(x,y))')
>>> c = read_expr('all x.(parent_of(x,john) -> ANSWER(x))')
>>> logic._counter._value = 0
>>> tp = ResolutionProverCommand(None, [p1,p2,p3,c])
>>> sorted(tp.find_answers())
[<ConstantExpression art>]
>>> print(tp.proof()) # doctest: +SKIP
[1] {father_of(art,john)}                  A
[2] {father_of(bob,kim)}                   A
[3] {-father_of(z3,z4), parent_of(z3,z4)}  A
[4] {-parent_of(z6,john), ANSWER(z6)}      A
[5] {parent_of(art,john)}                  (1, 3)
[6] {parent_of(bob,kim)}                   (2, 3)
[7] {ANSWER(z6), -father_of(z6,john)}      (3, 4)
[8] {ANSWER(art)}                          (1, 7)
[9] {ANSWER(art)}                          (4, 5)
<BLANKLINE>

#Multiple answers
>>> p1 = read_expr('father_of(art,john)')
>>> p2 = read_expr('mother_of(ann,john)')
>>> p3 = read_expr('all x.all y.(father_of(x,y) -> parent_of(x,y))')
>>> p4 = read_expr('all x.all y.(mother_of(x,y) -> parent_of(x,y))')
>>> c = read_expr('all x.(parent_of(x,john) -> ANSWER(x))')
>>> logic._counter._value = 0
>>> tp = ResolutionProverCommand(None, [p1,p2,p3,p4,c])
>>> sorted(tp.find_answers())
[<ConstantExpression ann>, <ConstantExpression art>]
>>> print(tp.proof()) # doctest: +SKIP
[ 1] {father_of(art,john)}                  A
[ 2] {mother_of(ann,john)}                  A
[ 3] {-father_of(z3,z4), parent_of(z3,z4)}  A
[ 4] {-mother_of(z7,z8), parent_of(z7,z8)}  A
[ 5] {-parent_of(z10,john), ANSWER(z10)}    A
[ 6] {parent_of(art,john)}                  (1, 3)
[ 7] {parent_of(ann,john)}                  (2, 4)
[ 8] {ANSWER(z10), -father_of(z10,john)}    (3, 5)
[ 9] {ANSWER(art)}                          (1, 8)
[10] {ANSWER(z10), -mother_of(z10,john)}    (4, 5)
[11] {ANSWER(ann)}                          (2, 10)
[12] {ANSWER(art)}                          (5, 6)
[13] {ANSWER(ann)}                          (5, 7)
<BLANKLINE>

            
###Examples - Nonmonotonic Reasoning
  
>>> from nltk import *
>>> from nltk.inference.nonmonotonic import *
>>> from nltk.sem import logic
>>> logic._counter._value = 0
>>> read_expr = logic.Expression.fromstring


#Closed Domain Assumption
#The only entities in the domain are those found in the assumptions or goal. 
#If the domain only contains "A" and "B", 
#then the expression "exists x.P(x)" can be replaced with "P(A) | P(B)" 
#and an expression "all x.P(x)" can be replaced with "P(A) & P(B)".

>>> p1 = read_expr(r'all x.(man(x) -> mortal(x))')
>>> p2 = read_expr(r'man(Socrates)')
>>> c = read_expr(r'mortal(Socrates)')
>>> prover = Prover9Command(c, [p1,p2])
>>> prover.prove()
True
>>> cdp = ClosedDomainProver(prover)
>>> for a in cdp.assumptions(): print(a) # doctest: +SKIP
(man(Socrates) -> mortal(Socrates))
man(Socrates)
>>> cdp.prove()
True

>>> p1 = read_expr(r'exists x.walk(x)')
>>> p2 = read_expr(r'man(Socrates)')
>>> c = read_expr(r'walk(Socrates)')
>>> prover = Prover9Command(c, [p1,p2])
>>> prover.prove()
False
>>> cdp = ClosedDomainProver(prover)
>>> for a in cdp.assumptions(): print(a) # doctest: +SKIP
walk(Socrates)
man(Socrates)
>>> cdp.prove()
True

>>> p1 = read_expr(r'exists x.walk(x)')
>>> p2 = read_expr(r'man(Socrates)')
>>> p3 = read_expr(r'-walk(Bill)')
>>> c = read_expr(r'walk(Socrates)')
>>> prover = Prover9Command(c, [p1,p2,p3])
>>> prover.prove()
False
>>> cdp = ClosedDomainProver(prover)
>>> for a in cdp.assumptions(): print(a) # doctest: +SKIP
(walk(Socrates) | walk(Bill))
man(Socrates)
-walk(Bill)
>>> cdp.prove()
True

>>> p1 = read_expr(r'walk(Socrates)')
>>> p2 = read_expr(r'walk(Bill)')
>>> c = read_expr(r'all x.walk(x)')
>>> prover = Prover9Command(c, [p1,p2])
>>> prover.prove()
False
>>> cdp = ClosedDomainProver(prover)
>>> for a in cdp.assumptions(): print(a) # doctest: +SKIP
walk(Socrates)
walk(Bill)
>>> print(cdp.goal()) # doctest: +SKIP
(walk(Socrates) & walk(Bill))
>>> cdp.prove()
True

>>> p1 = read_expr(r'girl(mary)')
>>> p2 = read_expr(r'dog(rover)')
>>> p3 = read_expr(r'all x.(girl(x) -> -dog(x))')
>>> p4 = read_expr(r'all x.(dog(x) -> -girl(x))')
>>> p5 = read_expr(r'chase(mary, rover)')
>>> c = read_expr(r'exists y.(dog(y) & all x.(girl(x) -> chase(x,y)))')
>>> prover = Prover9Command(c, [p1,p2,p3,p4,p5])
>>> print(prover.prove())
False
>>> cdp = ClosedDomainProver(prover)
>>> for a in cdp.assumptions(): print(a) # doctest: +SKIP
girl(mary)
dog(rover)
((girl(rover) -> -dog(rover)) & (girl(mary) -> -dog(mary)))
((dog(rover) -> -girl(rover)) & (dog(mary) -> -girl(mary)))
chase(mary,rover)
>>> print(cdp.goal()) # doctest: +SKIP
((dog(rover) & (girl(rover) -> chase(rover,rover)) & (girl(mary) -> chase(mary,rover))) | (dog(mary) & (girl(rover) -> chase(rover,mary)) & (girl(mary) -> chase(mary,mary))))
>>> print(cdp.prove())
True



#Unique Names Assumption
#No two entities in the domain represent the same entity 
#unless it can be explicitly proven that they do. 
#Therefore, if the domain contains "A" and "B", 
#then add the assumption "-(A = B)" if it is not the case that "<assumptions> |- (A = B)".

>>> p1 = read_expr(r'man(Socrates)')
>>> p2 = read_expr(r'man(Bill)')
>>> c = read_expr(r'exists x.exists y.-(x = y)')
>>> prover = Prover9Command(c, [p1,p2])
>>> prover.prove()
False
>>> unp = UniqueNamesProver(prover)
>>> for a in unp.assumptions(): print(a) # doctest: +SKIP
man(Socrates)
man(Bill)
-(Socrates = Bill)
>>> unp.prove()
True

>>> p1 = read_expr(r'all x.(walk(x) -> (x = Socrates))')
>>> p2 = read_expr(r'Bill = William')
>>> p3 = read_expr(r'Bill = Billy')
>>> c = read_expr(r'-walk(William)')
>>> prover = Prover9Command(c, [p1,p2,p3])
>>> prover.prove()
False
>>> unp = UniqueNamesProver(prover)
>>> for a in unp.assumptions(): print(a) # doctest: +SKIP
all x.(walk(x) -> (x = Socrates))
(Bill = William)
(Bill = Billy)
-(William = Socrates)
-(Billy = Socrates)
-(Socrates = Bill)
>>> unp.prove()
True



#Closed World Assumption
#The only entities that have certain properties are those 
#that is it stated have the properties. 
#We accomplish this assumption by "completing" predicates.

#If the assumptions contain "P(A)", then "all x.(P(x) -> (x=A))" is the completion of "P". 
#If the assumptions contain "all x.(ostrich(x) -> bird(x))", 
#then "all x.(bird(x) -> ostrich(x))" is the completion of "bird". 
#If the assumptions don't contain anything that are "P", then "all x.-P(x)" is the completion of "P".

>>> p1 = read_expr(r'walk(Socrates)')
>>> p2 = read_expr(r'-(Socrates = Bill)')
>>> c = read_expr(r'-walk(Bill)')
>>> prover = Prover9Command(c, [p1,p2])
>>> prover.prove()
False
>>> cwp = ClosedWorldProver(prover)
>>> for a in cwp.assumptions(): print(a) # doctest: +SKIP
walk(Socrates)
-(Socrates = Bill)
all z1.(walk(z1) -> (z1 = Socrates))
>>> cwp.prove()
True

>>> p1 = read_expr(r'see(Socrates, John)')
>>> p2 = read_expr(r'see(John, Mary)')
>>> p3 = read_expr(r'-(Socrates = John)')
>>> p4 = read_expr(r'-(John = Mary)')
>>> c = read_expr(r'-see(Socrates, Mary)')
>>> prover = Prover9Command(c, [p1,p2,p3,p4])
>>> prover.prove()
False
>>> cwp = ClosedWorldProver(prover)
>>> for a in cwp.assumptions(): print(a) # doctest: +SKIP
see(Socrates,John)
see(John,Mary)
-(Socrates = John)
-(John = Mary)
all z3 z4.(see(z3,z4) -> (((z3 = Socrates) & (z4 = John)) | ((z3 = John) & (z4 = Mary))))
>>> cwp.prove()
True

>>> p1 = read_expr(r'all x.(ostrich(x) -> bird(x))')
>>> p2 = read_expr(r'bird(Tweety)')
>>> p3 = read_expr(r'-ostrich(Sam)')
>>> p4 = read_expr(r'Sam != Tweety')
>>> c = read_expr(r'-bird(Sam)')
>>> prover = Prover9Command(c, [p1,p2,p3,p4])
>>> prover.prove()
False
>>> cwp = ClosedWorldProver(prover)
>>> for a in cwp.assumptions(): print(a) # doctest: +SKIP
all x.(ostrich(x) -> bird(x))
bird(Tweety)
-ostrich(Sam)
-(Sam = Tweety)
all z7.-ostrich(z7)
all z8.(bird(z8) -> ((z8 = Tweety) | ostrich(z8)))
>>> print(cwp.prove())
True



#Multi-Decorator Example
#Decorators can be nested to utilize multiple assumptions.

>>> p1 = read_expr(r'see(Socrates, John)')
>>> p2 = read_expr(r'see(John, Mary)')
>>> c = read_expr(r'-see(Socrates, Mary)')
>>> prover = Prover9Command(c, [p1,p2])
>>> print(prover.prove())
False
>>> cmd = ClosedDomainProver(UniqueNamesProver(ClosedWorldProver(prover)))
>>> print(cmd.prove())
True



#Default Reasoning

>>> logic._counter._value = 0
>>> premises = []

#define the taxonomy
>>> premises.append(read_expr(r'all x.(elephant(x)        -> animal(x))'))
>>> premises.append(read_expr(r'all x.(bird(x)            -> animal(x))'))
>>> premises.append(read_expr(r'all x.(dove(x)            -> bird(x))'))
>>> premises.append(read_expr(r'all x.(ostrich(x)         -> bird(x))'))
>>> premises.append(read_expr(r'all x.(flying_ostrich(x)  -> ostrich(x))'))

#default the properties using abnormalities
>>> premises.append(read_expr(r'all x.((animal(x)  & -Ab1(x)) -> -fly(x))')) #normal animals don't fly
>>> premises.append(read_expr(r'all x.((bird(x)    & -Ab2(x)) -> fly(x))'))  #normal birds fly
>>> premises.append(read_expr(r'all x.((ostrich(x) & -Ab3(x)) -> -fly(x))')) #normal ostriches don't fly

#specify abnormal entities
>>> premises.append(read_expr(r'all x.(bird(x)           -> Ab1(x))')) #flight
>>> premises.append(read_expr(r'all x.(ostrich(x)        -> Ab2(x))')) #non-flying bird
>>> premises.append(read_expr(r'all x.(flying_ostrich(x) -> Ab3(x))')) #flying ostrich

#define entities
>>> premises.append(read_expr(r'elephant(el)'))
>>> premises.append(read_expr(r'dove(do)'))
>>> premises.append(read_expr(r'ostrich(os)'))

print the augmented assumptions list
>>> prover = Prover9Command(None, premises)
>>> command = UniqueNamesProver(ClosedWorldProver(prover))
>>> for a in command.assumptions(): print(a) # doctest: +SKIP
all x.(elephant(x) -> animal(x))
all x.(bird(x) -> animal(x))
all x.(dove(x) -> bird(x))
all x.(ostrich(x) -> bird(x))
all x.(flying_ostrich(x) -> ostrich(x))
all x.((animal(x) & -Ab1(x)) -> -fly(x))
all x.((bird(x) & -Ab2(x)) -> fly(x))
all x.((ostrich(x) & -Ab3(x)) -> -fly(x))
all x.(bird(x) -> Ab1(x))
all x.(ostrich(x) -> Ab2(x))
all x.(flying_ostrich(x) -> Ab3(x))
elephant(el)
dove(do)
ostrich(os)
all z1.(animal(z1) -> (elephant(z1) | bird(z1)))
all z2.(Ab1(z2) -> bird(z2))
all z3.(bird(z3) -> (dove(z3) | ostrich(z3)))
all z4.(dove(z4) -> (z4 = do))
all z5.(Ab2(z5) -> ostrich(z5))
all z6.(Ab3(z6) -> flying_ostrich(z6))
all z7.(ostrich(z7) -> ((z7 = os) | flying_ostrich(z7)))
all z8.-flying_ostrich(z8)
all z9.(elephant(z9) -> (z9 = el))
-(el = os)
-(el = do)
-(os = do)

>>> UniqueNamesProver(ClosedWorldProver(Prover9Command(read_expr('-fly(el)'), premises))).prove()
True
>>> UniqueNamesProver(ClosedWorldProver(Prover9Command(read_expr('fly(do)'), premises))).prove()
True
>>> UniqueNamesProver(ClosedWorldProver(Prover9Command(read_expr('-fly(os)'), premises))).prove()
True

             
               

###Examples - Sentiment analysis
#Sentiment analysis (sometimes known as opinion mining or emotion AI) 
#refers to the use of natural language processing, text analysis, computational linguistics, and biometrics 
#to systematically identify, extract, quantify, and study affective states and subjective information.

#sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event. The attitude may be a judgment or evaluation (see appraisal theory), affective state (that is to say, the emotional state of the author or speaker), or the intended emotional communication (that is to say, the emotional effect intended by the author or interlocutor).


>>> from nltk.classify import NaiveBayesClassifier
>>> from nltk.corpus import subjectivity
>>> from nltk.sentiment import SentimentAnalyzer
>>> from nltk.sentiment.util import *

>>> n_instances = 100
>>> subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
>>> obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
>>> len(subj_docs), len(obj_docs)
(100, 100)


#Each document is represented by a tuple (sentence, label). 
#The sentence is tokenized, so it is represented by a list of strings:

>>> subj_docs[0]
(['smart', 'and', 'alert', ',', 'thirteen', 'conversations', 'about', 'one',
'thing', 'is', 'a', 'small', 'gem', '.'], 'subj')


#We separately split subjective and objective instances 
#to keep a balanced uniform class distribution in both train and test sets.

>>> train_subj_docs = subj_docs[:80]
>>> test_subj_docs = subj_docs[80:100]
>>> train_obj_docs = obj_docs[:80]
>>> test_obj_docs = obj_docs[80:100]
>>> training_docs = train_subj_docs+train_obj_docs
>>> testing_docs = test_subj_docs+test_obj_docs

>>> sentim_analyzer = SentimentAnalyzer()
>>> all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])


#We use simple unigram word features, handling negation:

>>> unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
>>> len(unigram_feats)
83
>>> sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


#We apply features to obtain a feature-value representation of our datasets:

>>> training_set = sentim_analyzer.apply_features(training_docs)
>>> test_set = sentim_analyzer.apply_features(testing_docs)


#We can now train our classifier on the training set, 
#and subsequently output the evaluation results:

>>> trainer = NaiveBayesClassifier.train
>>> classifier = sentim_analyzer.train(trainer, training_set)
Training classifier
>>> for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
...     print('{0}: {1}'.format(key, value))
Evaluating NaiveBayesClassifier results...
Accuracy: 0.8
F-measure [obj]: 0.8
F-measure [subj]: 0.8
Precision [obj]: 0.8
Precision [subj]: 0.8
Recall [obj]: 0.8
Recall [subj]: 0.8



#Vader

>>> from nltk.sentiment.vader import SentimentIntensityAnalyzer
>>> sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
        "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
        "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
        "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
        "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
        "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
        "The book was good.",         # positive sentence
        "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
        "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
        "A really bad, horrible book.",       # negative sentence with booster words
        "At least it isn't a horrible book.", # negated negative sentence with contraction
        ":) and :D",     # emoticons handled
        "",              # an empty string is correctly handled
        "Today sux",     #  negative slang handled
        "Today sux!",    #  negative slang with punctuation emphasis handled
        "Today SUX!",    #  negative slang with capitalization emphasis
        "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
    ]
>>> paragraph = "It was one of the worst movies I've seen, despite good reviews. \
    Unbelievably bad acting!! Poor direction. VERY poor production. \
    The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

>>> from nltk import tokenize
>>> lines_list = tokenize.sent_tokenize(paragraph)
>>> sentences.extend(lines_list)

>>> tricky_sentences = [
        "Most automated sentiment analysis tools are shit.",
        "VADER sentiment analysis is the shit.",
        "Sentiment analysis has never been good.",
        "Sentiment analysis with VADER has never been this good.",
        "Warren Beatty has never been so entertaining.",
        "I won't say that the movie is astounding and I wouldn't claim that \
        the movie is too banal either.",
        "I like to hate Michael Bay films, but I couldn't fault this one",
        "It's one thing to watch an Uwe Boll film, but another thing entirely \
        to pay for it",
        "The movie was too good",
        "This movie was actually neither that funny, nor super witty.",
        "This movie doesn't care about cleverness, wit or any other kind of \
        intelligent humor.",
        "Those who find ugly meanings in beautiful things are corrupt without \
        being charming.",
        "There are slow and repetitive parts, BUT it has just enough spice to \
        keep it interesting.",
        "The script is not fantastic, but the acting is decent and the cinematography \
        is EXCELLENT!",
        "Roger Dodger is one of the most compelling variations on this theme.",
        "Roger Dodger is one of the least compelling variations on this theme.",
        "Roger Dodger is at least compelling as a variation on the theme.",
        "they fall in love with the product",
        "but then it breaks",
        "usually around the time the 90 day warranty expires",
        "the twin towers collapsed today",
        "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
        under orders and in the ''least offensive way possible.''"
    ]
>>> sentences.extend(tricky_sentences)
>>> sid = SentimentIntensityAnalyzer()
>>> for sentence in sentences:
        print(sentence)
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
VADER is smart, handsome, and funny.
compound: 0.8316, neg: 0.0, neu: 0.254, pos: 0.746,
VADER is smart, handsome, and funny!
compound: 0.8439, neg: 0.0, neu: 0.248, pos: 0.752,
VADER is very smart, handsome, and funny.
compound: 0.8545, neg: 0.0, neu: 0.299, pos: 0.701,
VADER is VERY SMART, handsome, and FUNNY.
compound: 0.9227, neg: 0.0, neu: 0.246, pos: 0.754,
VADER is VERY SMART, handsome, and FUNNY!!!
compound: 0.9342, neg: 0.0, neu: 0.233, pos: 0.767,
VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!
compound: 0.9469, neg: 0.0, neu: 0.294, pos: 0.706,
The book was good.
compound: 0.4404, neg: 0.0, neu: 0.508, pos: 0.492,
The book was kind of good.
compound: 0.3832, neg: 0.0, neu: 0.657, pos: 0.343,
The plot was good, but the characters are uncompelling and the dialog is not great.
compound: -0.7042, neg: 0.327, neu: 0.579, pos: 0.094,
A really bad, horrible book.
compound: -0.8211, neg: 0.791, neu: 0.209, pos: 0.0,
At least it isn't a horrible book.
compound: 0.431, neg: 0.0, neu: 0.637, pos: 0.363,
:) and :D
compound: 0.7925, neg: 0.0, neu: 0.124, pos: 0.876,
<BLANKLINE>
compound: 0.0, neg: 0.0, neu: 0.0, pos: 0.0,
Today sux
compound: -0.3612, neg: 0.714, neu: 0.286, pos: 0.0,
Today sux!
compound: -0.4199, neg: 0.736, neu: 0.264, pos: 0.0,
Today SUX!
compound: -0.5461, neg: 0.779, neu: 0.221, pos: 0.0,
Today kinda sux! But I'll get by, lol
compound: 0.2228, neg: 0.195, neu: 0.531, pos: 0.274,
It was one of the worst movies I've seen, despite good reviews.
compound: -0.7584, neg: 0.394, neu: 0.606, pos: 0.0,
Unbelievably bad acting!!
compound: -0.6572, neg: 0.686, neu: 0.314, pos: 0.0,
Poor direction.
compound: -0.4767, neg: 0.756, neu: 0.244, pos: 0.0,
VERY poor production.
compound: -0.6281, neg: 0.674, neu: 0.326, pos: 0.0,
The movie was bad.
compound: -0.5423, neg: 0.538, neu: 0.462, pos: 0.0,
Very bad movie.
compound: -0.5849, neg: 0.655, neu: 0.345, pos: 0.0,
VERY bad movie.
compound: -0.6732, neg: 0.694, neu: 0.306, pos: 0.0,
VERY BAD movie.
compound: -0.7398, neg: 0.724, neu: 0.276, pos: 0.0,
VERY BAD movie!
compound: -0.7616, neg: 0.735, neu: 0.265, pos: 0.0,
Most automated sentiment analysis tools are shit.
compound: -0.5574, neg: 0.375, neu: 0.625, pos: 0.0,
VADER sentiment analysis is the shit.
compound: 0.6124, neg: 0.0, neu: 0.556, pos: 0.444,
Sentiment analysis has never been good.
compound: -0.3412, neg: 0.325, neu: 0.675, pos: 0.0,
Sentiment analysis with VADER has never been this good.
compound: 0.5228, neg: 0.0, neu: 0.703, pos: 0.297,
Warren Beatty has never been so entertaining.
compound: 0.5777, neg: 0.0, neu: 0.616, pos: 0.384,
I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.
compound: 0.4215, neg: 0.0, neu: 0.851, pos: 0.149,
I like to hate Michael Bay films, but I couldn't fault this one
compound: 0.3153, neg: 0.157, neu: 0.534, pos: 0.309,
It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it
compound: -0.2541, neg: 0.112, neu: 0.888, pos: 0.0,
The movie was too good
compound: 0.4404, neg: 0.0, neu: 0.58, pos: 0.42,
This movie was actually neither that funny, nor super witty.
compound: -0.6759, neg: 0.41, neu: 0.59, pos: 0.0,
This movie doesn't care about cleverness, wit or any other kind of intelligent humor.
compound: -0.1338, neg: 0.265, neu: 0.497, pos: 0.239,
Those who find ugly meanings in beautiful things are corrupt without being charming.
compound: -0.3553, neg: 0.314, neu: 0.493, pos: 0.192,
There are slow and repetitive parts, BUT it has just enough spice to keep it interesting.
compound: 0.4678, neg: 0.079, neu: 0.735, pos: 0.186,
The script is not fantastic, but the acting is decent and the cinematography is EXCELLENT!
compound: 0.7565, neg: 0.092, neu: 0.607, pos: 0.301,
Roger Dodger is one of the most compelling variations on this theme.
compound: 0.2944, neg: 0.0, neu: 0.834, pos: 0.166,
Roger Dodger is one of the least compelling variations on this theme.
compound: -0.1695, neg: 0.132, neu: 0.868, pos: 0.0,
Roger Dodger is at least compelling as a variation on the theme.
compound: 0.2263, neg: 0.0, neu: 0.84, pos: 0.16,
they fall in love with the product
compound: 0.6369, neg: 0.0, neu: 0.588, pos: 0.412,
but then it breaks
compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0,
usually around the time the 90 day warranty expires
compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0,
the twin towers collapsed today
compound: -0.2732, neg: 0.344, neu: 0.656, pos: 0.0,
However, Mr. Carter solemnly argues, his client carried out the kidnapping under orders and in the ''least offensive way possible.''
compound: -0.5859, neg: 0.23, neu: 0.697, pos: 0.074,


###Examples - Combinatory categorial grammar
#Combinatory categorial grammar (CCG) is an efficiently parsable, 
#yet linguistically expressive grammar formalism. 

#CCG relies on combinatory logic, which has the same expressive power as the lambda calculus, 
#but builds its expressions differently

#For example, the combinator B (the compositor) is useful in creating long-distance dependencies, 
#as in "Who do you think Mary is talking about?" 
#and the combinator W (the duplicator) is useful as the lexical interpretation of reflexive pronouns, 
#as in "Mary talks about herself". 

#Together with I (the identity mapping) and C (the permutator) these form a set of primitive, non-interdefinable combinators. 



#Relative Clauses

>>> from nltk.ccg import chart, lexicon


Construct a lexicon:

>>> lex = lexicon.parseLexicon('''
        :- S, NP, N, VP
    
        Det :: NP/N
        Pro :: NP
        Modal :: S\\NP/VP
    
        TV :: VP/NP
        DTV :: TV/NP
    
        the => Det
    
        that => Det
        that => NP
    
        I => Pro
        you => Pro
        we => Pro
    
        chef => N
        cake => N
        children => N
        dough => N
    
        will => Modal
        should => Modal
        might => Modal
        must => Modal
    
        and => var\\.,var/.,var
    
        to => VP[to]/VP
    
        without => (VP\\VP)/VP[ing]
    
        be => TV
        cook => TV
        eat => TV
    
        cooking => VP[ing]/NP
    
        give => DTV
    
        is => (S\\NP)/NP
        prefer => (S\\NP)/NP
    
        which => (N\\N)/(S/NP)
    
        persuade => (VP/VP[to])/NP
        ''')

>>> parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
>>> for parse in parser.parse("you prefer that cake".split()):  # doctest: +SKIP
        chart.printCCGDerivation(parse)
        break
    
 you    prefer      that   cake
 NP   ((S\NP)/NP)  (NP/N)   N
                  -------------->
                        NP
     --------------------------->
               (S\NP)
--------------------------------<
               S

>>> for parse in parser.parse("that is the cake which you prefer".split()):  # doctest: +SKIP
        chart.printCCGDerivation(parse)
        break
    
 that      is        the    cake      which       you    prefer
  NP   ((S\NP)/NP)  (NP/N)   N    ((N\N)/(S/NP))  NP   ((S\NP)/NP)
                                                 ----->T
                                              (S/(S\NP))
                                                 ------------------>B
                                                       (S/NP)
                                 ---------------------------------->
                                               (N\N)
                           ----------------------------------------<
                                              N
                   ------------------------------------------------>
                                          NP
      ------------------------------------------------------------->
                                 (S\NP)
-------------------------------------------------------------------<
                                 S


#Some other sentences to try: "that is the cake which we will persuade the chef to cook" "that is the cake which we will persuade the chef to give the children"

>>> sent = "that is the dough which you will eat without cooking".split()
>>> nosub_parser = chart.CCGChartParser(lex, chart.ApplicationRuleSet +
                        chart.CompositionRuleSet + chart.TypeRaiseRuleSet)


Without Substitution (no output)

>>> for parse in nosub_parser.parse(sent):
        chart.printCCGDerivation(parse)


With Substitution:

>>> for parse in parser.parse(sent):  # doctest: +SKIP
        chart.printCCGDerivation(parse)
        break
    
 that      is        the    dough      which       you     will        eat          without           cooking
  NP   ((S\NP)/NP)  (NP/N)    N    ((N\N)/(S/NP))  NP   ((S\NP)/VP)  (VP/NP)  ((VP\VP)/VP['ing'])  (VP['ing']/NP)
                                                  ----->T
                                               (S/(S\NP))
                                                                             ------------------------------------->B
                                                                                         ((VP\VP)/NP)
                                                                    ----------------------------------------------<Sx
                                                                                       (VP/NP)
                                                       ----------------------------------------------------------->B
                                                                               ((S\NP)/NP)
                                                  ---------------------------------------------------------------->B
                                                                               (S/NP)
                                  -------------------------------------------------------------------------------->
                                                                       (N\N)
                           ---------------------------------------------------------------------------------------<
                                                                      N
                   ----------------------------------------------------------------------------------------------->
                                                                 NP
      ------------------------------------------------------------------------------------------------------------>
                                                         (S\NP)
------------------------------------------------------------------------------------------------------------------<
                                                        S



#Conjunction

>>> from nltk.ccg.chart import CCGChartParser, ApplicationRuleSet, CompositionRuleSet
>>> from nltk.ccg.chart import SubstitutionRuleSet, TypeRaiseRuleSet, printCCGDerivation
>>> from nltk.ccg import lexicon


#Lexicons for the tests:

>>> test1_lex = '''
            :- S,N,NP,VP
            I => NP
            you => NP
            will => S\\NP/VP
            cook => VP/NP
            which => (N\\N)/(S/NP)
            and => var\\.,var/.,var
            might => S\\NP/VP
            eat => VP/NP
            the => NP/N
            mushrooms => N
            parsnips => N'''
>>> test2_lex = '''
            :- N, S, NP, VP
            articles => N
            the => NP/N
            and => var\\.,var/.,var
            which => (N\\N)/(S/NP)
            I => NP
            anyone => NP
            will => (S/VP)\\NP
            file => VP/NP
            without => (VP\\VP)/VP[ing]
            forget => VP/NP
            reading => VP[ing]/NP
            '''


#Tests handling of conjunctions. 
#Note that while the two derivations are different, they are semantically equivalent.

>>> lex = lexicon.parseLexicon(test1_lex)
>>> parser = CCGChartParser(lex, ApplicationRuleSet + CompositionRuleSet + SubstitutionRuleSet)
>>> for parse in parser.parse("I will cook and might eat the mushrooms and parsnips".split()):
...     printCCGDerivation(parse) # doctest: +NORMALIZE_WHITESPACE +SKIP
 I      will       cook               and                might       eat     the    mushrooms             and             parsnips
 NP  ((S\NP)/VP)  (VP/NP)  ((_var2\.,_var2)/.,_var2)  ((S\NP)/VP)  (VP/NP)  (NP/N)      N      ((_var2\.,_var2)/.,_var2)     N
    ---------------------->B
         ((S\NP)/NP)
                                                     ---------------------->B
                                                          ((S\NP)/NP)
                          ------------------------------------------------->
                                     (((S\NP)/NP)\.,((S\NP)/NP))
    -----------------------------------------------------------------------<
                                  ((S\NP)/NP)
                                                                                              ------------------------------------->
                                                                                                             (N\.,N)
                                                                                   ------------------------------------------------<
                                                                                                          N
                                                                           -------------------------------------------------------->
                                                                                                      NP
    ------------------------------------------------------------------------------------------------------------------------------->
                                                                (S\NP)
-----------------------------------------------------------------------------------------------------------------------------------<
                                                                 S
 I      will       cook               and                might       eat     the    mushrooms             and             parsnips
 NP  ((S\NP)/VP)  (VP/NP)  ((_var2\.,_var2)/.,_var2)  ((S\NP)/VP)  (VP/NP)  (NP/N)      N      ((_var2\.,_var2)/.,_var2)     N
    ---------------------->B
         ((S\NP)/NP)
                                                     ---------------------->B
                                                          ((S\NP)/NP)
                          ------------------------------------------------->
                                     (((S\NP)/NP)\.,((S\NP)/NP))
    -----------------------------------------------------------------------<
                                  ((S\NP)/NP)
    ------------------------------------------------------------------------------->B
                                      ((S\NP)/N)
                                                                                              ------------------------------------->
                                                                                                             (N\.,N)
                                                                                   ------------------------------------------------<
                                                                                                          N
    ------------------------------------------------------------------------------------------------------------------------------->
                                                                (S\NP)
-----------------------------------------------------------------------------------------------------------------------------------<
                                                                 S


#Tests handling subject extraction. 
#Interesting to point that the two parses are clearly semantically different.

>>> lex = lexicon.parseLexicon(test2_lex)
>>> parser = CCGChartParser(lex, ApplicationRuleSet + CompositionRuleSet + SubstitutionRuleSet)
>>> for parse in parser.parse("articles which I will file and forget without reading".split()):
        printCCGDerivation(parse)  # doctest: +NORMALIZE_WHITESPACE +SKIP
 articles      which       I      will       file               and             forget         without           reading
    N      ((N\N)/(S/NP))  NP  ((S/VP)\NP)  (VP/NP)  ((_var3\.,_var3)/.,_var3)  (VP/NP)  ((VP\VP)/VP['ing'])  (VP['ing']/NP)
                          -----------------<
                               (S/VP)
                                                                                        ------------------------------------->B
                                                                                                    ((VP\VP)/NP)
                                                                               ----------------------------------------------<Sx
                                                                                                  (VP/NP)
                                                    ------------------------------------------------------------------------->
                                                                               ((VP/NP)\.,(VP/NP))
                                           ----------------------------------------------------------------------------------<
                                                                                (VP/NP)
                          --------------------------------------------------------------------------------------------------->B
                                                                        (S/NP)
          ------------------------------------------------------------------------------------------------------------------->
                                                                 (N\N)
-----------------------------------------------------------------------------------------------------------------------------<
                                                              N
 articles      which       I      will       file               and             forget         without           reading
    N      ((N\N)/(S/NP))  NP  ((S/VP)\NP)  (VP/NP)  ((_var3\.,_var3)/.,_var3)  (VP/NP)  ((VP\VP)/VP['ing'])  (VP['ing']/NP)
                          -----------------<
                               (S/VP)
                                                    ------------------------------------>
                                                            ((VP/NP)\.,(VP/NP))
                                           ---------------------------------------------<
                                                              (VP/NP)
                                                                                        ------------------------------------->B
                                                                                                    ((VP\VP)/NP)
                                           ----------------------------------------------------------------------------------<Sx
                                                                                (VP/NP)
                          --------------------------------------------------------------------------------------------------->B
                                                                        (S/NP)
          ------------------------------------------------------------------------------------------------------------------->
                                                                 (N\N)
-----------------------------------------------------------------------------------------------------------------------------<
                                                              N




                                                      
###Examples - Alignment
   

#Corpus Reader

>>> from nltk.corpus import comtrans
>>> words = comtrans.words('alignment-en-fr.txt')
>>> for word in words:
        print(word)
Resumption
of
the
session
I
declare...
>>> als = comtrans.aligned_sents('alignment-en-fr.txt')[0]
>>> als  
AlignedSent(['Resumption', 'of', 'the', 'session'],
['Reprise', 'de', 'la', 'session'],
Alignment([(0, 0), (1, 1), (2, 2), (3, 3)]))



#Alignment Objects
#Aligned sentences are simply a mapping between words in a sentence:

>>> print(" ".join(als.words))
Resumption of the session
>>> print(" ".join(als.mots))
Reprise de la session
>>> als.alignment
Alignment([(0, 0), (1, 1), (2, 2), (3, 3)])


#Usually we look at them from the perpective of a source to a target languge, 
#but they are easilly inverted:

>>> als.invert() # doctest: +NORMALIZE_WHITESPACE
AlignedSent(['Reprise', 'de', 'la', 'session'],
['Resumption', 'of', 'the', 'session'],
Alignment([(0, 0), (1, 1), (2, 2), (3, 3)]))


#We can create new alignments, 
#but these need to be in the correct range of the corresponding sentences:

>>> from nltk.align import Alignment, AlignedSent
>>> als = AlignedSent(['Reprise', 'de', 'la', 'session'],
...                   ['Resumption', 'of', 'the', 'session'],
...                   Alignment([(0, 0), (1, 4), (2, 1), (3, 3)]))
Traceback (most recent call last):
    ...
IndexError: Alignment is outside boundary of mots


#You can set alignments with any sequence of tuples, 
#so long as the first two indexes of the tuple are the alignment indices:

als.alignment = Alignment([(0, 0), (1, 1), (2, 2, "boat"), (3, 3, False, (1,2))])

>>> Alignment([(0, 0), (1, 1), (2, 2, "boat"), (3, 3, False, (1,2))])
Alignment([(0, 0), (1, 1), (2, 2, 'boat'), (3, 3, False, (1, 2))])



#Alignment Algorithms
#EM for IBM Model 1
#Here is an example from Koehn, 2010:

>>> from nltk.align import IBMModel1
>>> corpus = [AlignedSent(['the', 'house'], ['das', 'Haus']),
...           AlignedSent(['the', 'book'], ['das', 'Buch']),
...           AlignedSent(['a', 'book'], ['ein', 'Buch'])]
>>> em_ibm1 = IBMModel1(corpus, 20)
>>> print(round(em_ibm1.probabilities['the']['das'], 1))
1.0
>>> print(round(em_ibm1.probabilities['book']['das'], 1))
0.0
>>> print(round(em_ibm1.probabilities['house']['das'], 1))
0.0
>>> print(round(em_ibm1.probabilities['the']['Buch'], 1))
0.0
>>> print(round(em_ibm1.probabilities['book']['Buch'], 1))
1.0
>>> print(round(em_ibm1.probabilities['a']['Buch'], 1))
0.0
>>> print(round(em_ibm1.probabilities['book']['ein'], 1))
0.0
>>> print(round(em_ibm1.probabilities['a']['ein'], 1))
1.0
>>> print(round(em_ibm1.probabilities['the']['Haus'], 1))
0.0
>>> print(round(em_ibm1.probabilities['house']['Haus'], 1))
1.0
>>> print(round(em_ibm1.probabilities['book'][None], 1))
0.5


#And using an NLTK corpus. We train on only 10 sentences, since it is so slow:

>>> from nltk.corpus import comtrans
>>> com_ibm1 = IBMModel1(comtrans.aligned_sents()[:10], 20)
>>> print(round(com_ibm1.probabilities['bitte']['Please'], 1))
0.2
>>> print(round(com_ibm1.probabilities['Sitzungsperiode']['session'], 1))
1.0



#Evaluation
#The evaluation metrics for alignments are usually not interested in the contents of alignments but more often the comparison to a "gold standard" alignment that has been been constructed by human experts. For this reason we often want to work just with raw set operations against the alignment points. This then gives us a very clean form for defining our evaluation metrics.
#The AlignedSent class has no distinction of "possible" or "sure" alignments. Thus all alignments are treated as "sure".

Consider the following aligned sentence for evaluation:

>>> my_als = AlignedSent(['Resumption', 'of', 'the', 'session'],
        ['Reprise', 'de', 'la', 'session'],
        [(0, 0), (3, 3), (1, 2), (1, 1), (1, 3)])



#Precision
#precision = |AnP| / |A|


>>> print(als.precision(set()))
0.0
>>> print(als.precision([(0,0), (1,1), (2,2), (3,3)]))
1.0
>>> print(als.precision([(0,0), (3,3)]))
0.5
>>> print(als.precision([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]))
1.0
>>> print(my_als.precision(als))
0.6



#Recall
#recall = |AnS| / |S|



>>> print(als.recall(set()))
None
>>> print(als.recall([(0,0), (1,1), (2,2), (3,3)]))
1.0
>>> print(als.recall([(0,0), (3,3)]))
1.0
>>> print(als.recall([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]))
0.66666666666...
>>> print(my_als.recall(als))
0.75



#Alignment Error Rate (AER)
#AER = 1 - (|AnS| + |AnP|) / (|A| + |S|)



>>> print(als.alignment_error_rate(set()))
1.0
>>> print(als.alignment_error_rate([(0,0), (1,1), (2,2), (3,3)]))
0.0
>>> print(my_als.alignment_error_rate(als))
0.33333333333...
>>> print(my_als.alignment_error_rate(als,
...     als.alignment | set([(1,2), (2,1)])))
0.22222222222...




 