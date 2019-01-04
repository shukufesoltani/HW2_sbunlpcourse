import io
import re
from hazm import Normalizer,Lemmatizer
import numpy as np
def process_text(text):
    normalize=Normalizer()
    text=normalize.normalize(text)
    text = text.replace("_", " ")
    text = text.replace(',', ' ')
    text=text.replace("\u220c","")
    text=text.replace("\u200c","")
    text=text.replace("-","")
    # text = text.replace('/', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('.', ' ')
    text=text.replace("،"," ")
    text=text.replace("«"," ")
    text=text.replace("»"," ")
    # Convert text string to a list of words
    t = re.findall("[\u0627-\u06FF]+|<S>|</s>|\?|//", text)  # just split word by space to space and omit other thing
    lemma=Lemmatizer()
    text=[lemma.lemmatize(x) for x in t]
    return text

def generate_ngrams(words_list, n):
    ngrams_list = []
    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
    return ngrams_list

def unigram_probability(words_list):
    unigram_language_model={}
    unigam_count={}
    word_set=set()
    for word in words_list:
        word_set.add(word)
    pure_word_list=list(word_set)
    for word in pure_word_list:
        w_count=words_list.count(word)
        prob=(w_count/len(words_list))
        unigram_language_model[word]=prob
        unigam_count[word]=w_count
    # unigram_language_model={word:count}
    return unigram_language_model,unigam_count


def bigram_probality(bigram_list,unigram_prob):
    unigram_language_model=unigram_prob
    bigram_language_model={}
    bigram_count={}
    biword_set=set()
    for word in bigram_list:
        biword_set.add(word)
    pure_biword_list=list(biword_set)
    for word in pure_biword_list:
        w_count=bigram_list.count(word)
        wordSplit=word.split()
        w=wordSplit[0]
        makhrag=unigram_language_model[w]
        bigram_prob=w_count/makhrag
        bigram_language_model[word]=bigram_prob
        bigram_count[word]=w_count

    return bigram_language_model,bigram_count

def trigram_probability(trigram_list,bigram_count):
    trigram_set=set(trigram_list)
    trigram_pure_list=list(trigram_set)
    trigram_language_model={}
    trigram_count={}
    for triword in trigram_pure_list:
        w_count=trigram_list.count(triword)
        wordSplit=triword.split()
        if not (len(wordSplit)==3):
            if len(wordSplit)==2:
                wordSplit=["<S>"]+wordSplit
            if len(wordSplit)==1:
                wordSplit = ["<S>"] + wordSplit
                wordSplit = ["<S>"] + wordSplit

        del wordSplit[-1]
        # print(wordSplit)
        wordSplit=" ".join(wordSplit)
        trigram_count[triword]=w_count
        if bigram_count.get(wordSplit) is None:
            break
        prob=w_count/bigram_count[wordSplit]
        trigram_language_model[triword]=prob
    return trigram_language_model,trigram_count
def quadgram_probability(quadgram_list,trigram_count):
    writer=io.open("quadgram.txt","w",encoding="UTF-8")
    quadgram_set=set(quadgram_list)
    quadgram_pure_list=list(quadgram_set)
    quadgram_count={}
    quadgram_language_model={}
    for word in quadgram_pure_list:
        w_count=quadgram_list.count(word)
        quadgram_count[word]=w_count
        wordSplit=word.split()
        del wordSplit[-1]
        wordSplit=" ".join(wordSplit)
    return quadgram_language_model, quadgram_count

def fivegram_probality(fivegram_list,quadgram_count):
    writer=io.open("fivegram.txt","w",encoding="UTF-8")
    fivegram_set=set(fivegram_list)
    fivegram_pure_list=list(fivegram_set)
    fivegram_count={}
    fivegram_language_model={}
    for word in fivegram_pure_list:
        w_count=fivegram_list.count(word)
        fivegram_count[word]=w_count
        wordSplit=word.split()
        del wordSplit[-1]
        wordSplit=" ".join(wordSplit)
        if not (wordSplit is None):
            if not(quadgram_count.get(wordSplit) is None):
                fivegram_prob=w_count/quadgram_count[wordSplit]
                fivegram_language_model[word]=fivegram_prob
                writer.write(str(word)+","+str(fivegram_prob)+"\n")
    return fivegram_language_model, fivegram_count

def unigram_generator(unigram_model,token_list):
    unigram_model = [(k, unigram_model[k]) for k in
                  sorted(unigram_model, key=unigram_model.get, reverse=True)]
    unigram_model = np.asarray(unigram_model)
    unigram_word = unigram_model[:, 0]  # list of all sorted word
    unigram_prob = unigram_model[:, 1]  # list of all sorted prob
    unigram_prob = unigram_prob.astype(np.float)
    print("language_model is done...")
    start_word_list = np.random.choice(unigram_word, 100, p=unigram_prob)
    # make empty matrix for saving generate sentences
    generating_sentences = [[] for count in range(0, 100)]
    writer = open("unigram.txt", "w", encoding="UTF-8")
    print("unigram generation...")
    j=0
    for each_row in generating_sentences:
        print("no of sent", j)
        each_row.append(start_word_list[j])
        j = j + 1
        flag = True
        i = 0
        while (flag):
            i = i + 1
            next_word = np.random.choice(unigram_word, 1, p=unigram_prob)
            if (i == 1 and next_word == "</S>" and next_word == start_word_list[i]):
                next_word = np.random.choice(unigram_word, 1, p=unigram_prob, replace=False)
            each_row.append(next_word[0])
            if (i == 50 or next_word == "</S>" or next_word == "."):
                flag = False
        for obj in each_row:
            writer.write(obj + " ")
        writer.write("\n")


if __name__ == '__main__':
    with io.open("testSet.txt","r",encoding="UTF-8") as f:
        text=f.read()

    words_list = process_text(text)
    unigrams = generate_ngrams(words_list, 1)
    unigram_language_model,unigram_count = unigram_probability(unigrams)
    unigram_generator(unigram_language_model,words_list)
    print('unigram_language_model is done')
    bigrams = generate_ngrams(words_list, 2)
    bigram_language_model,bigram_count = bigram_probality(bigrams,unigram_count )

    print('bigram is done')
    trigrams = generate_ngrams(words_list, 3)
    trigram_language_model,trigram_count=trigram_probability(trigrams,bigram_count)
    print('trigram is done')
    fourgrams = generate_ngrams(words_list, 4)
    quadgram_language_model,quadgram_count=quadgram_probability(fourgrams,trigram_count)

    print('quadgram is done')
    fivegrams = generate_ngrams(words_list, 5)
    fivegram_language_model,fivegram_count=fivegram_probality(fourgrams,quadgram_count)
    print('pentagram is done')

