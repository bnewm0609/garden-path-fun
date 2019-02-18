from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG, LMModel
from generate import append_batch
import torch
import math
import numpy as np
# from matplotlib import pyplot as plt
from text_utils import TextEncoder

import spacy
from collections import Counter

class SurprisalAnalyzer:

    def __init__(self):
        # initialize lm and text encoder and everything

        # set up the encoder to turn words into indices
        encoder_path = 'model/encoder_bpe_40000.json'
        bpe_path = 'model/vocab_40000.bpe'
        self.text_encoder = TextEncoder(encoder_path, bpe_path)

        self.nvocab = len(self.text_encoder.encoder)
        nctx = 512 # number of positional embeddings (nctx = number of context)
        vocab = self.nvocab + nctx

        # set up pretrained openai model
        args = DEFAULT_CONFIG
        self.lm_model = LMModel(args, vocab, nctx, return_probs = True)
        load_openai_pretrained_model(self.lm_model.transformer, n_ctx=nctx, n_special=0)
        self.lm_model.eval() # this line puts the model in eval mode so we don't do dropout :) 


        # set up spacy for pos tagging
        self.nlp = spacy.load('en', disable=['ner', 'textcat', 'parser'])

    def  make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        # add positional encodings - just second dimension that says which word is where
        pos_enc = np.arange(self.nvocab, self.nvocab + X.shape[-1])
        pos_enc = np.expand_dims(pos_enc, axis=0)
        batch = np.stack([X, pos_enc], axis=-1)
        batch = torch.tensor(batch, dtype=torch.long)
        return batch

    def _get_continuation_tensor(self, sent_vec):
        """
        Deals strictly with tensors
        """
        sent_batch = self.make_batch(sent_vec)
        sent_res = self.lm_model(sent_batch)
        return sent_res

    def tensor_to_probs(self, tensor):
        """
        converts torch tensor to clean numpy array holding probabilities
        (Basically just hides some nasty code)
        """
        return tensor[:, -1, :].flatten().detach().numpy()

    def get_continuation_probs(self, sentence):
        sent_vec = self.text_encoder.encode([sentence])
        tensor = self._get_continuation_tensor(sent_vec)
        return self.tensor_to_probs(tensor)

    def _get_continuations(self, sent_res, k=10, verbose=False):
        """
        Making this private so I can access it externally... that's awful

        This is a helper function for the `get_continuations` wrapper that 
        separates the actual processing of the sentence from getting top
        continuations
        """
        probs, decode = sent_res[:,-1,:].topk(k)
        if verbose:
            for p, d in zip(probs.flatten(), decode.flatten()):
                print("\t...%s (%.4f)"%(self.text_encoder.decoder[d.item()], p.item()))
        words = [self.text_encoder.decoder[d.item()] for d in decode.flatten()]
        # strip of the word ending tags if there are some - if it's not a full continuation, what to do?
        for i in range(len(words)):
            if words[i][-4:] == "</w>":
                words[i] = words[i][:-4]
        probs = probs.flatten().detach().numpy() # convert probs from tensor to numpy array
        return words, probs

    def get_continuations(self, sentence, k=10, verbose=False):
        """
        sentence: a string that you want to get next words for
        k: how many next words you want to get
        verbose: do you want to print the output
        """
        sent_vec = self.text_encoder.encode([sentence])
        sent_res = self._get_continuation_tensor(sent_vec)
        if verbose:
            print(sentence)

        return self._get_continuations(sent_res, k, verbose)


    def _get_pos_continuations(self, sentence, words, probs):
        """
        helper function for `get_pos_continuations` that takes the lists of words and
        probabilities and performs all the computation to get the most common pos
        tags independently of processing an individual sentence
        """
        # get POS of all of k continuations
        pos_counter = Counter()

        for word, prob in zip(words, probs):
            sentence_continuation = "{} {}".format(sentence, word)
            encoded = self.nlp(sentence_continuation)
            pos_counter[encoded[-1].pos_] += prob

        # format pos_counter most common output as two lists, one of probs and one of pos tags
        pos_counter_list = list(zip(*pos_counter.most_common()))
        pos_tags, pos_tag_probs = list(pos_counter_list[0]), np.array((pos_counter_list[1]), dtype=np.float32)
        return pos_tags, pos_tag_probs

    def get_pos_continuations(self, sentence, k=10, verbose=False):
        """
        sentence: string you want next parts of speech for
        k: how many top words to analyze 
        NOTE: unlike in the `get_continuation` function, the k is NOT how many
        unique POS tags you want to look at, it's how many words you want to consider
        """
        # get likely next words
        words, probs = self.get_continuations(sentence, k, verbose=False)
        return self._get_pos_continuations(sentence, words, probs)



    ################################################################################
    # The following three functions calculate entropy/surprisal of a SINGLE function
    ################################################################################
    def _get_surprisal(self, distribution, index):
        word_prob = distribution[index]
        return -np.log2(word_prob)
    
    def get_surprisal(self, sentence, word):
        """
        get the -log2 probability of the word following the sentence
        """
        all_probs = self.get_continuation_probs(sentence)
        # if the word is not in the vocabulary in full, represent its probability by the 
        # probability of the first part of its encoding (the 0 index)
        word_index = self.text_encoder.encode([word])[0]
        # word_prob = all_probs[word_index]
        return self._get_surprisal(all_probs, word_index)#-np.log2(word_prob)

    def _get_entropy(self, distribution):
        return -np.sum([p*np.log2(p) if p > 0 else 0 for p in distribution])

    def get_entropy(self, sentence):
        """
        finds the shannon entropy of predicting the word following sentence
        """
        all_probs = self.get_continuation_probs(sentence)
        return self._get_entropy(all_probs)#-np.sum([p*np.log2(p) if p > 0 else 0 for p in all_probs])

    def get_surprisal_entropy_ratio(self, sentence, word):
        "gets ratio betwen surprisal and entropy at the end of the sentence for a given word"
        all_probs = self.get_continuation_probs(sentence)
        word_index = self.text_encoder.encode([word])[0]
        entropy = self._get_entropy(all_probs)
        surprisal = self._get_surprisal(all_probs, word_index)
        return surprisal/entropy

    ####################################################################
    # Same as above but for part of speech
    ####################################################################
    def get_surprisal_pos(self, sentence, pos, k=1000):
        """
        Because we the language model is not a POS tagger, we cannot directly
        calculate the surprisal of the pos from a full probability distribution,
        instead we have to use the degenerate distribution computed from the 
        top k most probable POS continuations

        sentence is full sentence
        pos is pos we want to get surprisal of
        k is how many possible continuations to check
        """
        pos_tags, pos_tag_probs = self.get_pos_continuations(sentence, k)
        pos_index = pos_tags.index(pos) # assume the POS we want is in the list somewhere...
        return self._get_surprisal(pos_tag_probs, pos_index)

        
    def get_entropy_pos(self, sentence, k=1000):
        """
        Disclaimer about degenerate distribution same as above
        """
        pos_tags, pos_tag_probs = self.get_pos_continuations(sentence, k)
        return self._get_entropy(pos_tag_probs)


    
    #####################################################################
    # Gets all of the above metrics for every word in a single sentence #
    #####################################################################
    def get_surprisal_sentence(self, sentence, prepend=None, start=1):
        """
        A little uglier, but perhaps faster

        """
        surprisals = []
        sent_enc = self.text_encoder.encode([sentence])[0] # list of indices in enocder 1-d
        if prepend != None:
            sent_enc = prepend + sent_enc
        sent_dec = [self.text_encoder.decoder[ind] for ind in sent_enc]

        sent_batch = None

        # if you run the language model with the whole sentence the outputs for each
        # word are the probabilities for the next word!
        sent_batch = self.make_batch([sent_enc])
        sent_tensor = self.lm_model(sent_batch)
        for i in range(start, len(sent_enc)):
            surprisals.append(-np.log2(sent_tensor[:,i-1,sent_enc[i]].item()))
        return surprisals, sent_dec
        
    def get_s_h_shr_sentence(self, sentence, prepend=None, start=1):
        """
        calculates the surprisal, entropy, and surprisal-entropy-ratio at each word (defined by bpe)
        in the sentence

        returns, in order
        1. The list of surprisals (len(sentence) - 1)
        2. The list of entropies  (len(sentence) - 1)
        3. The list of rations between surprisals and entropies (len(sentence) - 1)
        4. The decoded tokens that are used by the BPE encoder wrapper
        """
        surprisals, entropies, surprisal_entropy_ratios = [],[],[]
        sent_enc = self.text_encoder.encode([sentence])[0] # list of indices in enocder 1-d
        if prepend != None:
            sent_enc = prepend + sent_enc
        sent_dec = [self.text_encoder.decoder[ind] for ind in sent_enc]

        # start = max(0, min(1, start)) # doesn't work because language model needs to condition on something
        start = 1

        for i in range(start, len(sent_enc)):
            partial_sent_enc = [sent_enc[:i]]
            cont_tensor = self._get_continuation_tensor(partial_sent_enc)
            partial_probs = self.tensor_to_probs(cont_tensor)

            surprisals.append(self._get_surprisal(partial_probs, sent_enc[i]))
            entropies.append(self._get_entropy(partial_probs))
            surprisal_entropy_ratios.append(surprisals[-1]/entropies[-1])

        return surprisals, entropies, surprisal_entropy_ratios, sent_dec







if __name__ == "__main__":
    sa = SurprisalAnalyzer()

    print("1. Testing `get_continuations`")
    print("The horse raced past the barn:")
    print(sa.get_continuations("The horse raced past the barn"))

    print("2. Testing `get_pos_continuations`")
    print("The horse raced past the barn:")
    print(sa.get_pos_continuations("The horse raced past the barn"))

    print("3. Testing `get_surprisal`")
    print("The horse raced past the barn:")
    print(sa.get_surprisal("The horse raced past the barn", "fell"))

    print("4. Testing `get_entropy`")
    print("The horse raced past the barn:")
    print(sa.get_entropy("The horse raced past the barn"))

    print("5. Testing `get_surprisal_entrop_ratio`")
    print("The horse raced past the barn:")
    print(sa.get_surprisal_entropy_ratio("The horse raced past the barn", "fell"))


    print("6. Testing `get_s_h_shr_sentences`")
    print("The horse raced past the barn fell:")
    print(sa.get_s_h_shr_sentence("The horse raced past the barn fell"))

    print("7. Testing `get_s_h_shr_sentences` on oov input")
    print("The horse raced past the barn took a downturn:")
    print(sa.get_s_h_shr_sentence("The horse raced past the barn took a downturn"))

    print("8. Testing `get_surprisal_pos`")
    print("The horse raced past the barn:")
    print(sa.get_surprisal_pos("The horse raced past the barn", "VERB"))

    print("9. Testing `get_entropy_pos`")
    print("The horse raced past the barn:")
    print(sa.get_entropy_pos("The horse raced past the barn"))













