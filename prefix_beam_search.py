from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import numpy as np

from typing import Dict, Sequence, Union, Optional, Any, List, Callable, Mapping
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset, Audio
from evaluate import load
import torch
import sys
import json

from argparse import ArgumentParser

pplx = load('perplexity')

class Memo(dict):
    def __init__(self, factory):
         self.factory = factory
    def __missing__(self, key):
         self[key] = self.factory(key)
         return self[key]

def prefix_beam_search(
        ctc: Sequence[Sequence[float]],
        lm: Optional[Callable]=None,
        alphabet: Optional[Mapping]=None,
        blank: str = '%',
        space: str = ' ',
        eos: str = '>',
        k: int = 25,
        alpha: float = 0.30,
        beta: int = 5,
        prune: float = 0.001
    ) -> str:
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

    Retruns:
        string: The decoded CTC output.
    """

    lm = (lambda _: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    lm_memo = Memo(lm)
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    if not alphabet:
        alphabet = {c:i for i, c in enumerate(ascii_lowercase)}
        for spec_tok in [blank, space, eos]:
            alphabet[spec_tok] = len(alphabet)
    reverse_alphabet = {v:k for k, v in alphabet.items()}
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1 # probability of prefix ending w/ blank
    Pnb[0][O] = 0 # probability of prefix ending w/ non-blank
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = {reverse_alphabet[i]: i for i in np.where(ctc[t] > prune)[0]}
        for l in A_prev:
            
            if len(l) > 0 and l[-1] == eos:
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue  

            for c in pruned_alphabet:
                c_ix = alphabet[c]
                # END: STEP 2
                
                # STEP 3: “Extending” with a blank
                if c == blank:
                    Pb[t][l] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3
                
                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    else:#elif len(l.replace(space, '')) > 0 and c in (space, eos):
                    # comment out condition bc we don't want to only run lm on full words
                    # for a char-based model
                        lm_prob = lm_memo[l_plus.strip(space+eos)] ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # else:
                    #     Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    return A_prev[0].strip(eos)

def wav_to_hf_audio(file: Union[str, Sequence[str]]) -> Dataset:
    if type(file) is str:
        file = [file]
    audio_dict = {'audio':file}
    ds = Dataset.from_dict(audio_dict).cast_column('audio', Audio(sampling_rate=16000))
    return ds

def decode_audio(
        file: Union[str, Sequence[str], None],
        asr: str,
        lm: str,
        input_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    if type(file) is str:
        file = [file,]
    # pipe = pipeline('automatic-speech-recognition', asr)
    # print('Running ASR pipeline on audio file...')
    # ctc_out = pipe(file)

    lm_funct = lambda s: pplx.compute(predictions=[s,], model_id=lm)['mean_perplexity']
    audio_ds = wav_to_hf_audio(file)
    print("Loading ASR model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(asr)
    tokenizer=processor.tokenizer
    asr_model = Wav2Vec2ForCTC.from_pretrained(asr)
    alphabet = tokenizer.vocab
    alphabet[tokenizer.pad_token] = tokenizer.pad_token_id
    alphabet[' '] = tokenizer.word_delimiter_token_id
    def get_ctc_logits(audio: torch.tensor):
        input_dict = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        with torch.no_grad():
            logits = asr_model(input_dict.input_values).logits
        return logits
    
    def map_labels(row: dict):
        audio = row['audio']['array']
        ctc_logits = get_ctc_logits(audio)
        for example in ctc_logits:
            beam_search_label = prefix_beam_search(
                example,
                lm_funct,
                alphabet=alphabet,
                blank = tokenizer.pad_token,
                space = ' ',
            )
        ctc_label = tokenizer.decode(ctc_logits)
        return {
            'beam_search': beam_search_label,
            'ctc': ctc_label,
        }

    print('Running beam search on file...')
    labels = audio_ds.map(map_labels, remove_columns=['audio'])

    return {
        'ctc': labels['ctc'],
        'beam_search': labels['beam_search'],
    }

def toy_case():
    ctc= np.array([
        [0.4,  0.4,  0.2,  0. ,  0. ,  0. ],
        [0.2,  0.2,  0.2,  0. ,  0. ,  0.4],
        [0.4,  0.4,  0.2,  0. ,  0. ,  0. ],
        [0.2,  0.2,  0.2,  0. ,  0. ,  0.4],
        [0.35, 0.35, 0.3,  0. ,  0. ,  0. ],
        [0.2, 0.2, 0.2, 0. , 0. , 0.4],
        [0.2, 0.2, 0.2, 0. , 0. , 0.4],
        [0.2, 0.2, 0.2, 0. , 0. , 0.4],
        [0.4, 0.4, 0.2, 0. , 0. , 0. ]
    ])
    alphabet = {'b': 0, 'a': 1, 'c': 2, ' ': 3, '>': 4, '%': 5}
    scores = {'bac': 0.9, 'bcc': 0.1, 'baa': 0.5}
    lm = lambda s: scores.get(s, 0.1)
    print(prefix_beam_search(ctc, lm, alphabet))

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser('Beam search runner')
    parser.add_argument('WAV', help='audio file to decode')
    parser.add_argument('ASR', help='asr model path')
    parser.add_argument('LM', help='lm model path')
    parser.add_argument('OUT', help='JSON path to save results to')

    args = parser.parse_args(argv)

    out = decode_audio(args.WAV, args.ASR, args.LM)
    with open(args.OUT, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()

    