from .prefix_beam_search import prefix_beam_search
from typing import Dict, Sequence, Union
from transformers import pipeline, Wav2Vec2Processor
from datasets import Dataset, Audio
from evaluate import load
import torch

from argparse import ArgumentParser

pplx = load('perplexity')

def wav_to_hf_audio(file: Union[str, Sequence[str]]) -> Dataset:
    if type(file) is str:
        file = [file]
    audio_dict = {'audio':file}
    ds = Dataset.from_dict(audio_dict).cast_column('audio', Audio(sampling_rate=16000))
    return ds

def decode_audio(
        file: Union[str, Sequence[str]],
        asr: str,
        lm: str,
) -> Dict[str, str]:
    if type(file) is str:
        file = [file,]
    pipeline = pipeline('automatic-speech-recognition', asr)
    ctc_out = pipeline(file)

    lm_funct = lambda s: pplx(predictions=[s,], model_id=lm)['mean_perplexity']
    audio_ds = wav_to_hf_audio(file)
    processor = Wav2Vec2Processor.from_pretrained(asr)
    def get_ctc_logits(audio: torch.tensor):
        input_dict = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        logits = asr(input_dict.input_values).logits
        return logits
    
    def map_beam_search(row: dict):
        audio = row['audio']
        ctc_logits = get_ctc_logits(audio)
        return {'label': prefix_beam_search(ctc_logits, lm_funct)}


    beam_search_out = audio_ds.map(map_beam_search, remove_columns=['audio'])

    return {
        'ctc': ctc_out,
        'beam_search': beam_search_out,
    }

if __name__ == '__main__':
    parser = ArgumentParser('Beam search runner')
    parser.add_argument('WAV', 'audio file to decode')
    parser.add_argument('ASR', 'asr model path')
    parser.add_argument('LM', 'lm model path')