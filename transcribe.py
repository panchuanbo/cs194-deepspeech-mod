import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json

import numpy as np
import scipy.io

unique_ages = ['fourties', 'twenties', 'seventies', 'teens', 'sixties', 'thirties', 'eighties', 'fifties']
unique_genders = ['male', 'female']
unique_accents = ['african', 'newzealand', 'malaysia', 'us', 'england', 'indian', 'wales', 'philippines', 'southatlandtic', 'bermuda', 'scotland', 'australia', 'canada', 'singapore', 'ireland', 'hongkong']
unique_cols = ["#191970", "#87CEEB",
               "#FF8C00", "#808080",
               "#98FB98", "#FF00FF",
               "#8C1515", "#FFA07A",
               "#000080", "#F0FFFF",
               "#D3D3D3", "#228B22",
               "#F88072", "#FFA500",
               "#808000", "#5F93A0"]

def out_to_preds(out):
    out = out.detach().numpy()
    out = out.reshape(len(unique_ages) + len(unique_genders) + len(unique_accents))

    print(out)
    out[:8] /= np.sum(out[:8])
    out[8:10] /= np.sum(out[8:10])
    out[10:] /= np.sum(out[10:])
    out *= 100

    print(out)

    # ages = {k: float(out[i]) for (i, k) in enumerate(unique_ages)}
    # genders = {k: float(out[8+i]) for (i, k) in enumerate(unique_genders)}
    # accents = {k: float(out[10+i]) for (i, k) in enumerate(unique_accents)}

    ages = [{"title": k, "value": round(float(out[i]), 2), "color": unique_cols[i]} for (i, k) in enumerate(unique_ages)]
    print(ages)
    genders = [{"title": k, "value": round(float(out[8+i]), 2), "color": unique_cols[i]} for (i, k) in enumerate(unique_genders)]
    print(genders)
    accents = [{"title": k, "value": round(float(out[10+i]), 2), "color": unique_cols[i]} for (i, k) in enumerate(unique_accents)]

    return {
        "ages": ages,
        "genders": genders,
        "accents": accents
    }

def transcribe(audio_path, parser, model, decoder, device):
    print(audio_path)
    wav_data = np.array(scipy.io.wavfile.read(audio_path))[1]

    spect = parser.parse_audio(wav_data)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([1]).int()
    out, output_sizes = model(spect, input_sizes)

    return out_to_preds(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser = add_inference_args(parser)
    parser.add_argument('--audio-path', default='audio.wav',
                        help='Audio file to predict on')
    parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.cuda)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    parser = SpectrogramParser(model.audio_conf, normalize=True)

    decoded_output, decoded_offsets = transcribe(args.audio_path, parser, model, decoder, device)
    print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
