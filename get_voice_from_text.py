import os
import string
import time
import argparse
import json
import numpy as np
import torch
from get_speaker_embds import compute_embeddings
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *


def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec



# create output path

# model vars 
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()

# load the config
C = load_config(CONFIG_PATH)


# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)


model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

#define args
parser = argparse.ArgumentParser(description='Synthesize text to speech')
parser.add_argument('--text', type=str, default="Synthesized text" required=True, help='Text to synthesize')
parser.add_argument('--ref_folder', type=str, required=True, help='Reference audio folder')
parser.add_argument('--out_folder', type=str, default='out/' required=True, help='Output audio folder')
parser.add_argument('--length_scale', type=float, default=1.6, help='scaler for the duration predictor. The larger it is, the slower the speech')
parser.add_argument('--inference_noise_scale', type=float, default=0.3, help='scaler for the duration predictor. The larger it is, the slower the speech')
parser.add_argument('--inference_noise_scale_dp', type=float, default=0.3, help='scaler for the duration predictor. The larger it is, the slower the speech')
parser.add_argument('--language_id', type=int, default=0, help='Language id')
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)

model.length_scale = args.length_scale
model.inference_noise_scale = args.inference_noise_scale
model.inference_noise_scale_dp = args.inference_noise_scale_dp

reference_emb = compute_embeddings(args.ref_folder)

wav, alignment, _, _ = synthesis(
                    model,
                    args.text,
                    C,
                    "cuda" in str(next(model.parameters()).device),
                    ap,
                    speaker_id=None,
                    d_vector=reference_emb,
                    style_wav=None,
                    language_id=args.language_id,
                    enable_eos_bos_chars=C.enable_eos_bos_chars,
                    use_griffin_lim=True,
                    do_trim_silence=False,
                ).values()
print("Generated Audio")
IPython.display.display(Audio(wav, rate=ap.sample_rate))
file_name = text.replace(" ", "_")
file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
out_path = os.path.join(args.out_folder, file_name)
print(" Saving output to {}".format(out_path))
ap.save_wav(wav, out_path)