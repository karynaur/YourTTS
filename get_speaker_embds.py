from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
import librosa
import argparse
import torch
import subprocess
import glob

def compute_embeddings(folder):
    CONFIG_SE_PATH = "config_se.json"
    CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"
    USE_CUDA = torch.cuda.is_available()


    SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)


    reference_files = glob.glob(folder + "/*.wav")
    for sample in reference_files:
        cmd = f"ffmpeg-normalize {sample} -nt rms -t=-27 -o {sample} -ar 16000 -f"
        subprocess.call(cmd.split())

    reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)
    return reference_emb
