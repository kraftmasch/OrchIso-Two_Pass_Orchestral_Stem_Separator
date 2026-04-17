import gradio as gr
import torch
import torch.nn as nn
import torchaudio
import subprocess
import os

SAMPLE_RATE  = 44100
CHUNK_SIZE   = 44100 * 4
N_FFT        = 2048
HOP_LENGTH   = 512
WIN_LENGTH   = 2048
DEMUCS_MODEL = 'mdx_extra'
MODEL_PATH   = 'orchestra_spectrogram_best.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class SpectrogramUNet(nn.Module):
    def __init__(self, features=[16, 32, 64, 128]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock2D(2, features[0])
        self.enc2 = ConvBlock2D(features[0], features[1])
        self.enc3 = ConvBlock2D(features[1], features[2])
        self.enc4 = ConvBlock2D(features[2], features[3])

        self.bottleneck = ConvBlock2D(features[3], features[3] * 2)

        self.up4  = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.dec4 = ConvBlock2D(features[3]*2, features[3])

        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = ConvBlock2D(features[2]*2, features[2])

        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = ConvBlock2D(features[1]*2, features[1])

        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = ConvBlock2D(features[0]*2, features[0])

        self.final = nn.Sequential(
            nn.Conv2d(features[0], 2, 1),
            nn.Sigmoid()
        )

    def _cat(self, up, skip):
        h = min(up.shape[2], skip.shape[2])
        w = min(up.shape[3], skip.shape[3])
        return torch.cat([up[:, :, :h, :w], skip[:, :, :h, :w]], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))

        return self.final(d1)


model = SpectrogramUNet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded")
else:
    print("WARNING: model not found")

def to_spectrogram(wav):
    window = torch.hann_window(WIN_LENGTH).to(device)
    results, phases = [], []
    for ch in range(wav.shape[0]):
        stft = torch.stft(
            wav[ch],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True
        )
        results.append(stft.abs())
        phases.append(stft.angle())
    return torch.stack(results), torch.stack(phases)

def process_audio(model, input_path, output_path):
    model.eval()
    wav, sr = torchaudio.load(input_path)
    wav = wav.to(device)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    wav = wav[:2]

    total = wav.shape[1]
    hop = CHUNK_SIZE // 2

    out_buf = torch.zeros_like(wav).to(device)
    w_buf   = torch.zeros(total).to(device)
    window  = torch.hann_window(CHUNK_SIZE).to(device)

    starts = list(range(0, total - CHUNK_SIZE + 1, hop))

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=1.0,
        n_iter=32,
    ).to(device)

    with torch.no_grad():
        for start in starts:
            chunk = wav[:, start:start + CHUNK_SIZE]

            mag, _ = to_spectrogram(chunk)
            mag = mag.to(device)

            mag_log = torch.log1p(mag).unsqueeze(0)

            mask = model(mag_log).squeeze(0)

            f_min = min(mask.shape[1], mag.shape[1])
            w_min = min(mask.shape[2], mag.shape[2])

            masked_mag = mask[:, :f_min, :w_min] * mag[:, :f_min, :w_min]

            expected_f = N_FFT // 2 + 1
            if masked_mag.shape[1] < expected_f:
                pad_f = expected_f - masked_mag.shape[1]
                masked_mag = torch.nn.functional.pad(masked_mag, (0, 0, 0, pad_f))

            channels = []
            for ch in range(masked_mag.shape[0]):
                recon = griffin_lim(masked_mag[ch].unsqueeze(0))
                channels.append(recon.squeeze(0))

            out_chunk = torch.stack(channels)

            n = min(out_chunk.shape[1], CHUNK_SIZE)

            out_buf[:, start:start + n] += out_chunk[:, :n] * window[:n].unsqueeze(0)
            w_buf[start:start + n] += window[:n]

    w_buf = w_buf.clamp(min=1e-8)
    out_buf = (out_buf / w_buf.unsqueeze(0)).clamp(-1.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, out_buf.cpu(), SAMPLE_RATE)


def orchiso(audio_file, progress=gr.Progress()):
    if audio_file is None:
        return None, "Upload file dulu"

    try:
        progress(0.05, desc="Prepare...")

        song_path = audio_file
        song_name = os.path.splitext(os.path.basename(song_path))[0]
        safe_name = song_name.replace(' ', '_')

        progress(0.10, desc="Demucs...")

        result = subprocess.run([
            'python', '-m', 'demucs',
            '--name', DEMUCS_MODEL,
            '--device', 'cuda',
            '--out', 'separated',
            song_path,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return None, result.stderr

        other_stem = f'separated/{DEMUCS_MODEL}/{song_name}/other.wav'

        progress(0.60, desc="U-Net...")

        output_path = f'output/{safe_name}_orchiso.wav'
        process_audio(model, other_stem, output_path)

        wav_out, sr_out = torchaudio.load(output_path)
        duration = wav_out.shape[1] / sr_out

        progress(1.0, desc="Done")

        return output_path, f"Berhasil! Durasi: {duration:.1f}s"

    except Exception as e:
        import traceback
        return None, traceback.format_exc()


# UI
with gr.Blocks(title="OrchIso") as demo:
    gr.Markdown("# OrchIso")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath")
        audio_output = gr.Audio(type="filepath")

    status = gr.Textbox()

    btn = gr.Button("Run")
    reset_btn = gr.Button("Reset")
    
    btn.click(
        fn=orchiso,
        inputs=[audio_input],
        outputs=[audio_output, status]
    )

    def reset_ui():
      return None, ""

    reset_btn.click(
    fn=reset_ui,
    inputs=[],
    outputs=[audio_output, status]
)

demo.launch()