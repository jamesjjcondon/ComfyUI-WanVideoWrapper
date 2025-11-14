import torch
import torch.nn as nn
import folder_paths
import os

from .mel_converter import get_mel_converter
from .vae.autoencoder import AutoEncoderModule
from .vae.distributions import DiagonalGaussianDistribution
import torchaudio

from ..utils import log

from comfy import model_management as mm
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: str,
        bigvgan_vocoder_ckpt = None,
        mode=['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                        vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                        mode=mode,
                                        need_vae_encoder=need_vae_encoder)

    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def wrapped_decode(self, z):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            mel_decoded = self.decode(z)
            audio = self.vocode(mel_decoded)

            return audio

    def wrapped_encode(self, audio):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            dist = self.encode_audio(audio)

            return dist.mean

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

class OviMMAudioVAELoader:
    """Loads MMAudio VAE for audio encoding/decoding in Ovi"""
    @classmethod
    def INPUT_TYPES(s):
        s.vae_files = folder_paths.get_filename_list("vae")
        s.mmaudio_files = folder_paths.get_filename_list("mmaudio")
        s.all_files = s.vae_files + s.mmaudio_files

        return {
            "required": {
                "vae": (s.all_files, {"tooltip": "MMAudio VAE 16k (v1-16.pth) model from models/vae or models/mmaudio"}),
                "vocoder": (s.all_files, {"tooltip": "BigVGAN vocoder (best_netG.pt) from models/vae or models/mmaudio"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MMAUDIOVAE",)
    RETURN_NAMES = ("mmaudio_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Loads MMAudio VAE for Ovi audio generation"

    def loadmodel(self, vae, vocoder, precision):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        vae_path = folder_paths.get_full_path("vae", vae) if vae in self.vae_files else folder_paths.get_full_path("mmaudio", vae)
        vocoder_path = folder_paths.get_full_path("vae", vocoder) if vocoder in self.vae_files else folder_paths.get_full_path("mmaudio", vocoder)

        vae = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            bigvgan_vocoder_ckpt=vocoder_path,
            mode='16k',
            need_vae_encoder=True
        )

        vae.to(device=offload_device, dtype=dtype)
        vae.eval()

        return (vae,)

class WanVideoDecodeOviAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mmaudio_vae": ("MMAUDIOVAE",),
                    "samples": ("LATENT",),
                }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, mmaudio_vae, samples):
        mm.soft_empty_cache()
        audio_latents = samples.get("latent_ovi_audio", None)
        if audio_latents is None:
            raise ValueError("No Ovi audio latents found in input samples")

        if not isinstance(audio_latents, torch.Tensor):
            audio_latents = torch.tensor(audio_latents)

        # MMAudio VAE expects [B, 20, L]
        if audio_latents.ndim == 2:
            if audio_latents.shape[1] == 20:
                z = audio_latents.transpose(0, 1).unsqueeze(0)   # [L,20] -> [1,20,L]
            elif audio_latents.shape[0] == 20:
                z = audio_latents.unsqueeze(0)                   # [20,L] -> [1,20,L]
            else:
                raise ValueError(f"Unexpected 2D latent shape {audio_latents.shape}")
        elif audio_latents.ndim == 3:
            if audio_latents.shape[1] == 20:
                z = audio_latents
            elif audio_latents.shape[2] == 20:
                z = audio_latents.permute(0, 2, 1)               # [B,L,20] -> [B,20,L]
            else:
                raise ValueError(f"Unexpected 3D latent shape {audio_latents.shape}")
        else:
            raise ValueError(f"Unexpected latent ndim {audio_latents.ndim}, expected 2 or 3")

        mmaudio_vae.to(device)
        z = z.to(device=device, dtype=mmaudio_vae.dtype)

        waveform = mmaudio_vae.wrapped_decode(z)

        # Normalise shape for Comfy AUDIO; assume [B, T] or [B, 1, T]
        waveform = waveform.detach().cpu()

        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]        # [B,1,T] -> [B,T]
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            # [1,T] is fine; Comfy usually copes with [C,T] or [T]
            waveform = waveform

        audio = {"waveform": waveform.float(), "sample_rate": 16000}

        mmaudio_vae.to(offload_device)
        mm.soft_empty_cache()

        return (audio,)


class WanVideoEncodeOviAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mmaudio_vae": ("MMAUDIOVAE",),
                    "audio": ("AUDIO",),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, mmaudio_vae, audio):
        mmaudio_vae.to(device)

        waveform = audio.get("waveform", None)
        sample_rate = audio.get("sample_rate", None)

        if waveform is None or sample_rate is None:
            raise ValueError("WanVideoEncodeOviAudio: audio dict must contain 'waveform' and 'sample_rate'")

        # Move to float32 on CPU for torchaudio, then to GPU
        waveform = waveform.detach().cpu().float()

        # Accept a few common shapes and convert to [1, T] mono
        # Possible shapes:
        #   [T]
        #   [C, T]
        #   [B, C, T]
        if waveform.ndim == 1:
            # [T] -> [1, T]
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            # [C, T] -> mixdown to mono [1, T]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 3:
            # [B, C, T] -> take first batch, mixdown channels -> [1, T]
            waveform = waveform[0]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"WanVideoEncodeOviAudio: Unexpected waveform shape {waveform.shape}")

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        # Now waveform is [1, T]
        waveform = waveform.to(device=device, dtype=mmaudio_vae.dtype)

        # MMAudio expects [B, T] / [B, 1, T] depending on implementation.
        # If your VAE expects [B, T], just squeeze channel dim if present.
        if waveform.ndim == 2:
            # [1, T] – fine
            pass
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            # [1, 1, T] -> [1, T]
            waveform = waveform[:, 0, :]
        else:
            raise ValueError(f"WanVideoEncodeOviAudio: waveform after processing has unexpected shape {waveform.shape}")

        samples = mmaudio_vae.wrapped_encode(waveform)

        mmaudio_vae.to(offload_device)
        mm.soft_empty_cache()

        # samples should be [B, 20, L] for your sampler
        return ({"latent_ovi_audio": samples},)

class WanVideoAddOviAudioToLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "original_samples": ("LATENT",),
                    "audio_samples": ("LATENT",),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, original_samples, audio_samples):
        samples = original_samples.copy()
        samples.update(audio_samples)

        return (samples,)
    
class WanVideoEmptyMMAudioLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "length": ("INT", {"default": 157, "min": 1, "max": 10000, "step": 1, "tooltip": "Length of the audio latent sequence"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, length):
        audio_latents = torch.zeros(
            (1, 20, length),
            device=device,
            dtype=torch.float32)  # 1, l c -> l, c

        return ({"latent_ovi_audio": audio_latents},)


class WanVideoOviCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "ovi_audio_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "ovi_negative_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Adds Ovi negative text embeddings and audio CFG scale to the text embeddings dictionary"

    def process(self, original_text_embeds, ovi_audio_cfg, ovi_negative_text_embeds=None):
        negative_text_embeds = None
        if ovi_negative_text_embeds is not None:
            negative_text_embeds = ovi_negative_text_embeds.get("prompt_embeds", None)
        if negative_text_embeds is None:
            negative_text_embeds = original_text_embeds["prompt_embeds"]
            log.info("WanVideoOviCFG: Ovi negative text embeddings not provided, using original prompt embeddings as negative embeddings")
        else:
            log.info("WanVideoOviCFG: Using provided Ovi audio negative text embeddings")
        log.info("WanVideoOviCFG: negative text embedding shape: {}".format(negative_text_embeds[0].shape))

        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "ovi_negative_prompt_embeds": negative_text_embeds,
                "ovi_audio_cfg": ovi_audio_cfg,
            })
        return (prompt_embeds_dict_copy,)

NODE_CLASS_MAPPINGS = {
    "OviMMAudioVAELoader": OviMMAudioVAELoader,
    "WanVideoDecodeOviAudio": WanVideoDecodeOviAudio,
    "WanVideoEncodeOviAudio": WanVideoEncodeOviAudio,
    "WanVideoOviCFG": WanVideoOviCFG,
    "WanVideoAddOviAudioToLatents": WanVideoAddOviAudioToLatents,
    "WanVideoEmptyMMAudioLatents": WanVideoEmptyMMAudioLatents,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OviMMAudioVAELoader": "Ovi MMAudio VAE Loader",
    "WanVideoDecodeOviAudio": "WanVideo Decode Ovi Audio",
    "WanVideoEncodeOviAudio": "WanVideo Encode Ovi Audio",
    "WanVideoOviCFG": "WanVideo Ovi CFG",
    "WanVideoAddOviAudioToLatents": "WanVideo Add MMAudio To Latents",
    "WanVideoEmptyMMAudioLatents": "WanVideo Empty MMAudio Latents",
}