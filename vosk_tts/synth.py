import json
import time
import logging
import re
import io
import numpy as np
import wave
from pydub import AudioSegment
from .g2p import convert
from deepmultilingualpunctuation import PunctuationModel


def load_breath(path='breath.wav') -> np.ndarray:
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.int16)


def flexible_text_split(text):
    pattern = r'(.*?[.,!?])'
    sentences = re.findall(pattern, text, flags=re.DOTALL)
    leftover = text[len(''.join(sentences)):].strip()
    return [s.strip() for s in sentences if s.strip()] + ([leftover] if leftover else [])


class Synth:
    def __init__(self, model):
        self.model = model
        self.punct_model = PunctuationModel()

    def audio_float_to_int16(self, audio: np.ndarray, max_wav_value: float = 32767.0) -> np.ndarray:
        audio = np.clip(audio * max_wav_value, -max_wav_value, max_wav_value)
        return audio.astype("int16")

    def get_word_bert(self, text):
        tokens = self.model.tokenizer.encode(text.replace("+", ""))
        bert = self.model.bert_onnx.run(
            None,
            {
                "input_ids": [tokens.ids],
                "attention_mask": [tokens.attention_mask],
                "token_type_ids": [tokens.type_ids],
            }
        )[0]
        selected = [i for i, t in enumerate(tokens.tokens) if not t.startswith('#')]
        return bert[[0] + selected]

    def synth_audio(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):
        noise_level = noise_level if noise_level is not None else self.model.config["inference"].get("noise_level", 0.8)
        speech_rate = speech_rate if speech_rate is not None else self.model.config["inference"].get("speech_rate", 1.0)
        duration_noise_level = duration_noise_level if duration_noise_level is not None else self.model.config["inference"].get("duration_noise_level", 0.8)
        scale = scale if scale is not None else self.model.config["inference"].get("scale", 1.0)

        text = text.replace("—", "-")

        if self.model.tokenizer:
            bert = self.get_word_bert(text)
            if self.model.config.get("no_blank", 0):
                phoneme_ids, bert_embs = self.g2p_noblank(text, bert)
            else:
                phoneme_ids, bert_embs = self.g2p(text, bert)
            bert_embs = np.expand_dims(np.transpose(np.array(bert_embs, dtype=np.float32)), 0)
        else:
            phoneme_ids = self.g2p_noembed(text)
            bert_embs = np.zeros((1, 768, len(phoneme_ids)), dtype=np.float32)

        input_tensor = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        args = {
            "input": input_tensor,
            "input_lengths": np.array([input_tensor.shape[1]], dtype=np.int64),
            "scales": np.array([noise_level, 1.0 / speech_rate, duration_noise_level], dtype=np.float32),
            "sid": np.array([speaker_id or 0], dtype=np.int64),
        }

        start = time.perf_counter()
        audio = self.model.onnx.run(None, args)[0].squeeze() * scale
        audio = self.audio_float_to_int16(audio)
        duration = time.perf_counter() - start
        rtf = duration / (len(audio) / 22050) if audio.size > 0 else 0

        logging.info(f"RTF: {rtf:.2f} (infer={duration:.2f}s, audio={len(audio)/22050:.2f}s)")
        return audio

    def fast_synth_streaming_bytes(self, text, pause_duration=0.4, breath_path='breath.wav', **kwargs) -> bytes:
        chunks = flexible_text_split(text)
        breath_audio = load_breath(breath_path)
        audio_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                audio = self.synth_audio(chunk, **kwargs)
                audio_chunks.append(audio)
                if i < len(chunks) - 1:
                    audio_chunks.append(breath_audio)
            except Exception as e:
                logging.error(f"Ошибка синтеза '{chunk}': {e}")

        if not audio_chunks:
            raise RuntimeError("Ни один фрагмент не был синтезирован.")

        final_audio = np.concatenate(audio_chunks)
        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(final_audio.tobytes())
            return buffer.getvalue()

    def punctuate_text(self, text: str) -> str:
        clean_text = re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?;:\-—()]', '', text)
        no_punct = re.sub(r'[,.!?;:…]', '', clean_text)
        return self.punct_model.restore_punctuation(no_punct)

    def g2p(self, text, embeddings):
        punctuated = self.punctuate_text(text)
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_index = 1

        for word in re.split(pattern, punctuated.lower()):
            if not word:
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
            else:
                phonemes.extend(self.model.dic.get(word, convert(word)).split())
            phone_embeddings.extend([embeddings[word_index]] * len(phonemes[-1]))
            if word != " ":
                word_index += 1

        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        ids = [self.model.config["phoneme_id_map"][phonemes[0]]]
        emb = [phone_embeddings[0]]
        for i in range(1, len(phonemes)):
            ids.extend([0, self.model.config["phoneme_id_map"][phonemes[i]]])
            emb.extend([phone_embeddings[i]] * 2)

        logging.info(f"Phonemes: {phonemes}")
        return ids, emb

    def g2p_noblank(self, text, embeddings):
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_index = 1

        for word in re.split(pattern, text.lower()):
            if not word:
                continue
            phonemes.extend(self.model.dic.get(word, convert(word)).split())
            phone_embeddings.extend([embeddings[word_index]] * len(phonemes[-1]))
            if word != " ":
                word_index += 1

        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        ids = [self.model.config["phoneme_id_map"][p] for p in phonemes]
        return ids, phone_embeddings

    def g2p_noembed(self, text):
        punctuated = self.punctuate_text(text)
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]

        for word in re.split(pattern, punctuated.lower()):
            if not word:
                continue
            phonemes.extend(self.model.dic.get(word, convert(word)).split())

        phonemes.append("$")
        ids = []
        map_ = self.model.config["phoneme_id_map"]
        ids.extend(map_[phonemes[0]] if isinstance(map_[phonemes[0]], list) else [map_[phonemes[0]]])
        for i in range(1, len(phonemes)):
            ids.append(0)
            next_ids = map_[phonemes[i]]
            ids.extend(next_ids if isinstance(next_ids, list) else [next_ids])

        return ids