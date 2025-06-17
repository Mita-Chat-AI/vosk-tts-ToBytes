import re
import io
import time
import wave

import logging
import numpy as np
import onnxruntime
from pydub import AudioSegment
from deepmultilingualpunctuation import PunctuationModel

from .g2p import convert


sess_options = onnxruntime.SessionOptions()
sess_options.enable_cpu_mem_arena = False
sess_options.enable_mem_pattern = False
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL


punct_model = PunctuationModel()


def load_breath(path='breath.wav') -> np.ndarray:
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)  # 16-bit PCM
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return samples


def flexible_text_split(text):
    pattern = r'(.*?[.,!?])'
    sentences = re.findall(pattern, text, flags=re.DOTALL)
    
    if not sentences:
        return [text.strip()] if text.strip() else []

    return [s.strip() for s in sentences if s.strip()]


class Synth:

    def __init__(self, model):
        self.model = model

    def audio_float_to_int16(self,
        audio: np.ndarray, max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * max_wav_value
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

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

        # Select only first token in multitoken words
        selected = [0]
        for i, t in enumerate(tokens.tokens):
            if t[0] != '#':
                selected.append(i)
        bert = bert[selected]
        return bert

    def synth_audio(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):

        if noise_level is None:
            noise_level = self.model.config["inference"].get("noise_level", 0.8)
        if speech_rate is None:
            speech_rate = self.model.config["inference"].get("speech_rate", 1.0)
        if duration_noise_level is None:
            duration_noise_level = self.model.config["inference"].get("duration_noise_level", 0.8)
        if scale is None:
            scale = self.model.config["inference"].get("scale", 1.0)

        text = re.sub("—", "-", text)

        if self.model.tokenizer != None and self.model.config.get("no_blank", 0) == 0:
            bert = self.get_word_bert(text)
            phoneme_ids, bert_embs = self.g2p(text, None)
            bert_embs = np.expand_dims(np.transpose(np.array(bert_embs, dtype=np.float32)), 0)
        elif self.model.tokenizer != None and self.model.config.get("no_blank", 0) != 0:
            bert = self.get_word_bert(text)
            phoneme_ids, bert_embs = self.g2p_noblank(text, bert)
            bert_embs = np.expand_dims(np.transpose(np.array(bert_embs, dtype=np.float32)), 0)
        else:
            phoneme_ids = self.g2p_noembed(text)
            bert_embs = np.zeros((1, 768, len(phoneme_ids)), dtype=np.float32)

        # Run main prediction
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array([noise_level, 1.0 / speech_rate, duration_noise_level], dtype=np.float32)

        # Assign first voice
        if speaker_id is None:
            speaker_id = 0
        sid = np.array([speaker_id], dtype=np.int64)

        args = {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
                "sid": sid,
        }
        if self.model.tokenizer != None:
            args["bert"] = bert_embs

        start_time = time.perf_counter()
        audio = self.model.onnx.run(
            None,
            args
        )[0]
        audio = audio.squeeze()
        audio = audio * scale

        audio = self.audio_float_to_int16(audio)
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / 22050
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        logging.info("Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)" % (real_time_factor, infer_sec, audio_duration_sec))
        return audio


    def synth(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):
        audio = self.synth_audio(text, speaker_id, noise_level, speech_rate, duration_noise_level, scale)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 байта для int16
            wf.setframerate(22050)
            wf.writeframes(audio.astype(np.int16).tobytes())  # <-- запись звука

        return buffer.getvalue()


    def fast_synth_streaming_bytes(synth_obj, text, breath_path='breath.wav', **kwargs) -> bytes:
        chunks = flexible_text_split(text)
        breath_audio = load_breath(breath_path)
        audio_chunks = []

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                audio = synth_obj.synth_audio(chunk, **kwargs)
                audio_chunks.append(audio)
                # Добавляем дыхание между кусками, кроме последнего
                if i < len(chunks) - 1:
                    audio_chunks.append(breath_audio)
            except Exception as e:
                print(f"Ошибка при синтезе '{chunk}': {e}")

        if not audio_chunks:
            raise RuntimeError("Не удалось сгенерировать аудио ни для одного фрагмента.")

        final_audio = np.concatenate(audio_chunks)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(final_audio.tobytes())

        return buffer.getvalue()


    def punctuate_text(self, text: str) -> str:
        keep_only_russian = re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?;:\-—()]', '', text)

        text = re.sub(r'[,.!?;:…]', '', keep_only_russian)

        print(text)
        result =punct_model.restore_punctuation(text)
        print(result)

        return result

    def g2p(self, text, _):
        punctuated_text = self.punctuate_text(text)
        embeddings = self.get_word_bert(punctuated_text)

        pattern = "([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_index = 1

        for word in re.split(pattern, punctuated_text.lower()):
            if word == "":
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
                phone_embeddings.append(embeddings[word_index])
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_index])
            else:
                for p in convert(word).split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_index])
            if word != " ":
                word_index += 1

        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        # Convert to ids and intersperse with blank
        phoneme_id_map = self.model.config["phoneme_id_map"]
        phoneme_ids = [phoneme_id_map[phonemes[0]]]
        phone_embeddings_is = [phone_embeddings[0]]
        for i in range(1, len(phonemes)):
            phoneme_ids.append(0)
            phoneme_ids.append(phoneme_id_map[phonemes[i]])
            phone_embeddings_is.append(phone_embeddings[i])
            phone_embeddings_is.append(phone_embeddings[i])

        logging.info(f"Text: {punctuated_text}")
        logging.info(f"Phonemes: {phonemes}")
        return phoneme_ids, phone_embeddings_is



    def g2p_noblank(self, text, embeddings):
        pattern = "([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_index = 1
        for word in re.split(pattern, text.lower()):
            if word == "":
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
                phone_embeddings.append(embeddings[word_index])
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_index])
            else:
                for p in convert(word).split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_index])
            if word != " ":
                word_index = word_index + 1
        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        # Convert to ids and intersperse with blank
        phoneme_id_map = self.model.config["phoneme_id_map"]
        phoneme_ids = [phoneme_id_map[p] for p in phonemes]

        logging.info(f"Text: {text}")
        logging.info(f"Phonemes: {phonemes}")
        return phoneme_ids, phone_embeddings


    def g2p_noembed(self, text):
        punctuate_text = self.punctuate_text(text)
        print(punctuate_text)
        pattern = "([,.?!;:\"() ])"
        phonemes = ["^"]
        for word in re.split(pattern, punctuate_text.lower()):
            if word == "":
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
            else:
                for p in convert(word).split():
                    phonemes.append(p)
        phonemes.append("$")

        # Convert to ids and intersperse with blank
        phoneme_id_map = self.model.config["phoneme_id_map"]
        if isinstance(phoneme_id_map[phonemes[0]], list):
            phoneme_ids = []
            phoneme_ids.extend(phoneme_id_map[phonemes[0]])
            for i in range(1, len(phonemes)):
                phoneme_ids.append(0)
                phoneme_ids.extend(phoneme_id_map[phonemes[i]])
        else:
            phoneme_ids = [phoneme_id_map[phonemes[0]]]
            for i in range(1, len(phonemes)):
                phoneme_ids.append(0)
                phoneme_ids.append(phoneme_id_map[phonemes[i]])

        logging.info(f"Text: {text}")
        logging.info(f"Phonemes: {phonemes}")
        return phoneme_ids
