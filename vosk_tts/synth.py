import json
import numpy as np
import onnxruntime
import wave
import time
import re
import logging
import io
import wave
sess_options = onnxruntime.SessionOptions()
sess_options.enable_cpu_mem_arena = False
sess_options.enable_mem_pattern = False
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

import io
import wave
import numpy as np
from .g2p import convert

import re

def flexible_text_split(text, max_chunk_len=100):
    """
    Гибко разбивает текст на фрагменты:
    1) Сначала на предложения по знакам [.?!]
    2) Если предложение длиннее max_chunk_len, режет его по словам на более мелкие части
    """

    # 1) Разбиваем на предложения с сохранением знаков
    pattern = r'([^.!?]*[.!?]+)'
    sentences = re.findall(pattern, text, flags=re.DOTALL)

    # Остаток после предложений
    remainder = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    if remainder:
        sentences.append(remainder)

    sentences = [s.strip() for s in sentences if s.strip()]

    # 2) Разбиваем длинные предложения на подфрагменты
    chunks = []
    for sentence in sentences:
        if len(sentence) <= max_chunk_len:
            chunks.append(sentence)
        else:
            # Разбиваем по пробелам
            words = sentence.split()
            current_chunk = ""
            for word in words:
                # Если добавление слова превышает лимит, сохраняем текущий и начинаем новый
                if len(current_chunk) + len(word) + 1 > max_chunk_len:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += " " + word
            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks


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
            phoneme_ids, bert_embs = self.g2p(text, bert)
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
                "sid": sid,  # ✅ ОСТАВИТЬ!
        }


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
    

    def safe_synth_audio(self, text, *args, **kwargs):
        """
        Разбивает длинный текст на предложения и синтезирует каждое отдельно, 
        чтобы не перегружать VRAM. Затем объединяет аудио.
        """
        # Разбиваем на предложения по знакам препинания
        segments = re.split(r'(?<=[.!?])\s+', text.strip())

        audio_chunks = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            try:
                audio = self.synth_audio(segment, *args, **kwargs)
                audio_chunks.append(audio)
            except Exception as e:
                logging.warning(f"Ошибка при синтезе сегмента '{segment}': {e}")

        if not audio_chunks:
            raise RuntimeError("Не удалось сгенерировать аудио ни для одного сегмента.")

        return np.concatenate(audio_chunks)


    def synth(self, text, oname, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):

        audio = self.safe_synth_audio(text, speaker_id, noise_level, speech_rate, duration_noise_level, scale)

        with wave.open(oname, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio.tobytes())


    def synth_bytes(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):
        """Синтезирует речь и возвращает результат в виде байтового WAV-потока"""
        audio = self.synth_audio(text, speaker_id, noise_level, speech_rate, duration_noise_level, scale)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio.tobytes())
        buffer.seek(0)
        return buffer.read()
    


 


    

    def synth_streaming_bytes(synth_obj, text, chunk_size=50, **kwargs) -> bytes:
        """
        Синтезирует речь кусками по chunk_size символов,
        возвращает весь результат в виде WAV-байтов.
        """
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        audio_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                audio_bytes = synth_obj.synth_bytes(chunk, **kwargs)  # возвращает WAV bytes для куска
                # Нужно из WAV bytes вырезать заголовок и взять только raw audio data
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_chunks.append(frames)
            except Exception as e:
                print(f"Ошибка при синтезе фрагмента '{chunk}': {e}")

        if not audio_chunks:
            raise RuntimeError("Не удалось сгенерировать аудио ни для одного фрагмента.")

        # Объединяем аудио фреймы
        combined_frames = b"".join(audio_chunks)

        # Создаем итоговый WAV в памяти с правильным заголовком
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes per sample
            wf.setframerate(22050)
            wf.writeframes(combined_frames)

        return buffer.getvalue()


    def synth_streaming_flexible(synth_obj, text, max_chunk_len=100, **kwargs) -> bytes:
        chunks = flexible_text_split(text, max_chunk_len=max_chunk_len)
        audio_chunks = []

        for chunk in chunks:
            try:
                audio_bytes = synth_obj.synth_bytes(chunk, **kwargs)
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_chunks.append(frames)
            except Exception as e:
                print(f"Ошибка при синтезе '{chunk}': {e}")

        if not audio_chunks:
            raise RuntimeError("Не удалось сгенерировать аудио ни для одного фрагмента.")

        combined_frames = b"".join(audio_chunks)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(combined_frames)

        return buffer.getvalue()



    def g2p(self, text, embeddings):
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
        phoneme_ids = [phoneme_id_map[phonemes[0]]]
        phone_embeddings_is = [phone_embeddings[0]]
        for i in range(1, len(phonemes)):
            phoneme_ids.append(0)
            phoneme_ids.append(phoneme_id_map[phonemes[i]])
            phone_embeddings_is.append(phone_embeddings[i])
            phone_embeddings_is.append(phone_embeddings[i])

        logging.info(f"Text: {text}")
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
        pattern = "([,.?!;:\"() ])"
        phonemes = ["^"]
        for word in re.split(pattern, text.lower()):
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
