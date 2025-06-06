# Vosk TTS

!!! Тут добавлена лишь одна функция: synth_bytes, которая возвращает не wav, а байтовый поток.
стрим не доступный

Simple TTS based on VITS with some old ideas

## Usage

### ToBytes
```
from vosk_tts import Model, Synth

model = Model(model_name="vosk-model-tts-ru-0.8-multi")
synth = Synth(model)

wav_bytes = synth.synth_bytes("Привет, мир!", speaker_id=2)

with open("output_from_bytes.wav", "wb") as f:
    print(f)
    #
    f.write(wav_bytes)
```

### Command line

```
pip3 install vosk-tts

vosk-tts -n vosk-model-tts-ru-0.8-multi -s 2 --input "Привет мир!" --output out.wav
```

### API

```
from vosk_tts import Model, Synth
model = Model(model_name="vosk-model-tts-ru-0.8-multi")
synth = Synth(model)

synth.synth("Привет мир!", "out.wav", speaker_id=2)
```

## Voices

For now we support several Russian voices 3 females and 2 males. Get the model here:

[vosk-model-tts-ru-0.8-multi](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.8-multi.zip)

You can use speaker IDs from 0 to 4 included.

We plan to add more voices and languages in the future.
