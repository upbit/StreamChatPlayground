import re
import torch
import struct
import librosa
import numpy as np
from typing import Dict, Any
from argparse import ArgumentParser, Namespace
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from my_utils import load_audio
from module.mel_processing import spectrogram_torch
from flask import Flask, request, Response, redirect
from pprint import pprint


app = Flask(__name__)
# Command line arguments parse by parse_argument()
configs: Namespace = None
model_mappings: Dict[str, Any] = {}

splits_flags = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}  # 不考虑省略号


def to_device(model, device=None):
    "A helper function to move a model to GPU"
    global configs
    if configs.half:
        model = model.half()
    if device:
        return model.to(device)
    else:
        return model.to(configs.device)


def get_bert_feature(text, word2ph):
    global configs, model_mappings
    bert = model_mappings["bert"]
    tokenizer, bert_model = bert["tokenizer"], bert["model"]

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(configs.device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def inference(ref_wav_path, prompt_text, prompt_language, text, text_language):
    global configs, model_mappings

    hps = model_mappings["hps"]

    # 先返回PCM的头部，将音频长度设置成较大的值以便后面分块发送音频数据
    yield pcm16_header(hps.data.sampling_rate)

    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if configs.half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = to_device(torch.from_numpy(wav16k))
        zero_wav_torch = to_device(torch.from_numpy(zero_wav))
        wav16k = torch.cat([wav16k, zero_wav_torch])

        ssl_model = model_mappings["ssl_model"]
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        vq_model = model_mappings["vq_model"]
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

    # step1
    if prompt_language == "en":
        phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
    else:
        phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)

    text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
    if text[-1] not in splits_flags:
        text += "。" if text_language != "en" else "."
    texts = text.split("\n")
    pprint(texts)

    # audio_opt = []
    if prompt_language == "en":
        bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
    else:
        bert1 = nonen_get_bert_inf(prompt_text, prompt_language)

    for text in texts:
        if len(text.strip()) == 0:
            continue
        if text_language == "en":
            phones2, word2ph2, norm_text2 = clean_text_inf(text, text_language)
        else:
            phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text, text_language)

        if text_language == "en":
            bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
        else:
            bert2 = nonen_get_bert_inf(text, text_language)

        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(configs.device).unsqueeze(0)
        bert = bert.to(configs.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(configs.device)
        prompt = prompt_semantic.unsqueeze(0).to(configs.device)

        # step2
        hz = 50
        t2s = model_mappings["t2s"]
        t2s_model, t2s_topk, t2s_max_sec = t2s["model"], t2s["top_k"], t2s["max_sec"]
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=t2s_topk,
                early_stop_num=hz * t2s_max_sec,
            )

        # step3
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0) #mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)
        refer = to_device(refer)

        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(configs.device).unsqueeze(0),
                refer,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分

        audio_raw = (np.concatenate([audio, zero_wav], 0) * 32768).astype(np.int16)
        yield np.int16(audio_raw / np.max(np.abs(audio_raw)) * 32767).tobytes()


def splite_en_inf(sentence, language):
    pattern = re.compile(r"[a-zA-Z. ]+")
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)

    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    global configs
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(configs.device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if configs.half == True else torch.float32,
        ).to(configs.device)

    return bert


def nonen_clean_text_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "en" or "ja":
            pass
        else:
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    # print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = " ".join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    # print(textlist)
    # print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def load_models():
    global configs, model_mappings

    # BERT
    tokenizer = AutoTokenizer.from_pretrained(configs.bert)
    model = AutoModelForMaskedLM.from_pretrained(configs.bert)
    model_mappings["bert"] = {"tokenizer": tokenizer, "model": to_device(model)}

    # HuBERT
    cnhubert.cnhubert_base_path = configs.hubert
    model_mappings["ssl_model"] = to_device(cnhubert.get_model())

    # SoVITS
    dict_s2 = torch.load(configs.sovits, map_location="cpu")
    hps = DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    model_mappings["hps"] = hps

    # VQ
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    vq_model = to_device(vq_model)
    vq_model.eval()
    model_mappings["vq_model"] = vq_model
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

    # GPT
    dict_s1 = torch.load(configs.gpt, map_location="cpu")
    config = dict_s1["config"]
    print(config)
    t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = to_device(t2s_model)
    t2s_model.eval()
    model_mappings["t2s"] = {
        "model": t2s_model,
        "top_k": config["inference"]["top_k"],
        "max_sec": config["data"]["max_sec"],
    }
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def load_configs():
    "Parse command line options and load config files."
    # TODO: load config
    parser = ArgumentParser(description="Server for TTS API")

    parser.add_argument(
        "--bert",
        type=str,
        default="pretrained_models/chinese-roberta-wwm-ext-large",
        help="Path to the pretrained BERT model",
    )
    parser.add_argument(
        "--hubert",
        type=str,
        default="pretrained_models/chinese-hubert-base",
        help="Path to the pretrained HuBERT model",
    )
    parser.add_argument(
        "--sovits",
        type=str,
        # default="pretrained_models/s2G488k.pth",
        default="my_models/xiaowu_e12_s108.pth",
        help="Path to the pretrained SoVITS",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        # default="pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        default="my_models/xiaowu-e10.ckpt",
        help="Path to the pretrained GPT",
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--half", action="store_true", help="Use half precision instead of float32")

    global configs
    try:
        configs = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        import os

        os._exit(e.code)


def pcm16_header(rate, size=1000000000, channels=1):
    # Header for 16-bit PCM, modify from scipy.io.wavfile.write
    fs = rate
    # size = data.nbytes  # length * sizeof(nint16)

    header_data = b"RIFF"
    header_data += struct.pack("i", size + 44)
    header_data += b"WAVE"

    # fmt chunk
    header_data += b"fmt "
    format_tag = 1  # PCM
    bit_depth = 2 * 8  # 2 bytes
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)
    fmt_chunk_data = struct.pack("<HHIIHH", format_tag, channels, fs, bytes_per_second, block_align, bit_depth)

    header_data += struct.pack("<I", len(fmt_chunk_data))
    header_data += fmt_chunk_data

    header_data += b"data"

    return header_data + struct.pack("<I", size)


def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/")
def main():
    global model_mappings
    hps = model_mappings["hps"]

    text = request.args.get("text", type=str)
    if not text:
        return Response("Param `text` is required", 400)
    text_lang = request.args.get("lang", type=str, default="zh")  # zh, en, jp

    # TODO: change to params
    ref_wav_path = "my_models/sample.wav"
    raw_text = load_text("my_models/sample.txt")
    lang_tag, ref_text = raw_text.split("|")
    params = {
        "ref_wav_path": ref_wav_path,
        "prompt_text": ref_text,
        "prompt_language": lang_tag,
        "text": text,
        "text_language": text_lang,
    }
    pprint(params)

    return inference(**params), {"Content-Type": "audio/x-wav"}


if __name__ == "__main__":
    load_configs()
    load_models()
    app.run(host="127.0.0.1", debug=True)
