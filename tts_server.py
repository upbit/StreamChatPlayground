import torch
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

app = Flask(__name__)
# Command line arguments parse by parse_argument()
configs: Namespace = None
model_mappings: Dict[str, Any] = {}


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

    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = to_device(torch.from_numpy(wav16k))

        ssl_model = model_mappings["ssl_model"]
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        vq_model = model_mappings["vq_model"]
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

    # step1
    dict_language = {"中文": "zh", "英文": "en", "日文": "ja"}

    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    texts = text.split("\n")

    hps = model_mappings["hps"]
    audio_opt = []
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if configs.half else np.float32,
    )
    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(configs.device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype=torch.float16 if configs.half else torch.float32,
            ).to(configs.device)
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2).to(configs.device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
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
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

    return hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)


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

    import tqdm

    bar = tqdm.tqdm(total=5)

    # BERT
    tokenizer = AutoTokenizer.from_pretrained(configs.bert)
    model = AutoModelForMaskedLM.from_pretrained(configs.bert)
    model_mappings["bert"] = {"tokenizer": tokenizer, "model": to_device(model)}
    bar.update(1)

    # HuBERT
    cnhubert.cnhubert_base_path = configs.hubert
    model_mappings["ssl_model"] = to_device(cnhubert.get_model())
    bar.update(2)

    # SoVITS
    dict_s2 = torch.load(configs.sovits, map_location="cpu")
    hps = DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    model_mappings["hps"] = hps
    bar.update(3)

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
    bar.update(4)

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
    bar.update(5)


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
        default="my_models/xiaowu_e12_s84.pth",
        help="Path to the pretrained SoVITS",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        # default="pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        default="my_models/xiaowu-e15.ckpt",
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


def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/")
def main():
    text = request.args.get("text", type=str)
    if not text:
        return Response("Param `text` is required", 400)

    # TODO: change to params
    ref_wav_path = "D:\Software\StreamChatPlayground\Characters\XiaoWu\sample.wav"
    ref_text = load_text("D:\Software\StreamChatPlayground\Characters\XiaoWu\sample.txt")
    lang_tag = "中文"
    rate, adata = inference(ref_wav_path, ref_text, lang_tag, text, lang_tag)
    scaled = np.int16(adata / np.max(np.abs(adata)) * 32767)

    from scipy.io.wavfile import write

    write("temp.wav", rate, scaled)

    def generate():
        with open("temp.wav", "rb") as fwav:
            data = fwav.read(4096)
            while data:
                yield data
                data = fwav.read(4096)

    return Response(generate(), mimetype="audio/x-wav")


if __name__ == "__main__":
    # TODO: multithreads
    load_configs()
    load_models()
    app.run(host="127.0.0.1", debug=True)
