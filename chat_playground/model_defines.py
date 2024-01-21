from typing import List


class ModelType:
    OpenAI = "OpenAI"
    ZhipuAI = "智普AI"


MODEL_DEFINES = {
    ModelType.OpenAI: ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-vision-preview"],
    ModelType.ZhipuAI: ["glm-4", "glm-4v", "glm-3-turbo"],
}


def GetModels(mtype: ModelType) -> List:
    return MODEL_DEFINES[mtype]
