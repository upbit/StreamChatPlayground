import os
from metagpt.config2 import config
from metagpt.provider.zhipuai_api import ZhiPuAILLM
# from metagpt.utils.cost_manager import CostManager
from metagpt.logs import logger
from langchain.prompts import PromptTemplate
from utils.common import load_plaintext

# default LLM
llm = None

FORMAT_PROMPTS = {
    "json": '以json数组的形式输出。json字段名请转换成英文小写下划线格式，且保持一致。如："融资类型" -> "investment_type"',
    "markdown": '以markdown表格的形式输出。',
}

async def llm_extractor(guidance: str, content: str, format: str) -> str:
    """
    Perform extraction on the 'content' text using a large language model.

    Args:
        guidance (str): Guide the extraction process.
        content (str): The text content that needs to be extracted.
        format (str): The output format, can be "json" or "markdown".

    Returns:
        str: text extracted from content.
    """
    output_format = FORMAT_PROMPTS.get(format.lower(), None)
    print(output_format)
    if not output_format:
        raise ValueError(f"Unsupported format: {format}")

    template = load_plaintext(os.path.dirname(os.path.abspath(__file__)), "template.yaml")
    prompt = PromptTemplate(input_variables=[], template=template)
    template = prompt.format(
        guidance=guidance,
        content=content,
        output=output_format,
    )
    logger.debug(template)

    rsp = await llm.aask(template)
    logger.debug(rsp)
    return rsp

def init_llm():
    global llm
    llm = ZhiPuAILLM(config.llm)
    llm.use_system_prompt = False

init_llm()
