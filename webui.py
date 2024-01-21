import os
import zhipuai
import openai
import streamlit as st
from dotenv import dotenv_values
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.schema import AgentAction, AgentFinish
from chat_playground import *


@st.cache_resource()
def init_config(env_name=".env"):
    current_path = os.path.split(os.path.realpath(__file__))[0]
    cfg = dotenv_values(os.path.join(current_path, env_name))

    # set api keys
    if "ZHIPUAI_API_KEY" in cfg:
        zhipuai.api_key = cfg["ZHIPUAI_API_KEY"]
        cfg["zhipu"] = True
    if "OPENAI_API_KEY" in cfg:
        openai.api_key = cfg["OPENAI_API_KEY"]
        cfg["openai"] = True
    return cfg


def display_messages():
    "Display chat messages from history"
    avators = {
        "user": "🧑‍💻",
        "assistant": "🤖",
        "human": "🧑‍💻",
        "ai": "🤖",
        "system": "🦖",
    }
    for message in st.session_state.messages:
        with st.chat_message(message.type, avatar=avators[message.type]):
            st.markdown(message.content)


def create_stream_llm(
    model_type: ModelType,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> BaseChatModel:
    if model_type == ModelType.OpenAI:
        llm = ChatOpenAI(temperature=temperature, top_p=top_p, max_tokens=max_tokens, streaming=True)
    elif model_type == ModelType.ZhipuAI:
        llm = ChatZhipuAI(
            api_key=zhipuai.api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            model_kwargs={"top_p": top_p},
        )
    else:
        raise Exception(f"Unsupport model type: {model_type}")
    return llm


def main():
    st.set_page_config(
        page_title="Chat LLM Playground",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = init_config()

    st.title("Chat Playground")
    # st.markdown(
    #     "<sub>Code Interpreter 使用 jupyter kernel 执行代码，请确认执行环境安全后再与大模型交互</sub>",
    #     unsafe_allow_html=True,
    # )

    with st.sidebar:
        temperature = st.slider("temperature", 0.0, 1.5, 0.9, step=0.01)
        top_p = st.slider("top_p", 0.0, 1.0, 0.7, step=0.01)
        max_new_token = st.slider("Output length", 5, 32000, 2048, step=1)

        cols = st.columns(2)
        export_btn = cols[0]
        clear_history = cols[1].button("Clear History", use_container_width=True)
        retry = export_btn.button("Retry", use_container_width=True)

        # system_prompt = st.text_area(label="System Prompt", height=300, value=CHATGPT_SYSTEM_PROMPT)

    model_cols = st.columns([0.2, 0.2, 0.6])
    with model_cols[0]:
        models = []
        if cfg.get("zhipu"):
            models.append(ModelType.ZhipuAI)
        if cfg.get("openai"):
            models.append(ModelType.OpenAI)
        if len(models) == 0:
            st.error("没有找到可用的API，请至少配置 ZHIPUAI_API_KEY/OPENAI_API_KEY 的其中一项")
            st.stop()
        model_type = st.selectbox("LLM", models)
    with model_cols[1]:
        model_name = st.selectbox("model", GetModels(model_type))
    with model_cols[2]:
        st.markdown(
            f"**当前模型**: {model_type}[{model_name}]\n\n"
            f"`temperature={temperature} top_p={top_p} max_new_token={max_new_token}`"
        )
    chat_history = st.container()

    llm = create_stream_llm(
        model_type=model_type,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_token,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if prompt := st.chat_input():
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        result = run_chat(
            llm=llm,
            messages=st.session_state.messages + [user_message],
            st_callback=StreamlitCallbackHandler(st.container()),
        )
        st.session_state.messages.append(result.message)

    if clear_history:
        print("\n== Clean ==\n")
        st.session_state.messages = []
        return

    # alway refresh message when streaming complete
    with chat_history:
        display_messages()


def cutoff_historys(messages, max_history_length):
    results = []
    if messages[0].type == "system":
        results.append(messages[0])
    append_length = max_history_length - len(results)
    results += messages[-1 * append_length :]
    return results


def run_chat(llm, messages, st_callback, max_history_length=10):
    messages = cutoff_historys(messages, max_history_length)
    res = llm.generate(messages=[messages], callbacks=[st_callback], stream=True)
    print(f"size={len(res.generations[0])} llm_output={res.llm_output}")

    result = res.generations[0][0]

    # TODO: 增加action/tools 的处理
    # st_callback.on_agent_action()

    finish = AgentFinish(result, f"{res.llm_output}")
    st_callback.on_agent_finish(finish)
    return result


if __name__ == "__main__":
    main()
