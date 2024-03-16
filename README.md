# Stream Chat Playground

Yet Another Chat Playground for Large Language Models (with LangChain and Streamlit)

## Clone repo

```bash
git clone https://github.com/upbit/StreamChatPlayground
git submodule update --init --recursive
```

## Prepare venv

**PDM**:

```bash
pip install pdm
pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple # 使用清华国内源
pdm install

export METAGPT_PROJECT_ROOT=$(pwd)/MetaGPT
pip install -e MeteGPT
```

**pip**:

```bash
pip install -r requirements.txt

export METAGPT_PROJECT_ROOT=$(pwd)/MetaGPT
pip install -e MeteGPT
```

## Run

* `streamlit run webui.py`

## FAQ

### Playwright Host validation warning

```
Playwright Host validation warning:
╔══════════════════════════════════════════════════════╗
║ Host system is missing dependencies to run browsers. ║
║ Please install them with the following command:      ║
║                                                      ║
║     playwright install-deps                          ║
║                                                      ║
║ Alternatively, use apt:                              ║
║     apt-get install libnss3\                         ║
║         libnspr4\                                    ║
║         libatk1.0-0\                                 ║
║         libatk-bridge2.0-0\                          ║
║         libcups2\                                    ║
║         libatspi2.0-0\                               ║
║         libxcomposite1\                              ║
║         libxdamage1                                  ║
║                                                      ║
║ <3 Playwright Team                                   ║
╚══════════════════════════════════════════════════════╝
```

Run: `apt-get install libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libatspi2.0-0 libxcomposite1 libxdamage1`
