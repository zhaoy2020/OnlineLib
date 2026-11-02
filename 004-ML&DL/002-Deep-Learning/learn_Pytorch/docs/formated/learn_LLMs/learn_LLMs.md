**Table of contents**<a id='toc0_'></a>    
- 1. [HuggingFace](#toc1_)    
- 2. [什么是RAG？](#toc2_)    
  - 2.1. [文本知识检索](#toc2_1_)    
    - 2.1.1. [知识库构建](#toc2_1_1_)    
    - 2.1.2. [查询构建](#toc2_1_2_)    
    - 2.1.3. [如何检索？-文本检索](#toc2_1_3_)    
    - 2.1.4. [如何喂给大模型？-生成增强](#toc2_1_4_)    
  - 2.2. [多模态知识检索](#toc2_2_)    
  - 2.3. [应用](#toc2_3_)    
- 3. [调用大语言模型](#toc3_)    
  - 3.1. [OpenAI Python SDK](#toc3_1_)    
    - 3.1.1. [文本生成](#toc3_1_1_)    
    - 3.1.2. [代码补全](#toc3_1_2_)    
    - 3.1.3. [图像生成（DALL-E）](#toc3_1_3_)    
    - 3.1.4. [图像识别](#toc3_1_4_)    
    - 3.1.5. [语音转文本（Whisper）](#toc3_1_5_)    
    - 3.1.6. [错误处理与最佳实践](#toc3_1_6_)    
    - 3.1.7. [openai库的高级用法](#toc3_1_7_)    
      - 3.1.7.1. [异步支持](#toc3_1_7_1_)    
      - 3.1.7.2. [微调（Fine-tuning）：](#toc3_1_7_2_)    
      - 3.1.7.3. [流式响应：](#toc3_1_7_3_)    
  - 3.2. [deepseek-ai的SDK](#toc3_2_)    
  - 3.3. [curl接口](#toc3_3_)    
- 4. [部署大模型](#toc4_)    
  - 4.1. [下载模型](#toc4_1_)    
  - 4.2. [ollama](#toc4_2_)    
    - 4.2.1. [Install and run model](#toc4_2_1_)    
    - 4.2.2. [API on web port](#toc4_2_2_)    
    - 4.2.3. [Python ollama module](#toc4_2_3_)    
      - 4.2.3.1. [demo：翻译中文为英文](#toc4_2_3_1_)    
  - 4.3. [ktransformers](#toc4_3_)    
    - 4.3.1. [Docker安装](#toc4_3_1_)    
    - 4.3.2. [编译安装](#toc4_3_2_)    
      - 4.3.2.1. [prepare](#toc4_3_2_1_)    
      - 4.3.2.2. [方式一：pip3 install whl](#toc4_3_2_2_)    
      - 4.3.2.3. [方式二：编译](#toc4_3_2_3_)    
    - 4.3.3. [虚拟机中编译安装](#toc4_3_3_)    
    - 4.3.4. [使用](#toc4_3_4_)    
- 5. [转格式](#toc5_)    

<!-- vscode-jupyter-toc-config
	numbering=true
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# 1. <a id='toc1_'></a>[HuggingFace](#toc0_)


```python
import os 


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```


```bash
%%bash
export HF_ENDPOINT="https://hf-mirror.com"
```

# 2. <a id='toc2_'></a>[什么是RAG？](#toc0_)

RAG的分类：

|Model | 检索器微调 | 大预言模型微调| 例如 |
|---|---|---| --- |
| 黑盒 | - | - | e.g. In-context ralm |
| 黑盒 | 是 | - | e.g. Rplug |
| 白盒 | - | 是 | e.g. realm, self-rag |
| 白盒 | 是 | 是 | e.g. altas |

## 2.1. <a id='toc2_1_'></a>[文本知识检索](#toc0_)
如何检索出相关信息来辅助改善大语言模型生成质量的系统。知识检索通常包括知识库构建、查询构建、文本检索和检索结果重排四部分。

### 2.1.1. <a id='toc2_1_1_'></a>[知识库构建](#toc0_)
文本块的知识库构建，如维基百科、新闻、论文等。

文本分块：将文本分成多个块，每个块包含一个或多个句子。
- 固定大小块：将文本分成固定大小的块，如每个块包含512个字符。
- 基于内容块：将文本分成基于内容的块，如每个块包含一个句子。
  - 通过句子分割符分割句子。
  - 用LLM进行分割

知识库增强：知识库增强是通过改进和丰富知识库的内容和结构，为查询提供"抓手”，包括查询生成与标题生成两种方法。
- 伪查询生成
- 标题生成

### 2.1.2. <a id='toc2_1_2_'></a>[查询构建](#toc0_)
查询构建：旨在通过查询增强的方式，扩展和丰富用户查询的语义和内容，提高检索结果的准确性和全面性，“钩"出相应内容。增强方式可分为语义增强与内容增强。
- 语义增强：同一句话多种表达方式
- 内容增强：增加背景知识

### 2.1.3. <a id='toc2_1_3_'></a>[如何检索？-文本检索](#toc0_)
`检索器`：给定知识库和用户查询，文本检索旨在找到知识库中与用户查询相关的知识文本;检索效率增强旨在解决检索时的性能瓶颈问题。所以检索质量、检索效率很重要。常见检索器有三类：
- 判别式检索器：
  - 稀疏检索器，e.g. TF-IDF
  - 双向编码检索器，e.g. 用bert预先将文本块进行编码成向量
  - 交叉编码检索器，e.g. 
- 生成式检索器：器直接将知识库中的文档信息记忆在模型参数中。然后，在接收到查询请求时，能够直接生成相关文档的标识符夺（即Doc ID），以完成检索。
- 图检索器：图检索器的知识库为图数据库，包括开放知识图谱和自建图两种，它们一般由<主体、谓词和客体>三元组构成。这样做不仅可以捕捉概念间的语义关系，还允许人类和机器可以共同对知识进行理解与推理。

`重排器`：检索阶段为了保证检索速度通常会损失一定的性能，可能检索到质量较低的文档。重排的目的是对检索到的段落进行进一步的排序精选。重排可以分为基于交叉编码的方法和基于上下文学习的方法。

### 2.1.4. <a id='toc2_1_4_'></a>[如何喂给大模型？-生成增强](#toc0_)
RAG增强比较：

|架构分类|优点|缺点|
|-|-|-|
|输入端prompt|简单|tokens太多|
|中间层|高效|耗GPU资源|
|输出端|-|-|

## 2.2. <a id='toc2_2_'></a>[多模态知识检索](#toc0_)
## 2.3. <a id='toc2_3_'></a>[应用](#toc0_)
对话机器人、知识库文答...

# 3. <a id='toc3_'></a>[大语言模型SDK](#toc0_)

## 3.1. <a id='toc3_1_'></a>[OpenAI Python SDK](#toc0_)

### 3.1.1. <a id='toc3_1_1_'></a>[文本生成](#toc0_)


```python
# Please install OpenAI SDK first: `pip3 install openai`

import os
import openai 


client = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"), 
    base_url = "https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False, 
    max_tokens=2048,
)

print(response.choices[0].message.content)

# # stream=True的时候，启用流示返回
# for chunk in response:
#     print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.1.2. <a id='toc3_1_2_'></a>[代码补全](#toc0_)


```python
response = client.chat.completions.create(
    messages = [
        {'role': 'user', 'content': "1 + 1"},
    ],
    model = "gpt-3.5-turbo", 
    stream = False, 
    max_tokens = 2048,
)

print(response.choices[0].message.content)
```

### 3.1.3. <a id='toc3_1_3_'></a>[图像生成（DALL-E）](#toc0_)

调用images.generate生成图像：


```python
response = client.images.generate(
    prompt="一只穿着宇航服的猫",
    n=1,
    size="1024x1024"
)

print(response.data.url)

```

### 3.1.4. <a id='toc3_1_4_'></a>[图像识别](#toc0_)


```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "这是什么？",
                    "type": "text"
                },
                {
                    "image_url": {
                        "url": "data:image/png;base64,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"
                    },
                    "type": "image_url"
                }
            ]
        }
    ],
    model='gpt-4o-2024-05-13',
    stream=False,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### 3.1.5. <a id='toc3_1_5_'></a>[语音转文本（Whisper）](#toc0_)

使用audio.transcriptions处理音频文件：


```python
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

print(transcript.text)

```

### 3.1.6. <a id='toc3_1_6_'></a>[错误处理与最佳实践](#toc0_)

异常捕获：处理API请求中的常见错误（如认证失败、超时）：


```python
from openai import APIError
try:
    response = client.chat.completions.create(...)
except APIError as e:
    print(f"API请求失败: {e}")

```

### 3.1.7. <a id='toc3_1_7_'></a>[openai库的高级用法](#toc0_)

#### 3.1.7.1. <a id='toc3_1_7_1_'></a>[异步支持](#toc0_)

SDK提供了异步客户端AsyncOpenAI,使用方法与同步客户端类似,只需在API调用前加上await:


```python

```

#### 3.1.7.2. <a id='toc3_1_7_2_'></a>[微调（Fine-tuning）：](#toc0_)

支持对基础模型进行定制化训练，需准备数据集并调用fine_tuning.jobs.create

#### 3.1.7.3. <a id='toc3_1_7_3_'></a>[流式响应：](#toc0_)

通过stream=True实现逐词实时输出，适用于交互式场景：


```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    stream=True
)

for chunk in stream:
    print(chunk.choices.delta.content or "", end="")

```

## 3.2. <a id='toc3_2_'></a>[deepseek-ai的SDK](#toc0_)

- Temperature 设置，参数默认为 1.0。

|场景	|温度|
|-|-|
|代码生成/数学解题   	|0.0|
|数据抽取/分析	|1.0|
|通用对话	|1.3|
|翻译	|1.3|
|创意类写作/诗歌创作	|1.5|


```python
# Please install OpenAI SDK first: `pip3 install openai`

import os 
from openai import OpenAI 


client = OpenAI(
    api_key = "sk-", 
    base_url = "https://api.deepseek.com/v1",            # 出于与 OpenAI 兼容考虑，此处 v1 与模型版本无关。
)

```


```python
print(client.models.list())
```

    SyncPage[Model](data=[Model(id='deepseek-chat', created=None, object='model', owned_by='deepseek'), Model(id='deepseek-reasoner', created=None, object='model', owned_by='deepseek')], object='list')



```python


response = client.chat.completions.create(
    # model="deepseek-chat",        # DeepSeek-V3
    model = "deepseek-reasoner",    # DeepSeek 最新推出的推理模型 DeepSeek-R1
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```


    ---------------------------------------------------------------------------

    APIStatusError                            Traceback (most recent call last)

    Cell In[15], line 1
    ----> 1 response = client.chat.completions.create(
          2     # model="deepseek-chat",        # DeepSeek-V3
          3     model = "deepseek-reasoner",    # DeepSeek 最新推出的推理模型 DeepSeek-R1
          4     messages=[
          5         {"role": "system", "content": "You are a helpful assistant"},
          6         {"role": "user", "content": "Hello"},
          7     ],
          8     stream=False
          9 )
         11 print(response.choices[0].message.content)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_utils/_utils.py:279, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
        277             msg = f"Missing required argument: {quote(missing[0])}"
        278     raise TypeError(msg)
    --> 279 return func(*args, **kwargs)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/resources/chat/completions.py:863, in Completions.create(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, reasoning_effort, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, extra_headers, extra_query, extra_body, timeout)
        821 @required_args(["messages", "model"], ["messages", "model", "stream"])
        822 def create(
        823     self,
       (...)
        860     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        861 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        862     validate_response_format(response_format)
    --> 863     return self._post(
        864         "/chat/completions",
        865         body=maybe_transform(
        866             {
        867                 "messages": messages,
        868                 "model": model,
        869                 "audio": audio,
        870                 "frequency_penalty": frequency_penalty,
        871                 "function_call": function_call,
        872                 "functions": functions,
        873                 "logit_bias": logit_bias,
        874                 "logprobs": logprobs,
        875                 "max_completion_tokens": max_completion_tokens,
        876                 "max_tokens": max_tokens,
        877                 "metadata": metadata,
        878                 "modalities": modalities,
        879                 "n": n,
        880                 "parallel_tool_calls": parallel_tool_calls,
        881                 "prediction": prediction,
        882                 "presence_penalty": presence_penalty,
        883                 "reasoning_effort": reasoning_effort,
        884                 "response_format": response_format,
        885                 "seed": seed,
        886                 "service_tier": service_tier,
        887                 "stop": stop,
        888                 "store": store,
        889                 "stream": stream,
        890                 "stream_options": stream_options,
        891                 "temperature": temperature,
        892                 "tool_choice": tool_choice,
        893                 "tools": tools,
        894                 "top_logprobs": top_logprobs,
        895                 "top_p": top_p,
        896                 "user": user,
        897             },
        898             completion_create_params.CompletionCreateParams,
        899         ),
        900         options=make_request_options(
        901             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        902         ),
        903         cast_to=ChatCompletion,
        904         stream=stream or False,
        905         stream_cls=Stream[ChatCompletionChunk],
        906     )


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:1283, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1269 def post(
       1270     self,
       1271     path: str,
       (...)
       1278     stream_cls: type[_StreamT] | None = None,
       1279 ) -> ResponseT | _StreamT:
       1280     opts = FinalRequestOptions.construct(
       1281         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1282     )
    -> 1283     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:960, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        957 else:
        958     retries_taken = 0
    --> 960 return self._request(
        961     cast_to=cast_to,
        962     options=options,
        963     stream=stream,
        964     stream_cls=stream_cls,
        965     retries_taken=retries_taken,
        966 )


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:1064, in SyncAPIClient._request(self, cast_to, options, retries_taken, stream, stream_cls)
       1061         err.response.read()
       1063     log.debug("Re-raising status error")
    -> 1064     raise self._make_status_error_from_response(err.response) from None
       1066 return self._process_response(
       1067     cast_to=cast_to,
       1068     options=options,
       (...)
       1072     retries_taken=retries_taken,
       1073 )


    APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance', 'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}


## 3.3. <a id='toc3_3_'></a>[curl接口](#toc0_)


```bash
%%bash 
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DeepSeek API Key>" \
  -d '{
        "model": "deepseek-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'
```

# 4. <a id='toc4_'></a>[本地部署大模型](#toc0_)


## 4.1. <a id='toc4_1_'></a>[下载模型](#toc0_)


```python
# https://huggingface.co/deepseek-ai/DeepSeek-R1
hfd.sh deepseek-ai/DeepSeek-R1 -x 10 -j 10 

# https://huggingface.co/unsloth/DeepSeek-R1-GGUF
hfd.sh unsloth/DeepSeek-R1-GGUF -x 10 -j 10 --include DeepSeek-R1-Q8_0

```

## 4.2. <a id='toc4_2_'></a>[ollama](#toc0_)
### 4.2.1. <a id='toc4_2_1_'></a>[Install and run model](#toc0_)


```python
# start the serve
ollama serve

# list all model images
ollama list 

# run model from image
ollama run model_card
```

### 4.2.2. <a id='toc4_2_2_'></a>[API on web port](#toc0_)
communicatation with local model via web port.

`generate` and `chat`.


```bash
%%bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:7b",
  "prompt": "Who are you?",
  "stream": false,
  "options": {
    "temperature": 0.6
  },
  "format": "json"
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   456  100   315  100   141    142     64  0:00:02  0:00:02 --:--:--   206


    {"model":"deepseek-r1:7b","created_at":"2025-02-18T02:49:32.744934023Z","response":"{\"}\u003cthink\u003e{\"\n\n\n\n\n\n\n\n\n\n:\n\n{\n\n}\n\n}\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n","done":false}


```bash
%%bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:7b",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ],
  "stream": false
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  5034    0  4905  100   129    326      8  0:00:16  0:00:15  0:00:01   719


    {"model":"deepseek-r1:7b","created_at":"2025-02-18T02:49:26.56791501Z","message":{"role":"assistant","content":"\u003cthink\u003e\nOkay, so I just read that \"Why is the sky blue?\" and now I'm trying to figure it out myself. Let me think through this step by step.\n\nFirst off, when you look at the sky on a clear day, it's usually blue, especially during the day when the sun is out. But sometimes I've seen it turn other colors too, like red in the evening or during sunrise. So why is it mostly blue?\n\nI know that light travels through the atmosphere, but how does it get colored? I remember learning about something called Rayleigh scattering from my science class. Let me try to recall what that was about. Rayleigh scattering involves light interacting with particles much smaller than the wavelength of light itself. When sunlight enters the Earth's atmosphere, it reaches tiny molecules in the air, like nitrogen and oxygen.\n\nWait, so these small particles scatter the sunlight in all directions. But why does this result in a blue sky? I think it has something to do with the wavelengths of light. Visible light ranges from violet to red, right? And I remember that violet light has a shorter wavelength than blue. So maybe the shorter wavelengths are scattered more.\n\nBut if blue is scattered more, wouldn't it be easier to see at night when there's less atmosphere overhead? Hmm, but then why does the sky turn red in the evening or during sunrise?\n\nOh right! During sunrise and sunset, the light has to pass through a much thicker layer of atmosphere. That makes sense because we're looking at the light after it's been scattered through more particles. The longer path means that all the shorter wavelengths (like violet) are scattered out, leaving red to dominate because it has a longer wavelength.\n\nSo during the day, when the sun is directly overhead, blue and green wavelengths get scattered away by Rayleigh scattering, making the sky appear blue. But in the early morning or late afternoon, as the sun is near the horizon, the light has to pass through more atmosphere, so red comes through because it's not scattered as much.\n\nWait, but isn't there also something called Mie scattering? I think that happens when particles are larger than the wavelength of light. Does that affect the color of the sky too?\n\nI believe Mie scattering is more significant for larger particles, like dust or droplets in clouds. So it might cause some effects we see during sunrise and sunset, but not as much as Rayleigh does for small molecules.\n\nSo to summarize: The sky appears blue on a clear day because blue light scatters more in the atmosphere due to Rayleigh scattering by tiny gas molecules. During sunrise and sunset, the longer path allows red light to dominate, giving the sky its reddish hues.\n\nBut wait, what about when we see other colors? I mean, sometimes during the day it's not just blue; I've seen green or yellow in some places. Is that because of Rayleigh scattering changing as the atmosphere gets thicker?\n\nOr maybe it's due to other factors like pollution or particles in the air affecting light differently. That might complicate things.\n\nAlso, does humidity play a role? Sometimes when it's humid, does the sky appear clearer instead of blue? I think higher humidity can affect how light scatters because more water vapor means more molecules to scatter off.\n\nSo maybe on days with high humidity, the Rayleigh scattering is less effective, making the sky look different. But that's probably a secondary effect compared to the basic reason for the color being blue.\n\nIn any case, I think the primary reason is Rayleigh scattering by nitrogen and oxygen in the atmosphere causing shorter wavelengths like blue to scatter more, resulting in a blue sky during clear days.\n\u003c/think\u003e\n\nThe sky appears blue primarily due to a phenomenon known as Rayleigh scattering. When sunlight enters Earth's atmosphere, it interacts with tiny molecules of nitrogen and oxygen. These particles scatter shorter wavelengths of light, such as blue and violet, which have shorter wavelengths than red or orange. This scattering is more effective for shorter wavelengths, causing the sky to appear blue during the day.\n\nDuring sunrise and sunset, the light passes through a thicker layer of atmosphere, where all shorter wavelengths are scattered away, leaving longer wavelengths like red to dominate, resulting in the reddish hues observed at these times.\n\nOther factors such as humidity, pollution, or atmospheric particles can influence how light scatters, but the primary reason for the sky's blue color remains Rayleigh scattering by nitrogen and oxygen molecules."},"done_reason":"stop","done":true,"total_duration":15024265818,"load_duration":22057058,"prompt_eval_count":9,"prompt_eval_duration":7000000,"eval_count":901,"eval_duration":14993000000}

### 4.2.3. <a id='toc4_2_3_'></a>[Python ollama module](#toc0_)


```python
import ollama


texts = '''
详细比较deepseek母公司和openAI公司的区别
'''

# model_card = "deepseek-r1:7b"
model_card = "modelscope.cn/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"

# 方式一（非流式输出）：
# outputs = ollama.generate(model_card, inputs)
# print(f'{outputs['response']}')

# 方式二（流式输出）：
outputs = ollama.generate(
    stream=True,
    model=model_card,
    prompt=texts,
)
for chunk in outputs:
    if not chunk['done']:
        print(f'{chunk['response']}', end='', flush=True)
```

#### 4.2.3.1. <a id='toc4_2_3_1_'></a>[demo：翻译中文为英文](#toc0_)


```python
import ollama 


class zh2en():
    def __init__(self, model_card):
        self.model_card = model_card
        
    def build_prompt(self, texts):
        # with open(prompt_template_path, 'r') as f:
        #     prompt_template = f.read()
        #     # str with replace function
        #     prompt = prompt_template.replace(var, texts)
        prompt_template = """
        专业翻译：\n
        ---\n
        {Chinese_words} \n
        --- \n
        作为翻译专家，将上述中文准确翻译为英文。 \n
        """
        prompt = prompt_template.replace("{Chinese_words}", texts)
        return prompt

    def translate(self, texts):
        prompt = self.build_prompt(texts = texts)
        # key step
        outputs = ollama.generate(
            stream=True,
            model=self.model_card,
            prompt=prompt,
        )
        for chunk in outputs:
            if not chunk['done']:
                print(f'{chunk['response']}', end='', flush=True)
            else:
                print('⚡')


translater = zh2en(model_card='modelscope.cn/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF')

translater.translate('基于深度学习的对枯草芽胞杆菌芽胞形成相关基因的研究。')
translater.translate('通过宏基因组研究微生物与植物相互作用的机制。')
```

    <think>
    嗯，首先我要理解这个题目的意思。“基于深度学习”指的是使用深度学习技术来进行研究。而“对枯草芽胞杆菌芽胞形成相关基因的研究”则是具体的研究内容，涉及到枯草芽胞杆菌在形成芽胞过程中相关的基因。
    
    我需要把这整个句子准确地翻译成英文。首先，“基于深度学习”可以直接翻译为“Based on deep learning”。接下来是“研究”，对应的英文是“study”。然后是“枯草芽胞杆菌”，这个应该是一个专有名词，可能需要查一下正确的英译名称，比如“Bacillus subtilis”。
    
    接着是“芽胞形成相关基因”，这部分可以翻译为“genes related to spore formation”。最后，把整个句子连贯起来，就是“Based on deep learning study of genes related to spore formation in Bacillus subtilis.”
    
    这样组合起来，既准确传达了原意，又符合英文的表达习惯。我觉得这个翻译应该是比较专业和准确的。
    </think>
    
    Study of Genes Related to Spore Formation in *Bacillus subtilis* Based on Deep Learning⚡
    <think>
    好的，首先我要理解用户的需求。他给了一个中英对照的句子，要求专业翻译，并且需要将中文句子“通过宏基因组研究微生物与植物相互作用的机制。”准确地翻译成英文。
    
    接下来，我需要分析原文的意思。句子的主干是“通过宏基因组研究…”，这里的关键词有“宏基因组”、“微生物”、“植物”以及“相互作用的机制”。所以，首先要确定这些术语在英文中的准确对应词。
    
    “宏基因组”通常翻译为“metagenome”或者“meta-genomics”，但更常见的是使用“metagenomics”来表示这一研究领域。因此，这里选择“metagenomics”作为翻译。
    
    然后，“通过…研究…”的结构在英文中可以用“through”或者“by means of”来表达，但为了简洁和专业，直接使用“Through”比较合适。
    
    接下来是“微生物与植物相互作用的机制”。这里需要注意语序和用词。整体结构应该是“the mechanisms underlying the interactions between microorganisms and plants.” 这样不仅清晰，而且符合学术写作的规范。
    
    最后，把这些部分组合起来，确保句子通顺且准确。所以，最终翻译为：“Through metagenomics research on the mechanisms of interaction between microorganisms and plants.”
    
    在整个思考过程中，我注意到用户可能是在撰写学术论文或者准备研究报告，因此准确性和专业性是关键。此外，保持句子的简洁也是必要的，以便读者能够快速理解内容。
    
    最后，再检查一遍翻译是否忠实于原文，并且符合英语表达习惯。确认无误后，就可以将这个翻译结果提供给用户了。
    </think>
    
    Through metagenomics research on the mechanisms of interaction between microorganisms and plants.⚡


## 4.3. <a id='toc4_3_'></a>[ktransformers](#toc0_)


### 4.3.1. <a id='toc4_3_1_'></a>[Docker安装](#toc0_)

[https://github.com/kvcache-ai/ktransformers-private/blob/main/doc/en/Docker.md](https://github.com/kvcache-ai/ktransformers-private/blob/main/doc/en/Docker.md)


```python
# pull the image from docker hub 
# about 19 GB
# docker pull approachingai/ktransformers:0.1.1
docker pull approachingai/ktransformers:0.2.1

# docker run \
#     --gpus all \
#     -v /path/to/models:/models \
#     -p 10002:10002 \
#     approachingai/ktransformers:v0.1.1 \
#     --port 10002 \
#     --gguf_path /models/path/to/gguf_path \
#     --model_path /models/path/to/model_path \
#     --web True

# Directly run
docker run  \
    --gpus all \
    -v /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai:/models \
    -p 10002:10002 \
    approachingai/ktransformers:0.1.1 \
    --port 10002 \
    --model_path /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai/DeepSeek-R1 \
    --gguf_path /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai/DeepSeek-R1-Q4_K_M_GGUF \
    --web True

# or
docker run \
    # -d \
    --gpus all \
    -it \
    -p 10002:10002 \
    -v /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai:/models \
    approachingai/ktransformers:0.1.1 \
    /bin/bash 

## and then
docker exec -it container_ID /bin/bash
```

QA:

- Q: Docker运行后出现 Illegal instruction (core dumped)报错
  - [https://github.com/kvcache-ai/ktransformers/issues/356](https://github.com/kvcache-ai/ktransformers/issues/356)
  - 重新编译以下:
    ```bash
    USE_NUMA=1
    bash install.sh
    ```

### 4.3.2. <a id='toc4_3_2_'></a>[编译安装](#toc0_)

[https://kvcache-ai.github.io/ktransformers/en/install.html](https://kvcache-ai.github.io/ktransformers/en/install.html)

#### 4.3.2.1. <a id='toc4_3_2_1_'></a>[prepare](#toc0_)


```python
name="ktransformers"

conda create -n $name python=3.11 -y 
conda activate $name


# Install CudaToolkit and nvcc ...
conda install nvidia/label/cuda-12.4.0::cuda -y --channel nvidia/label/cuda-12.4.0

# Anaconda provides a package called `libstdcxx-ng` that includes a newer version of `libstdc++`, which can be installed via `conda-forge`.
conda install -c conda-forge libstdcxx-ng -y 

strings ~/miniconda3/envs/${name}/lib/libstdc++.so.6 | grep GLIBCXX


# Install PyTorch via pip ...
# pip3 install torch torchvision torchaudio
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch -y

# pip3 install packaging ninja cpufeature numpy
conda install conda-forge::ninja conda-forge::packaging anaconda::numpy -y


```

#### 4.3.2.2. <a id='toc4_3_2_2_'></a>[方式一：pip3 install whl](#toc0_)


```python
# ktransformers
pip3 install ktransformers-0.2.1.post1+cu124torch24avx2-cp311-cp311-linux_x86_64.whl

# flash_attn
# pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# flashinfer
pip3 install flashinfer_python-0.2.2+cu124torch2.4-cp38-abi3-linux_x86_64.whl
python -c "import torch; print(torch.cuda.get_device_capability())"
export TORCH_CUDA_ARCH_LIST="8.0"
```

#### 4.3.2.3. <a id='toc4_3_2_3_'></a>[方式二：编译](#toc0_)


```python
# Make sure your system has dual sockets and double size RAM than the model's size (e.g. 1T RAM for 512G model)
export USE_NUMA=1
bash install.sh # or `make dev_install`
```

### 4.3.3. <a id='toc4_3_3_'></a>[虚拟机中编译安装](#toc0_)

由于`GLIBCxxx`报错，改用虚拟机中ubuntu安装ktransformers


```python
# 拉取包含GLIBC 2.29+的镜像（如Ubuntu 20.04）
docker pull ubuntu:20.04

# 启动容器并挂载项目目录 
# docker run -it -v /path/to/your/code:/app ubuntu:20.04 /bin/bash 
docker run \
    --gpus all \
    -p 10002:10002\
    -it \
    ubuntu:20.04 /bin/bash 

# Install wget and then 
apt-get update 
apt-get install wget 

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh

docker exec --gpus all -it ubuntu:20.04 /bin/bash
```

### 4.3.4. <a id='toc4_3_4_'></a>[使用](#toc0_)


```python
# 将主机和容器的10002端口做映射
docker run --gpus all -p 10002:10002 -it \
    ubuntu_2004:ktransformers /bin/bash


# 检查映射
docker port [容器ID]
```


```python
# As we use dual socket, we set cpu_infer to 65
python -m ktransformers.local_chat \
    --model_path DeepSeek-R1 \
    --gguf_path DeepSeek-R1-Q4_K_M_GGUF \
    --cpu_infer 60 \
    --max_new_tokens 10000 \
    --cache_lens 50000 \
    --total_context 50000 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --force_think \
    --use_cuda_graph \
    --port 10002
    # --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    # --host 127.0.0.1 \
```


```python
# 32K
# 64K
32768 * 2
```




    65536




```python
2 ** 13
```




    8192




```python
# http://localhost:10002/web/index.html#/chat
# using multi-GPU with DeepSeek-V3-configures (V3 is the same as R1)
# increace cache_lens to take up more GPU memory, when get long context
ktransformers \
    --model_path /models/DeepSeek-R1/ \
    --gguf_path /models/DeepSeek-R1-Q4_K_M_GGUF/ \
    --use_cuda_graph \
    --temperature 0.6 \
    --port 10002 \
    --cpu_infer 62 \
    --cache_lens 32768 \
    --total_context 32768 \
    --optimize_config_path /workspace/ktransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml \
    # --optimize_config_path /workspace/ktransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    # --web True \
```


```bash
%%bash
curl -X 'GET' \
  'http://localhost:10002/api/tags' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "DeepSeek-R1",
  "stream": true
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   213  100    81  100   132  27000  44000 --:--:-- --:--:-- --:--:-- 71000


    {"models":[{"name":"DeepSeek-Coder-V2-Instruct","modified_at":"123","size":123}]}


```bash
%%bash
curl -X 'POST' \
  'http://localhost:10002/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "DeepSeek-R1",
  "stream": true
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed


    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"<think>\n\n","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"</think>\n\n","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"Sure! ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"Here's ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"a ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"light-hearted ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"joke ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"for ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"you:\n\n","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"Why ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"don’t ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"skeletons ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"fight ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"each ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"other?  \n","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"…Because ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"they ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"don’t ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"have ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"the ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"*guts*!  \n\n","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"😄 ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"Got ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"any ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"more ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    


    100 20102    0 19970  100   132    737      4  0:00:33  0:00:27  0:00:06  1213


    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"requests? ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"I’m ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"here ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"to ","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"help!","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[{"delta":{"content":"","function_call":null,"refusal":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":null}
    
    data: {"id":"6d273293-d06b-4ae0-acdb-574d38242696","choices":[],"created":1742001772,"model":"DeepSeek-Coder-V2-Instruct","object":"chat.completion.chunk","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":53,"prompt_tokens":7,"total_tokens":60,"completion_tokens_details":null,"prompt_tokens_details":null}}
    
    data: [DONE]
    



```bash
%%bash
curl -X 'POST' \
  'http://localhost:10002/api/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  "model": "DeepSeek-R1",
  "prompt": "tell me a joke",
  "stream": true
}"
```

# 5. <a id='toc5_'></a>[转格式](#toc0_)


```bash
%%bash
# ipynb to html
jupyter nbconvert \
    --to html \
    --output-dir=../docs/formated/learn_LLMs \
    LLMs.ipynb 
    # --NbConvertApp.log_level=ERROR

cp -rf Pytorch_Pictures ../docs/formated/learn_LLMs/
# browse translate html to pdf
```

    [NbConvertApp] Converting notebook LLMs.ipynb to html
    [NbConvertApp] Writing 482390 bytes to Format/LLMs/LLMs.html



```python
# ipynb to markdown
!jupyter nbconvert --to markdown --output-dir=../docs/formated/learn_LLMs learn_LLMs.ipynb
```

    [NbConvertApp] WARNING | pattern 'LLMs.ipynb' matched no files
    [NbConvertApp] Making directory ../docs/formated/learn_LLMs
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

