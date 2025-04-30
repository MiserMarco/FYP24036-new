step1:按照MinerU官方教程安装MinerU
> [!WARNING]
> **安装前必看——软硬件环境支持说明**
> 
> 为了确保项目的稳定性和可靠性，我们在开发过程中仅对特定的软硬件环境进行优化和测试。这样当用户在推荐的系统配置上部署和运行项目时，能够获得最佳的性能表现和最少的兼容性问题。
>
> 通过集中资源和精力于主线环境，我们团队能够更高效地解决潜在的BUG，及时开发新功能。
>
> 在非主线环境中，由于硬件、软件配置的多样性，以及第三方依赖项的兼容性问题，我们无法100%保证项目的完全可用性。因此，对于希望在非推荐环境中使用本项目的用户，我们建议先仔细阅读文档以及FAQ，大多数问题已经在FAQ中有对应的解决方案，除此之外我们鼓励社区反馈问题，以便我们能够逐步扩大支持范围。

<table>
    <tr>
        <td colspan="3" rowspan="2">操作系统</td>
    </tr>
    <tr>
        <td>Linux after 2019</td>
        <td>Windows 10 / 11</td>
        <td>macOS 11+</td>
    </tr>
    <tr>
        <td colspan="3">CPU</td>
        <td>x86_64 / arm64</td>
        <td>x86_64(暂不支持ARM Windows)</td>
        <td>x86_64 / arm64</td>
    </tr>
    <tr>
        <td colspan="3">内存</td>
        <td colspan="3">大于等于16GB，推荐32G以上</td>
    </tr>
    <tr>
        <td colspan="3">存储空间</td>
        <td colspan="3">大于等于20GB，推荐使用SSD以获得最佳性能</td>
    </tr>
    <tr>
        <td colspan="3">python版本</td>
        <td colspan="3">>=3.10</td>
    </tr>
    <tr>
        <td colspan="3">Nvidia Driver 版本</td>
        <td>latest(专有驱动)</td>
        <td>latest</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CUDA环境</td>
        <td>11.8/12.4/12.6/12.8</td>
        <td>11.8/12.4/12.6/12.8</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CANN环境(NPU支持)</td>
        <td>8.0+(Ascend 910b)</td>
        <td>None</td>
        <td>None</td>
    </tr>
    <tr>
        <td rowspan="2">GPU/MPS 硬件支持列表</td>
        <td colspan="2">显存6G以上</td>
        <td colspan="2">
        Volta(2017)及之后生产的全部带Tensor Core的GPU <br>
        6G显存及以上</td>
        <td rowspan="2">apple slicon</td>
    </tr>
</table>

# 构建conda环境
conda create -n fina 'python>=3.10' -y
conda activate fina
pip install -U "magic-pdf[full]" -i https://mirrors.aliyun.com/pypi/simple

模型下载分为首次下载和更新模型目录，请参考对应的文档内容进行操作

# 首次下载模型文件

模型文件可以从 Hugging Face 或 Model Scope 下载，由于网络原因，国内用户访问HF可能会失败，请使用 ModelScope。

<details>
  <summary>方法一：从 Hugging Face 下载模型</summary>
  <p>使用python脚本 从Hugging Face下载模型文件</p>
  <pre><code>pip install huggingface_hub
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py</code></pre>
  <p>python脚本会自动下载模型文件并配置好配置文件中的模型目录</p>
</details>

## 方法二：从 ModelScope 下载模型

### 使用python脚本 从ModelScope下载模型文件

```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```
python脚本会自动下载模型文件并配置好配置文件中的模型目录

配置文件可以在用户目录中找到，文件名为`magic-pdf.json`

> [!TIP]
> windows的用户目录为 "C:\\Users\\用户名", linux用户目录为 "/home/用户名", macOS用户目录为 "/Users/用户名"


# 此前下载过模型，如何更新

## 1. 通过 Hugging Face 或 Model Scope 下载过模型

如此前通过 HuggingFace 或 Model Scope 下载过模型，可以重复执行此前的模型下载python脚本，将会自动将模型目录更新到最新版本。


#### 3. 修改配置文件以进行额外配置

完成[下载模型权重文件]步骤后，脚本会自动生成用户目录下的magic-pdf.json文件，并自动配置默认模型路径。
您可在【用户目录】下找到magic-pdf.json文件。

> [!TIP]
> windows的用户目录为 "C:\\Users\\用户名", linux用户目录为 "/home/用户名", macOS用户目录为 "/Users/用户名"

您可修改该文件中的部分配置实现功能的开关，如表格识别功能：

> [!NOTE]
>如json内没有如下项目，请手动添加需要的项目，并删除注释内容（标准json不支持注释）

```json
{
    // other config
    "layout-config": {
        "model": "doclayout_yolo" 
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": true  // 公式识别功能默认是开启的，如果需要关闭请修改此处的值为"false"
    },
    "table-config": {
        "model": "rapid_table",
        "sub_model": "slanet_plus",
        "enable": true, // 表格识别功能默认是开启的，如果需要关闭请修改此处的值为"false"
        "max_time": 400
    }
}
```

### 使用GPU

如果您的设备支持CUDA，且满足主线环境中的显卡要求，则可以使用GPU加速，请根据自己的系统选择适合的教程：

修改配置文件中的字段为如下：

{
  "device-mode":"cuda"
}

# 命令行测试minerU
## command line example

magic-pdf -p {some_pdf} -o {some_output_dir} -m auto

2、配置我们的年报智能分析RAG系统
# 关键代码 
app-V2.py

# 安装关键依赖包
requirments.txt

# 在命令行启动前后端
python app-V2.py

# 通过访问url：http://127.0.0.1:7860/

now start
