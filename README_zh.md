# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

**Segment Anything Model (SAM)** 可以根据输入提示（比如点和框）生成高质量的目标掩码，并可以用于为图像中所有的目标生成掩码。它已经在包含1100万张图像和11亿个掩码的[数据集](https://segment-anything.com/dataset/index.html)上进行了训练，并在各种分割任务上表现出非常强的zero-shot性能。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## 安装

该代码需要 python>=3.8 以及 pytorch>=1.7 和 torchvision>=0.8。请按照[此处](https://pytorch.org/get-started/locally/)的说明安装PyTorch和TorchVision依赖项。强烈建议安装带有CUDA支持的PyTorch和TorchVision。

安装 Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

或者在本地克隆仓库并使用以下方式安装:

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

以下可选依赖项对于掩码后处理、以COCO格式保存掩码、示例 notebooks 和以 ONNX 格式导出模型是必要的。要运行示例的 notebooks ，`jupter` 也是必要的。

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="开始"></a>开始


首先下载一个 [模型检查点](#model-checkpoints)。然后,该模型只需要几行代码就可以根据给定的提示获取掩码:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或者对整张图产生掩码:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

此外，图像的掩码可以用命令行生成：

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

有关使用提示和自动生成掩码的更多详细信息, 请参阅示例笔记本[使用带提示的SAM](/notebooks/predictor_example.ipynb)和[自动生成掩码](/notebooks/automatic_mask_generator_example.ipynb)。

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX导出

SAM 的轻量级掩码解码器可以导出到 ONNX 格式,以便它可以在任何支持ONNX 运行时的环境中运行, 例如在浏览器中展示的[演示](https://segment-anything.com/demo)。使用以下命令导出模型: 
```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

请参阅[示例笔记本](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)，了解如何通过 SAM 的主干进行图像预处理并使用 ONNX 模型进行掩码预测的详细信息。建议使用最新的稳定版本的 PyTorch 进行 ONNX 导出。

### 网页Demo

`demo/` 文件夹中有一个简单的单页 React 应用，演示了如何在具有多线程的 Web 浏览器中使用导出的 ONNX 模型进行掩码预测。请参阅 [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) 以获取更多详细信息。

## <a name="Models"></a>Model Checkpoints

这个模型有三个版本，具有不同的主干（backbone）尺寸。可以通过运行相应的代码来实例化这些模型。

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

点击下面的链接下载相应模型类型的checkpoint。

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## 数据集

请查看[此处](https://ai.facebook.com/datasets/segment-anything/)以获取数据集的概览。可以在[这里](https://ai.facebook.com/datasets/segment-anything-downloads/)下载数据集。通过下载数据集，您同意已阅读并接受 SA-1B 数据集研究许可的条款。

我们将每个图像的掩码保存为一个 json 文件。它可以作为一个字典在 Python 中以下格式加载。

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

图像 ids 可以在 sa_images_ids.txt 中找到，同样可以通过上述[链接](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。

将 COCO RLE 格式的掩码解码为二进制：

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

请参阅[此处](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)，以获取更多关于操作以 RLE 格式存储的掩码的说明。

## 许可证

该模型根据 [Apache 2.0](LICENSE) 许可获得许可。

## 贡献

请看 [contributing](CONTRIBUTING.md) 和 [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

Segment Anything 项目在许多贡献者（按字母顺序排列）的帮助下得以实现：

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

如果您在研究中使用了 SAM 或 SA-1B，请使用以下 BibTeX 条目。

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
