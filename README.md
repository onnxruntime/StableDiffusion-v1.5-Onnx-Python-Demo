# Stable Diffusion with ONNX Runtime & DirectML

This Python application uses ONNX Runtime with DirectML to run an image inference loop based on a provided prompt. This app works by generating images based on a textual prompt using a trained ONNX model. The program also includes a simple GUI for an interactive experience if desired.

## Setup

First, obtain the Olive-optimized models by following the [Olive Stable Diffusion Optimization tutorial](https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion). Once you've optimized the models, you should copy the output directory (`<olive_clone_path>/examples/directml/stable_diffusion/models/optimized/runwayml/stable-diffusion-v1-5/`) to this project directory (`<python_demo_clone_path>/stable-diffusion-v1-5/`).

Ensure you have Python 3.9 or later installed on your system. You can download it from [here](https://www.python.org/downloads/). 

Clone this repository and navigate to its location in your terminal.

You should also have the following packages installed:
- PySimpleGUI
- onnxruntime
- packaging
- diffusers

You can install these via pip:

```sh
pip install -r requirements.txt
```

## Usage

You can run the script using the following command:

```sh
python stable_diffusion.py --prompt "castle surrounded by water and nature, village, volumetric lighting, detailed, photorealistic, fantasy, epic cinematic shot, mountains, 8k ultra hd" --num_images 2 --batch_size 1 --num_steps 50 --non_interactive
```

### Command Line Arguments

The script accepts the following command line arguments:

- `--prompt`: The textual prompt to generate the image from. Default is `"castle surrounded by water and nature, village, volumetric lighting, detailed, photorealistic, fantasy, epic cinematic shot, mountains, 8k ultra hd"`.
- `--num_images`: The number of images to generate in total. Default is `2`.
- `--batch_size`: The number of images to generate per inference. Default is `1`.
- `--num_steps`: The number of steps in the diffusion process. Default is `50`.
- `--non_interactive`: A flag that, if present, runs the script without a GUI.

### Running with GUI

To run the script with the GUI, simply omit the `--non_interactive` argument:

```sh
python stable_diffusion.py 
```

In the GUI, you can provide the text prompt and click "Generate" to start the image generation process.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
