# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import warnings
import PySimpleGUI as sg
import onnxruntime as ort
import time
from diffusers import OnnxStableDiffusionPipeline
from packaging import version


def run_inference_loop(
    pipeline, prompt, num_images, batch_size, num_steps, image_callback=None, step_callback=None
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_steps + step)

    while images_saved < num_images:
        # Run a batch through the pipeline
        start_time = time.perf_counter()
        result = pipeline(
            [prompt] * batch_size,
            num_inference_steps=num_steps,
            callback=update_steps if step_callback else None,
        )
        batch_time = time.perf_counter() - start_time

        # Save outputs that passed the safety checker
        passed_safety_checker = 0
        for image_index in range(batch_size):
            if not result.nsfw_content_detected[image_index]:
                passed_safety_checker += 1
                if images_saved < num_images:
                    output_path = f"result_{images_saved}.png"
                    result.images[image_index].save(output_path)
                    if image_callback:
                        image_callback(images_saved, output_path)
                    images_saved += 1

        print(f"Batch completed in {batch_time:.2f} sec; {passed_safety_checker}/{batch_size} passed the safety checker.")


def run_inference_interactive(pipeline, prompt, num_images, batch_size, num_steps):
    sg.theme("SystemDefault")

    if num_images > 9:
        print("WARNING: interactive UI only supports displaying up to 9 images")
        num_images = 9

    image_size = (512, 512)
    image_rows = 1 + (num_images - 1) // 3
    image_cols = 2 if num_images == 4 else min(num_images, 3)
    image_index = 0
    min_batches_required = 1 + (num_images - 1) // batch_size

    # Create GUI
    layout = []
    for _ in range(image_rows):
        ui_row = []
        for _ in range(image_cols):
            ui_row.append(sg.Image(key=f"sd_output{image_index}", size=image_size, background_color="black"))
            image_index += 1
        layout.append(ui_row)
    layout.append([sg.ProgressBar(num_steps * min_batches_required, key="sb_progress", expand_x=True, size=(8, 8))])
    layout.append([sg.InputText(key="sd_prompt", default_text=prompt, expand_x=True), sg.Button("Generate")])
    window = sg.Window("Stable Diffusion with ONNX Runtime & DirectML", layout)

    # Run GUI event loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == "Generate":
            def update_progress_bar(total_steps_completed):
                window["sb_progress"].update_bar(total_steps_completed)

            def image_completed(index, path):
                window[f"sd_output{index}"].update(filename=path)

            def generate_image():
                run_inference_loop(
                    pipeline,
                    values["sd_prompt"],
                    num_images,
                    batch_size,
                    num_steps,
                    image_completed,
                    update_progress_bar,
                )

            window["Generate"].update(disabled=True)
            window.start_thread(generate_image, "image_generation_done")
        elif event == "image_generation_done":
            window["Generate"].update(disabled=False)

def run_inference(prompt, num_images, batch_size, num_steps, non_interactive):
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
   
    # These lines are optional. Setting free dimension overrides makes the input shapes static, 
    # which allows further optimizations in ONNX Runtime and the DirectML execution provider.
    sess_options.add_free_dimension_override_by_name("unet_sample_batch", 2 * batch_size)
    sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
    sess_options.add_free_dimension_override_by_name("unet_sample_height", 64)
    sess_options.add_free_dimension_override_by_name("unet_sample_width", 64)
    sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
    sess_options.add_free_dimension_override_by_name("unet_hidden_batch", 2 * batch_size)
    sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

    with warnings.catch_warnings():
        # Ignore warning about deprecated CLIPFeatureExtractor in diffusers library
        warnings.simplefilter("ignore")

        print("Loading pipeline...", end=" ", flush=True)
        pipeline = OnnxStableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5",  # TODO: replace with path to downloaded models
            provider="DmlExecutionProvider",
            sess_options=sess_options
        )
        print("done!")

    if non_interactive:
        run_inference_loop(pipeline, prompt, num_images, batch_size, num_steps)
    else:
        run_inference_interactive(pipeline, prompt, num_images, batch_size, num_steps)


if __name__ == "__main__":
    if version.parse(ort.__version__) < version.parse("1.15.0"):
        print("This script requires onnxruntime-directml 1.15.0 or newer. Please 'pip install -r requirements.txt'.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="castle surrounded by water and nature, village, volumetric lighting, detailed, photorealistic, fantasy, epic cinematic shot, mountains, 8k ultra hd", type=str)
    parser.add_argument("--num_images", default=2, type=int, help="Number of images to generate in total")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per inference")
    parser.add_argument("--num_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--non_interactive", action="store_true", help="Run without a GUI")
    args = parser.parse_args()

    ort.set_default_logger_severity(3)

    run_inference(
        args.prompt,
        args.num_images,
        args.batch_size,
        args.num_steps,
        args.non_interactive,
    )
