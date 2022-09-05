# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import cv2
import glob
from collections import OrderedDict

import paddle
from paddle import inference

from utils import util_calculate_psnr_ssim as util

def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="SwinIR DENOISING", add_help=add_help)

    parser.add_argument(
        '--folder_gt',
        type=str,
        default="./test_tipc/data/CBSD68/val_mini",
        help='path to clean data')

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=True, type=str2bool, help="use_gpu")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")

    args = parser.parse_args()
    return args


class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "model.pdmodel"),
            os.path.join(args.model_dir, "model.pdiparams"))


    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(10000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, path):
        """preprocess
        Preprocess to the input.
        Args:
            path: gt img path
        Returns: Input data after preprocess.
        """
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, 15. / 255., img_gt.shape)
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = np.expand_dims(img_lq, 0)

        return img_lq, img_gt

    def postprocess(self, output, img_gt):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            output: Inference denoised image.
            img_gt: Clean image
        Returns: Output denoised image.
        """
        output= np.clip(output, 0., 1.)
        output = np.squeeze(output)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)

        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = np.squeeze(img_gt)
        border = 0
        psnr = util.calculate_psnr(output, img_gt, crop_border=border)
        ssim = util.calculate_ssim(output, img_gt, crop_border=border)

        return psnr, ssim

    def run(self, img_lq, window_size = 8, tile = 136, tile_overlap = 16):
        """run
        Inference process using inference engine.
        Args:
            img_lq: Input data after preprocess.
        Returns: Inference engine output
        """
        _, _, h_old, w_old = img_lq.shape
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = np.concatenate((img_lq, np.flip(img_lq, 2)), axis=2)[:, :, :h_old + h_pad, :]
        img_lq =np.concatenate((img_lq, np.flip(img_lq, 3)), 3)[:, :, :, :w_old + w_pad]
        img_lq = img_lq.astype(np.float32)

        b,c, h, w = img_lq.shape
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        sf = 1 # 尺度
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = np.zeros([b, c, h*sf, w*sf], dtype=np.float32)
        W = np.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                self.input_tensor.copy_from_cpu(in_patch)
                self.predictor.run()
                out_patch = self.output_tensor.copy_to_cpu()
                out_patch_mask = np.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask

        output = np.true_divide(E, W)
        output = output[..., :h_old * sf, :w_old * sf]

        return output


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="SwinIR_denoising",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids=0 if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    psnr, ssim = 0, 0

    # dataset preprocess
    image_paths = sorted(glob.glob(os.path.join(args.folder_gt, "*.png")))
    for image_path in image_paths:

        img_lq, img_gt = inference_engine.preprocess(image_path)

        if args.benchmark:
            autolog.times.stamp()

        output = inference_engine.run(img_lq)

        if args.benchmark:
            autolog.times.stamp()

         # postprocess
        psnr, ssim = inference_engine.postprocess(output, img_gt)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        if args.benchmark:
            autolog.times.stamp()
            autolog.times.end(stamp=True)
            autolog.report()

        print(f"image_name: {image_path}, psnr: {psnr}, ssim:{ssim}")

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print('\n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(ave_psnr, ave_ssim))


if __name__ == "__main__":
    args = get_args()
    infer_main(args)