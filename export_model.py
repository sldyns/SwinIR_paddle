import argparse
import os

import paddle
from models.network_swinir import SwinIR as net

parser = argparse.ArgumentParser(description="NAFNet_test")
parser.add_argument("--save-inference-dir", type=str, default="./test_tipc/output/swinir_denoising", help='path of model for export')
parser.add_argument("--model_path", type=str, default="model_best.pdparams", help='path of model checkpoint')

opt = parser.parse_args()

def main(opt):

    model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = paddle.load(opt.model_path)
    model.set_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model)

    print('Loaded trained params of model successfully.')

    shape = [1, 3, 136, 136]

    new_model = model

    new_model.eval()
    new_net = paddle.jit.to_static(
        new_model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(opt.save_inference_dir, 'model')
    paddle.jit.save(new_net, save_path)


    print(f'Model is saved in {opt.save_inference_dir}.')


if __name__ == '__main__':
    main(opt)


