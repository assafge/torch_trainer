import argparse

from TorchTrainer import TorchTrainer
from time import time
from threading import Thread


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    parser.add_argument('-v', '--debug_prints', help='path to output directory')
    new = parser.add_argument_group('new model')
    new.add_argument('-m', '--model_cfg', help='path to model cfg file')
    new.add_argument('-o', '--optimizer_cfg', help='path to optimizer cfg file')
    new.add_argument('-d', '--dataset_cfg', help='path to dataset cfg file')
    new.add_argument('-w', '--out_path', help='path to output directory')
    new.add_argument('-e', '--exp_name', default='')
    pre_train = parser.add_argument_group('pre-trained')
    pre_train.add_argument('-r', '--model_path', help='path to pre-trained model')
    pre_train.add_argument('-i', '--inference_img', default=None)
    pre_train.add_argument('-gt', '--GT', default=None)

    args = parser.parse_args()
    return args


def main():
    t0 = time()
    args = get_args()
    if args.model_path:
        trainer = TorchTrainer.warm_startup(root=args.model_path, gpu_index=args.gpu_index)
    else:
        trainer = TorchTrainer.new_train(out_path=args.out_path, model_cfg=args.model_cfg, optimizer_cfg=args.optimizer_cfg,
                                         dataset_cfg=args.dataset_cfg, gpu_index=args.gpu_index, exp_name=args.exp_name)
    # if not args.inference_img:
    #     print('training is about to start, in order to stop => type "stop" + ↵')
    #     process = Thread(target=trainer.train)
    #     process.start()
    #     while process.is_alive() and trainer.running:
    #         if 'stop' in input():
    #             trainer.running = False
    #             print('stopping...')
    #     process.join()
    #     print()
    if not args.inference_img:
        trainer.train()

    else:
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        segmentation = True
        trainer.model.eval()
        with torch.no_grad():
            if 'tif' in args.inference_img:
                im_raw = cv2.imread(args.inference_img, cv2.IMREAD_UNCHANGED).astype(np.float64)
                max16bit = np.iinfo(np.uint16).max
                im_raw = np.clip(im_raw, 0, max16bit)
                im_raw = ((im_raw / (2**16)) * 255).astype(np.uint8)
                color = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB)
            else:
                color = cv2.cvtColor(cv2.imread(args.inference_img), cv2.COLOR_BGR2RGB)
            width, height = color.shape[:2]
            # pil_im = Image.fromarray(im)
            # color = color.transpose(1, 2, 0)
            # inputs = trainer.dataset.transform(color/255).float().to(trainer.device)
            inputs = torch.from_numpy(np.transpose(color/255, (2, 0, 1))).float().to(trainer.device)
            inputs = inputs.unsqueeze(0)

            out = trainer.model(inputs)
            if segmentation:
                print(out.shape)
                outs = out.argmax(dim=1).squeeze()
                print(out.shape)
                out_im = outs.cpu().numpy()
            if not segmentation:
                out_np = out.cpu().numpy()
                out_im = np.squeeze(out_np, axis=0)
                out_im = out_im.transpose(1, 2, 0)
                out_im = np.clip(out_im, 0, 1)
                out_im = (out_im * 255).astype(np.uint8)

        print('calc time = %.2f' % (time() - t0))

        if args.GT:
            gt = trainer.dataset.depth_read(args.GT)
            rows = 3
        else:
            rows = 2
        fig, axes = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=True)
        axes[0].imshow(color), axes[0].set_title('in')
        axes[1].imshow(out_im), axes[1].set_title('out')
        if args.GT:
            axes[2].imshow(gt), axes[2].set_title('GT')
        plt.show()


if __name__ == '__main__':
    main()
