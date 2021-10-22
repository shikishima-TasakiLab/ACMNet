import torch.nn
import numpy as np
import os
from PIL import Image
import cv2
from options.test_options import TestOptions
from data import create_dataloader, create_h5_dataloader
from models import create_model
from tqdm import tqdm
import util

DEPTH_MAX = 80.0

def ToFalseColors(depth, mask=None):
    color_map = np.array([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
                     [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]],dtype=np.float32)
    sum = 0.0
    for i in range(8):
        sum += color_map[i][3]

    weights = np.zeros([8], dtype=np.float32)
    cumsum = np.zeros([8], dtype=np.float32)
    for i in range(7):
        weights[i] = sum / color_map[i][3]
        cumsum[i+1] = cumsum[i] + color_map[i][3] / sum
    H, W = depth.shape
    image = np.ones([H, W, 3], dtype=np.uint8)
    max_depth = np.max(depth)
    for i in range(int(H)):
        for j in range(int(W)):
            val = np.min([depth[i, j]/max_depth, 1.0])
            for k in range(7):
                if val<cumsum[k+1]:
                    break
            w = 1.0 - (val-cumsum[k]) * weights[k]
            r = int((w*color_map[k][0]+(1.0-w)*color_map[k+1][0]) * 255.0)
            g = int((w*color_map[k][1]+(1.0-w)*color_map[k+1][1]) * 255.0)
            b = int((w*color_map[k][2]+(1.0-w)*color_map[k+1][2]) * 255.0)
            image[i, j, 0] = r
            image[i, j, 1] = g
            image[i, j, 2] = b
    if mask is not None:
        image[:,:,0] = image[:,:,0] * mask + 255 * (1-mask)
        image[:,:,1] = image[:,:,1] * mask + 255 * (1-mask)
        image[:,:,2] = image[:,:,2] * mask + 255 * (1-mask)
    return image.astype(np.uint8)

def main():
    np.random.seed(seed=0)
    torch.random.manual_seed(seed=0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=0)

    opt = TestOptions().parse()
    # data_loader = create_dataloader(opt)
    data_loader = create_h5_dataloader(opt)
    num_samples = len(data_loader)
    print('#test images = %d' % num_samples)

    model = create_model(opt)
    model.setup(opt)
    total_steps = 0
    model.eval()

    if opt.save:
        if opt.suffix != '':
            opt.suffix = '_' + opt.suffix
        dirs = os.path.join('results', opt.model+opt.suffix)
        os.makedirs(dirs, exist_ok=True)

    # mae    = np.zeros(num_samples, np.float32)
    # rmse   = np.zeros(num_samples, np.float32)
    # imae   = np.zeros(num_samples, np.float32)
    # irmse  = np.zeros(num_samples, np.float32)
    # a1     = np.zeros(num_samples, np.float32)
    # a2     = np.zeros(num_samples, np.float32)
    # a3     = np.zeros(num_samples, np.float32)
    # a4     = np.zeros(num_samples, np.float32)

    sum_ae = 0.0
    sum_se = 0.0
    sum_iae = 0.0
    sum_ise = 0.0
    sum_ape = 0.0
    cnt = 0.0

    for ind, data in tqdm(enumerate(data_loader)):
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()

        gt_depth = np.squeeze(data['gt'].data.cpu().numpy())
        pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
        s_depth = np.squeeze(data['sparse'].data.cpu().numpy())

        pred_depth[pred_depth<=0.0] = 0.0
        pred_depth[pred_depth>1.0] = 1.0
        mask = (gt_depth > 0.0) & (gt_depth<=1.0)

        gt = gt_depth[mask]
        pred = pred_depth[mask]

        sum_ae += np.sum(np.abs(pred - gt))
        sum_se += np.sum((pred - gt) ** 2.0)
        sum_iae += np.sum(np.abs(1.0 / pred - 1e3 / gt))
        sum_ise += np.sum((1.0 / pred - 1.0 / gt) ** 2.0)
        sum_ape += np.sum(np.abs(pred - gt) / gt)
        cnt += np.sum(mask)

        # mae[ind], rmse[ind], imae[ind], irmse[ind], a1[ind], \
        #     a2[ind], a3[ind], a4[ind] = util.compute_errors(gt_depth[mask], pred_depth[mask])

        if opt.save:
            # gt_depth = gt_depth[96:,:]
            # s_depth = s_depth[96:,:]
            # pred_depth = pred_depth[96:,:]
            # gt_image = ToFalseColors(gt_depth, mask=(gt_depth>0).astype(np.float32))
            # pred_image = ToFalseColors(pred_depth)
            # s_image = ToFalseColors(s_depth, mask=(s_depth>0).astype(np.float32))

            img = visuals['img'].data.cpu().numpy().squeeze().transpose(1, 2, 0) * 255.0

            gt_image = cv2.applyColorMap((gt_depth * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            gt_image[gt_depth > 1.0] = [0, 0, 0]

            pred_image = cv2.applyColorMap((pred_depth * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            pred_image[pred_depth > 1.0] = [0, 0, 0]

            s_image = cv2.applyColorMap((s_depth * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            s_image[s_depth > 1.0] = [0, 0, 0]

            cv2.imwrite(os.path.join(dirs, f'{ind:05d}_gt.png'), gt_image)
            cv2.imwrite(os.path.join(dirs, f'{ind:05d}_pred.png'), pred_image)
            cv2.imwrite(os.path.join(dirs, f'{ind:05d}_s.png'), s_image)
            cv2.imwrite(os.path.join(dirs, f'{ind:05d}_img.png'), img.astype(np.uint8))

            # gt_img = Image.fromarray(gt_image, 'RGB')
            # pred_img = Image.fromarray(pred_image, 'RGB')
            # s_img = Image.fromarray(s_image, 'RGB')
            # gt_img.save('%s/%05d_gt.png'%(dirs, ind))
            # pred_img.save('%s/%05d_pred.png'%(dirs, ind))
            # s_img.save('%s/%05d_sparse.png'%(dirs, ind))
            # im = util.tensor2im(visuals['img'])
            # util.save_image(im, '%s/%05d_img.png'%(dirs, ind), 'RGB')

    mae = sum_ae / cnt * DEPTH_MAX
    rmse = np.sqrt(sum_se / cnt) * DEPTH_MAX
    imae = sum_iae / cnt * DEPTH_MAX
    irmse = np.sqrt(sum_ise / cnt) * DEPTH_MAX
    mape = sum_ape / cnt

    print(f'{"MAE":15s}:{"RMSE":15s}:{"iMAE":15s}:{"iRMSE":15s}:{"MAPE":15s}')
    print(f'{mae:15.6f}:{rmse:15.6f}:{imae:15.6f}:{irmse:15.6f}:{mape:15.6f}')

if __name__ == '__main__':
    main()
