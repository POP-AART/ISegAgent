import torchio as tio
import numpy as np
import torch
from torchio import ZNormalization
import torch.nn.functional as F
import surface_distance
from surface_distance import metrics


def compute_iou(pred_mask, gt_semantic_seg):
    if gt_semantic_seg.sum() == 0:
        return -1.0
        # return -np.inf  # TODO: need to be discussed
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    if out_mask.sum() == 0:
        return 0.0
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou


def compute_nsd(mask_gt, mask_pred, spacing=(1, 1, 1)):

    ssd_score = surface_distance.compute_surface_distances(mask_pred==1, mask_gt==1, spacing_mm=[x.cpu().numpy() for x in spacing])
    nsd_score = metrics.compute_surface_dice_at_tolerance(ssd_score, tolerance_mm=5)

    return nsd_score


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def generate_metrics_of_each_slice(img3D, gt3D):
    """
    img3D: (1, 1, W, H, D)
    gt3D: (1, 1, W, H, D)
    compare each slice
    """
    num_slices = img3D.shape[-1]
    
    iou_list = []
    dice_list = []
    nsd_list = []

    # TODO: high computational cost, add a new parameter for specifying the number of slices
    for i in range(num_slices):
        img_slice = img3D[0, 0, :, :, i].detach().cpu().numpy()
        gt_slice = gt3D[0, 0, :, :, i].detach().cpu().numpy()

        # TODO: clean the NaN values?
        # iou_list.append(compute_iou(img_slice, gt_slice) if np.sum(gt_slice) > 0 else 0.0)
        iou_list.append(compute_iou(img_slice, gt_slice))
        # dice_list.append(compute_dice(img_slice, gt_slice))
        # nsd_list.append(compute_nsd(img_slice, gt_slice))
    
    iou_list = np.array(iou_list)
    dice_list = np.array(dice_list)
    nsd_list = np.array(nsd_list)
    
    return iou_list, dice_list, nsd_list


def finetune_model_predict3D(img3D, gt3D, spacing, sam_model_tune, 
                             device='cuda', click_method='random', 
                             num_clicks=10, prev_masks=None, 
                             crop_size: int=128,
                             click_methods: dict=None,
                             slices: list=None,
                             norm_transform=tio.ZNormalization(masking_method=lambda x: x > 0)):
    
    num_clicks = len(slices) if slices is not None else num_clicks

    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    nsd_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4, crop_size//4, crop_size//4))  # [1,1,32,32,32]

    # TODO: not so sure about this "all_P" part
    # Assuming gt3D has a shape of [depth, height, width]
    depth, height, width = gt3D.shape[-3:]
    # Initialize all_P to store all interaction predictions, adding one channel for the background, 
    # TODO: need to be reported, what I found is there is only one object when calculating the metrics
    all_P = torch.zeros((depth, 2, height, width), device=device)  # Second dimension for background and one foreground
    


    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device), slice=slices[num_click])

            if len(batch_points) == 0:
                continue

            points_co = torch.cat(batch_points, dim=0).to(device)  
            points_la = torch.cat(batch_labels, dim=0).to(device)  

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)


            # Update all_P for each depth slice
            # TODO: should be replaced after the for loop?, since the final result is only needed
            for d in range(depth):
                all_P[d, 1, :, :] = torch.from_numpy(medsam_seg[d, :, :]).to(device)  # foreground mask
                all_P[d, 0, :, :] = torch.from_numpy(1 - medsam_seg[d, :, :]).to(device)  # background as inverse


            # metrics
            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
            nsd_list.append(round(compute_nsd(medsam_seg, gt3D[0][0].detach().cpu().numpy(), spacing), 4))

    # TODO: add a way to return the all_P for each depth slice!!!!, like the pred_list
    # other wise I should not put the for loop in another for loop
    return pred_list, click_points, click_labels, iou_list, dice_list, nsd_list, all_P
