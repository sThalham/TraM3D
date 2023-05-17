import os, time
import torch
import torch.nn.functional as F
from lib.utils.metrics import AverageValueMeter
from lib.datasets.tless.inout import save_results
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms as transforms

invTrans = transforms.Compose(
                [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                 transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), ])


def test(query_data, template_data, model, epoch, logger, tb_logger, id_obj, save_prediction_path, is_master):
    print("Testing object {}".format(id_obj))
    start_time = time.time()

    query_size, query_dataloader = len(query_data), iter(query_data)
    template_size, template_dataloader = len(template_data), iter(template_data)
    timing_text = "Validation time for epoch {}: {:.02f} minutes"

    model.eval()
    with torch.no_grad():
        list_feature_template, list_synthetic_pose, list_mask, list_idx_template, list_inplane = [], [], [], [], []
        list_templates = []
        for i in tqdm(range(template_size)):
            # read all templates and its poses
            miniBatch = next(template_dataloader)

            template = miniBatch["template"].cuda()
            obj_pose = miniBatch["obj_pose"].cuda()
            idx_template = miniBatch["idx_template"].cuda()
            inplane = miniBatch["inplane"].cuda()
            mask = miniBatch["mask"].cuda().float()
            feature_template = model(template)

            #for bat in range(template.shape[0]):

            #    tmp_img = invTrans(template).cpu().numpy()[bat, ...]*255
            #    tmp_img = np.transpose(tmp_img, (1, 2, 0))

            #    mask_img = mask.cpu().numpy()[bat, ...]
            #    mask_img = np.transpose(mask_img, (1, 2, 0))
            #    mask_img = np.repeat(mask_img, axis=2, repeats=3)*255

            #    cv2.imwrite('/home/stefan/debug_viz/template_' + str(i) + '_' + str(bat) + '.png', tmp_img)
            #    cv2.imwrite('/home/stefan/debug_viz/mask_' + str(i) + '_' + str(bat) + '.png', mask_img)

            list_synthetic_pose.append(obj_pose)
            list_mask.append(mask)
            list_feature_template.append(feature_template)
            list_idx_template.append(idx_template)
            list_inplane.append(inplane)

            ##viz
            list_templates.append(invTrans(template).cpu().numpy())

        list_feature_template = torch.cat(list_feature_template, dim=0)
        list_synthetic_pose = torch.cat(list_synthetic_pose, dim=0)
        list_mask = torch.cat(list_mask, dim=0)
        list_idx_template = torch.cat(list_idx_template, dim=0)
        list_inplane = torch.cat(list_inplane, dim=0)

        ## viz
        list_templates = np.concatenate(list_templates, axis=0)

        names = ["obj_pose", "id_obj", "id_scene", "id_frame", "idx_frame", "idx_obj_in_scene", "visib_fract",
                 "gt_idx_template", "gt_inplane",
                 "pred_template_pose", "pred_idx_template", "pred_inplane"]
        results = {names[i]: [] for i in range(len(names))}
        for i in tqdm(range(query_size)):
            miniBatch = next(query_dataloader)

            query = miniBatch["query"].cuda()
            feature_query = model(query)

            #id = miniBatch["id_obj"].detach().numpy()
            #for bat in range(query.shape[0]):

            #    query_img = invTrans(query).cpu().numpy()[bat, ...]*255
            #    query_img = np.transpose(query_img, (1, 2, 0))

            #    cv2.imwrite('/home/stefan/debug_viz/query_' + str(id[bat]) + str(i) + '_' + str(bat) + '.png', query_img)

            # get best template
            matrix_sim = model.calculate_similarity_for_search(feature_query, list_feature_template, list_mask,
                                                               training=False)
            weight_sim, pred_index = matrix_sim.topk(k=1)
            pred_template_pose = list_synthetic_pose[pred_index.reshape(-1)]
            pred_idx_template = list_idx_template[pred_index.reshape(-1)]
            pred_inplane = list_inplane[pred_index.reshape(-1)]

            matched_templates = list_templates[pred_index.reshape(-1)]
            matched_masks = list_mask[pred_index.reshape(-1)]
            id = miniBatch["id_obj"].detach().numpy()
            for bat in range(query.shape[0]):

                query_img = invTrans(query).cpu().numpy()[bat, ...]*255
                query_img = np.transpose(query_img, (1, 2, 0))
                template_img = matched_templates[bat, ...] * 255
                template_img = np.transpose(template_img, (1, 2, 0))

                mask_img = matched_masks.cpu().numpy()[bat, ...]
                mask_img = np.transpose(mask_img, (1, 2, 0))
                mask_img = np.repeat(mask_img, axis=2, repeats=3)*255

                cv2.imwrite('/hdd/TraM3D/tless_samples_useen/query_' + str(id[bat]) + str(i) + '_' + str(bat) + '.png', query_img)
                cv2.imwrite('/hdd/TraM3D/tless_samples_useen/template_' + str(id[bat]) + str(i) + '_' + str(bat) + '.png', template_img)
                cv2.imwrite('/hdd/TraM3D/tless_samples_useen/mask_' + str(id[bat]) + str(i) + '_' + str(bat) + '.png', mask_img)

            for name in names[:-3]:
                data = miniBatch[name].detach().numpy()
                results[name].extend(data)
            results["pred_template_pose"].extend(pred_template_pose.cpu().detach().numpy())
            results["pred_idx_template"].extend(pred_idx_template.cpu().detach().numpy())
            results["pred_inplane"].extend(pred_inplane.cpu().detach().numpy())

    save_results(results, save_prediction_path)
    logger.info("Prediction of epoch {} of object {} is saved at {}".format(epoch, id_obj, save_prediction_path))
    if is_master:
        logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))