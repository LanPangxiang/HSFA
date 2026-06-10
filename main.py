import os
import numpy as np
import logging
import random
import torch
from torch.utils.data import DataLoader
from args_get import parameter_parser
from models.model import HSFA as MODEL
from dataset import POIDataset
from runner import Runner
from utils.tools import MyTool


if __name__ == '__main__':
    args = parameter_parser()
    args.save = True
    args.log = True

    model_name = MODEL.__name__
    args.model_name = model_name
    save_path = MyTool.set_save_path(args.data_name, args.model_name)

    MyTool.set_logging(save_path, args.model_name)

    logging.info(f"seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        args.device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset_path = 'dataset'
    data = POIDataset(args.data_name, args.min_len, args.max_len)


    train_loader = DataLoader(data.traj_dict['train'], batch_size=args.batch, shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    val_loader = DataLoader(data.traj_dict['val'], batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    test_loader = DataLoader(data.traj_dict['test'], batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)

    args = data.set_nums(args)

    # process auxiliary POI information
    logging.info(f"start splitting regions...")
    region_labels = data.get_region_information(num_region=args.num_region)
    logging.info(f"splitting regions done...")

    cat_labels = data.get_category_information()
    args.num_cat = cat_labels.max().item()

    # args.k_list = [1, 2, 5, 10]
    args.k_list = [1, 5, 10]
    args.required_metrics = ['Acc', 'NDCG', 'MRR']
    args.model_params = MyTool.load_model_params(args.data_name)
    if args.pre_hsfa_num_bands is not None:
        args.model_params['pre_hsfa_num_bands'] = int(args.pre_hsfa_num_bands)
    if args.pre_hsfa_low_band_count is not None:
        args.model_params['pre_hsfa_low_band_count'] = int(args.pre_hsfa_low_band_count)
    logging.info(
        "Frequency settings override: K=%s, c=%s",
        args.model_params.get('pre_hsfa_num_bands', None),
        args.model_params.get('pre_hsfa_low_band_count', None),
    )

    model = MODEL(args)

    run = Runner(args, model, region_labels, cat_labels)

    check_patience = 0
    best_model_path = os.path.join(save_path, 'best_model.pth')
    latest_model_path = os.path.join(save_path, 'latest_model.pth')
    code_zip_path = os.path.join(save_path, 'code.zip')
    
    if args.save and args.log:
        import zipfile
        import pathlib
        from utils.tools import zipdir
        zipf = zipfile.ZipFile(code_zip_path, 'w', zipfile.ZIP_DEFLATED)
        zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
        zipf.close()
        logging.info(f"Code saved as zip at {code_zip_path}")

    
    for epoch in range(args.epoch):
        # run.train(train_loader)
        ###############print loss############################################
        train_loss_array = run.train(train_loader)
        epoch_current_mean_batch_loss = float(train_loss_array.mean()) if train_loss_array.size > 0 else 0.0
        logging.info(
            f"epoch {epoch + 1:>03d} train current_mean_batch_loss: {epoch_current_mean_batch_loss:.6f}"
        )
        print(
            f"epoch {epoch + 1:>03d} train current_mean_batch_loss: {epoch_current_mean_batch_loss:.6f}"
        )
        ###############print loss############################################

        if args.save:
            torch.save(model.state_dict(), latest_model_path)

        metric_dict = run.valid(val_loader)
        logging.info(f"epoch {epoch + 1:>03d} valid:")
        MyTool.print_metrics(metric_dict)

        # ==========================================
        # (Modified: Add Test)
        # ==========================================
        test_metric_dict = run.test(test_loader)
        logging.info(f"epoch {epoch + 1:>03d} test:")
        MyTool.print_metrics(test_metric_dict)
        # ==========================================

        # valid result check
        check_result = run.check_best_result(metric_dict)
        if check_result and args.save:
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"update valid score and save best model at epoch {run.best_epoch}")
            check_patience = 0  # reset
        else:
            check_patience += 1
            if check_patience > args.patience:
                break

        run.current_epoch += 1

    if args.save and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"Loaded best model weights from epoch {run.best_epoch} for final test")
    
    final_metric_dict = run.test(test_loader)
    logging.info(f"Final test result with best model from epoch {run.best_epoch}")

    MyTool.print_metrics(final_metric_dict)
    best_result = run.best_metric
    logging.info(f"Final validation result at epoch {run.best_epoch}")
    MyTool.print_metrics(best_result)
