
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_batch_metrics
from utils.tools import MyTool


class Runner:
    def __init__(self, args, model, region_labels, cat_labels):
        self.args = args
        self.model_name = args.model_name
        self.device = args.device

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.poi_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_poi)
        self.cat_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_cat)
        self.geo_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_region)

        self.region_labels = region_labels.to(self.device)
        self.cat_labels = cat_labels.to(self.device)


        self.PAD_VALUE = args.num_poi
        self.CAT_PAD = args.num_cat
        self.GEO_PAD = args.num_region

        self.num_epoch = args.epoch
        self.k_list = args.k_list
        self.required_metrics = args.required_metrics

        self.current_epoch = 1
        self.best_score = 0.0
        self.best_epoch = 0
        self.best_metric = None

    def check_best_result(self, metric_dict, target_metric="NDCG"):
        check_scores = metric_dict[target_metric]
        score = sum(check_scores.values())
        if (self.best_metric is None) or (self.best_score < score):
            self.best_score = score
            self.best_metric = metric_dict
            self.best_epoch = self.current_epoch
            return True
        return False

    def process_batch(self, batch):
        """
        TS-NPR: target time aligned with labels.
        - history inputs: dow[:-1], hour[:-1]
        - labels        : poi[1:]
        - target time   : dow[1:], hour[1:]  (same length as labels)
        """
        poi_inputs, poi_labels = [], []
        user_list = []
        loc_inputs, dt_inputs, dow_inputs, hour_inputs = [], [], [], []
        dow_targets, hour_targets = [], []

        for sample in batch:
            poi_seq, cat_seq, geo_seq, user_id, locations, time_deltas, dow, hour = sample

            # History inputs (length L-1)
            poi_input = poi_seq[:-1]
            loc_input = locations[:-1]
            dt_input = time_deltas[:-1]
            dow_input = dow[:-1]
            hour_input = hour[:-1]

            # Supervision labels (length L-1)
            poi_label = poi_seq[1:]

            # Target time aligned with labels (length L-1)
            dow_target = dow[1:]
            hour_target = hour[1:]

            poi_inputs.append(torch.as_tensor(poi_input, dtype=torch.long))
            poi_labels.append(torch.as_tensor(poi_label, dtype=torch.long))
            user_list.append(user_id)

            loc_inputs.append(torch.as_tensor(loc_input, dtype=torch.float32))
            dt_inputs.append(torch.as_tensor(dt_input, dtype=torch.float32))
            dow_inputs.append(torch.as_tensor(dow_input, dtype=torch.long))
            hour_inputs.append(torch.as_tensor(hour_input, dtype=torch.long))

            dow_targets.append(torch.as_tensor(dow_target, dtype=torch.long))
            hour_targets.append(torch.as_tensor(hour_target, dtype=torch.long))

        # Pad sequences
        poi_inputs = pad_sequence(poi_inputs, batch_first=True, padding_value=self.PAD_VALUE)
        poi_labels = pad_sequence(poi_labels, batch_first=True, padding_value=self.PAD_VALUE)

        cat_inputs = self.cat_labels[poi_inputs]
        geo_inputs = self.region_labels[poi_inputs]
        cat_labels = self.cat_labels[poi_labels]
        geo_labels = self.region_labels[poi_labels]

        loc_inputs = pad_sequence(loc_inputs, batch_first=True, padding_value=0.0)
        dt_inputs = pad_sequence(dt_inputs, batch_first=True, padding_value=0.0).unsqueeze(-1)
        dow_inputs = pad_sequence(dow_inputs, batch_first=True, padding_value=0)
        hour_inputs = pad_sequence(hour_inputs, batch_first=True, padding_value=0)

        # Target time tensors (same shape as poi_labels)
        dow_targets = pad_sequence(dow_targets, batch_first=True, padding_value=0)
        hour_targets = pad_sequence(hour_targets, batch_first=True, padding_value=0)

        user_list = torch.as_tensor(user_list, dtype=torch.long)

        all_input = [
            poi_inputs.to(self.device),
            cat_inputs.to(self.device),
            geo_inputs.to(self.device),
            user_list.to(self.device),
            loc_inputs.to(self.device),
            dt_inputs.to(self.device),
            dow_inputs.to(self.device),
            hour_inputs.to(self.device),
            dow_targets.to(self.device),  # NEW
            hour_targets.to(self.device),  # NEW
        ]
        all_label = [
            poi_labels.to(self.device),
            cat_labels.to(self.device),
            geo_labels.to(self.device),
        ]
        return all_input, all_label


    def train(self, train_loader):
        self.model.train()
        epoch_loss = []

        with tqdm(total=len(train_loader), desc=f"train process [{self.current_epoch:>03d}/{self.num_epoch:>03d}]") as progress_bar:
            for batch in train_loader:
                inputs, labels = self.process_batch(batch)
                pred_poi, pred_cat, pred_geo = self.model(inputs)
                label_pois, label_cats, label_geos = labels[0], labels[1], labels[2]

                poi_loss = self.poi_loss(pred_poi.transpose(2, 1), label_pois)
                cat_loss = self.cat_loss(pred_cat.transpose(2, 1), label_cats)
                geo_loss = self.geo_loss(pred_geo.transpose(2, 1), label_geos)
                loss = poi_loss + cat_loss + geo_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix(current_mean_batch_loss=loss.item())

        return np.array(epoch_loss)

    def valid(self, valid_loader):
        return self.test(valid_loader)

    def test(self, test_loader):
        self.model.eval()
        total_metrics = MyTool.init_metric_dict(self.required_metrics, self.k_list)

        with tqdm(total=len(test_loader), desc=f"test process [{self.current_epoch:>03d}/{self.num_epoch:>03d}]") as progress_bar:
            for batch in test_loader:
                inputs, labels = self.process_batch(batch)
                ## Updates during TTT testing are explicit/functional updates within the forward layer; autograd is not required.
                with torch.no_grad():
                    pred_poi, pred_cat, pred_geo = self.model(inputs)

                label_pois, _, _ = labels
                total_metrics = calculate_batch_metrics(total_metrics, pred_poi, label_pois, self.PAD_VALUE, self.k_list)
                progress_bar.update(1)

        return MyTool.get_average_metric(total_metrics)



