import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from utils.constants import col_name
from utils.tools import MyTool


def show_clusters(gps_locations, k, labels):
    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = gps_locations[labels == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i + 1}')
    plt.title("KMeans Clustering of POI Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def split_region(locations, num_region=50, method='KMeans'):
    kmeans = KMeans(n_clusters=num_region, random_state=0)
    kmeans.fit(locations)
    labels = kmeans.labels_
    return labels


class SpatioTemporalTrajectoryDataset:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        return self.trajectories[index]


def build_spatiotemporal_trajectories(df, min_len, max_len, data_name):
    trajectories = []
    tag = df['tag'].unique().tolist()[0]

    for traj_id in tqdm(set(df[col_name.trajectory_id].tolist()), desc=f"building {tag} spatiotemporal trajectories..."):
        traj_df = df[df[col_name.trajectory_id] == traj_id]
        
        if len(traj_df) < min_len or (data_name != 'NYC' and len(traj_df) > max_len):
            continue
        # #####################把最长的轨迹从后往前选###############################
        # if len(traj_df) < min_len:
        #     continue
        #
        # if data_name != 'NYC' and len(traj_df) > max_len:
        #     traj_df = traj_df.iloc[-max_len:]
        # #####################把最长的轨迹从后往前选###############################

        # Extract sequences
        poi_seq = traj_df[col_name.poi_id].to_list()
        cat_seq = traj_df[col_name.cat_id].to_list()
        geo_seq = traj_df[col_name.region_id].to_list()
        user_id = traj_df.user_id.tolist()[0]
        
        locations = traj_df[[col_name.latitude, col_name.longitude]].values
        timestamps = traj_df['timestamp'].values
        
        # Calculate time deltas (in hours)
        time_deltas = np.diff(timestamps, prepend=timestamps[0]) / 3600.0
        
        day_of_week = traj_df[col_name.local_time].dt.dayofweek.to_list()
        hour_of_day = traj_df[col_name.local_time].dt.hour.to_list()

        # Pack into a tuple
        trajectory_data = (
            poi_seq, cat_seq, geo_seq, user_id, 
            locations, time_deltas, day_of_week, hour_of_day
        )
        trajectories.append(trajectory_data)

    return trajectories


class POIDataset:
    def __init__(self, data_name, min_len=3, max_len=101):
        self.data_name = data_name
        self.min_len = min_len
        self.max_len = max_len
        self.valid_transition_time = 6 * 60 * 60

        root_path = MyTool.get_root_path()
        self.data_path = os.path.join(root_path, "dataset", data_name)
        self.df = pd.read_csv(os.path.join(self.data_path, f"{data_name}.csv"))
        self.df[col_name.local_time] = pd.to_datetime(self.df[col_name.local_time])
        self.df['timestamp'] = self.df[col_name.local_time].apply(lambda x: x.timestamp())

        # Pre-calculate region and category info
        self._precompute_cat_and_region()

        self.split_tags = self.df['tag'].unique().tolist()
        self.tables = {tag: self.df[self.df['tag'] == tag] for tag in self.split_tags}

        traj_dict = {}
        for tag in self.split_tags:
            sub_df = self.tables[tag]
            trajectories = build_spatiotemporal_trajectories(sub_df, self.min_len, self.max_len, self.data_name)
            traj_dict[tag] = SpatioTemporalTrajectoryDataset(trajectories)
        self.traj_dict = traj_dict

    def _precompute_cat_and_region(self, num_region=50):
        # This ensures that region and category IDs are consistent across the dataset
        train_df = self.df[self.df['tag'] == 'train'].copy()
        
        # Factorize category ID
        cat_map = {cat: i for i, cat in enumerate(train_df[col_name.cat_id].unique())}
        self.df[col_name.cat_id] = self.df[col_name.cat_id].map(cat_map).fillna(-1).astype(int)
        self.num_cat = len(cat_map)

        # Compute regions based on training set POIs
        poi_locations = train_df.drop_duplicates(subset=[col_name.poi_id]).set_index(col_name.poi_id)
        poi_locations = poi_locations.sort_index()
        locations = poi_locations[[col_name.latitude, col_name.longitude]].values
        region_labels = split_region(locations, num_region)
        
        poi_to_region = {poi_id: region for poi_id, region in zip(poi_locations.index, region_labels)}
        self.df[col_name.region_id] = self.df[col_name.poi_id].map(poi_to_region).fillna(-1).astype(int)
        self.num_region = num_region

    def set_nums(self, config):
        train_df = self.tables['train']
        config.num_poi = train_df[col_name.poi_id].nunique()
        config.num_user = train_df[col_name.user_id].nunique()
        config.num_cat = self.num_cat
        config.num_region = self.num_region
        return config

    def get_user_poi_edges(self):
        df = self.tables['train']
        edges = df[[col_name.user_id, col_name.poi_id]].values
        edges = torch.LongTensor(edges)
        return edges

    def get_poi_poi_edges(self):
        df = self.tables['train']
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        user_group = df.groupby([col_name.user_id])
        edges = []
        for user_id, user_df in user_group:
            time_list = user_df['timestamp'].tolist()
            poi_list = user_df[col_name.poi_id].tolist()
            for i in range(len(time_list)-1):
                time_diff = time_list[i+1] - time_list[i]
                if time_diff <= self.valid_transition_time:
                    valid_edge = (poi_list[i], poi_list[i+1])
                    edges.append(valid_edge)
        edges = torch.LongTensor(edges)
        return edges

    def get_region_information(self, num_region=None, method='KMeans'):
        df = self.tables['train']
        df = df.drop_duplicates(subset=[col_name.poi_id], keep='first')
        df = df.sort_values(by=col_name.poi_id, ascending=True).reset_index(drop=True)
        region_ids = df[col_name.region_id].values
        region_ids = torch.LongTensor(region_ids)
        pad_val = torch.LongTensor([self.num_region if num_region is None else num_region])
        region_ids = torch.cat((region_ids, pad_val), dim=0)
        return region_ids

    def get_category_information(self):
        df = self.tables['train']
        df = df.drop_duplicates(subset=[col_name.poi_id], keep='first')
        df = df.sort_values(by=col_name.poi_id, ascending=True).reset_index(drop=True)
        cat_ids = df[col_name.cat_id].values
        cat_ids = torch.LongTensor(cat_ids)
        pad_val = torch.LongTensor([self.num_cat])
        cat_ids = torch.cat((cat_ids, pad_val), dim=0)
        return cat_ids
