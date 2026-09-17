import os
import pandas as pd

# SECTION 1: DATA LOADING

# Load GTFS files
stops = pd.read_csv("stops.txt")
stop_times = pd.read_csv("stop_times.txt")

#print(stops.head())
#print(stop_times.head())

# merge stop_times to stops so we can get the parent_station for each stop_id in stop_times.
stop_times = stop_times.merge(
    stops[["stop_id", "parent_station"]],
    on="stop_id",
    how="left"
)

stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

stop_times["next_station"] = (
    stop_times.groupby("trip_id")["parent_station"].shift(-1)
)

#print(stop_times[["stop_id", "parent_station", "next_station"]].head(10))
# 1. Shift arrival_time to pair departure time of stop A with arrival time of stop B
stop_times["next_arrival_time"] = stop_times.groupby("trip_id")["arrival_time"].shift(-1)


#print(stop_times[["stop_id", "parent_station", "next_station","arrival_time","next_arrival_time"]].head(10))

# 2. Define a function to convert GTFS 'HH:MM:SS' time strings to total minutes
def gtfs_time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    # Split hours, minutes, seconds and convert to float minutes
    h, m, s = map(int, str(time_str).split(':'))
    return h * 60 + m + s / 60.0

# 3. Convert departure and next arrival times from text to numeric minutes
dep_mins = stop_times['departure_time'].apply(gtfs_time_to_minutes)
arr_mins = stop_times['next_arrival_time'].apply(gtfs_time_to_minutes)

# 4. Subtract departure from next arrival to get exact track travel time
stop_times['hop_time_mins'] = arr_mins - dep_mins

# 7. Extract parent stations from stops.txt for spatial coordinate lookups
parent_stations_df = stops[stops['parent_station'].isna() | (stops['parent_station'] == "")].copy()

valid_hops = stop_times[
    (stop_times["next_station"].notna()) & 
    (stop_times["parent_station"] != stop_times["next_station"]) & 
    (stop_times["hop_time_mins"] > 0) &
    (stop_times["hop_time_mins"] < 30)
]

subway_edges_df = valid_hops.groupby(["parent_station", "next_station"])["hop_time_mins"].mean().reset_index()
subway_edges_df.columns = ["from_station_id", "to_station_id", "travel_time_mins"]

parent_stations_df = parent_stations_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
parent_stations_df["stop_lat"] = parent_stations_df["stop_lat"].astype(float)
parent_stations_df["stop_lon"] = parent_stations_df["stop_lon"].astype(float)





#SECTION 2: DATA CLUSTERING
import numpy as np
from sklearn.cluster import DBSCAN

#Load the attractions CSV
pois_df = pd.read_csv("pois.csv")

#Physical parameters in miles
CLUSTER_RADIUS_MILES = 1.5
EARTH_RADIUS_MILES = 3958.8
# Convert radius (1.5 miles) to radians for Haversine
cluster_radius_radians = CLUSTER_RADIUS_MILES / EARTH_RADIUS_MILES
#make all poi lat lon into radians for haversine distance calculation
poi_coordinates_in_radians = np.radians(pois_df[['lat', 'lon']].values)

# Run DBSCAN (min_samples=1 ensures single isolated POIs remain as 1-item clusters)
db = DBSCAN(eps=cluster_radius_radians, min_samples=1, metric='haversine')
pois_df['cluster_id'] = db.fit_predict(poi_coordinates_in_radians)


# Compute centroids and aggregate properties for each cluster
cluster_list = []

for cluster_id, group in pois_df.groupby('cluster_id'):
    centroid_lat = group['lat'].mean()
    centroid_lon = group['lon'].mean()
    total_weight = group['weight'].sum()
    poi_count = len(group)
    poi_names_list = ", ".join(group['poi_name'].tolist())
    
    cluster_list.append({
        'cluster_id': cluster_id,
        'cluster_name': f"Cluster {cluster_id + 1}",
        'centroid_lat': centroid_lat,
        'centroid_lon': centroid_lon,
        'total_weight': total_weight,
        'poi_count': poi_count,
        'poi_names': poi_names_list
    })

# Convert list of cluster into a pandas DataFrame
clusters_df = pd.DataFrame(cluster_list)

print(f"  - Grouped {len(pois_df)} POIs into {len(clusters_df)} distinct 1.5-mile clusters.")
print(clusters_df[['cluster_name', 'centroid_lat', 'centroid_lon', 'poi_count', 'total_weight']])



# SECTION 3. GRAPHING CONSTRUCTION

import networkx as nx
from haversine import haversine, Unit


#Build the NetworkX Directed Subway Track Graph
Subway_Graph = nx.DiGraph() #(directed graph)

# Add all track edges with exact GTFS runtimes
for _, edge in subway_edges_df.iterrows():
    Subway_Graph.add_edge(
        str(edge['from_station_id']),
        str(edge['to_station_id']),
        weight=edge['travel_time_mins']
    )

print("\nSUBWAY NETWORK RESULTS:")
print(subway_edges_df.head(5).to_string(index=False))














