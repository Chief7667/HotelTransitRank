import os
import math
import pandas as pd
import folium
from haversine import haversine, Unit

# 1. DATA INIT


stations_df = pd.DataFrame([
    {"stop_name": "Times Sq - 42 St", "stop_lat": 40.7549, "stop_lon": -73.9868},
    {"stop_name": "Grand Central - 42 St", "stop_lat": 40.7527, "stop_lon": -73.9772},
    {"stop_name": "34 St - Penn Station", "stop_lat": 40.7506, "stop_lon": -73.9935},
    {"stop_name": "W 4 St - Wash Sq", "stop_lat": 40.7323, "stop_lon": -74.0001},
    {"stop_name": "Atlantic Av - Barclays Ctr", "stop_lat": 40.6839, "stop_lon": -73.9788}
])

clusters_df = pd.DataFrame([
    {
        "cluster_name": "Cluster 1: Midtown Core",
        "poi_names": "Times Square, Rockefeller Center, Bryant Park",
        "total_weight": 8.5,
        "poi_count": 3,
        "lat": 40.7580,
        "lon": -73.9855
    },
    {
        "cluster_name": "Cluster 2: Lower Manhattan",
        "poi_names": "One World Trade, 9/11 Memorial, Battery Park",
        "total_weight": 6.2,
        "poi_count": 3,
        "lat": 40.7115,
        "lon": -74.0125
    },
    {
        "cluster_name": "Cluster 3: Central Park South",
        "poi_names": "Central Park, Plaza Hotel, Carnegie Hall",
        "total_weight": 5.0,
        "poi_count": 3,
        "lat": 40.7651,
        "lon": -73.9760
    }
])

hotels_df = pd.DataFrame([
    {"hotel_name": "Pod 39", "lat": 40.7494, "lon": -73.9772, "nearest_station": "Grand Central - 42 St"},
    {"hotel_name": "Moxy NYC Times Square", "lat": 40.7512, "lon": -73.9885, "nearest_station": "Times Sq - 42 St"},
    {"hotel_name": "The Standard High Line", "lat": 40.7408, "lon": -74.0079, "nearest_station": "W 4 St - Wash Sq"},
    {"hotel_name": "The William Vale", "lat": 40.7222, "lon": -73.9571, "nearest_station": "Atlantic Av - Barclays Ctr"},
    {"hotel_name": "Ace Hotel Brooklyn", "lat": 40.6892, "lon": -73.9822, "nearest_station": "Atlantic Av - Barclays Ctr"}
])


# 2. HOTEL SCORING 


raw_hotel_scores = []

for idx, hotel in hotels_df.iterrows():
    h_lat, h_lon = hotel['lat'], hotel['lon']
    
    # Walk time to nearest station
    station_match = stations_df[stations_df['stop_name'] == hotel['nearest_station']]
    if not station_match.empty:
        s_lat, s_lon = station_match.iloc[0]['stop_lat'], station_match.iloc[0]['stop_lon']
        walk_dist_km = haversine((h_lat, h_lon), (s_lat, s_lon), unit=Unit.KILOMETERS)
        walk_time_mins = max(1.5, (walk_dist_km / 4.8) * 60)
    else:
        walk_time_mins = 3.5

    cluster_scores = []
    total_raw_score = 0.0

    for c_idx, cluster in clusters_df.iterrows():
        dist_km = haversine((h_lat, h_lon), (cluster['lat'], cluster['lon']), unit=Unit.KILOMETERS)
        trip_mins = max(3.0, dist_km * 4.2 + walk_time_mins)
        
        # Exponential distance decay factor (-0.08 penalizes distance more heavily)
        proximity_factor = math.exp(-0.08 * trip_mins)
        cluster_pts = cluster['total_weight'] * proximity_factor * 10.0
        
        total_raw_score += cluster_pts
        cluster_scores.append({
            'name': cluster['cluster_name'].split(':')[1].strip() if ':' in cluster['cluster_name'] else cluster['cluster_name'],
            'pts': cluster_pts,
            'mins': trip_mins
        })

    cluster_scores.sort(key=lambda x: x['pts'], reverse=True)
    
    raw_hotel_scores.append({
        'hotel_name': hotel['hotel_name'],
        'lat': h_lat,
        'lon': h_lon,
        'raw_score': total_raw_score,
        'nearest_station': hotel['nearest_station'],
        'walk_time_mins': walk_time_mins,
        'top_clusters': cluster_scores[:2]
    })

scored_hotels_df = pd.DataFrame(raw_hotel_scores)

# RELATIVE NORMALIZATION (0 to 100 Scale relative to best/worst hotel)
min_s = scored_hotels_df['raw_score'].min()
max_s = scored_hotels_df['raw_score'].max()

scored_hotels_df['hotel_access_score'] = scored_hotels_df['raw_score'].apply(
    lambda x: round(((x - min_s) / (max_s - min_s)) * 100.0, 2) if max_s > min_s else 100.0
)

scored_hotels_df = scored_hotels_df.sort_values(by='hotel_access_score', ascending=False).reset_index(drop=True)

# 3. FOLIUM MAP 

output_dir = "output_v3"
os.makedirs(output_dir, exist_ok=True)

# OpenStreetMap tiles remove Carto API key watermark
hotel_map = folium.Map(location=[40.7450, -73.9850], zoom_start=13, tiles="OpenStreetMap")

# 1. Add Stations
for _, station in stations_df.iterrows():
    folium.CircleMarker(
        location=[station['stop_lat'], station['stop_lon']],
        radius=3,
        color='black',
        fill=True,
        fill_color='black',
        fill_opacity=0.8,
        tooltip=station['stop_name']
    ).add_to(hotel_map)

# 2. Add POI Clusters 
for _, cluster in clusters_df.iterrows():
    cluster_title = cluster['cluster_name'].split(':')[1].strip() if ':' in cluster['cluster_name'] else cluster['cluster_name']
    poi_items = "".join([f"<li style='margin-bottom: 2px;'>{p.strip()}</li>" for p in cluster['poi_names'].split(',')])

    cluster_popup_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; width: 500px;max-height: 250px; overflow-y: auto; line-height: 1.4; color: #222;">
        <h4 style="margin: 0 0 6px 0; font-size: 14px; font-weight: bold; color: #0056b3;">{cluster_title}</h4>
        
        <div style="margin-bottom: 8px;">
            <b>Cluster Weight:</b> 
            <span style="background-color: #007bff; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px;">
                {cluster['total_weight']:.2f} pts
            </span>
        </div>
        
        <div style="border-top: 1px solid #e0e0e0; padding-top: 6px; margin-top: 6px;">
            <div style="font-weight: bold; margin-bottom: 4px; color: #333;">Grouped Attractions ({cluster['poi_count']}):</div>
            <ul style="margin: 0; padding-left: 18px; color: #444;">
                {poi_items}
            </ul>
        </div>
    </div>
    """

    folium.Marker(
        location=[cluster['lat'], cluster['lon']],
        popup=folium.Popup(cluster_popup_html, max_width=270),
        tooltip=cluster['cluster_name'],
        icon=folium.Icon(color='blue', icon='star', prefix='fa')
    ).add_to(hotel_map)

# 3. Add Scored Hotels 
for idx, row in scored_hotels_df.iterrows():
    score = row['hotel_access_score']
    hotel_name = row['hotel_name']
    station_name = row['nearest_station']
    walk_time = row['walk_time_mins']
    
    # Relative Color Tiers
    if score >= 70.0:
        badge_color, tier_text, icon_color = "#28a745", "Top Tier", "green"
    elif score >= 40.0:
        badge_color, tier_text, icon_color = "#fd7e14", "Mid Tier", "orange"
    else:
        badge_color, tier_text, icon_color = "#dc3545", "Low Tier", "red"

    top_clusters = row['top_clusters']
    cluster_items_html = "".join([
        f"<li style='margin-bottom: 3px;'><b>{c['name']}:</b> +{c['pts']:.1f} pts <i>({c['mins']:.0f} min trip)</i></li>"
        for c in top_clusters
    ])

    hotel_popup_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; width: 240px; line-height: 1.4; color: #222;">
        <h4 style="margin: 0 0 6px 0; font-size: 14px; font-weight: bold; color: #111;">{hotel_name}</h4>
        
        <div style="margin-bottom: 8px;">
            <b>Score:</b> 
            <span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px;">
                {score:.1f} / 100
            </span>
            <span style="color: #666; font-size: 11px; margin-left: 4px;">({tier_text})</span>
        </div>
        
        <div style="margin-bottom: 8px; color: #444;">
            <b>Subway:</b> {station_name} <span style="color: #666; font-size: 11px;">({walk_time:.1f} min walk)</span>
        </div>
        
        <div style="border-top: 1px solid #e0e0e0; padding-top: 6px; margin-top: 6px;">
            <div style="font-weight: bold; margin-bottom: 4px; color: #333;">Top Attractions Served:</div>
            <ul style="margin: 0; padding-left: 18px; color: #444;">
                {cluster_items_html}
            </ul>
        </div>
    </div>
    """

    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(hotel_popup_html, max_width=280),
        tooltip=hotel_name,
        icon=folium.Icon(color=icon_color, icon='home', prefix='fa')
    ).add_to(hotel_map)

# 4. SAVE OUTPUTS

map_path = os.path.join(output_dir, "nyc_hotel_map.html")
csv_path = os.path.join(output_dir, "scored_hotels.csv")

hotel_map.save(map_path)
scored_hotels_df.to_csv(csv_path, index=False)

print(f"Outputs saved successfully to: {output_dir}")
print(f"  - Map: {map_path}")
print(f"  - CSV: {csv_path}")



#naming git push
