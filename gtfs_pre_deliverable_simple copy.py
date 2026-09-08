import folium
import math
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit


# SETTINGS

BASE_DIR = Path(__file__).parent
STOPS_FILE = BASE_DIR / "stops.txt"

OUTPUT_DIR = BASE_DIR / "output_v3"
OUTPUT_DIR.mkdir(exist_ok=True)

WALKING_SPEED_METERS_PER_MIN = 80
ROUGH_SUBWAY_SPEED_METERS_PER_MIN = 500


#  POI WEIGHT TIERS


POI_WEIGHT_TIERS = [
    1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,
]


#  REAL NYC POI LIST

POIS = [
    {"poi_name": "Statue of Liberty", "lat": 40.689211, "lon": -74.044643},
    {"poi_name": "Times Square", "lat": 40.756360, "lon": -73.986440},
    {"poi_name": "Empire State Building", "lat": 40.748817, "lon": -73.985428},
    {"poi_name": "Central Park", "lat": 40.781200, "lon": -73.966500},
    {"poi_name": "Grand Central Terminal", "lat": 40.752655, "lon": -73.977295},
    {"poi_name": "The Metropolitan Museum of Art", "lat": 40.779434, "lon": -73.963402},
    {"poi_name": "World Trade Center", "lat": 40.712742, "lon": -74.013382},
    {"poi_name": "Brooklyn Bridge", "lat": 40.706086, "lon": -73.996864},
    {"poi_name": "Rockefeller Center", "lat": 40.758740, "lon": -73.978674},
    {"poi_name": "Bryant Park", "lat": 40.753597, "lon": -73.983233},
    {"poi_name": "Penn Station", "lat": 40.750568, "lon": -73.993519},
    {"poi_name": "MoMA", "lat": 40.761433, "lon": -73.977622},
    {"poi_name": "Chelsea Market", "lat": 40.742439, "lon": -74.006143},
    {"poi_name": "Washington Square Park", "lat": 40.730823, "lon": -73.997332},
    {"poi_name": "SoHo", "lat": 40.723301, "lon": -74.002988},
    {"poi_name": "Williamsburg", "lat": 40.708116, "lon": -73.957070},
]


#  REAL NYC HOTEL LIST

HOTELS = [
    {"hotel_name": "New York Marriott Marquis", "area": "Times Square", "lat": 40.758617, "lon": -73.986221},
    {"hotel_name": "The Plaza Hotel", "area": "Central Park South", "lat": 40.764700, "lon": -73.974600},
    {"hotel_name": "The New Yorker, A Wyndham Hotel", "area": "Penn Station / Midtown West", "lat": 40.752272, "lon": -73.993319},
    {"hotel_name": "Moxy NYC Times Square", "area": "Times Square / Herald Square", "lat": 40.752697, "lon": -73.990737},
    {"hotel_name": "Ace Hotel New York", "area": "NoMad", "lat": 40.745866, "lon": -73.988850},
    {"hotel_name": "Lotte New York Palace", "area": "Midtown East", "lat": 40.758108, "lon": -73.974709},
    {"hotel_name": "Conrad New York Downtown", "area": "Battery Park City", "lat": 40.715032, "lon": -74.015642},
    {"hotel_name": "The Standard, High Line", "area": "Meatpacking District", "lat": 40.740853, "lon": -74.007861},
    {"hotel_name": "Arlo SoHo", "area": "SoHo / Hudson Square", "lat": 40.724757, "lon": -74.006152},
    {"hotel_name": "citizenM New York Bowery", "area": "Bowery / Lower East Side", "lat": 40.720756, "lon": -73.993195},
    {"hotel_name": "The William Vale", "area": "Williamsburg", "lat": 40.722293, "lon": -73.956623},
    {"hotel_name": "Hampton Inn Brooklyn Downtown", "area": "Downtown Brooklyn", "lat": 40.695927, "lon": -73.984275},
]



#  POI WEIGHT FUNCTIONS

def assign_default_poi_weights(pois):
    weighted_pois = []
    for index, poi in enumerate(pois):
        poi_copy = poi.copy()
        if index < len(POI_WEIGHT_TIERS):
            poi_copy["weight"] = POI_WEIGHT_TIERS[index]
        else:
            poi_copy["weight"] = POI_WEIGHT_TIERS[-1]
        weighted_pois.append(poi_copy)
    return weighted_pois


def recalculate_weights_by_current_order(pois):
    for index, poi in enumerate(pois):
        if index < len(POI_WEIGHT_TIERS):
            poi["weight"] = POI_WEIGHT_TIERS[index]
        else:
            poi["weight"] = POI_WEIGHT_TIERS[-1]
    return pois


def display_poi_weights(pois):
    print("\nCurrent POI weight list:")
    print("-" * 55)
    for index, poi in enumerate(pois):
        print(f"{index:2d}. {poi['poi_name']:<35} weight = {poi['weight']:.2f}")
    print("-" * 55)


def swap_poi_positions_by_index(pois, index_a, index_b):
    if index_a < 0 or index_a >= len(pois):
        print(f"Invalid first index: {index_a}")
        return pois
    if index_b < 0 or index_b >= len(pois):
        print(f"Invalid second index: {index_b}")
        return pois

    first_name = pois[index_a]["poi_name"]
    second_name = pois[index_b]["poi_name"]

    pois[index_a], pois[index_b] = pois[index_b], pois[index_a]
    pois = recalculate_weights_by_current_order(pois)
    print(f"\nMoved POIs: {first_name} <-> {second_name}")
    return pois


def update_poi_weights_in_terminal(pois):
    print("\nBefore hotel scoring starts, you can update POI importance.")
    print("You will swap weights by index number.")
    print("Example: typing 0 and 10 swaps Statue of Liberty and Penn Station positions.")

    keep_updating = "yes"
    while keep_updating.lower() in ["yes", "y"]:
        display_poi_weights(pois)
        first_value = input("Enter first POI index to swap: ").strip()
        second_value = input("Enter second POI index to swap: ").strip()

        try:
            index_a = int(first_value)
            index_b = int(second_value)
        except ValueError:
            print("Please enter numbers only, like 0 and 10.")
            continue

        pois = swap_poi_positions_by_index(pois, index_a, index_b)
        display_poi_weights(pois)
        keep_updating = input("Do you want to keep updating? yes/no: ").strip()

    print("\nFinal POI weights locked in. Starting hotel calculations...")
    return pois


# DISTANCE / SCORING HELPERS

def haversine_meters(lat1, lon1, lat2, lon2):
    """Finds straight-line distance between two lat/lon points in meters using the haversine library."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def walking_minutes(distance_meters):
    return distance_meters / WALKING_SPEED_METERS_PER_MIN


def rough_subway_minutes(distance_meters):
    return max(3, distance_meters / ROUGH_SUBWAY_SPEED_METERS_PER_MIN)


def time_to_score(total_minutes):
    return 100 * math.exp(-total_minutes / 35)


# CLUSTERING HELPER

def cluster_pois_dbscan(pois_df, max_walk_meters=800):
    """
    Groups POIs within `max_walk_meters` into a single cluster.
    Returns a new DataFrame of centroids and combined weights.
    """
    EARTH_RADIUS = 6371000  # meters
    coords = np.radians(pois_df[['lat', 'lon']])
    eps_radians = max_walk_meters / EARTH_RADIUS
    
    dbscan = DBSCAN(eps=eps_radians, min_samples=1, metric='haversine', algorithm='ball_tree')
    pois_df['cluster_id'] = dbscan.fit_predict(coords)
    
    clustered_data = []
    
    for cluster_id, group in pois_df.groupby('cluster_id'):
        centroid_lat = group['lat'].mean()
        centroid_lon = group['lon'].mean()
        combined_weight = group['weight'].sum()
        
        if len(group) > 1:
            poi_name = f"Cluster ({len(group)} POIs): " + ", ".join(group['poi_name'].tolist())
        else:
            poi_name = group['poi_name'].iloc[0]
            
        clustered_data.append({
            "poi_name": poi_name,
            "lat": centroid_lat,
            "lon": centroid_lon,
            "weight": combined_weight,
            "original_poi_count": len(group)
        })
        
    # Sort the new clusters by their combined weight descending
    return pd.DataFrame(clustered_data).sort_values(by="weight", ascending=False).reset_index(drop=True)


# GTFS STATION FUNCTIONS

def load_station_nodes():
    if not STOPS_FILE.exists():
        raise FileNotFoundError("Missing stops.txt. Put this script in the same folder as stops.txt.")

    stops = pd.read_csv(STOPS_FILE)
    if "location_type" in stops.columns:
        location_type_numeric = pd.to_numeric(stops["location_type"], errors="coerce")
        stations = stops[location_type_numeric == 1].copy()
        if stations.empty and "parent_station" in stops.columns:
            stations = stops[stops["parent_station"].isna()].copy()
    else:
        stations = stops.copy()

    stations = stations.rename(
        columns={"stop_id": "station_id", "stop_name": "station_name", "stop_lat": "lat", "stop_lon": "lon"}
    )
    return stations[["station_id", "station_name", "lat", "lon"]].dropna().reset_index(drop=True)


def find_nearest_station(lat, lon, stations):
    best_station = None
    best_distance = float("inf")

    for _, station in stations.iterrows():
        distance = haversine_meters(lat, lon, station["lat"], station["lon"])
        if distance < best_distance:
            best_distance = distance
            best_station = station

    if best_station is None:
        raise ValueError("No stations found. Check that stops.txt loaded correctly.")

    return {
        "station_id": best_station["station_id"],
        "station_name": best_station["station_name"],
        "station_lat": best_station["lat"],
        "station_lon": best_station["lon"],
        "distance_meters": best_distance,
        "walk_minutes": walking_minutes(best_distance),
    }


def create_map(hotels_df, pois_df, stations, hotel_scores_df):
    nyc_map = folium.Map(location=[40.7580, -73.9855], zoom_start=12)

    for _, station in stations.iterrows():
        folium.CircleMarker(
            location=[station["lat"], station["lon"]],
            radius=2, color="black", fill=True, fill_opacity=0.7, popup=station["station_name"],
        ).add_to(nyc_map)

    for _, poi in pois_df.iterrows():
        folium.Marker(
            location=[poi["lat"], poi["lon"]],
            popup=f"<b>{poi['poi_name']}</b><br>Weight: {poi['weight']:.2f}",
            icon=folium.Icon(color="blue", icon="star"),
        ).add_to(nyc_map)

    for _, hotel in hotel_scores_df.iterrows():
        score = hotel["hotel_access_score"]
        if score >= 80: color = "green"
        elif score >= 70: color = "orange"
        else: color = "red"

        folium.Marker(
            location=[hotel["lat"], hotel["lon"]],
            popup=f"<b>{hotel['hotel_name']}</b><br>Area: {hotel['area']}<br>Score: {hotel['hotel_access_score']}<br>Nearest Station: {hotel['nearest_station']}<br>Walk: {hotel['walk_to_station_min']} min",
            icon=folium.Icon(color=color, icon="home"),
        ).add_to(nyc_map)

    nyc_map.save(OUTPUT_DIR / "nyc_hotel_map.html")
    print("Map saved!")


# MAIN 

def main():
    print("Loading GTFS station data...")
    stations = load_station_nodes()

    # Let the user rank the POIs
    weighted_pois = assign_default_poi_weights(POIS)
    weighted_pois = update_poi_weights_in_terminal(weighted_pois)

    hotels_df = pd.DataFrame(HOTELS)
    pois_df = pd.DataFrame(weighted_pois)
    
    # ---------------------------------------------------------
    # NEW: Cluster the POIs based on 800m proximity
    # ---------------------------------------------------------
    pois_df = cluster_pois_dbscan(pois_df, max_walk_meters=800)

    print(f"\nStations loaded: {len(stations)}")
    print(f"Hotels loaded: {len(hotels_df)}")
    print(f"Clustered POIs loaded: {len(pois_df)}")

    # ---------------------------------------------------------
    # NEW: Pre-calculate (Cache) nearest stations to avoid massive loop lag
    # ---------------------------------------------------------
    print("Pre-calculating nearest stations...")
    poi_station_cache = {}
    for _, poi in pois_df.iterrows():
        poi_station_cache[poi["poi_name"]] = find_nearest_station(poi["lat"], poi["lon"], stations)

    hotel_station_cache = {}
    for _, hotel in hotels_df.iterrows():
        hotel_station_cache[hotel["hotel_name"]] = find_nearest_station(hotel["lat"], hotel["lon"], stations)


    stations.to_csv(OUTPUT_DIR / "station_nodes.csv", index=False)
    hotels_df.to_csv(OUTPUT_DIR / "nyc_hotels.csv", index=False)
    pois_df.to_csv(OUTPUT_DIR / "nyc_pois_ranked.csv", index=False)

    hotel_scores = []
    trip_details = []

    print("\nScoring hotels...\n")

    for _, hotel in hotels_df.iterrows():
        # Load from cache instantly instead of recalculating
        hotel_station = hotel_station_cache[hotel["hotel_name"]]

        weighted_score_total = 0
        weight_total = 0
        trip_time_total = 0

        top_score_total = 0
        top_weight_total = 0
        
        # Track normalized coverage score variables
        coverage_points = 0
        max_possible_coverage_points = 0

        for _, poi in pois_df.iterrows():
            # Load from cache instantly
            poi_station = poi_station_cache[poi["poi_name"]]

            station_to_station_distance = haversine_meters(
                hotel_station["station_lat"],
                hotel_station["station_lon"],
                poi_station["station_lat"],
                poi_station["station_lon"],
            )

            estimated_subway_min = rough_subway_minutes(station_to_station_distance)

            # Use REAL walk time to calculate the trip accurately (teleportation bug fixed)
            hotel_walk = hotel_station["walk_minutes"]
            total_trip_min = hotel_walk + estimated_subway_min + poi_station["walk_minutes"]
            access_score = time_to_score(total_trip_min)

            # Top POI Score (Grab first 5 clusters)
            if poi.name < 5:
                top_score_total += access_score * poi["weight"]
                top_weight_total += poi["weight"]

            # Normalized Coverage Points
            if total_trip_min <= 15:
                coverage_points += poi["weight"] * 3
            elif total_trip_min <= 30:
                coverage_points += poi["weight"] * 2
            elif total_trip_min <= 45:
                coverage_points += poi["weight"] * 1
                
            max_possible_coverage_points += poi["weight"] * 3

            weighted_score_total += access_score * poi["weight"]
            weight_total += poi["weight"]
            trip_time_total += total_trip_min

            trip_details.append({
                "hotel_name": hotel["hotel_name"],
                "hotel_area": hotel["area"],
                "poi_name": poi["poi_name"],
                "poi_weight": poi["weight"],
                "hotel_nearest_station": hotel_station["station_name"],
                "poi_nearest_station": poi_station["station_name"],
                "hotel_walk_min": round(hotel_station["walk_minutes"], 2),
                "poi_walk_min": round(poi_station["walk_minutes"], 2),
                "rough_subway_min": round(estimated_subway_min, 2),
                "estimated_total_trip_min": round(total_trip_min, 2),
                "access_score": round(access_score, 2),
            })

        travel_score = weighted_score_total / weight_total
        top_poi_score = top_score_total / top_weight_total if top_weight_total > 0 else 0

        # Normalized Coverage Score
        if max_possible_coverage_points > 0:
            coverage_score = (coverage_points / max_possible_coverage_points) * 100
        else:
            coverage_score = 0

        # Dedicated Walk Score (Grace period correctly applied here)
        walk = hotel_station["walk_minutes"]
        effective_walk = max(0, walk - 10) 
        
        if effective_walk == 0:
            walk_score = 100
        elif effective_walk <= 10:
            walk_score = 80
        elif effective_walk <= 20:
            walk_score = 60
        else:
            walk_score = 40

        final_score = (
            travel_score * 0.40
            + top_poi_score * 0.30
            + coverage_score * 0.20
            + walk_score * 0.10
        )

        average_trip_time = trip_time_total / len(pois_df)

        hotel_scores.append({
            "hotel_name": hotel["hotel_name"],
            "area": hotel["area"],
            "lat": hotel["lat"],
            "lon": hotel["lon"],
            "nearest_station": hotel_station["station_name"],
            "walk_to_station_min": round(hotel_station["walk_minutes"], 2),
            "average_trip_time_to_pois_min": round(average_trip_time, 2),
            "travel_score": round(travel_score, 2),
            "top_poi_score": round(top_poi_score, 2),
            "coverage_score": round(coverage_score, 2),
            "walk_score": round(walk_score, 2),
            "hotel_access_score": round(final_score, 2),
        })

    hotel_scores_df = pd.DataFrame(hotel_scores).sort_values("hotel_access_score", ascending=False).reset_index(drop=True)
    trip_details_df = pd.DataFrame(trip_details)

    hotel_scores_df.to_csv(OUTPUT_DIR / "hotel_access_scores.csv", index=False)
    trip_details_df.to_csv(OUTPUT_DIR / "hotel_to_poi_details.csv", index=False)

    create_map(hotels_df, pois_df, stations, hotel_scores_df)

    print("\nTop Hotels:\n")
    print(hotel_scores_df[["hotel_name", "hotel_access_score", "nearest_station"]])
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()