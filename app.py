import os
# Configure headless environment variables to prevent Segmentation Fault on cloud hosts
os.environ["MPLBACKEND"] = "Agg"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import urllib.parse
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Clean garbage memory on script initialization
gc.collect()

# ==============================================================================
# 1. Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Japan Milk Run Optimization",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Japan Milk Run - Route Optimization App")
st.write("Vehicle Routing Problem (VRP / Milk Run) Solver powered by Google OR-Tools")

# ==============================================================================
# 2. Sidebar File Uploaders
# ==============================================================================
st.sidebar.header("📁 Upload Master Data Files")
file_loc = st.sidebar.file_uploader("1. locations_master", type=["csv", "xlsx"])
file_dem = st.sidebar.file_uploader("2. demands_flows", type=["csv", "xlsx"])
file_fleet = st.sidebar.file_uploader("3. fleet_master", type=["csv", "xlsx"])

def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# ==============================================================================
# 3. Distance Matrix & Optimization Engine
# ==============================================================================
def compute_haversine_matrix(df):
    """Calculates travel duration matrix (in minutes) between coordinates."""
    coords = list(zip(df['Latitude'], df['Longitude']))
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
                lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                matrix[i][j] = int(round(dist_km * 1.35 / 45.0 * 60))
    return matrix

def solve_pdvrp_engine(df_loc, df_dem, df_fleet, time_matrix, start_mode='AUTOMATIC', custom_start_list=None):
    gc.collect()

    df_dem_sub = df_dem[df_dem['Schedule'].isin(['MWF', 'Daily', 'TT'])].reset_index(drop=True)
    loc_to_idx = {loc_id: idx for idx, loc_id in enumerate(df_loc['Location_ID'])}

    num_reqs = len(df_dem_sub)
    num_vehicles = len(df_fleet)

    node_matrix_indices = []
    node_demands = []

    # 1. Create 0-demand Start Nodes for each vehicle in the available fleet
    starts = list(range(num_vehicles))
    start_locations = []
    for v_idx in range(num_vehicles):
        if start_mode == 'USER_DEFINED' and custom_start_list:
            loc_name = custom_start_list[v_idx % len(custom_start_list)]
        else:
            loc_name = 'OURA'  # Original Central Base Depot
        start_locations.append(loc_name)
        node_matrix_indices.append(loc_to_idx.get(loc_name, 0))
        node_demands.append(0)

    # 2. Create 0-demand Shared Return Depot Node (OURA)
    end_depot_node = num_vehicles
    ends = [end_depot_node] * num_vehicles
    node_matrix_indices.append(loc_to_idx['OURA'])
    node_demands.append(0)

    # 3. Create Pickup and Delivery Request Nodes
    pickup_delivery_pairs = []
    for i, row in df_dem_sub.iterrows():
        qty = int(row['Pallets'])
        p_matrix_idx = loc_to_idx[row['Origin_Location_ID']]
        d_matrix_idx = loc_to_idx[row['Destination_Location_ID']]

        p_node_idx = len(node_matrix_indices)
        node_matrix_indices.append(p_matrix_idx)
        node_demands.append(qty)

        d_node_idx = len(node_matrix_indices)
        node_matrix_indices.append(d_matrix_idx)
        node_demands.append(-qty)

        pickup_delivery_pairs.append((p_node_idx, d_node_idx))

    num_nodes = len(node_matrix_indices)

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_idx, to_idx):
        fn, tn = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return time_matrix[node_matrix_indices[fn]][node_matrix_indices[tn]]

    tc_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tc_idx)

    # 13-Hour Maximum Driver Working Limit (780 Minutes)
    routing.AddDimension(tc_idx, 0, 780, True, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')
    time_dim.SetGlobalSpanCostCoefficient(100)

    def demand_cb(from_idx):
        return node_demands[manager.IndexToNode(from_idx)]

    dc_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    v_caps = df_fleet['Capacity_Pallets'].tolist()
    routing.AddDimensionWithVehicleCapacity(dc_idx, 0, v_caps, True, 'Capacity')

    solver = routing.solver()
    for p_node, d_node in pickup_delivery_pairs:
        p_idx = manager.NodeToIndex(p_node)
        d_idx = manager.NodeToIndex(d_node)
        routing.AddPickupAndDelivery(p_idx, d_idx)
        solver.Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
        solver.Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

    for v in range(num_vehicles):
        routing.SetFixedCostOfVehicle(1000, v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 3

    sol = routing.SolveWithParameters(params)

    if not sol:
        del routing, manager
        gc.collect()
        return None, 0

    active_routes, total_time = [], 0
    colors = ['#E63946', '#2A9D8F', '#F4A261', '#9C27B0', '#3F51B5', '#009688', '#FF5722']

    for v in range(num_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue
        v_code, v_type, v_cap = df_fleet.loc[v, 'Vehicle_ID'], df_fleet.loc[v, 'Vehicle_Type'], v_caps[v]
        stops = []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            m_idx = node_matrix_indices[node_idx]
            cumul_mins = sol.Min(time_dim.CumulVar(idx))
            stops.append({
                'loc_id': df_loc.loc[m_idx, 'Location_ID'],
                'lat': df_loc.loc[m_idx, 'Latitude'],
                'lon': df_loc.loc[m_idx, 'Longitude'],
                'demand': node_demands[node_idx],
                'cumul_mins': cumul_mins
            })
            idx = sol.Value(routing.NextVar(idx))
            
        end_cumul_mins = sol.Min(time_dim.CumulVar(idx))
        stops.append({
            'loc_id': 'OURA', 
            'lat': df_loc.loc[0, 'Latitude'], 
            'lon': df_loc.loc[0, 'Longitude'], 
            'demand': 0,
            'cumul_mins': end_cumul_mins
        })
        duration = end_cumul_mins
        total_time += duration
        
        active_routes.append({
            'truck_num': len(active_routes) + 1,
            'v_code': v_code,
            'v_type': v_type,
            'v_cap': v_cap,
            'start_loc': start_locations[v],
            'duration_mins': duration,
            'color': colors[len(active_routes) % len(colors)],
            'stops': stops
        })

    del sol, routing, manager
    gc.collect()

    return active_routes, total_time

# ==============================================================================
# 4. Route Visualizers & Exporters
# ==============================================================================
def render_combined_master_map(df_loc, routes):
    """Renders a single master map with all vehicle routes plotted together."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.grid(True, linestyle='--', alpha=0.5)

    depot_mask = (df_loc['Location_ID'] == 'OURA')
    
    ax.scatter(df_loc.loc[~depot_mask, 'Longitude'], df_loc.loc[~depot_mask, 'Latitude'], color='#D0D0D0', s=300, zorder=2, label='Locations')
    ax.scatter(df_loc.loc[depot_mask, 'Longitude'], df_loc.loc[depot_mask, 'Latitude'], color='#FF2222', s=600, zorder=3, label='Central Depot (OURA)')

    for _, row in df_loc.iterrows():
        l_id, lat, lon = row['Location_ID'], row['Latitude'], row['Longitude']
        if l_id == 'OURA':
            ax.text(lon + 0.005, lat + 0.003, "OURA [DEPOT]", fontsize=10, weight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFE6E6", ec="red"))
        else:
            ax.text(lon + 0.003, lat + 0.002, l_id, fontsize=8, weight='bold', bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray", alpha=0.8))

    for r in routes:
        stops = r['stops']
        for k in range(len(stops) - 1):
            sx, sy, ex, ey = stops[k]['lon'], stops[k]['lat'], stops[k + 1]['lon'], stops[k + 1]['lat']
            dx, dy = ex - sx, ey - sy
            ax.plot([sx, ex], [sy, ey], color=r['color'], linewidth=2.5, alpha=0.85, zorder=4, label=f"Truck {r['truck_num']} ({r['v_code']})" if k == 0 else "")
            if (dx != 0 or dy != 0):
                ax.arrow(sx + dx * 0.45, sy + dy * 0.45, dx * 0.08, dy * 0.08, shape='full', lw=0, length_includes_head=True, head_width=0.008, color=r['color'], zorder=5)

    ax.set_title("🌐 Master Route Map - All Vehicles Combined Loop", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    return fig

def render_single_route_plot(df_loc, r):
    """Renders a detailed single-vehicle route plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, linestyle='--', alpha=0.5)

    depot_mask = (df_loc['Location_ID'] == 'OURA')
    ax.scatter(df_loc.loc[~depot_mask, 'Longitude'], df_loc.loc[~depot_mask, 'Latitude'], color='#E0E0E0', s=250, zorder=2)
    ax.scatter(df_loc.loc[depot_mask, 'Longitude'], df_loc.loc[depot_mask, 'Latitude'], color='#FF2222', s=500, zorder=3)

    stops = r['stops']
    visited = set([s['loc_id'] for s in stops])
    loads = {}
    for s in stops:
        loc, dem = s['loc_id'], s['demand']
        if loc not in loads:
            loads[loc] = {'p': 0, 'd': 0}
        if dem > 0:
            loads[loc]['p'] += dem
        elif dem < 0:
            loads[loc]['d'] += abs(dem)

    v_lons = [df_loc[df_loc['Location_ID'] == l]['Longitude'].values[0] for l in visited if l != 'OURA']
    v_lats = [df_loc[df_loc['Location_ID'] == l]['Latitude'].values[0] for l in visited if l != 'OURA']
    ax.scatter(v_lons, v_lats, color='#3377FF', s=500, zorder=4)

    for k in range(len(stops) - 1):
        sx, sy, ex, ey = stops[k]['lon'], stops[k]['lat'], stops[k + 1]['lon'], stops[k + 1]['lat']
        dx, dy = ex - sx, ey - sy
        ax.plot([sx, ex], [sy, ey], color=r['color'], linewidth=3.0, alpha=0.9, zorder=5)
        if (dx != 0 or dy != 0):
            ax.arrow(sx + dx * 0.4, sy + dy * 0.4, dx * 0.1, dy * 0.1, shape='full', lw=0, length_includes_head=True, head_width=0.008, color=r['color'], zorder=6)

    for l in visited:
        lon = df_loc[df_loc['Location_ID'] == l]['Longitude'].values[0]
        lat = df_loc[df_loc['Location_ID'] == l]['Latitude'].values[0]
        p, d = loads[l]['p'], loads[l]['d']
        if l == 'OURA':
            ax.text(lon + 0.005, lat + 0.003, "OURA [DEPOT END]", fontsize=9, weight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFE6E6", ec="red"))
        elif l == r['start_loc']:
            ax.text(lon + 0.004, lat + 0.002, f"START: {l}\n(+{p} load, -{d} unload)", fontsize=8.5, weight='bold', color='darkgreen', bbox=dict(boxstyle="square,pad=0.2", fc="#E6FFE6", ec="green"))
        else:
            ax.text(lon + 0.004, lat + 0.002, f"{l}\n(+{p} load, -{d} unload)", fontsize=8.5, weight='bold', bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="gray", alpha=0.85))

    ax.set_title(f"Truck {r['truck_num']}: {r['v_code']} ({r['v_type']}) - [START: {r['start_loc']} -> END: OURA] ({r['duration_mins']} mins)", fontsize=11, fontweight='bold', color=r['color'])
    plt.tight_layout()
    return fig

def generate_google_maps_url(stops):
    """Generates a direct Google Maps Directions URL for a given sequence of stops."""
    if len(stops) < 2:
        return ""
    origin = f"{stops[0]['lat']},{stops[0]['lon']}"
    destination = f"{stops[-1]['lat']},{stops[-1]['lon']}"
    waypoints = "|".join([f"{s['lat']},{s['lon']}" for s in stops[1:-1]])
    
    base_url = "https://www.google.com/maps/dir/?api=1"
    params = {
        "origin": origin,
        "destination": destination,
        "travelmode": "driving"
    }
    if waypoints:
        params["waypoints"] = waypoints
        
    return f"{base_url}&{urllib.parse.urlencode(params)}"

def generate_kml_file(routes):
    """Generates KML xml file content to import into Google My Maps."""
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('  <Document>')
    kml.append('    <name>Milk Run Optimized Routes</name>')
    
    for r in routes:
        kml.append('    <Placemark>')
        kml.append(f'      <name>Truck {r["truck_num"]} ({r["v_code"]})</name>')
        kml.append(f'      <description>Vehicle Type: {r["v_type"]} | Duration: {r["duration_mins"]} mins</description>')
        kml.append('      <LineString>')
        kml.append('        <tessellate>1</tessellate>')
        kml.append('        <coordinates>')
        for s in r['stops']:
            kml.append(f'          {s["lon"]},{s["lat"]},0')
        kml.append('        </coordinates>')
        kml.append('      </LineString>')
        kml.append('    </Placemark>')
        
    kml.append('  </Document>')
    kml.append('</kml>')
    return "\n".join(kml)

# Dialog Modal for Full Screen & Timed Stop Activity Breakdown
@st.dialog("🔍 Route Activity & Arrival Time Breakdown", width="large")
def show_route_detail_modal(df_loc, r):
    st.subheader(f"🚚 Truck {r['truck_num']}: {r['v_code']} ({r['v_type']})")
    st.write(f"**Max Capacity:** {r['v_cap']} Pallets | **Total Duration:** {r['duration_mins']} Mins")

    # Display large individual map
    fig = render_single_route_plot(df_loc, r)
    st.pyplot(fig)
    plt.close(fig)

    # Activity sequence breakdown table with timestamps
    st.markdown("### 📋 Timed Sequential Stop & Cargo Activity Log")
    st.caption("Calculates arrival timestamps assuming dispatch begins at 08:00 AM.")

    activity_log = []
    current_onboard = 0
    start_hour = 8

    for idx, s in enumerate(r['stops']):
        loc = s['loc_id']
        dem = s['demand']
        mins = s.get('cumul_mins', 0)
        
        arrival_hh = start_hour + (mins // 60)
        arrival_mm = mins % 60
        clock_time = f"{arrival_hh:02d}:{arrival_mm:02d}"

        if dem > 0:
            action = f"📦 PICKUP (+{dem} Pallets)"
            current_onboard += dem
        elif dem < 0:
            action = f"📦 UNLOAD (-{abs(dem)} Pallets)"
            current_onboard -= abs(dem)
        else:
            action = "🏁 DEPARTURE DEPOT" if idx == 0 else ("🏁 FINAL RETURN DEPOT" if idx == len(r['stops']) - 1 else "🔄 TRANSIT / DROP-OFF")

        activity_log.append({
            "Stop #": idx + 1,
            "Location": loc,
            "Elapsed Driving Time": f"+{mins} mins",
            "Est. Arrival Clock Time": clock_time,
            "Activity": action,
            "Pallets Changed": f"+{dem}" if dem > 0 else (f"{dem}" if dem < 0 else "0"),
            "Onboard Load After Stop": f"{current_onboard} / {r['v_cap']} Pallets"
        })

    st.dataframe(pd.DataFrame(activity_log), use_container_width=True)

# ==============================================================================
# 5. Main Execution Flow
# ==============================================================================
if file_loc and file_dem and file_fleet:
    df_loc = load_uploaded_file(file_loc)
    df_dem = load_uploaded_file(file_dem)
    df_fleet = load_uploaded_file(file_fleet)

    st.success("✅ Files uploaded successfully!")

    # Patch missing location coordinates if omitted in user inputs
    missing_locs = [
        {'Location_ID': 'GUNDAI', 'Latitude': 36.262843, 'Longitude': 139.223466},
        {'Location_ID': 'SANKO_KASEI', 'Latitude': 36.230514, 'Longitude': 139.159021},
        {'Location_ID': 'KASUGAI', 'Latitude': 36.227094, 'Longitude': 139.351622},
        {'Location_ID': 'NUKABE', 'Latitude': 36.243419, 'Longitude': 138.953182},
        {'Location_ID': 'TONEX', 'Latitude': 36.362337, 'Longitude': 139.259837},
        {'Location_ID': 'MARUNAKA', 'Latitude': 36.268254, 'Longitude': 139.217019},
        {'Location_ID': 'SAN_S', 'Latitude': 36.263706, 'Longitude': 139.221558}
    ]
    existing = set(df_loc['Location_ID'])
    new_locs = [loc for loc in missing_locs if loc['Location_ID'] not in existing]
    if new_locs:
        df_loc = pd.concat([df_loc, pd.DataFrame(new_locs)], ignore_index=True)

    time_matrix = compute_haversine_matrix(df_loc)

    st.subheader("⚙️ Solver Settings")
    start_option = st.radio(
        "Select Vehicle Dispatch Start Mode:",
        ["Option 1: User-Defined Starts", "Option 2: Default Base Start (OURA)"]
    )

    all_locations = df_loc['Location_ID'].tolist()
    custom_starts = []

    # Dynamic starting location selection for Option 1
    if "Option 1" in start_option:
        num_starts = st.number_input(
            "Enter Number of Starting Locations to Define:",
            min_value=1,
            max_value=10,
            value=4,
            step=1
        )

        st.markdown("##### 📍 Select Starting Locations:")
        default_defaults = ['OURA', 'HIDAKA', 'OGURA', 'NUKABE', 'OURA', 'KOHNAN', 'TONEX', 'MARUNAKA']
        cols = st.columns(min(int(num_starts), 5))

        for s_idx in range(int(num_starts)):
            default_loc = default_defaults[s_idx % len(default_defaults)]
            default_index = all_locations.index(default_loc) if default_loc in all_locations else 0

            with cols[s_idx % len(cols)]:
                selected_loc = st.selectbox(
                    f"Start Location #{s_idx + 1}:",
                    options=all_locations,
                    index=default_index,
                    key=f"custom_start_{s_idx}"
                )
                custom_starts.append(selected_loc)

    mode_key = 'USER_DEFINED' if "Option 1" in start_option else 'AUTOMATIC'

    if st.button("🚀 Run Optimization"):
        with st.spinner("Calculating optimal routes using Google OR-Tools..."):
            routes, total_time = solve_pdvrp_engine(df_loc, df_dem, df_fleet, time_matrix, mode_key, custom_starts)

        if routes:
            st.session_state['routes'] = routes
            st.session_state['total_time'] = total_time
            st.session_state['df_loc'] = df_loc
        else:
            st.error("No feasible solution found within vehicle capacity and driver time limits.")

    # Render results if present in session state
    if 'routes' in st.session_state:
        routes = st.session_state['routes']
        total_time = st.session_state['total_time']
        df_loc = st.session_state['df_loc']

        st.success(f"🎉 Optimization Complete! Active Fleet: {len(routes)} trucks | Total Driving Duration: {total_time} mins ({round(total_time / 60, 2)} hrs)")

        # 1. Combined Master Map
        st.subheader("🌐 Master Fleet Coverage Map")
        st.caption("Shows all vehicle trajectories together to verify all locations are included in the overall loop.")
        master_fig = render_combined_master_map(df_loc, routes)
        st.pyplot(master_fig)
        plt.close(master_fig)

        # 2. Summary Table with Elapsed Driving Timestamps
        st.subheader("📊 Fleet Dispatch Summary")
        summary_data = []
        for r in routes:
            sequence_with_times = " -> ".join([f"{s['loc_id']} (+{s.get('cumul_mins', 0)}m)" for s in r['stops']])
            summary_data.append({
                "Vehicle ID": r['v_code'],
                "Vehicle Type": r['v_type'],
                "Capacity (Pallets)": r['v_cap'],
                "Start Location": r['start_loc'],
                "Duration (Mins)": r['duration_mins'],
                "Route Sequence (Location & Arrival Elapsed Minutes)": sequence_with_times
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # 3. Individual Route Inspection Controls
        st.subheader("🔍 Inspect & Analyze Individual Vehicle Routes")
        st.caption("Click any button below to open a full-screen view and examine timed step-by-step load/unload activities.")

        cols = st.columns(min(len(routes), 3))
        for idx, r in enumerate(routes):
            with cols[idx % 3]:
                if st.button(f"🔍 Open Truck {r['truck_num']} ({r['v_code']}) Activity Modal", key=f"btn_modal_{r['truck_num']}"):
                    show_route_detail_modal(df_loc, r)

        # 4. Google Maps Integration
        st.subheader("📍 Google Maps Integration")
        kml_data = generate_kml_file(routes)
        st.download_button(
            label="🗺️ Download KML File for Google My Maps",
            data=kml_data,
            file_name="optimized_milk_run_routes.kml",
            mime="application/vnd.google-earth.kml+xml",
            help="Import this file into Google My Maps (mymaps.google.com) to view all routes together!"
        )

        st.markdown("##### 🔗 Direct Google Maps Navigation Links:")
        map_cols = st.columns(min(len(routes), 3))
        for idx, r in enumerate(routes):
            gmaps_url = generate_google_maps_url(r['stops'])
            with map_cols[idx % 3]:
                st.link_button(
                    f"🚚 Truck {r['truck_num']} ({r['v_code']}) Direct Route", 
                    gmaps_url
                )

else:
    st.info("👈 Please upload `locations_master`, `demands_flows`, and `fleet_master` CSV/Excel files using the sidebar.")
