import gc
import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ==============================================================================
# 1. Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Japan Milk Run Optimization",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Japan Milk Run - Route Optimization App")
st.write("แอปพลิเคชันจัดเส้นทางรถขนส่ง (VRP/Milkrun) คำนวณด้วย Google OR-Tools")

# ==============================================================================
# 2. Sidebar File Uploaders
# ==============================================================================
st.sidebar.header("📁 อัปโหลดไฟล์ Master Data")
file_loc = st.sidebar.file_uploader("1. locations_master", type=["csv", "xlsx"])
file_dem = st.sidebar.file_uploader("2. demands_flows", type=["csv", "xlsx"])
file_fleet = st.sidebar.file_uploader("3. fleet_master", type=["csv", "xlsx"])

def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# ==============================================================================
# 3. Helper Functions & Solver Engine
# ==============================================================================
def compute_haversine_matrix(df):
    coords = list(zip(df['Latitude'], df['Longitude']))
    n = len(coords)
    matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
                lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                matrix[i][j] = int(round(dist_km * 1.35 / 45.0 * 60))
    return matrix

def solve_pdvrp_engine(data, start_mode='AUTOMATIC', custom_start_list=None):
    df_loc, df_dem, df_fleet, time_matrix = data['df_loc'], data['df_dem'], data['df_fleet'], data['time_matrix']
    df_dem_sub = df_dem[df_dem['Schedule'].isin(['MWF', 'Daily', 'TT'])].reset_index(drop=True)
    loc_to_idx = {loc_id: idx for idx, loc_id in enumerate(df_loc['Location_ID'])}

    num_reqs = len(df_dem_sub)
    num_nodes = 1 + 2 * num_reqs
    num_vehicles = len(df_fleet)

    node_matrix_indices = [loc_to_idx['OURA']]
    node_demands = [0]
    for i, row in df_dem_sub.iterrows():
        qty = int(row['Pallets'])
        node_matrix_indices.append(loc_to_idx[row['Origin_Location_ID']])
        node_demands.append(qty)
        node_matrix_indices.append(loc_to_idx[row['Destination_Location_ID']])
        node_demands.append(-qty)

    ends = [0] * num_vehicles
    starts = []
    if start_mode == 'USER_DEFINED' and custom_start_list:
        for v_idx in range(num_vehicles):
            loc_name = custom_start_list[v_idx % len(custom_start_list)]
            target_m_idx = loc_to_idx.get(loc_name, 0)
            matched_node = 0
            for n_idx, m_idx in enumerate(node_matrix_indices):
                if m_idx == target_m_idx and node_demands[n_idx] >= 0:
                    matched_node = n_idx
                    break
            starts.append(matched_node)
    else:
        starts = [0] * num_vehicles

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_idx, to_idx):
        fn, tn = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return time_matrix[node_matrix_indices[fn]][node_matrix_indices[tn]]

    tc_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tc_idx)

    # Add Time Dimension (13 Hours = 780 Mins)
    routing.AddDimension(tc_idx, 0, 780, True, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')
    time_dim.SetGlobalSpanCostCoefficient(100)

    # Add Capacity Dimension
    def demand_cb(from_idx):
        return node_demands[manager.IndexToNode(from_idx)]

    dc_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    v_caps = df_fleet['Capacity_Pallets'].tolist()
    routing.AddDimensionWithVehicleCapacity(dc_idx, 0, v_caps, True, 'Capacity')

    # Pickup & Delivery Pair Constraints
    solver = routing.solver()
    for i in range(num_reqs):
        p_idx, d_idx = manager.NodeToIndex(2 * i + 1), manager.NodeToIndex(2 * i + 2)
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
        return None, 0

    active_routes, total_time = [], 0
    colors = ['#E63946', '#2A9D8F', '#F4A261', '#9C27B0', '#3F51B5']

    for v in range(num_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue
        v_code, v_type, v_cap = df_fleet.loc[v, 'Vehicle_ID'], df_fleet.loc[v, 'Vehicle_Type'], v_caps[v]
        stops = []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            m_idx = node_matrix_indices[node_idx]
            stops.append({
                'loc_id': df_loc.loc[m_idx, 'Location_ID'],
                'lat': df_loc.loc[m_idx, 'Latitude'],
                'lon': df_loc.loc[m_idx, 'Longitude'],
                'demand': node_demands[node_idx]
            })
            idx = sol.Value(routing.NextVar(idx))
        stops.append({'loc_id': 'OURA', 'lat': df_loc.loc[0, 'Latitude'], 'lon': df_loc.loc[0, 'Longitude'], 'demand': 0})
        duration = sol.Min(time_dim.CumulVar(routing.End(v)))
        total_time += duration
        active_routes.append({
            'truck_num': len(active_routes) + 1, 'v_code': v_code, 'v_type': v_type, 'v_cap': v_cap,
            'start_loc': stops[0]['loc_id'], 'duration_mins': duration, 'color': colors[len(active_routes) % len(colors)], 'stops': stops
        })
    return active_routes, total_time

def render_subplots(df_loc, routes, title):
    cols = 2
    rows = math.ceil(len(routes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes_flat = [axes] if len(routes) == 1 else axes.flatten()
    depot_mask = (df_loc['Location_ID'] == 'OURA')

    for i, r in enumerate(routes):
        ax = axes_flat[i]
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.scatter(df_loc.loc[~depot_mask, 'Longitude'], df_loc.loc[~depot_mask, 'Latitude'], color='#D0D0D0', s=300, zorder=2)
        ax.scatter(df_loc.loc[depot_mask, 'Longitude'], df_loc.loc[depot_mask, 'Latitude'], color='#FF2222', s=550, zorder=3)

        stops = r['stops']
        visited = set([s['loc_id'] for s in stops])
        loads = {}
        for s in stops:
            loc, dem = s['loc_id'], s['demand']
            if loc not in loads: loads[loc] = {'p': 0, 'd': 0}
            if dem > 0: loads[loc]['p'] += dem
            elif dem < 0: loads[loc]['d'] += abs(dem)

        v_lons = [df_loc[df_loc['Location_ID'] == l]['Longitude'].values[0] for l in visited if l != 'OURA']
        v_lats = [df_loc[df_loc['Location_ID'] == l]['Latitude'].values[0] for l in visited if l != 'OURA']
        ax.scatter(v_lons, v_lats, color='#3377FF', s=550, zorder=4)

        for k in range(len(stops) - 1):
            sx, sy, ex, ey = stops[k]['lon'], stops[k]['lat'], stops[k+1]['lon'], stops[k+1]['lat']
            dx, dy = ex - sx, ey - sy
            ax.plot([sx, ex], [sy, ey], color=r['color'], linewidth=2.5, alpha=0.9, zorder=5)
            ax.arrow(sx + dx*0.4, sy + dy*0.4, dx*0.1, dy*0.1, shape='full', lw=0, length_includes_head=True, head_width=0.008, color=r['color'], zorder=6)

        for l in visited:
            lon = df_loc[df_loc['Location_ID'] == l]['Longitude'].values[0]
            lat = df_loc[df_loc['Location_ID'] == l]['Latitude'].values[0]
            p, d = loads[l]['p'], loads[l]['d']
            if l == 'OURA':
                ax.text(lon + 0.005, lat + 0.003, "OURA [DEPOT END]", fontsize=8.5, weight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.2", fc="#FFE6E6", ec="red"))
            elif l == r['start_loc']:
                ax.text(lon + 0.004, lat + 0.002, f"START: {l}\n(+{p} load, -{d} unload)", fontsize=8, weight='bold', color='darkgreen', bbox=dict(boxstyle="square,pad=0.2", fc="#E6FFE6", ec="green"))
            else:
                ax.text(lon + 0.004, lat + 0.002, f"{l}\n(+{p} load, -{d} unload)", fontsize=8, weight='bold', bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="gray", alpha=0.85))

        ax.set_title(f"Truck {r['truck_num']}: {r['v_code']} ({r['v_type']})\n[START: {r['start_loc']} -> END: OURA] ({r['duration_mins']} mins)", fontsize=10, fontweight='bold', color=r['color'])

    for j in range(len(routes), len(axes_flat)): fig.delaxes(axes_flat[j])
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig

# ==============================================================================
# 4. Main App Execution Workflow
# ==============================================================================
if file_loc and file_dem and file_fleet:
    df_loc = load_uploaded_file(file_loc)
    df_dem = load_uploaded_file(file_dem)
    df_fleet = load_uploaded_file(file_fleet)

    st.success("✅ อัปโหลดไฟล์ข้อมูลเรียบร้อยแล้ว")

    # Patch missing coordinates if needed
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
    data = {'df_loc': df_loc, 'df_dem': df_dem, 'df_fleet': df_fleet, 'time_matrix': time_matrix}

    st.subheader("⚙️ ตัวเลือกการคำนวณ")
    start_option = st.radio(
        "เลือกโหมดกำหนดจุดเริ่มต้นของรถ:",
        ["Option 2: Program-Optimized Starts (คำนวณอัตโนมัติ)", "Option 1: User-Defined Starts (กำหนดจุดเริ่มเอง)"]
    )

    custom_starts = ['OURA', 'TSUBAKIMOTO', 'OGURA', 'NUKABE']
    mode_key = 'AUTOMATIC' if "Option 2" in start_option else 'USER_DEFINED'

    if st.button("🚀 คำนวณจัดเส้นทาง (Run Optimization)"):
        with st.spinner("กำลังคำนวณเส้นทางด้วย Google OR-Tools..."):
            routes, total_time = solve_pdvrp_engine(data, mode_key, custom_starts)

            if routes:
                st.success(f"🎉 คำนวณสำเร็จ! ใช้รถทั้งหมด {len(routes)} คัน | เวลาขับรวม: {total_time} นาที ({round(total_time/60, 2)} ชั่วโมง)")

                # Summary Table
                st.subheader("📊 ตารางสรุปเส้นทางแต่ละคัน")
                summary_data = []
                for r in routes:
                    sequence = " -> ".join([s['loc_id'] for s in r['stops']])
                    summary_data.append({
                        "Vehicle ID": r['v_code'],
                        "Vehicle Type": r['v_type'],
                        "Capacity (Pallets)": r['v_cap'],
                        "Start Location": r['start_loc'],
                        "Duration (Mins)": r['duration_mins'],
                        "Route Sequence": sequence
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

                # Subplot Chart Render
                st.subheader("🗺️ แผนที่แสดงเส้นทางรถขนส่ง (Subplots)")
                fig = render_subplots(df_loc, routes, f"Milkrun Optimization View ({start_option})")
                st.pyplot(fig)
                plt.close('all')
                gc.collect()
            else:
                st.error("ไม่สามารถจัดเส้นทางได้ตามเงื่อนไขความจุและเวลาที่กำหนด")

else:
    st.info("👈 กรุณาอัปโหลดไฟล์ CSV/Excel ทั้ง 3 ไฟล์ทางเมนูด้านซ้ายเพื่อเริ่มคำนวณ")
