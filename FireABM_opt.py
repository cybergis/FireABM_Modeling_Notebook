import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString
from collections import deque, Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.lines as mp_lines
from IPython.display import HTML
import networkx as nx
from networkx.utils import generate_unique_node
import random
import copy
import ipywidgets as widgets
from traitlets import traitlets
import os
import csv
import time
import math
import pytz
from datetime import datetime
from heapq import heappush, heappop
from itertools import count

# plt.rcParams['animation.writer']='avconv'
plt.rcParams['animation.writer'] = 'ffmpeg'

DEFAULT_VEHICLE_LENGTH = 3  # meters, couting from the head of the vehicle, no other vehicles after [length] meters
DEFAULT_ROAD_LANES = 1  # For viz puporses (and convenience), limiting road_lanes to 1. Different lanes could be modeled as different keys of the same i,j
DEFAULT_ROAD_SPEED = 15  # m/s, ~35 mph

time_zone = pytz.timezone('America/Chicago')
seed_number = None

def set_seeds(in_seed_number):  # sets seed for reproducibility, both python and numpy random generators used
    np.random.seed(in_seed_number)
    random.seed(in_seed_number)
    global seed_number
    seed_number = in_seed_number
    return seed_number

def check_seed():
    print(seed_number)

def gen_seeds(tot_size, sel_size, seed_number):
    seeds = []
    np.random.seed(seed_number)
    while len(seeds) < sel_size:
        r = np.random.randint(tot_size)
        if r not in seeds:
            seeds.append(r)
    return seeds

def time_stamp(start_time=None):
    print(datetime.now(time_zone).strftime("%H:%M:%S"))
    if start_time is None:
        return time.time()
    else:
        return abm_timer(start_time, time.time())

def abm_timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def setup_sim(g, seed=51, no_seed=False):
    fig, ax = ox.plot_graph(g, node_size=0, fig_height=15, show=False, margin=0)
    fig.tight_layout()
    if not no_seed:
        set_seeds(seed)
    return fig, ax

def str_replace_mult(string, rep_list):
    for r in rep_list:
        string = string.replace(*r)
    return string

def mph2ms(mph):
    return mph * 0.44704

def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def get_node_edge_gdf(g):
    nodes = ox.graph_to_gdfs(g, nodes=True, edges=False)
    edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
    return(nodes, edges)

def load_road_graph(pkl_file):
    g = nx.read_gpickle(path=pkl_file)
    return g

def create_bboxes(gdf_nodes, buffer_pct=0.15, buff_adj=None):
    if buff_adj:
        xposadj, xnegadj, yposadj, ynegadj = buff_adj[0], buff_adj[1], buff_adj[2], buff_adj[3]
    else:
        xposadj, xnegadj, yposadj, ynegadj = 1, 1, 1, 1
    # find buffer in xy coordinates to use in map
    xbuff = (max(gdf_nodes['x']) - min(gdf_nodes['x'])) * buffer_pct
    ybuff = (max(gdf_nodes['y']) - min(gdf_nodes['y'])) * buffer_pct
    bbox = [min(gdf_nodes['x']) + (xbuff * xposadj), min(gdf_nodes['y']) + (ybuff * yposadj),
        max(gdf_nodes['x']) - (xbuff * xnegadj), max(gdf_nodes['y']) - (ybuff * ynegadj)]

    # find buffer in lat/long to use in simulation
    latbuff = (max(gdf_nodes['lat']) - min(gdf_nodes['lat'])) * buffer_pct
    lonbuff = (max(gdf_nodes['lon']) - min(gdf_nodes['lon'])) * buffer_pct
    lbbox = [min(gdf_nodes['lon']) + (lonbuff * xposadj), min(gdf_nodes['lat']) + (latbuff * yposadj),
        max(gdf_nodes['lon']) - (lonbuff * xnegadj), max(gdf_nodes['lat']) - (latbuff * ynegadj)]

    # fill out coordinates into a rectangle and extract xy for plotting
    poly = shapely.geometry.box(*bbox)
    x, y = poly.exterior.xy
    return (bbox, lbbox, poly, x, y)

def check_graphs(gdf_edges, x=None, y=None, shpfile=None, is_fire=True, zorder=4, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    gdf_edges.plot(ax=ax)
    if x and y:
        ax.plot(x, y, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    if shpfile is not None:
        if is_fire:
            shpfile.plot(column="SimTime", ax=ax, zorder=zorder)
        else:
            shpfile.plot(ax=ax, cmap='Greens', zorder=zorder)
    return (fig, ax)

def inspect(g, nid, mid=None, radius=300, showMap=False, fullMap=True):  # Becky add g, showMap
    # if 'x' in g.nodes[nid]: # Becky added if statement, fix graph
    #     loc=(g.nodes[nid]['y'],g.nodes[nid]['x'])
    # else:
    #     loc=(g.nodes[nid]['lat'],g.nodes[nid]['lon'])  # Becky change y, x to lat, lon
    if nid is not None:
        loc = (g.nodes[nid]['lat'], g.nodes[nid]['lon'])  # Becky change y, x to lat, lon
        if fullMap:
            t = g
        else:
            t = ox.graph_from_point(loc, distance=radius, distance_type='bbox', network_type='drive')
        if not mid:
            nc = ['r' if node == nid else '#336699' for node in t.nodes()]
            ns = [50 if node == nid else 8 for node in t.nodes()]
            ox.plot_graph(t, node_size=ns, node_color=nc, node_zorder=2)
            if showMap:  # Becky add showMap logic
                return ox.plot_graph_folium(t)
        else:
            ec = ['r' if u == nid and v == mid else '#336699' for u, v, key, data in t.edges(keys=True, data=True)]
            es = [3 if u == nid and v == mid else 1 for u, v, key, data in t.edges(keys=True, data=True)]
            ox.plot_graph(t, node_size=30, edge_color=ec, edge_linewidth=es, edge_alpha=0.5)
            if showMap:  # Becky add showMap logic
                return ox.plot_graph_folium(t)

def highlight(g, edgelist, showMap=False):
    ec = ['r' if (u, v, key) in edgelist else '#336699' for u, v, key in g.edges(keys=True)]
    es = [3 if (u, v, key) in edgelist else 1 for u, v, key in g.edges(keys=True)]
    ox.plot_graph(g, node_size=30, edge_color=ec, edge_linewidth=es, edge_alpha=0.5)
    if showMap:
        return ox.plot_graph_folium(g)

class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

def select_nearest_node(g, show_info=False):
    global fig, ax
    fig, ax = ox.plot_graph(g, node_size=30, edge_alpha=0.5)

    global coords
    coords = [None, None]
    global near_node_id
    near_node_id = None

    def print_coords(button_inst):
        if show_info:
            print(coords)

    def print_node_id(button_inst):
        if show_info:
            print(near_node_id)

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        global coords
        coords = (ix, iy)
        revcords = (iy, ix)

        global near_node_id
        near_node_id = ox.geo_utils.get_nearest_node(g, revcords, method='euclidean')
        print(near_node_id)


    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    cords_button = LoadedButton(description='Get coordinates!')
    cords_button.on_click(print_coords)
    display(cords_button)  # noqa: F821

    node_button = LoadedButton(description='Get node!')
    node_button.on_click(print_node_id)
    display(node_button)  # noqa: F821


def view_node_attrib(g, attrib, show_null=False):
    if attrib == 'culdesacs':
        culdesacs = [key for key, value in g.graph['streets_per_node'].items() if value == 1]
        nc = ['r' if node in culdesacs else 'none' for node in g.nodes()]
    ox.plot_graph(g, node_color=nc)

def view_edge_attrib(g, attrib, fig_height=8, show_null=False, show_edge_values=False, edge_value_rm=None, show_val=False, val=None, num_bins=5, set_range=None, breaks=None, set_colors=None, node_size=5, cmap='viridis'):
    gdf_edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
    if attrib in gdf_edges:
        print('Attribute: ' + attrib + ', Type: ' + str(gdf_edges[attrib].dtype))
        elw = [1 for u, v, key, data in g.edges(keys=True, data=True)]

        legend_items = {'colors': [], 'names': []}
        if show_null:
            ec = ['red' if attrib not in data else 'grey' for u, v, key, data in g.edges(keys=True, data=True)]
            legend_items['colors'] = ['red', 'grey']
            legend_items['names'] = ['Null attribute', 'Attribute exists']

        elif set_range:
            if set_colors:
                colors = set_colors
            else:
                colors = ox.get_colors(len(set_range), cmap=cmap)
            ec = []
            elw = []
            for u, v, key, data in g.edges(keys=True, data=True):
                included = False
                for rindx, range_ele in enumerate(set_range):
                    if data[attrib] > range_ele[0] and data[attrib] < range_ele[1]:
                        ec.append(colors[rindx])
                        included = True
                        elw.append(3)
                if included is False:
                    ec.append('grey')
                    elw.append(1)

            legend_items['colors'] = colors
            legend_items['names'] = [str(rng[0]) + '-' + str(rng[1]) for rng in set_range]

        elif show_val:
            if isinstance(val, str):
                ec = ['red' if data[attrib] == val else 'grey' for u, v, key, data in g.edges(keys=True, data=True)]
                legend_items['colors'] = ['red', 'grey']
                legend_items['names'] = [val, 'not ' + str(val)]
            else:
                cats = list(range(len(val)))
                cat_names = val
                colors = ox.get_colors(len(val), cmap=cmap)
                ec = [colors[val.index(data[attrib])] if data[attrib] in val else 'grey' for u, v, key, data in g.edges(keys=True, data=True)]
                legend_items['colors'] = colors
                legend_items['names'] = list(cat_names)
                if len(colors) > len(cat_names):
                    legend_items['names'].append('Not in list')

        else:
            if gdf_edges[attrib].dtype in ['int64', 'float64', 'float', 'int'] and attrib is not 'key':
                print('min', round(gdf_edges[attrib].min(), 2), 'max', round(gdf_edges[attrib].max(), 2))
                ec, colors, bins = get_edge_colors_by_attr(g, attr=attrib, num_bins=num_bins, cmap=cmap, bin_cuts=breaks)
                # print('c and b', colors, bins)
                legend_items['colors'] = colors
                for indx in range(len(colors)):
                    leg_item = str(round(bins[indx], 2)) + " - " + str(round(bins[indx + 1], 2))
                    legend_items['names'].append(leg_item)

            elif gdf_edges[attrib].dtype == 'O' or attrib is 'key':
                edge_series = copy.deepcopy(gdf_edges[attrib])
                for index, edgs in enumerate(edge_series):
                    if type(edgs) == list:
                        edge_series[index] = edgs[0]
                cats = edge_series.astype("category").cat.codes
                cat_names = edge_series.astype("category").cat.categories
                colors = ox.get_colors(len(edge_series.unique()), cmap=cmap)
                ec = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
                legend_items['colors'] = colors
                legend_items['names'] = list(cat_names)
                if len(colors) > len(cat_names):
                    legend_items['names'].append('Null')

            else:
                try:
                    cats = gdf_edges[attrib].astype("category").cat.codes
                    cat_names = gdf_edges[attrib].astype("category").cat.categories
                    colors = ox.get_colors(len(gdf_edges[attrib].unique()), cmap=cmap)
                    ec = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
                    legend_items['colors'] = colors
                    legend_items['names'] = list(cat_names)
                    if len(colors) > len(cat_names):
                        legend_items['names'].append('Null')

                except:
                    ec = 'blue'
                    legend_items['colors'] = ['blue']
                    legend_items['names'] = ['Unable to parse variable']


        if show_edge_values:
            fig_height = 20

        fig, ax = ox.plot_graph(g, fig_height=fig_height, node_size=node_size, edge_color=ec, edge_alpha=0.5, edge_linewidth=elw, show=False, close=False)

        if show_edge_values:
            for index, row in gdf_edges.iterrows():
                if attrib in row:
                    edge_val_txt = str(row[attrib])
                    if edge_val_txt != 'nan':
                        if edge_value_rm:
                            for repl in edge_value_rm:
                                edge_val_txt = edge_val_txt.replace(repl, '')
                        plt.text(row.geometry.centroid.x, row.geometry.centroid.y, edge_val_txt, zorder=4)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
        proxies = [make_proxy(item, linewidth=5) for item in legend_items['colors']]
        ax.legend(proxies, legend_items['names'], loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        return (fig, ax)

    else:
        print('attribute not found in edges')

# adapted from https://stackoverflow.com/questions/19877666/add-legends-to-linecollection-plot/19881647#19881647
def make_proxy(color, **kwargs):
    return mp_lines.Line2D([0, 1], [0, 1], color=color, **kwargs)

def get_edge_colors_by_attr(g, attr, num_bins=5, cmap='viridis', start=0, stop=1, na_color='none', bin_cuts=None):  # overloaded osnmx function (support both continuous and non continuous vars)
    if num_bins is None:
        num_bins = len(g.edges())
    if bin_cuts:
        num_bins = len(bin_cuts) + 1

    bin_labels = range(num_bins)

    attr_values = pd.Series([data[attr] for u, v, key, data in g.edges(keys=True, data=True)])

    try:
        if not bin_cuts:
            cats, bins = pd.qcut(x=attr_values, q=num_bins, labels=bin_labels, retbins=True)
        else:
            print('bin_cuts!!', bin_cuts)
            cats, bins = pd.cut(x=attr_values, bins=bin_cuts, labels=bin_labels, retbins=True)
    except:  # added to support non continuous vars
        cats, bins = pd.cut(x=attr_values, bins=num_bins, labels=bin_labels, retbins=True)

    colors = ox.get_colors(num_bins, cmap, start, stop)
    edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]

    return edge_colors, colors, bins

def load_shpfile(g, path_list):
    in_file = gpd.read_file(os.path.join(*path_list))
    ox_crs = find_UTM_crs(g.nodes[list(g.nodes)[0]]['lat'], g.nodes[list(g.nodes)[0]]['lon'])
    prj_file = in_file.to_crs(ox_crs)
    return prj_file

def show_fire(g, fire_perim, show_graph=True, sim_time_num=None):  # Becky added
    if sim_time_num:
        fire_perim = fire_perim[fire_perim["SimTime"] == sim_time_num]
    if show_graph is True:
        fig, ax = ox.plot_graph(g, node_size=30, edge_alpha=0.5, show=False, axis_off=False, close=False)
        fire_perim.plot(ax=ax, cmap='YlOrRd', alpha=0.1, zorder=4)
        fire_perim.boundary.plot(ax=ax, color='grey', alpha=0.75, zorder=4)
    else:
        fire_perim.plot(cmap='YlOrRd', alpha=0.2, edgecolor='grey')
    plt.show()

def show_shpfile(g, shpfile, show_graph=True, is_fire=True, sim_time_num=None):  # Becky added
    if is_fire:
        if sim_time_num:
            shpfile = shpfile[shpfile["SimTime"] == sim_time_num]
    if show_graph is True:
        fig, ax = ox.plot_graph(g, node_size=30, edge_alpha=0.5, show=False, axis_off=False, close=False)
        shpfile.plot(ax=ax, cmap='YlOrRd', alpha=0.1, zorder=4)
        shpfile.boundary.plot(ax=ax, color='grey', alpha=0.75, zorder=4)
    else:
        shpfile.plot(cmap='YlOrRd', alpha=0.2, edgecolor='grey')
    plt.show()

def convert_fire_time(fire_perim, spread_num=None, sim_time_num=None, length=False):
    fire_list = sorted(list(set(fire_perim["SimTime"])))
    if spread_num:
        if spread_num < len(fire_list):
            return fire_list[spread_num]
        else:
            print('index out of range for fire_list')
    elif sim_time_num:
        if sim_time_num in fire_list:
            return fire_list.index(sim_time_num)
        else:
            print('fire_list not in index')
    elif length:
        return len(fire_list)
    else:
        return fire_list

def setup_graph(g):  # Becky added
    fig, ax = ox.plot_graph(g, node_size=0, fig_height=15, show=False, margin=0)
    fig.tight_layout()
    return fig, ax

def resolve_deadend(g):
    deadEnds = [e for e in g.edges(keys=True, data=True) if len(g.adj[e[1]]) == 0]  # Becky update fix
    for d in deadEnds:
        g.add_edge(d[1], d[0], key=0)
    return g

def fillPhantom(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' not in e[3]:
            u = e[0]
            v = e[1]
            e[3]['geometry'] = LineString([Point(g.nodes[u]['x'], g.nodes[u]['y']), Point(g.nodes[v]['x'], g.nodes[v]['y'])])
    return g

def adjustLength(g):
    for e in g.edges(keys=True, data=True):
        if 'geometry' in e[3]:
            e[3]['length'] = e[3]['geometry'].length
    return g

def add_unit_speed(g):  # Becky add for routing
    for e in g.edges(keys=True, data=True):
        if 'maxspeed' in e[3]:
            s = e[3]['maxspeed']
            if type(s) == list:
                s = s[0]
            # sm = mph2ms(int(s[:2]))
            e[3]['speed'] = mph2ms(int(s[:2]))  # -> Quickest
            if e[3]['speed'] > 0:
                e[3]['seg_time'] = e[3]['length'] / e[3]['speed']
            else:
                e[3]['seg_time'] = e[3]['length'] / DEFAULT_ROAD_SPEED
                e[3]['speed'] = DEFAULT_ROAD_SPEED
        else:
            e[3]['seg_time'] = e[3]['length'] / DEFAULT_ROAD_SPEED
            e[3]['speed'] = DEFAULT_ROAD_SPEED

        e[3]['ett'] = e[3]['length'] / e[3]['speed']  # -> Quickest
    return g

def add_road_type_weights(g, rt_weights=[1, 1, 5, 10, 15, 20, 20]):  # Becky add for routing
    # https://wiki.openstreetmap.org/wiki/Key:highway
    for e in g.edges(keys=True, data=True):
        if 'highway' in e[3]:
            rt = e[3]['highway']
            if rt in ['motorway', 'motorway_link']:
                e[3]['rt_weight'] = rt_weights[0]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[0]
            elif rt in ['trunk', 'trunk_link']:
                e[3]['rt_weight'] = rt_weights[1]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[1]
            elif rt in ['primary', 'primary_link']:
                e[3]['rt_weight'] = rt_weights[2]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[2]
            elif rt in ['secondary', 'secondary_link']:
                e[3]['rt_weight'] = rt_weights[3]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[3]
            elif rt in ['tertiary', 'tertiary_link']:
                e[3]['rt_weight'] = rt_weights[4]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[4]
            elif rt == 'unclassified':
                e[3]['rt_weight'] = rt_weights[5]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[5]
            elif rt == 'residential':
                e[3]['rt_weight'] = rt_weights[6]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[6]
            else:
                e[3]['rt_weight'] = rt_weights[5]
                e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[5]
        else:
            e[3]['rt_weight'] = rt_weights[5]
            e[3]['rt_weighted_len'] = e[3]['length'] * rt_weights[5]
    return g

def add_households(g, hh_shp, shp_name, hh_col_name, cut_off_len=10, num_col=10, bbox_poly=None):  # Becky add for placing vehicles

    gdf_edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
    edges_with_hhs = gpd.sjoin(gdf_edges, hh_shp, how="inner", op='intersects')

    name_list = [shp_name[:cut_off_len - (len(str(i)) + 1)] + "_" + str(i) for i in range(1, num_col)]
    name_list.append(shp_name[:cut_off_len])

    for col_name in [d_name for d_name in name_list if d_name in edges_with_hhs.columns and d_name != hh_col_name]:
        edges_with_hhs = edges_with_hhs.drop([col_name], axis=1)

    hh_dict = {}
    tract_dict = {}

    for i in list(zip(edges_with_hhs['u'], edges_with_hhs['v'], edges_with_hhs['key'], edges_with_hhs[hh_col_name])):
        hh_dict[(i[0], i[1], i[2])] = i[3]

    for i in list(zip(edges_with_hhs['u'], edges_with_hhs['v'], edges_with_hhs['key'], edges_with_hhs['NAME'])):
        tract_dict[(i[0], i[1], i[2])] = i[3]

    nx.set_edge_attributes(g, hh_dict, 'Tot_Est_HH_uncpd')
    nx.set_edge_attributes(g, tract_dict, 'Tract_name')

    return_list = [g]

    if bbox_poly:

        overlap = hh_shp['geometry'].intersection(bbox_poly)
        hh_shp['full_tract_area'] = hh_shp.geometry.area
        hh_shp_pts = hh_shp.copy()

        overlap_poly = overlap[~overlap.is_empty].copy()
        ovlp_gdf = gpd.GeoDataFrame(overlap_poly, geometry=overlap_poly)
        ovlp_gdf["cp_area"] = ovlp_gdf.geometry.area

        hh_shp_pts['geometry'] = ovlp_gdf['geometry'].centroid
        ovlp_gdf_join = gpd.sjoin(ovlp_gdf, hh_shp_pts, how="left", op='intersects')
        ovlp_gdf_join['Est_Area_Cpd_Ratio'] = ovlp_gdf_join['cp_area'] / ovlp_gdf_join['full_tract_area']

        ovlp_pts = ovlp_gdf.copy()
        ovlp_pts['geometry'] = ovlp_pts['geometry'].centroid

        cpd_hh_join = gpd.sjoin(hh_shp, ovlp_pts, how="left", op='intersects')
        cpd_hh_join['Est_Area_Cpd_Ratio'] = cpd_hh_join['cp_area'] / cpd_hh_join.geometry.area

        cpd_hh_join_sel = cpd_hh_join[['NAME', 'Est_Area_Cpd_Ratio']]
        edges_with_hhs_cpd = edges_with_hhs.merge(cpd_hh_join_sel, on='NAME')

        edges_with_hhs_cpd['Tot_Est_HH_cpd'] = edges_with_hhs_cpd[hh_col_name] * edges_with_hhs_cpd['Est_Area_Cpd_Ratio']
        edges_with_hhs_cpd['HH_Cpd_pct'] = edges_with_hhs_cpd['Tot_Est_HH_cpd'] / edges_with_hhs_cpd['Tot_Est_HH_cpd'].sum()
        extra_prc = edges_with_hhs_cpd.loc[0]['HH_Cpd_pct'] + (1 - edges_with_hhs_cpd['HH_Cpd_pct'].sum())
        edges_with_hhs_cpd.loc[0, 'HH_Cpd_pct'] = extra_prc
        # print(edges_with_hhs_cpd['HH_Cpd_pct'].sum())

        cpt_dict = {}
        tot_hh_cpt_dict = {}
        tot_hh_cpt_pct_dict = {}

        for i in list(zip(edges_with_hhs_cpd['u'], edges_with_hhs_cpd['v'], edges_with_hhs_cpd['key'], edges_with_hhs_cpd['Est_Area_Cpd_Ratio'])):
            cpt_dict[(i[0], i[1], i[2])] = i[3]

        for i in list(zip(edges_with_hhs_cpd['u'], edges_with_hhs_cpd['v'], edges_with_hhs_cpd['key'], edges_with_hhs_cpd['Tot_Est_HH_cpd'])):
            tot_hh_cpt_dict[(i[0], i[1], i[2])] = i[3]

        for i in list(zip(edges_with_hhs_cpd['u'], edges_with_hhs_cpd['v'], edges_with_hhs_cpd['key'], edges_with_hhs_cpd['HH_Cpd_pct'])):
            tot_hh_cpt_pct_dict[(i[0], i[1], i[2])] = i[3]

        nx.set_edge_attributes(g, cpt_dict, 'Est_Area_Cpd_Ratio')
        nx.set_edge_attributes(g, tot_hh_cpt_dict, 'Tot_Est_HH_Cpd')
        nx.set_edge_attributes(g, tot_hh_cpt_pct_dict, 'Pct_HH_Cpd')

        gdf_edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
        return_list.append(ovlp_gdf_join)

    return return_list

def set_test_hh_ratio(g, rat_dict):
    for e in g.edges(keys=True, data=True):
        if 'Tract_name' in e[3]:
            if e[3]['Tract_name'] in rat_dict.keys():
                e[3]['Test_Cpd_Ratio'] = rat_dict[e[3]['Tract_name']]
            else:
                e[3]['Test_Cpd_Ratio'] = 0.0
    return g

def poly_to_gdf(poly):
    poly_gdf = gpd.GeoDataFrame([1], geometry=[poly])
    return poly_gdf

def adj_speed_value(g, populate=False, overwrite=False):  # Becky Add for routing
    for e in g.edges(keys=True, data=True):
        if 'maxspeed' in e[3]:
            s = e[3]['maxspeed']
            if type(s) == list:
                s = s[0]
            sa = int(s[:2])
            e[3]['adj_speed'] = sa
        else:
            if populate:
                if 'highway' in e[3]:
                    if e[3]['highway'] == 'road':
                        e[3]['adj_speed'] = 0
                else:
                    e[3]['adj_speed'] = DEFAULT_ROAD_SPEED
            else:
                e[3]['adj_speed'] = 0
    return g

def add_fire_distance(g, fire_df, norm=False, inv=False):
    # print('adding fire distance, norm', norm, 'inv', inv)
    for e in g.edges(keys=True, data=True):
        if 'geometry' in e[3]:
            s = e[3]['geometry'].distance(fire_df.unary_union)
            e[3]['fire_dist'] = s
        else:
            e[3]['fire_dist'] = None
    if norm:
        gdf_edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
        g = normalize_edge_attribute(g, 'fire_dist', min(gdf_edges.fire_dist),
                                     max(gdf_edges.fire_dist), 'fire_dist_n')
    if inv:
        g = invert_norm_edge_attribute(g, 'fire_dist_n', 'inv_fire_dist_n')
    return g

def normalize_edge_attribute(g, attr, att_min, att_max, new_name):
    for e in g.edges(keys=True, data=True):
        if attr in e[3]:
            e[3][new_name] = (e[3][attr] - att_min) / (att_max - att_min)
        else:
            e[3][new_name] = None
    return g

def invert_norm_edge_attribute(g, attr, new_name):
    for e in g.edges(keys=True, data=True):
        if attr in e[3]:
            e[3][new_name] = max(min((1 - e[3][attr]), 1), 0)
        else:
            e[3][new_name] = None
    return g

def combine_attribute(g, attrs, weights, new_name):
    # print('combining', str(attrs), 'at weights', str(weights), 'to', new_name)
    for e in g.edges(keys=True, data=True):
        if all(attr in e[3] for attr in attrs):
            e[3][new_name] = sum([(weights[i] * e[3][a]) for i, a in enumerate(attrs)])
        else:
            e[3][new_name] = None
    return g

def cleanUp(g):
    return adjustLength(fillPhantom(g))

def project_lat_lon(lat, lon):
    return ox.project_geometry(Point(lon, lat))[0].coords[0]

# Becky added function
def project_UTM(y, x, crs):
    return ox.project_geometry(Point(y, x), crs=crs, to_latlong=True)[0].coords[0]

# Becky added function
def find_UTM_crs(lat, lon):
    return ox.project_geometry(Point(lon, lat))[1]

def isNodeInBbox(node, bbox):
    return bbox[0][0] < node['x'] < bbox[1][0] and bbox[0][1] < node['y'] < bbox[1][1]

def view_path(g, start_point, exit_point, strategy=[], showMap=False, norm=True):  # Becky added, show path choosen by driving strategy
    path_list = []

    if strategy == []:
        sweight = 'length'
        path = nx.shortest_path(g, source=start_point, target=exit_point, weight=sweight)
        if showMap:
            ox.plot.plot_graph_route(g, path)
        path_list.append(['dist', path])

    else:

        colors = ox.get_colors(len(strategy))
        rc = [colors[strategy.index(x)] for x in strategy for y in x]
        for rs in strategy:
            if rs == 'dist':
                if norm:
                    sweight = 'length_n'
                else:
                    sweight = 'length'
            elif rs == 'dist+speed':
                if norm:
                    print('not implemented')
                else:
                    sweight = 'seg_time'
            elif rs == 'dist+road_type_weight':
                if norm:
                    sweight = 'rt_wght_len_n'
                else:
                    sweight = 'rt_weighted_len'
            elif rs == 'dist+from+fire':
                if norm:
                    sweight = 'inv_fire_dist_n'
                else:
                    print('not implemented')
            elif rs == 'fire+dist':
                if norm:
                    sweight = 'fire_leng_n'
                else:
                    print('not implemented')
            elif rs == 'fire+rdty_weight':
                if norm:
                    sweight = 'fire_rd_wght_n'
                else:
                    print('not implemented')
            
            path = nx.shortest_path(g, source=start_point, target=exit_point, weight=sweight)
            if showMap:
                ox.plot.plot_graph_route(g, path, route_color=rc)
            path_list.append([rs, path])

    return path_list

def strategy_opts():
    ops = ['dist', 'dist+speed', 'dist+road_type_weight',
           'dist+from+fire', 'fire+dist', 'fire+rdty_weight']
    return ops

def convt_strategy_opts_to_weights(ops):
    for idx, strat in enumerate(ops):
        if strat == 'dist':
            # ops[idx] = 'length'
            ops[idx] = 'length_n'
        elif strat == 'dist+speed':
            # ops[idx] = 'seg_time'
            ops[idx] = 'seg_time_n'
        elif strat == 'dist+road_type_weight':
            # ops[idx] = 'rt_weighted_len'
            ops[idx] = 'rt_wght_len_n'
        elif strat == 'dist+from+fire':
            ops[idx] = 'inv_fire_dist_n'
        elif strat == 'fire+dist':
            ops[idx] = 'fire_leng_n'
        elif strat == 'fire+rdty_weight':
            ops[idx] = 'fire_rd_wght_n'

    return ops

def compare_paths(g, paths, strategies=None, showMap=False):  # Becky added, show path choosen by driving strategy
    colors = ox.get_colors(len(paths))

    rc = [colors[paths.index(x)] for x in paths for y in x]
    nc = [val for val in colors for _ in (0, 1)]

    # plot the routes
    if showMap:
        fig, ax = ox.plot_graph_routes(g, paths, route_color=rc, orig_dest_node_color=nc, node_size=0)

    if strategies:
        lengths = [nx.shortest_path_length(g, p, strategies[i]) for (i, p) in enumerate(paths)]
        return lengths


############################################################################
############################################################################

class Road:
    def __init__(self, g, node_i, node_j, key, attr, lanes=DEFAULT_ROAD_LANES, speed=DEFAULT_ROAD_SPEED):
        self.g=g
        self.node_i = node_i
        self.node_j = node_j
        self.key = key
        self.idx = (node_i, node_j, key)
        self.attr = attr
        self.geom = self.attr['geometry']
        self.length = self.geom.length
        self.vehicles = deque()
        self.requests = []
        self.isBlocked = False
        if 'maxspeed' in self.attr:
            s=self.attr['maxspeed']
            if type(s)==list:
                s=s[0]
            self.speed=mph2ms(int(s[:2]))
        else:
            self.speed=speed*(np.random.random()*2+0.5) # 0.5-2.5 of inital speed
        self.normal_speed = self.speed
        assert self.speed>=0

        ## -> quickest time
        self.checkins = {} # the entry time and position of current vehicles
        self.ett = {} # estimated travel time based on exit vehicles, at time t, ett updated by v
        ## <- quickest time 
        
        if 'Tract_name' in self.attr:
            self.tract = self.attr['Tract_name']
        else:
            self.tract = None
            
        if 'Test_Cpd_Ratio' in self.attr:
            self.test_hh_ratio = self.attr['Test_Cpd_Ratio']
            if self.test_hh_ratio is None:
                self.test_hh_ratio = 0 
        else:
            self.test_hh_ratio = 0
            
        if 'Pct_HH_Cpd' in self.attr:
            self.hh_ratio = self.attr['Pct_HH_Cpd']
            if self.hh_ratio is None:
                self.hh_ratio = 0 
        else:
            self.hh_ratio = 0

    ## -> quickest time
    #ett = length/speed
    def check_in(self, vid, pos, frame_number):
        #print('checkin vid:', vid, 'fn:', frame_number, 'rdidx:', self.idx)
        self.checkins[vid]=(pos, frame_number)

    def check_out(self, vid, last_pos, frame_number):
        assert vid in self.checkins
        delta = last_pos - self.checkins[vid][0] #Last pos - first pos
        if abs(delta) > 1e-7:
            #print('checkout vid:', vid, 'fn:', frame_number, 'first pos', self.checkins[vid][1], 'rdidx:', self.idx, 'delta:', delta)
            self.ett[frame_number] = ((frame_number - 1 - self.checkins[vid][1]) / delta * self.length, vid, delta/(frame_number - 1 - self.checkins[vid][1]))
            #print('full ett: ', self.ett)
        del self.checkins[vid]
    ## <- quickest time
        
    def set_block(self): # set to block road by fire spread
        self.isBlocked = True
        
    def mutate_block(self, prob=0.1):
        if not self.isBlocked:
            if np.random.random() < prob:
            #if not self.isBlocked:
                self.isBlocked = True
                return True
        return False
            #else:
            #    self.isBlocked = False
            #    collection.remove(self.idx)
        
    def mutate_speed(self, prob=0.9):
        if np.random.random() < prob: # stay unchanged for most of time
            if self.normal_speed == self.speed:
                self.speed = 1.0 if np.random.random() > 0.5 else 30 # Road mutate
            else:
                self.speed = self.normal_speed # Road restored
        
    def show(self, vehicles=None):
        if vehicles is None:
            vehicles = self.vehicles
        return shapely.geometry.GeometryCollection([self.geom]+[self.geom.interpolate(_.pos) for _ in vehicles])
    
    def tail_space(self):
        if len(self.vehicles) == 0:
            return self.length
        return max(self.vehicles[-1].pos - self.vehicles[-1].length, 0)
    #def remain_space(self):
    #    return self.length-sum(_.length for _ in self.vehicles)
    
    def add_vehicle(self, v):
        if self.tail_space() <= 0:
            return False
        if v.pos is 0:
            v.pos = np.random.random()*self.tail_space() ##!!! This is not true random
        self.vehicles.append(v)
        return True
    
    #def adjust(self):
    #    n=len(self.vehicles)
    #    if n == 0:
    #        return
    #    poses = np.random.random(n)*self.remain_space()
    #    poses.sort()
    #    acc_length = 0
    #    for i in range(n):
    #        acc_length += self.vehicles[n-1-i].length
    #        self.vehicles[n-1-i].pos=poses[i]+acc_length
            
    def move(self, frame_number, timestep=1):
        if len(self.vehicles) == 0:
            return

        distance = self.speed * timestep
        previous = None
        for v in self.vehicles:
            v.new_pos = v.pos + distance
            
            if previous is None: # first car
                if v.new_pos > self.length:  # need transit
                    if v.next_road is None or v.next_road.isBlocked: # in dead end or blocked road, wait at the beginning
                        v.new_pos = self.length 
                        v.last_move = v.new_pos - v.pos
                    else:
                        v.next_road.request_join(v.new_pos - self.length, v, self, timestep, frame_number) # timestep instead of frame_number                                         
                else:
                    v.last_move = distance
            else:
                v.new_pos = min(v.new_pos, previous)
                v.last_move = v.new_pos - v.pos
                
            previous = v.new_pos - v.length
    
    def request_join(self, momentum, v, from_road, timestep, frame_number):
        self.requests.append((momentum, v, from_road, timestep, frame_number))
    
    def resolve_requests(self, frame_number): # being called after all moves() done 
        if len(self.requests) == 0:
            return
        self.requests.sort()
        if len(set(_[1].vid for _ in self.requests)) < len(self.requests):
            raise ValueError('Duplicate vehicle in requests')
            
        while self.tail_space() > 0 and len(self.requests) > 0: # receive new vehicles as much as possible
            momentum, v, from_road, timestep, frame_number = self.requests.pop()
            from_road.check_out(v.vid, v.pos, frame_number) # quickest path
            v.new_pos = min(self.tail_space(), momentum)
            v.last_move = v.new_pos + from_road.length - v.pos
            v.pos = v.new_pos
            from_road.vehicles.popleft()
            self.check_in(v.vid, v.pos, frame_number) 
            self.vehicles.append(v)
            v.road = self
            v.trajectory.append((self.idx, frame_number))
            v.choose_next(frame_number)
            
        for momentum, v, from_road, timestep, frame_number in self.requests: # Those failed to udpate
            v.new_pos = from_road.length # wait at the top of their original road
            v.last_move = v.new_pos - v.pos
        
        self.requests=[]
     
    def sync_pos(self):
        for v in self.vehicles:
            v.pos = v.new_pos
            
    def report_veh_num(self):
        if self.length > 0:
            return (len(self.vehicles), len(self.vehicles)/self.length)
        else:
            return (len(self.vehicles), None)
        
class Vehicle:
    def __init__(self, g, vid, road=None, length=DEFAULT_VEHICLE_LENGTH, pos=0, target=None, st_weight='length'):
        self.g=g
        self.vid=vid
        self.road = road
        self.timer=0
        self.length=length
        self.pos=pos
        self.target=target
        self.st_weight = st_weight
        self.goal_time = None
        self.trajectory = []
        if target is None:
            self.isStuck = True
        else:
            self.isStuck = False
        self.set_route_times = 0
        self.is_clear = False
        self.init_paths = None
        #if self.road:
        #    self.addTo(self.road)
        #    self.navigate()
            
        #if self.road:
        #    self.addTo(self.road, self.pos)
        #    if not pos:
        #        self.pos=np.random.random()*self.road.length
    
    def choose_target(self, target_list=None):
        #print('choose_target navigate seting routes')
        if target_list is None:
            if type(self.target) == list:
                target_list = self.target
            else:
                self.isStuck = True
                return
            
        ans = [float('inf'), None, None] # length, target, path
        for t in target_list:
            #print('choose_target find targets')
            if nx.has_path(self.g, self.road.node_j, t):
                ### Shortest Path
                #print('path weight,', self.st_weight,'is used')
                paths = nx.shortest_path(self.g, self.road.node_j, t, weight=self.st_weight)
                if self.st_weight == 'dist': # Becky added if statements to determine overall shortest path by adjusted weight
                    #length = sum(self.g.adj[paths[i]][paths[i+1]][0]['length'] for i in range(len(paths)-1))
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['length_n'] for i in range(len(paths)-1)) 
                elif self.st_weight == 'dist+speed':
                    #length = sum(self.g.adj[paths[i]][paths[i+1]][0]['seg_time'] for i in range(len(paths)-1))
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['seg_time_n'] for i in range(len(paths)-1))
                elif self.st_weight == 'dist+road_type_weight':
                    #length = sum(self.g.adj[paths[i]][paths[i+1]][0]['rt_weighted_len'] for i in range(len(paths)-1))
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['rt_wght_len_n'] for i in range(len(paths)-1))
                elif self.st_weight == 'dist+from+fire':
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['inv_fire_dist_n'] for i in range(len(paths)-1))
                elif self.st_weight == 'fire+dist':
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['fire_leng_n'] for i in range(len(paths)-1))
                elif self.st_weight == 'fire+rdty_weight':
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['fire_rd_wght_n'] for i in range(len(paths)-1))
                elif self.st_weight == 'quickest':
                    #print('quickest weight, ett is used')
                    length = sum(self.g.adj[paths[i]][paths[i+1]][0]['ett'] for i in range(len(paths)-1))
                else:
                    print('BAD weight key!!! :', self.st_weight)
                
                if length < ans[0]:
                    ans[0] = length
                    ans[1] = t
                    ans[2] = paths        
                
        if ans[1] is None:
            self.isStuck = True
        else:
            self.goal = ans[1]
            self.routes = ans[2]
            self.paths = set((self.routes[i],self.routes[i+1],0) for i in range(len(self.routes)-1))
            self.route_pos = 1
            #print('!!', self.vid, 'rd', self.road.idx, 'route', self.paths, 'end', self.goal)
            self.set_route_times += 1
    
    def addTo(self, road, init=False):
        if road.isBlocked == False: #Becky don't add vehicle to blocked road
            if road.add_vehicle(self):
                road.check_in(self.vid, self.pos, 0) ## -> quickest path
                self.road=road
                if not init:
                    self.trajectory.append((self.road.idx, frame_number))
                else:
                    self.trajectory.append((self.road.idx, 0))
                if self.pos > self.road.length:
                    self.pos = self.road.length
                if self.st_weight: # becky added for routing weight
                    self.navigate(weight=self.st_weight)
                else:
                    self.navigate()
                return True
            else:
                return False
        else:
            return False
    
    def navigate(self, frame_number=None, target=None, weight=None): # return true if stucked after navigation
        #print('navigate, weight', weight)
        if weight is not None:
            #print('v', self.vid, 'weight', weight)
            if weight == 'dist': #Becky added
                #weight = 'length'
                weight = 'length_n'
            elif weight == 'dist+speed':
                #weight = 'seg_time'
                weight = 'seg_time_n'
            elif weight == 'dist+road_type_weight':
                #weight = 'rt_weighted_len'
                weight = 'rt_wght_len_n'
            elif weight == 'dist+from+fire':
                weight = 'inv_fire_dist_n'
            elif weight == 'fire+dist':
                weight = 'fire_leng_n'
            elif weight == 'fire+rdty_weight':
                weight = 'fire_rd_wght_n'
            elif weight == 'quickest':
                weight = 'ett'    
        else:
            weight = 'length_n'
        if target is None:
            target = self.target
        if type(target)==list:
            #print('choosing target')
            self.choose_target(target)            
        elif target is None or (not nx.has_path(self.g, self.road.node_j, target)):
            self.isStuck = True
            self.trajectory.append('STUCK')
        else:
            print('navigate seting routes')
            ### Shortest Path
            self.routes = nx.shortest_path(self.g,self.road.node_j,target,weight=weight)
            self.paths = set((self.routes[i],self.routes[i+1],0) for i in range(len(self.routes)-1))
            self.route_pos = 1
            
        self.choose_next(frame_number)
        return self.isStuck
        
    def handle_blocks(self, blocked_roads, target=None):
        if self.isStuck:
            return
        
        if any(r in self.paths for r in blocked_roads):
            return self.navigate(weight=self.st_weight, target=target)
        
    def choose_next(self, frame_number):
        if self.isStuck:
            self.next_road = None # Freeze if cannot reach destination
            
            #if len(self.road.nexts) == 0:
                #print 'Vehicle %4d entered dead end: (%d, %d, %d)'%(self.vid, self.road.node_i, self.road.node_j, self.road.key)
            #    self.next_road = None
            #else:
            #    self.next_road = np.random.choice(self.road.nexts.values())
        else:
            if self.road.node_j == self.goal:
                self.next_road = None # reach destination
                self.length = 0 # "disappear" after reaching destination
                self.goal_time = frame_number
                self.trajectory.append(('GOAL', self.road.node_j, self.goal_time))
                self.is_clear = True
            else:
                try:
                    self.next_road = self.road.nexts[(self.routes[self.route_pos],0)] # Go to the zero-indexed key since nx doesn't return key for shortest path
                except KeyError:
                    print(self.routes, self.route_pos, self.road.nexts)
                else:
                    self.route_pos += 1

    def xy(self, out=None):
        if out:
            return str(self.road.geom.interpolate(self.pos).coords[0][0])+";"+str(self.road.geom.interpolate(self.pos).coords[0][1])
        else:
            return self.road.geom.interpolate(self.pos).coords[0]
    
class NetABM():
    
    def __init__(self, g, n, bbox=None, fire_perim=None, fire_ignit_time=None, fire_act_ts_min=60, fire_des_ts_sec=10, sim_type="main", sim_number=0, reset_interval=False, start_vehicle_positons=None, nav_weight_list=None, placement_prob=None, init_strategies=None):
        self.g=g
        #self.g2=copy.deepcopy(g)
        #self.g3=copy.deepcopy(g)
        #self.g4=copy.deepcopy(g)
        self.roads={(_[0],_[1],_[2]):Road(g, *_) for _ in g.edges(keys=True, data=True) if 'geometry' in _[3]} 
        #self.roads2=copy.deepcopy(self.roads)
        #print('num roads 1', len(self.roads), len(self.roads2))
        
        self.bbox = bbox # added by Becky
        self.project_bbox=(project_lat_lon(bbox[1],bbox[0]),project_lat_lon(bbox[3],bbox[2]))
        self.targets=[e[1] for e in g.edges(keys=True,data=True) if isNodeInBbox(g.nodes[e[0]], self.project_bbox) and not isNodeInBbox(g.nodes[e[1]], self.project_bbox)]
        
        self.roads_in_bbox = [r for r in self.roads.values() if isNodeInBbox(g.nodes[r.node_i], self.project_bbox) or isNodeInBbox(g.nodes[r.node_j], self.project_bbox)]
        if len(self.roads_in_bbox) == 0:
            raise ValueError("No roads in the given bbox")
            
        self.in_roads_prob = np.array([r.length for r in self.roads_in_bbox])
        self.in_roads_prob/= sum(self.in_roads_prob)
        self.in_roads_prob[0] += 1.0-sum(self.in_roads_prob)
        
        if placement_prob is not None: # sets road probabilities by household ratio
            if placement_prob == 'Pct_HH_Cpd':
                self.in_roads_prob = self.in_roads_prob * [r.hh_ratio for r in self.roads_in_bbox]
                self.in_roads_prob[np.isnan(self.in_roads_prob)] = 0
                self.in_roads_prob/= sum(self.in_roads_prob)
            else:
                self.in_roads_prob = self.in_roads_prob * [r.test_hh_ratio for r in self.roads_in_bbox]
                self.in_roads_prob[np.isnan(self.in_roads_prob)] = 0
                self.in_roads_prob/= sum(self.in_roads_prob)
            
            self.in_roads_prob[0] += 1.0-sum(self.in_roads_prob)
         
        self.closed_roads = set()
        self.last_closed_roads = set()
        self.start_time = None
        self.sim_number = sim_number
        self.start_vehicle_positons = start_vehicle_positons
        
        self.sim_type = sim_type # main simulation vs optimization simulation
        #print("sim type: ", self.sim_type)
        self.fire_perim = fire_perim # expecting geodataframe (main sim - full, sub - 1 record of current perim)
        if self.fire_perim is not None:
            self.fire_perim['geom'] = self.fire_perim['geometry'].buffer(0)
            #print("fire found")
        self.fire_slice = None
        self.fire_db_slice = None
        self.fire_ignit_time = fire_ignit_time # row id for ignition feature, if None use row with least elapsed time
        if self.fire_ignit_time == None:
            if self.fire_perim is not None:
                self.fire_ignit_time = self.fire_perim['SimTime'].min()
        self.fire_act_ts_min = fire_act_ts_min # how many minutes between actual output updates (usually 60)?
        self.fire_des_ts_sec = fire_des_ts_sec # how many seconds desired between updates - can "speedup" or "slowdown" spread
        if reset_interval==True:
            self.fire_perim['SimTime'].replace(sorted(list(set(self.fire_perim['SimTime'].values))), 
                                [(x+1)*fire_act_ts_min for x in range(len(set(self.fire_perim['SimTime'].values)))], 
                                inplace=True)
        self.init_strategies = init_strategies
        self.congestion_dict = {}
        self.tot_num_roads_in_bbox = len(self.roads_in_bbox)
        self.veh_status = []
        self.edge_congestion = []
        
        #if target_lat_lon is not None:
        #    self.target_xy = project_lat_lon(*target_lat_lon)
        #    self.target = ox.get_nearest_node(g, self.target_xy[::-1], method='euclidean')
        #else:
        #    self.target_xy = None
        #    self.target = None
        
        for r in self.roads:
            self.roads[r].nexts={(x,k):self.roads[r[1],x,k] for x in g.adj[r[1]] for k in g.adj[r[1]][x] if 'geometry' in g.adj[r[1]][x][k]}
        #print('Initial ett test')
        for r in self.roads_in_bbox:
            self.g[r.idx[0]][r.idx[1]][r.idx[2]]['ett'] = r.length/r.speed
            assert self.g[r.idx[0]][r.idx[1]][r.idx[2]]['ett'] > 0
            self.g[r.idx[0]][r.idx[1]][r.idx[2]]['o_ett'] = self.g[r.idx[0]][r.idx[1]][r.idx[2]]['ett']
            
        self.road_length=np.array([r.geom.length for r in self.roads.values()])
        self.road_length/=sum(self.road_length)
            
        if sim_type == "main":
            
            if self.fire_perim is not None:
                self.spread_initial_fire(1.0)
            
            if self.init_strategies is not None:
                self.strat_per_veh = np.random.choice(list(self.init_strategies.keys()), n, p=list(self.init_strategies.values()))
                self.vehicles=[Vehicle(g,i,target=self.targets, st_weight=self.strat_per_veh[i]) for i in range(n)]
            else:
                self.vehicles=[Vehicle(g,i,target=self.targets) for i in range(n)]
  
            for v in self.vehicles:
                while not v.addTo(np.random.choice(self.roads_in_bbox, p=self.in_roads_prob), init=True):
                    pass
                
        else:
            # take vehicle positions from current location in main simulation
            self.vehicles=[Vehicle(g,i[0],pos=i[2],target=self.targets, st_weight=nav_weight_list[i[0]]) for i in self.start_vehicle_positons]
            for i, v in enumerate(self.vehicles):
                for r in self.roads.values():
                    if r.idx == self.start_vehicle_positons[i][1]:
                        v.addTo(r, init=True)
                        
        self.vehicle_colors=[plt.cm.rainbow(float(_)/n) if not self.vehicles[_].isStuck else (0.0,0.0,0.0,1.0) for _ in range(n)]
        
        self.initial_veh_coords = self.list_veh_coords()
        self.report_congestion(0)
        print('END init')
        
    def move(self, frame_number, timestep=1):
        #print('move frame_number:',frame_number, 'timestep:', timestep)
        for r in self.roads.values():
            r.move(frame_number, timestep)
        for r in self.roads.values():
            r.resolve_requests(frame_number)
        for r in self.roads.values():
            r.sync_pos()
                             
    def mutate_speed(self, prob=0.8):
        for r in self.roads.values():
            r.mutate_speed(prob)
    
    def mutate_block(self, prob=0.1):
        self.last_closed_roads = set(r.idx for r in self.roads_in_bbox if r.mutate_block(prob))
        if len(self.last_closed_roads) > 0: # could be accelarated with if self.last_closed_roads
            # TODO!! spread information to vehicles and navigate upon request            
            self.g.remove_edges_from(self.last_closed_roads)            
            self.closed_roads.update(self.last_closed_roads)
            for i in range(len(self.vehicles)):
                if (not self.vehicles[i].isStuck) and self.vehicles[i].handle_blocks(self.last_closed_roads):
                    self.vehicle_colors[i]=(0.0,0.0,0.0,1.0) # black                                        
        #assert all(self.roads[r].isBlocked for r in self.closed_roads)
        #assert all(not self.roads[r].isBlocked for r in self.roads if r not in self.closed_roads)
        
    def update_quickest(self): # -> quickest route
        #print('update quickest')
        changed_ett = set()
        for r in self.roads_in_bbox:
            if len(r.ett) < 1: # no vehicles
                continue
            try:
                avg_ett = sum(_[0] for _ in r.ett.values())/len(r.ett)
                net_edge = self.g[r.idx[0]][r.idx[1]][r.idx[2]]
                
                #print('rid:', r.idx, 'ett val', r.ett, 'average ett:', round(avg_ett, 3), 'netedge ett:', round(net_edge['ett'], 3), 'road speed:', r.speed, 'rlength', r.length, 'rett:', r.length/r.speed)
                if (net_edge['o_ett'] - avg_ett) > 1e-7:
                    print('***',  round(net_edge['o_ett'], 3),  round(avg_ett, 3))
                if abs(avg_ett - net_edge['ett'])/net_edge['ett'] > 0.5:
                    changed_ett.add(r.idx+(avg_ett,))
                    print('Changing rid:', r.idx, 'from', round(net_edge['ett'], 3), 'to', round(avg_ett, 3), '..Orig=', round(net_edge['o_ett'], 3))
            except KeyError:
                pass

        for i,j,k,t in changed_ett:
            try:
                self.g[i][j][k]['ett'] = t
            except KeyError:
                pass

        if 'quickest' in self.init_strategies.keys(): # renavigate with updated weights
            if len(changed_ett) > 0:
                for i in range(len(self.vehicles)):
                    self.vehicles[i].navigate(weight=self.vehicles[i].st_weight, target=self.targets)


    def select_fire_slice(self, update_number):
        now_time = self.fire_ignit_time + (update_number * self.fire_act_ts_min)
        #print("fire sliced....", update_number, ", ", now_time)
        #self.fire_slice = self.fire_perim[self.fire_perim["SimTime"]==now_time][self.fire_perim[self.fire_perim["SimTime"]==now_time].is_valid].unary_union # BECKY in future check for validity before inport? Why not valid
        self.fire_slice = self.fire_perim[self.fire_perim["SimTime"]==now_time]
        #print ("slice", type(self.fire_slice))
               
    def spread_initial_fire(self, update_number):
        print("fire initially spreads....", update_number)
        self.select_fire_slice(update_number)
        temp_closed_roads = set()
        for r in self.roads_in_bbox:
            for fidx in range(len(self.fire_slice)):
                if r.geom.intersects(self.fire_slice.iloc[fidx].geom):
                    temp_closed_roads.add(r.idx)
                    r.set_block()
        self.last_closed_roads = temp_closed_roads
        if len(self.last_closed_roads) > 0:           
            self.g.remove_edges_from(self.last_closed_roads)            
            self.closed_roads.update(self.last_closed_roads)
            
        if any(skey in self.init_strategies.keys() for skey in ['fire+dist', 'fire+rdty_weight']):
            self.g = add_fire_distance(self.g, self.fire_slice, norm=True, inv=True)
            if self.init_strategies.get('fire+dist', 0) > 0:
                self.g = combine_attribute(self.g, ['length_n', 'inv_fire_dist_n'], [0.5, 0.5], 'fire_leng_n')
            if self.init_strategies.get('fire+rdty_weight', 0) > 0:
                self.g = combine_attribute(self.g, ['rt_wght_len_n', 'inv_fire_dist_n'], [0.5, 0.5], 'fire_rd_wght_n')
                    
    def spread_fire(self, update_number):
        print("fire spreads....", update_number)
        self.select_fire_slice(update_number)
        temp_closed_roads = set()
        for r in self.roads_in_bbox:
            for fidx in range(len(self.fire_slice)):
                #print(type(r), type(fidx)) 
                if r.geom.intersects(self.fire_slice.iloc[fidx].geom):
                    temp_closed_roads.add(r.idx)
                    r.set_block()
        self.last_closed_roads = temp_closed_roads
        #self.last_closed_roads = set(r.idx for r in self.roads_in_bbox if r.geom.intersects(self.fire_slice))
        if len(self.last_closed_roads) > 0:           
            self.g.remove_edges_from(self.last_closed_roads)            
            self.closed_roads.update(self.last_closed_roads)
            
            if any(skey in self.init_strategies.keys() for skey in ['fire+dist', 'fire+rdty_weight']):
                self.g = add_fire_distance(self.g, self.fire_slice, norm=True, inv=True)
                if self.init_strategies.get('fire+dist', 0) > 0:
                    self.g = combine_attribute(self.g, ['length_n', 'inv_fire_dist_n'], [1.0, 0.0], 'fire_leng_n')
                if self.init_strategies.get('fire+rdty_weight', 0) > 0:
                    self.g = combine_attribute(self.g, ['rt_wght_len_n', 'inv_fire_dist_n'], [1.0, 0.0], 'fire_rd_wght_n')
            
            for i in range(len(self.vehicles)):
                if (not self.vehicles[i].isStuck) and self.vehicles[i].handle_blocks(self.last_closed_roads, target=self.targets):
                    self.vehicle_colors[i]=(0.0,0.0,0.0,1.0) # black 
                    
        
                            
    def show(self, ax, number_veh=False):
        #return ax.scatter(*zip(*[_.xy() for _ in self.vehicles]), s=20) 
        
        #ax.scatter(*zip(*[_.xy() for _ in self.vehicles]), c=self.vehicle_colors, s=20, zorder=4)
    
        if number_veh:
            ax.scatter(*zip(*[_.xy() for _ in self.vehicles]), c=self.vehicle_colors, s=20, zorder=4)
            #[print(_.xy()[0], _.xy()[1], _.vid) for _ in self.vehicles]
            [ax.text(_.xy()[0], _.xy()[1], str(_.vid), zorder=4) for _ in self.vehicles]
                #ax.text(_.xy()[0], _.xy()[1], str(_.vid), zorder=4)
        else:
            return ax.scatter(*zip(*[_.xy() for _ in self.vehicles]), c=self.vehicle_colors, s=20, zorder=4)
            
        #temp_vehicle_colors = self.vehicle_colors.copy()
        #for i, v in enumerate(self.vehicles):
        #    if v.road is None:
        #        temp_vehicle_colors.pop(i)
                
        #return ax.scatter(*zip(*[_.xy() for _ in self.vehicles if _.road is not None]), c=temp_vehicle_colors, s=20) # Becky Someting weird with geometry~   
        
        # Vehicle generation should take capacity constraint as well
        #inds=np.random.choice(range(len(self.roads)), n ,replace=True)
        #self.vehicles=[Vehicle(g, self.roads[_]) for _ in inds]
    
    def list_veh_coords(self, out=None):
        if out:
            pos_lost = ["("+str(_.vid)+';'+str(_.xy(out='yes'))+")" for _ in self.vehicles]
            return (" ").join(pos_lost)
        else:
            return ([(_.vid, _.xy()) for _ in self.vehicles])
    
    
    def list_veh_positions(self):
        return ([(_.vid, _.road.idx, _.pos) for _ in self.vehicles])
        
    def pickExample(self):
        for i in self.roads.values():
            if len(i.vehicles)>1:
                return i
            
    def report_congestion(self, time_stamp):
        roads_with_veh = 0
        tc_tot_veh = len(self.vehicles)
        rc_tot_clear_veh = sum([_.is_clear for _ in self.vehicles])
        rc_tot_stuck_veh = sum([_.isStuck for _ in self.vehicles])
        
        for r in self.roads_in_bbox:
            (num_veh, veh_per_len) = r.report_veh_num()
            if num_veh > 0:
                roads_with_veh = roads_with_veh + 1
                if r.idx in self.congestion_dict.keys():
                    self.congestion_dict[str(r.idx)].append({'ts':time_stamp, 'numv':num_veh, 'vplen':round(veh_per_len, 5)})
                else:
                    self.congestion_dict[str(r.idx)] = [{'ts':time_stamp, 'numv':num_veh, 'vplen':round(veh_per_len, 5)}]
                    
        self.veh_status.append({'ts':time_stamp, 'clear':rc_tot_clear_veh, 'stuck':rc_tot_stuck_veh, 'pctclear':round(rc_tot_clear_veh/tc_tot_veh, 3)})
        
        self.edge_congestion.append({'ts':time_stamp, 'tot_rd_veh':roads_with_veh, 'ave_veh_p_rd':round(tc_tot_veh/roads_with_veh, 3)})

    def run(self, nsteps=None, mutate_rate=0.005, update_interval=100, save_args=None, opt_interval=35, strategies=['dist', 'dist+speed'], opt_reps=1, mix_strat=True, congest_time=25):
        self.start_time = time.time()
        #print('num roads 2', len(self.roads))
        #self.opt_counter = 2
        self.drive_strat_list = None
        self.full_opt_results = {}
        self.strategies = strategies
        self.mix_strat = mix_strat
        self.congest_time = congest_time
            
        #print('start_main_sim body')
        if self.fire_perim is not None:
            update_interval = self.fire_des_ts_sec
        save = True
        self.isFinished = False
        if save_args is None or len(save_args) < 3:
            save=False
            print("data from this run won't be saved!")
            for i in range(steps):
                self.move()
                if i % update_interval == 0:
                    self.mutate_block(mutate_rate)
        else:  
            if len(save_args) == 3:
                fig, ax, filename = save_args 
            else:
                fig, ax, filename, result_file, folder, treatment_no, rep_no, seed, treat_desc, exp_desc, exp_no, nb_no, rd_grph_pkl = save_args 
                
                results_cont_folder = str(exp_no)+'files'
                movie_cont_folder = str(exp_no)+'videos'
                traj_cont_folder = str(exp_no)+'trajs'
                if not os.path.isdir(os.path.join(folder, results_cont_folder)):
                    os.mkdir(os.path.join(folder, results_cont_folder))
                if not os.path.isdir(os.path.join(folder, movie_cont_folder)):
                    os.mkdir(os.path.join(folder, movie_cont_folder))
                if not os.path.isdir(os.path.join(folder, traj_cont_folder)):
                    os.mkdir(os.path.join(folder, traj_cont_folder))
                
                filename = filename+"_"+str(treatment_no)+"_"+str(rep_no)+"_seed"+str(seed)+'_tspt_'+str(datetime.now().strftime("%d-%m-%y_%H-%M"))+"_nbno_"+str(nb_no)+"_expno_"+str(exp_no)+".mp4"
                
                result_file = result_file+"_"+str(treatment_no)+"_"+str(rep_no)+"_seed"+str(seed)+'_tspt_'+str(datetime.now().strftime("%d-%m-%y_%H-%M"))+"_nbno_"+str(nb_no)+"_expno_"+str(exp_no)+".txt"
                
                trajectory_file = filename+"_traj_"+str(treatment_no)+"_"+str(rep_no)+"_seed"+str(seed)+'_tspt_'+str(datetime.now().strftime("%d-%m-%y_%H-%M"))+"_nbno_"+str(nb_no)+"_expno_"+str(exp_no)+".txt"
                
                filename = os.path.join(folder, movie_cont_folder, filename)
          
            ax.add_patch(patches.Rectangle(self.project_bbox[0], self.project_bbox[1][0]-self.project_bbox[0][0], self.project_bbox[1][1]-self.project_bbox[0][1], fill=False, edgecolor='r'))
            #if self.target_xy is not None:
            #    ax.scatter(*zip(self.target_xy),marker='x',s=800)
            r=self.show(ax)

            def runFrame():
                i=0
                while not self.isFinished:
                    i+=1
                    yield i
                    
                         
            self.one_flag = []
            def update(frame_number):
                #print('.', end="")
                if frame_number == 1: #catch duplicate frame number
                    self.one_flag.append(frame_number)
                #print('frame_number', frame_number)
                if self.isFinished:
                    return
 # !!! reset closed roads, need to confirm no removal
                
                #initial wildfire location
        
                    
                #if self.fire_perim is not None:
                #    if frame_number == 1:
                #        self.spread_fire(1.0)
                
                if len(self.one_flag) == 2:
                    self.one_flag = [] 
                    #print('duplcate frame no 1')
                    return
                self.move(frame_number)
                r.set_offsets([_.xy() for _ in self.vehicles])
                
                if not any(v.last_move for v in self.vehicles):
                    self.isFinished = True
                    self.report_congestion(math.ceil(frame_number/self.congest_time)*self.congest_time)
                    #print 'Evacuation completed at time: %d'%frame_number   
                    
                   # if seed_number:
                   #    print("Seed_number: ", seed_number)
                    self.veh_clear_times = [(v.vid, v.goal_time) for v in self.vehicles]
                    
                    self.veh_strat = [(v.vid, v.st_weight) for v in self.vehicles]
                    
                    self.strat_list = [v.st_weight for v in self.vehicles]
                    self.strat_counter = Counter(self.strat_list)
                    
                    if result_file:
                        with open(os.path.join(folder, results_cont_folder, result_file), 'w') as out_file:
                            csv_writer = csv.writer(out_file, delimiter='\t')
                            
                         #   result_header = ['Video_fn', 'Result_fn', 'RG_file', 
                         #                    'Treat_no', 'Rep_no', 'Seed',
                         #                    'Elpsd_time_sec', 'Elpsd_time_TS',
                         #                    'Total_cars', 'Stuck_cars', 
                         #                    'Num_rds_in_bbox'
                         #                    'Finish_time', 
                         #                    'Treat_desc', 'veh_by_strat',
                         #                    'Veh_strat', 'Veh_clear_times', 'Exp_desc',
                         #                    'Exp_no', 'NB_no', 'Veh_stat_by_time',
                         #                    'Init_Veh_coords', 'Veh_by_edge']
                            result_header = ['Exp_no', 'NB_no',
                                             'Treat_no', 'Rep_no', 'Seed', 
                                             'Elpsd_time_TS',
                                             'Total_cars', 'Stuck_cars', 
                                             'Num_rds_in_bbox',
                                             'veh_by_strat',
                                             'Finish_time',
                                             'Treat_desc', 'Exp_desc', 'RG_file', 
                                             'Veh_stat_by_time',
                                             'Cong_by_time',
                                             'Veh_by_edge',
                                             'Init_Veh_coords'
                                            ]
                            
                         #   result_row = [str(i) for i in [
                         #       filename, result_file, rd_grph_pkl,
                         #       treatment_no, rep_no, seed,
                         #       round(time.time()-self.start_time, 2), str(frame_number),
                         #       len(self.vehicles) , str(sum([_.isStuck for _ in self.vehicles])),
                         #       self.tot_num_roads_in_bbox,
                         #       datetime.now().strftime("%d-%m-%y_%H-%M"), 
                         #       treat_desc, self.strat_counter, 
                         #       self.veh_strat, self.veh_clear_times, exp_desc,
                         #       exp_no, nb_no, self.veh_status,
                         #       self.initial_veh_coords, self.congestion_dict]] 
                            result_row = [str(i) for i in [exp_no, nb_no,
                                treatment_no, rep_no, seed,
                                str(frame_number),
                                len(self.vehicles), str(sum([_.isStuck for _ in self.vehicles])), 
                                self.tot_num_roads_in_bbox,
                                [(st, self.strat_counter[st]) for st in self.strat_counter.keys()],
                                datetime.now().strftime("%d-%m-%y_%H-%M"),
                                treat_desc, exp_desc, rd_grph_pkl,
                                self.veh_status,
                                self.edge_congestion,                           
                                self.congestion_dict,
                                self.initial_veh_coords
                            ]] 
                            
                            csv_writer.writerow(result_header)
                            csv_writer.writerow(result_row)
                        
                        with open(os.path.join(folder, traj_cont_folder, trajectory_file), 'w') as out_file:
                            csv_writer = csv.writer(out_file, delimiter='\t')
                            for v in self.vehicles:
                                csv_writer.writerow([v.vid, v.trajectory])
                        
                    else:
                        print(self.sim_type, "simulation complete!\n--Elapsed time (sec):", round(time.time()-self.start_time, 2), "Time steps -", ':'+str(frame_number)+':', "File -", filename, "Update interval -", update_interval, "Mutation rate -", ':'+str(mutate_rate)+':', "Total Cars -", len(self.vehicles), "Stuck Cars -", ':'+str(sum([_.isStuck for _ in self.vehicles])))
                    
                #r.set_color([plt.cm.RdYlGn(_.last_move) for _ in self.vehicles])  ## set vehicle colors based on speed
                
                if not self.isFinished:
                    if frame_number % self.congest_time == 0:
                        #print('record congestion at', frame_number)
                        self.report_congestion(frame_number)
                    if frame_number % update_interval == 0:
                        if self.fire_perim is None:
                            self.mutate_block(mutate_rate)
                        else:
                            #print('spreading, frame_number', frame_number)
                            self.spread_fire((frame_number/update_interval)+1)

                    if 'quickest' in self.init_strategies.keys():
                        self.update_quickest()
                  
                # displays closed roads
                ax.add_collection(collections.LineCollection([list(zip(*self.roads[idx].geom.xy)) for idx in self.last_closed_roads], colors='black')) # Becky changed s to self, changed zip function
                ax.scatter([self.g.nodes[idx[0]]['x'] for idx in self.last_closed_roads],
                           [self.g.nodes[idx[0]]['y'] for idx in self.last_closed_roads],
                           marker='D',
                           c='orange',
                           s=200)
                
                # display fire
                if self.fire_slice is not None:
                    for fidx in range(len(self.fire_slice)):
                        if self.fire_slice.iloc[fidx].geom.geom_type == 'MultiPolygon':
                            for fsp in self.fire_slice.iloc[fidx].geom:
                                if fsp.exterior:
                                    zipped = list(zip(fsp.exterior.xy[0], fsp.exterior.xy[1]))
                                    ax.add_collection(collections.LineCollection([zipped]))
                        elif self.fire_slice.iloc[fidx].geom.geom_type == 'Polygon':
                            if self.fire_slice.iloc[fidx].geom.exterior:
                                zipped = list(zip(self.fire_slice.iloc[fidx].geom.exterior.xy[0], self.fire_slice.iloc[fidx].geom.exterior.xy[1]))
                                ax.add_collection(collections.LineCollection([zipped]))
                        else:
                            print("fire slice something else")

            if nsteps is None:
                frame=runFrame
            else:
                frame=nsteps
            anim = animation.FuncAnimation(fig, update, frames=frame, interval=100, save_count=1000)
            anim.save(filename)
            
            #if self.sim_type == "secondary":
            #    # {vid:(strategy, time), vid:(strategy, time)...}
            #    return [(_.vid, _.goal_time, _.st_weight) for _ in self.vehicles]
            
            #return HTML('<video width="800" controls><source src="%s" type="video/mp4"></video>'%filename)
