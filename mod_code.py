from plot_graph import plot_cairo
import copy
import urllib
#from cobamp.wrappers import KShortestMCSEnumeratorWrapper
import seaborn as sns
import random
import numpy as np
import cobra as cb
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.signal as ss
import igraph as ig
import networkx as nx
from collections import defaultdict
from matplotlib import rc
rc('font',**{'family':'serif','serif':['STIX']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = 12,10
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from scipy.optimize import curve_fit
from bioservices.kegg import KEGG
from bioservices import uniprot
from bioservices import QuickGO
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from genes_ncbi_511145_proteincoding import GENEID2NT as GeneID2nt_ecoli
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS 
from goatools.anno.factory import get_objanno
from goatools.semantic import TermCounts, get_info_content 

def get_adj(mod):
	stoich = cb.util.array.create_stoichiometric_matrix(mod)
	adj = np.matmul(stoich, stoich.transpose())
	adj[adj != 0] = 1
	return adj, stoich

def find_hubs(adj):
	adj[adj != 0] = 1
	deg = adj.sum(axis = 1)
	return deg 

def fit_expon(adj):
        #preds = pd.read_pickle('preds_wild.pickle')
        smean = adj.sum(axis=1).mean()
        rate = 1./smean
        x = np.linspace(0, adj.sum(axis =1).max()-1, adj.sum(axis=1).max())
        dist_exp = st.expon.pdf(x, scale = 1./rate)
        point = st.expon.interval(0.95, loc=0, scale = 1./rate)
        fig, ax = plt.subplots()
        plt.hist(adj.sum(axis=1), bins = 100, density=True) 
        ax.plot(x, dist_exp)
        ks = st.kstest(adj.sum(axis = 1), 'expon')
        return point[1]

def fit_expon_two(adj):
	X = adj.sum(axis = 1)
	P = st.expon.fit(X)
	rX = np.linspace(1, 200, 199)
	rP = st.expon.pdf(rX, *P)
	point = st.expon.interval(0.95, *P)
	obvs, bounds = np.histogram(X, bins = 200)
	expect = st.expon.pdf(bounds, *P)
	stat, p = st.chisquare(obvs, expect[:-1] * X.sum())
	plt.hist(X, bins = 200, density=True)
	plt.plot(rX[1:], rP[1:])
	return point[1], stat, p

def find_outliers(adj_full):
	adj = adj_full.copy()
	out = st.gaussian_kde(adj.sum(axis = 1))
	x = np.linspace(0, 200, 500)
	y = out.pdf(x)
	peaks = ss.find_peaks(1/y)
	sns.distplot(adj.sum(axis = 1), bins = 100, kde = False, norm_hist = True)
	plt.scatter(x, y, marker = '.')
	plt.scatter(x[peaks[0]], y[peaks[0]])
	inds = np.where(adj.sum(axis = 1) > x[peaks[0][0]])
	adj[:, inds[0]] = np.zeros((len(adj), len(inds[0])))
	adj[inds[0], :] = np.zeros((len(inds[0]), len(adj)))
	#adj_trim = np.delete(np.delete(adj, inds[0], axis = 1), inds[0], axis = 0)
	np.fill_diagonal(adj, 0)
	#z_inds = np.where(adj_trim.sum(axis = 1) == 0)
	#adj_trim = np.delete(np.delete(adj_trim, z_inds[0], axis = 1), z_inds[0], axis = 0)
	return adj

def find_hairs(adj):
	deg = adj.sum(axis = 1)
	inds_one = np.where(deg == 1)[0]
	inds_two = np.where(deg == 2)[0]
	hairs = []
	hair_dict = {}
	neighbours_dict = defaultdict(None)
	hair_dict[1] = inds_one
	i = 2
	while(i != 0):
		neighbours = list(set(np.where(adj[:, hair_dict[i-1]].sum(axis = 1))[0]).intersection(inds_two))    
		#big_neighbours = list(set(np.where(adj[:, hair_dict[i-1]].sum(axis = 1))[0]).difference(inds_two))
		if neighbours:
			hair_dict[i] = neighbours
			inds_two = np.array(list(set(inds_two).difference(neighbours))) 
			i = i+1
			#for x in neighbours:
			#	neighbours_dict[x] = list(set(np.where(adj[:, hair_dict[i-1]].sum(axis = 1))[0]).difference(inds_two))
		else:
			i = 0
			
	#for x in sorted(hair_dict.keys(), reverse=True): 
	#	vals = hair_dict[x]
	#	chain_dict = defaultdict(None)
	#	for v in vals:
	#		small_neighbours = list(set(np.where(adj[:, v].sum(axis = 1))[0]).intersection(inds_two))
	#		big_neighbours = list(set(np.where(adj[:, v].sum(axis = 1))[0]).difference(inds_two))
	#		chain_dict[big_neighbours] = v
	#		chain_dict[big_neighbours].append(small_neighbours)
	return hair_dict, inds_two

def find_branch_hairs(adj, hair_dict, inds_two):
	r_G = nx.from_numpy_matrix(adj)
	G = ig.Graph()
	G.add_vertices(r_G.nodes)
	G.add_edges(r_G.edges)
	clusts = G.clusters()
	members = clusts.membership	
	num = len(G.components().sizes())
	c = G.components()
	burn = np.array(())
	burn_dict = {}
	for n in inds_two:
		neighbours = np.where(adj[:, n])[0]
		for x in neighbours:
			if x not in hair_dict:
				tmp = G.copy()
				tmp.delete_edges(tmp.get_eid(n, x))
				c = len(tmp.components().sizes())
				if c > num:
					mems_full = np.array(G.clusters().membership)
					mems_cut = np.array(tmp.clusters().membership)
					mems_full[mems_full != 0] = 1
					mems_cut[mems_cut != 0] = 1
					inds = np.where((mems_full - mems_cut) == -1)
					if x not in inds[0]:
						burn_dict[x] = inds[0]
					burn = np.append(burn, inds[0])
	burn = np.array([int(x) for x in burn])
	return burn, burn_dict


def follow_chain(x, adj_trim): 
	l = []  
	full = [] 
	d = 2 
	node = x 
	neigh = np.where(adj_trim[:, node]) 
	for c in neigh[0]: 
		l = [] 
		l.append(x) 
		d = adj_trim[:, c].sum() 
		deg = d 
		l.append(c) 
		node = c 
		while d == 2: 
			new_neigh = np.where(adj_trim[:, node])[0] 
			node = list(set(new_neigh).difference(l)) 
			if not node: 
				break 
			l.append(node[0]) 
			d = adj_trim[:, node].sum() 
		if deg == 2: 
			full.append(l) 
		else: 
			l = [] 
	return full, l 




def del_arcs(adj_trim, inds_two, burn):
	arcs = list(set(inds_two).difference(burn))
	store = defaultdict(None)
	arcs_del = []
	edges_add = []
	neighbours = np.where(adj_trim[:, arcs].sum(axis = 1)) #all of these are neighbours of an arc node
	deg = adj_trim[:, neighbours].sum(axis = 0) #these are the degrees of neighbours of an arc node
	#need to find the neighbours where the degree is > 2 and set to remove the node
	big_inds = np.where(deg[0] != 2)[0]
	arc_con = neighbours[0][big_inds]
	full_makes = np.empty((0, 2), int)
	full_breaks = np.empty((0, 2), int)
	for x in arc_con:
		arc, l = follow_chain(x, adj_trim)
		#print(x)
		#print(arc)
		for x in arc:
			make = [x[0], x[-1]]
			start = x.copy()[0:-1]
			end = x.copy()[1:]
			cut = np.vstack((start, end))
			full_makes = np.vstack((full_makes, make)) 
			full_breaks = np.vstack((full_breaks, cut.transpose()))
			store[make[0]] = cut.transpose()
	return full_makes, full_breaks, store
		#get list of edges to add (first and last of each sublist)
		#get list of edges to delete (consecutive inds)

def top_overlap(adj_trim):
	i = np.where(np.any(adj_trim, axis = 0))
	adj = adj_trim[:,i[0]][i[0], :] 
	overlap = adj + np.matmul(adj, adj)
	diag = np.diagonal(overlap)
	over = np.minimum(np.tile(diag, (len(overlap), 1)), np.transpose(np.tile(diag, (len(overlap), 1))))
	#recip = 1/over
	#recip[np.isinf(recip)] = 0
	#fin = overlap*recip
	fin = overlap/over
	return fin, i, adj

def top_overlap_two(adj_trim):
	i = np.where(np.any(adj_trim, axis = 0))
	adj = adj_trim[:,i[0]][i[0], :]
	l = np.matmul(adj, adj)
	np.fill_diagonal(l, 0)
	over = (l + adj_l)/np.minimum(np.tile(adj_l.sum(axis = 1), (len(adj_l),1)), np.transpose(np.tile(adj_l.sum(axis = 1), (len(adj_l),1)))) + 1 - adj_l
	return over, i, adj

def find_clusters(fin, p, thresh):
	full = []
	for x in range(len(fin)):
		color = '#' + '%06x' % random.randint(0, 0xFFFFFF)
		full.append(color)
	hierarchy.set_link_color_palette(full)
	Z = linkage(fin, method = 'complete')
	dn = dendrogram(Z, p = p, truncate_mode = 'level', above_threshold_color = '#9cadb5', color_threshold = 4.5, no_labels = True)
	plt.savefig('./figs/dend_full.pdf')
	plt.clf()
	dn = dendrogram(Z, above_threshold_color = '#9cadb5', color_threshold = thresh, no_labels = True)
	#plt.savefig('./figs/dend' + str(thresh) + '.pdf')
	plt.clf()
	return dn, max(Z[:,2]), full

def define_clusters(dn, i, names, mod, go_df):
	full_clust = []
	paths_full = []
	funcs_full = []
	mets_full = []
	for col in np.unique(dn['color_list']):
		inds = np.where(np.array(dn['color_list']) == col)
		clust_names = [names[x] for x in i[0][np.array(dn['leaves'])[inds[0]]]]
		met_inds = [x for x in i[0][np.array(dn['leaves'])[inds[0]]]]
		full_clust.append(clust_names)
		clust_paths = [list(mod.metabolites[x].reactions) for x in i[0][np.array(dn['leaves'])[inds[0]]]]
		paths = [x.subsystem for l in clust_paths for x in l]
		ids = set([x.id for l in clust_paths for x in l]).intersection(set(go_df.index.values))	
		funcs = [go_df.loc[x].annotations for x in list(ids)]
		flat_funcs = [x for l in funcs for x in l]
		funcs_full.append(flat_funcs)
		paths_full.append(paths)
		mets_full.append(np.array(met_inds))

	return full_clust, paths_full, mets_full, funcs_full

#def cluster_func(dn, mod):
#	paths_full = []
#	for col in np.unique(dn['color_list']):
#		inds = np.where(np.array(dn['color_list']) == col)
#		clust_paths = [list(mod.metabolites[x].reactions) for x in i[0][np.array(dn['leaves'])[inds[0]]]]
#		paths = [x.subsystem for l in clust_paths for x in l]	
#		paths_full.append(paths)
#	return paths_full


def get_green_subgraphs(hair_dict, burn, adj):
	reac_graph = nx.from_numpy_matrix(adj)
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	num = len(G.components().sizes())
	green_subs = defaultdict(None)
	g = [x for sublist in hair_dict.values() for x in sublist]
	green = np.hstack((g, burn))
	for i in green:
		n = np.where(adj[:, int(i)])
		join = set(n[0]).difference(set(green))
		if join:
			join_int = join.pop()
			tmp = G.copy()
			tmp.delete_edges(tmp.get_eid(int(i), join_int))
			c = len(tmp.components().sizes())
			if c > num:
				mems_full = np.array(G.clusters().membership)
				mems_cut = np.array(tmp.clusters().membership)
				mems_full[mems_full != 0] = 1
				mems_cut[mems_cut != 0] = 1
				inds = np.where((mems_full - mems_cut) == -1)
				green_subs[join_int] = inds
	return green_subs

def draw_graph(adj, hair_dict, inds_two, burn, full_makes, full_breaks):
	g = [x for sublist in hair_dict.values() for x in sublist]  
	green = np.hstack((g, burn))
	green = [int(x) for x in green]
	adj[:, green] = 0
	adj[green,:] = 0
	adj[full_breaks[:, 0], full_breaks[:, 1]] = 0
	adj[full_breaks[:, 1], full_breaks[:, 0]] = 0
	adj[full_makes[:, 0], full_makes[:, 1]] = 1
	adj[full_makes[:, 1], full_makes[:, 0]] = 1
	#adj = adj[~np.all(adj == 0, axis=1)]
	#adj = adj[:, ~np.all(adj == 0, axis=0)]
	reac_graph = nx.from_numpy_matrix(adj)
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	#G.vs['color'] = 'red'
	blue = list(set(inds_two).difference(burn))
	col_dict = dict.fromkeys(list(range(0, len(G.vs))), '#c81a1a')
	#for i in g:
		#G.vs['color'][i] = 'green'
	#	col_dict[i] = '#378749'
	#for i in blue:
		#G.vs['color'][i] = 'blue'
	#	col_dict[i] = '#4b8bc1'
	#for i in burn:
		#G.vs['color'][i] = 'green'  
	#	col_dict[i] = '#378749'
	G.vs["color"] = [col_dict[x] for x in G.vs.indices]
	H = G.copy()
	for i in reversed(range(len(G.vs.indices))):
		#print(G.vs.degree()[i])
		if G.vs.degree()[i] == 0:
			G.delete_vertices(i)
	layout = G.layout_mds()
	fig = ig.plot(G, './figs/network_edited.pdf', layout = layout)
	#fig.savefig('./figs/network.pdf')
	return G, H, adj

def plot_colours(adj_small, dn, fin):
	reac_graph = nx.from_numpy_matrix(adj_small)
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	G.vs['color'] = dn['color_list']
	layout = G.layout_mds()
	fig = ig.plot(G, './figs/network_clusts.pdf', layout = layout)
	return G

#metrics: - evenness of cluster sizes?
#	  - diversity of pathways (frequency + number?)

def connect_clusters(dn, adj_small, thresh, full):
	simp_adj = np.zeros((len(np.unique(dn['color_list'])), len(np.unique(dn['color_list']))))
	col_list = dn['color_list'].copy()
	col_list.insert(0, dn['color_list'][0])
	col_dict = defaultdict(None)
	i = 0
	f = np.array(full)
	for x in f[0:len(simp_adj)]:
		col_dict[x] = i
		i = i+1 
	#create new matrix that is the size of the clusters 
	for n in f[0:len(simp_adj)]:
	#for cluster:
		inds = np.where(np.array(col_list) == n)
		neighbours = np.where(adj_small[:, np.array(dn['leaves'])[inds[0]]])
		col_neighbours = np.array(col_list)[neighbours[0]]
		row_inds = np.array(col_list)[inds[0][neighbours[1]]]
		rows = []
		for x in row_inds:
			rows.append(int(col_dict[x]))
		cols = []
		for x in col_neighbours:
			cols.append(int(col_dict[x]))
		#print(np.unique(np.vstack((np.array(cols), np.array(rows))), return_counts = True, axis = 1))
		num = np.unique(np.vstack((np.array(cols), np.array(rows))), return_counts = True, axis = 1)
		simp_adj[num[0][0], num[0][1]] = num[1] #np.unique(np.vstack((np.array(cols), np.array(rows))), return_counts = True, axis = 1)[1] 
		simp_adj[num[0][1], num[0][0]] = num[1] 
		#find all adjacent edges
		#get those not within own cluster
		#find which clusters they are in
		#add to new ajacency matrix
		#plot
	np.fill_diagonal(simp_adj, 0)
	reac_graph = nx.from_numpy_matrix(simp_adj)
	weights = np.triu(simp_adj).flatten()
	visual_style = {}
	visual_style['edge_width'] = 10*(weights[np.where(weights)[0]]/max(weights))
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	G.vs['color'] = list(col_dict.keys())
	G.vs['size'] = 50
	G.vs['name'] = list(col_dict.values())
	visual_style["vertex_label"] = G.vs["name"]
	layout = G.layout_kamada_kawai()
	fig = ig.plot(G, './figs/network_conc_' + str(thresh) + '.pdf', layout = layout, **visual_style)
	return simp_adj, G, visual_style

def get_go_terms(mod, ind):
	g = QuickGO(verbose = False)
	godag = GODag('/home/sl13479/Documents/PhD_Docs/modules/go-basic.obo')
	met = mod.reactions[ind]
	u = uniprot.UniProt()
	get = u.search('organism:Escherichia coli + ' + met.gene_name_reaction_rule, frmt = 'tab', columns = 'organism, id, genes', limit = 3)
	if type(get) is str:
		id_split = get.split('\t')
		if len(id_split) > 3:
			id_ = id_split[3]
		else:
			return None, None, None
	else:
		return None, None, None
	GO = u.get_df(id_)['Gene ontology (GO)']
	go_ids = [x.split('[')[1][:-1] for x in GO[0]]
	for i in go_ids: 
		gosubdag_r0 = GoSubDag([i], godag, prt=None)
		try:
			ancestors = gosubdag_r0.rcntobj.go2ancestors[i]
			annotations = [g.goterms(x)[0]['name'] for x in ancestors]
			#print(i)
			#print(ancestors)
			#print(annotations)
		except:
			ancestors = None
	return go_ids, ancestors, annotations

def get_clust_reacs(mets_full, stoich, mod, go_df, kegg_df):
	clust_dict = defaultdict(None)
	clust_kegg = defaultdict(None)
	clust_path = defaultdict(None)
	genes = defaultdict(None)
	gene_names = defaultdict(None)
	#key = 0
	all_annotations = []
	for c, val in enumerate(mets_full):
		r = np.where(stoich[mets_full[c], :])[1]
		num = np.unique(r, return_counts=True)
		r_inds = num[0][np.where(num[1] > 1)]
		#for x in r_inds:
			#GO, ancestors, annotations = get_go_terms(mod, x)
			#all_ancestors.append(ancestors)
		ids = list(set([mod.reactions[x].id for x in r_inds]).intersection(set(go_df.index.values)))
		grr = [mod.reactions[x].gene_reaction_rule for x in r_inds]
		if len(grr[0]) != 5:
			grr = process_d_dict(grr)
		names = list(set(grr).intersection(set(kegg_df.index.values)))
		g = list(set([mod.reactions[x].gene_name_reaction_rule for x in r_inds]))
		g = process_d_dict(g)
		ann = [go_df.loc[x].annotations for x in list(ids)]
#			all_annotations.append(ann)
		terms = [x for l in ann for x in l]
		k_ann = [kegg_df.loc[x].annotations for x in list(names)]
		k_paths = [kegg_df.loc[x].paths for x in list(names)]	
		k_terms = [x for l in k_ann for x in l]
		gene_ids = [x for x in names]
		#paths = [x for l in k_paths for x in l]
		k_freqs = np.unique(k_terms, return_counts=True)
		freqs = np.unique(terms, return_counts=True)
		gene_names[c] = gene_ids
		clust_dict[c] = freqs
		clust_kegg[c] = k_freqs
		clust_path[c] = k_paths
		genes[c] = g
		#key = key+1
	#get full_inds of metabolites in cluster
	#this will give reaction inds
	#get frequency (or counts)
	#find any that are > 1
	#get inds of these and pass to get_go_terms
	return clust_dict, clust_kegg, clust_path, genes, gene_names

def make_go_dict(mod):
	#go_dict = defaultdict(None)
	GO_full = []
	ancestors_full = []
	annotations_full = []
	for idx, met in enumerate(mod.reactions):
		GO, ancestors, annotations = get_go_terms(mod, idx)
		if GO:
			GO_full.append(GO)
			#print(GO)
		else:
			GO_full.append(None)
		if ancestors:
			ancestors_full.append(ancestors)
		else:
			annotations_full.append(None)
		if annotations:
			annotations_full.append(annotations)
		else:
			annotations_full.append(None)
	new_go = [x for x in GO_full if x is not None] 
	new_ancestors = [x for x in ancestors_full if x is not None] 
	new_annotations = [x for x in annotations_full if x is not None] 
 
	r = [x.id for x in mod.reactions]
	booll = np.where(np.array([x is None for x in GO_full])*1)
	all_r = [x for x in r if x not in r[booll]]	
	df = pd.DataFrame(np.array([new_go, new_ancestors, new_annotations, all_r]).transpose(), columns = ['terms', 'ancestors', 'annotations', 'ids'])
	df = df.set_index('ids')

	return df

def make_kegg_dict(mod):
	k = KEGG(verbose = False)
	full_genes = []
	full_codes = []
	full_labels = []
	full_paths = []
	full_pcodes = []
	for x in mod.genes:
		#info = k.get('eco:' + x.id)
		info = k.get('eco:' + x.name)
		if not isinstance(info, int):
			ids = re.findall(r' \d{5} [\w ]+', info) 
			p = info.split('PATHWAY')
			if len(p) > 1:
				path = p[1].split('\n')[0] 
				full_paths.append(re.split('\d{5}', path)[-1].strip())
			else:
				full_paths.append(None)
			full_labels.append([re.split(' \d{5} ', x)[1] for x in ids])
			full_codes.append([re.findall('\d{5} ', x)[0] for x in ids])
			full_pcodes.append(re.findall('eco\d{5}', path))
			full_genes.append(x.id)
			print(x.id)
	return full_labels, full_codes, full_pcodes, full_genes

def add_subs(mets_full, store, green_subs):
	small_list = []
	new_mets = mets_full.copy()
	for n in new_mets:
		for x in n:
			if x in store.keys():
				arcs = store[x]
				small_list = small_list + list(np.unique(arcs))
				#get the arc nodes and add to the cluster
			if x in green_subs.keys():
				hairs = green_subs[x]
				small_list = small_list + (list(hairs[0]))
				#get the hairs and add to the cluster
		#print(len(small_list))
		n = np.append(n, small_list)
		small_list = []
	for idx, val in enumerate(new_mets):
		new_mets[idx] = np.unique(val)
	return new_mets

#def filter(full_clusts, clust_dict):
#	for idx, val in enumerate(full_clusts):
#		s = len(n)/2
#		loc = np.where(clust_dict[idx][1] > s)
#		mains = clust_dict[idx][0][loc]
#		print(mains)

def plot_freqs(clust_dict, thresh):
	legend_full = []
	for n in clust_dict:
		legend_str = 'Cluster ' + str(n)
		legend_full.append(legend_str)
		nums = clust_dict[n][1]
		sns.lineplot(x = np.linspace(0, len(nums)-1, len(nums)), y = np.sort(nums)/nums.sum(), marker="o")
	plt.xlabel('GO Term Index')
	plt.ylabel('Percentage of GO Term frequency within cluster')
	plt.legend(legend_full, ncol = 3)
	plt.savefig('./figs/clusts/dist_' + str(thresh) + '.pdf')
	return

def shape_hierarchy(fin, m):
	space = np.linspace(0, m, 100)
	full_n = []
	for x in space:
		dn, Z, full = find_clusters(fin, 15, m - x)
		n_clust = len(np.unique(dn['color_list']))
		full_n.append(n_clust)
		print(n_clust)
	
	ind = np.argmax(full_n)
	#x = np.array(full_n[:ind])
	#y = np.flipud(space[(len(full_n) - ind):])
	#x = np.flipud(space)[:ind]
	x = m - space[:ind]
	y = np.array(full_n[:ind])
	plt.plot(x, y, 'ko-') 	
	return x, y

def fit_clust(x, y):
	#p = np.polyfit(x, np.log(y), 1)
	##new_y = p[0]*x**2 + p[1]*x + p[2]
	#new_y = np.exp(p[1]) * np.exp(p[0] * x)
	#fit = curve_fit(lambda t,a,b,c: a+b*np.log(c * t),  x,  new_y)
	#plt.plot(x, fit[0][0] + fit[0][1] * np.log(x), 'ko-') 
	p = np.polyfit(x, np.log(y), 1)
	#new_y = p[0]*x**2 + p[1]*x + p[2]
	new_y = np.exp(p[1]) * np.exp(p[0] * x)
	plt.plot(x, new_y, 'ko-') 
	return p

def get_go_level(p, clust_dict, Z, x, n, thresh):
	plt.clf()
	nums = clust_dict[n][1]
	x_data = np.linspace(1, max(x), len(nums))
	y_clust = np.exp(p[1]) * np.exp(p[0] * x_data)
	y_data = np.sort(nums)/nums.sum()
	y_data = y_data * ((max(y_clust) - min(y_clust))/max(y_data)) + min(y_clust)
	sns.lineplot(x = x_data, y = np.flipud(y_data), marker="o")
	plt.plot(x_data, y_clust, 'ko-')
	plt.show()
	p = np.argmin(np.abs(y_clust - thresh))
	i = np.array([p-1, p, p+1])
	#g_ind = int(round(x_data[p]))
	go = clust_dict[n][0][np.flipud(np.argsort(clust_dict[n][1]))][i[i >=0]]
	print(go) 
	return go

def modularity(adj, thresh, names, dist):
	d_names = defaultdict(None)
	d_ids = defaultdict(None)
	r_G = nx.from_numpy_matrix(adj)
	G = ig.Graph()
	G.add_vertices(r_G.nodes)
	G.add_edges(r_G.edges)
	#G.es['weights'] = adj.flatten()
	#d = G.community_fastgreedy()
	#clust = d.as_clustering(d.optimal_count)
	#m = np.array(clust.membership)
	#for idx, val in enumerate(np.unique(m)):
	#	ids = np.where(val == m)
	#	d_ids[idx] = ids[0]
	#	d_names[idx] = [names[x] for x in ids[0]]	
	#fin, i, adj_small = top_overlap(adj)
	#fin = cdist(adj, adj, metric = 'cosine')
	Z = linkage(dist, method = 'complete')
	dn = dendrogram(Z, color_threshold = thresh, no_plot = True, no_labels=True)
	dn['color_list'].insert(0, dn['color_list'][0])
	col_dict = dict(zip(np.unique(dn['color_list']), list(range(0, len(np.unique(dn['color_list'])), 1))))
	#inds = np.where(np.array(dn['color_list']) == col)
	#[names[x] for x in i[0][np.array(dn['leaves'])[inds[0]]]]
	#full_clusts, paths_full, mets_full, funcs_full = define_clusters(dn, i, names, mod, go)
	entropy = mutual_info(paths_full, mets_full)
	m = np.array([col_dict[x] for x in dn['color_list']])
	#order_m = [m[x] for x in dn['leaves']]
	order = np.transpose(np.vstack((np.array(dn['leaves']), m)))
	c = ig.clustering.VertexClustering(G, membership = order[order[:,0].argsort()][:,1])
	#c = G.clusters(mode='WEAK')
	#print(len(np.unique(c.membership)))
	q = c.q
	print(str(len(adj)) + ': ' + str(q))
	return q #, d_ids, d_names #, max(Z[:,2])

def plot_entropy(mat1, mat2, m, names, mod, go):
	fin1, i1, adj_small = top_overlap(mat1)
	Z1 = linkage(fin, method = 'complete')
	fin2, i2, adj_small = top_overlap(mat2)
	Z2 = linkage(fin, method = 'complete')

	full_h1 = []
	full_h2 = []
	t = np.linspace(0, m, 200)
	for x in t:
		dn = dendrogram(Z1, color_threshold = x, no_plot = True, no_labels=True)
		full_clusts, paths_full, mets_full, funcs_full = define_clusters(dn, i1, names, mod, go)
		entropy1 = mutual_info(paths_full, mets_full)
		full_h1.append(entropy1)
		print(entropy1)
		
		dn = dendrogram(Z2, color_threshold = x, no_plot = True, no_labels=True)
		full_clusts, paths_full, mets_full, funcs_full = define_clusters(dn, i2, names, mod, go)
		entropy2 = mutual_info(paths_full, mets_full)
		full_h2.append(entropy2)
		print(entropy2)
		print(x)
	
	plt.plot(t, full_h1, 'ko-')
	plt.plot(t, full_h2, 'co-')
	plt.xlabel('Clustering threshold')
	plt.ylabel('Entropy across pathway labels')
	plt.legend(['Full metabolic network', 'Reduced metabolc network'])
	plt.savefig('./figs/entropy/entropy.pdf')
	plt.savefig('./figs/entropy/entropy.svg')
	plt.savefig('./figs/entropy/entropy.png')
	return




def plot_modularity(mat1, mat2,  Z):
	#condense matrices?
	i = np.linspace(0, Z, 200)
	dist1 = cdist(mat1, mat1, metric = 'jaccard')
	dist2 = cdist(mat2, mat2, metric = 'jaccard')
	full_q1 = []
	full_q2 = []
	#dist1, inds, adj_small = top_overlap(mat1)
	#dist2, inds, adj_small = top_overlap(mat2)
	for x in i:
		print(x)
		q1 = modularity(mat1, x, names, dist1)
		#plt.plot(x, q1, 'ko')
		full_q1.append(q1)
		q2 = modularity(mat2, x, names, dist2)
		#plt.plot(x, q2, 'co')
		full_q2.append(q2)
	plt.plot(i, full_q1, 'ko-')
	plt.plot(i, full_q2, 'co-')
	plt.xlabel('Clustering threshold')
	plt.ylabel('Modularity')
	plt.legend(['Full metabolic network', 'Reduced metabolic network'])
	plt.savefig('./figs/modularity/modularity_jaccard.pdf')
	plt.savefig('./figs/modularity/modularity_jaccard.svg')
	plt.savefig('./figs/modularity/modularity_jaccard.png')
	return

def mutual_info(clust_path, mets_full):
#	paths = np.unique([x for l in clust_path for x in l])
	entropy_full = []
	#for key, val in clust_path.items():
	for idx, val in enumerate(clust_path):
		#val = list(filter(None.__ne__, val))
		f = np.unique(val, return_counts = True)
		p = f[1]/f[1].sum()
		H = np.sum(p * np.log(p))
		entropy_full.append(H * -1)
		#print(H/len(p))
	#sns.distplot(entropy_full)
	#plt.savefig('./figs/entropy/reduced.pdf')
	#plt.savefig('./figs/entropy/reduced.svg')
	return np.array(entropy_full).mean()

def process_d_dict(grr):
	d = list(set(' '.join(grr).replace('(', '').replace(')', '').replace('and ', '').replace('or ', '').split(' ')))
	return d

def enrichment(genes):
	genes_df = pd.read_csv('gene_result.txt', delimiter = '\t')
	genes_df = genes_df[['GeneID', 'Symbol']].set_index('Symbol')
	check = list(set(genes).intersection(genes_df.index.values))
	gene_ids = genes_df.loc[check]
	gene_ids = gene_ids['GeneID'].values
	obodag = GODag("go-basic.obo")
	fin_gene2go = download_ncbi_associations()
	objanno = Gene2GoReader(fin_gene2go, taxids=[511145])
	ns2assoc = objanno.get_ns2assc()
	goeaobj = GOEnrichmentStudyNS(
        GeneID2nt_ecoli.keys(), # List of ecoli protein-coding genes
        ns2assoc, # geneid/GO associations
        obodag, # Ontologies
        propagate_counts = False,
        alpha = 0.05, # default significance cut-off
        methods = ['fdr_bh']) # defult multipletest correction method
	gene_ints = [int(x) for x in gene_ids]
	goea_results_all = goeaobj.run_study(gene_ints) #l is a list of the gene IDs --- format is six digits, and I think will require a lookup from the gene_results.txt file to get mapping of genes to numerical IDs   
	goea_quiet_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
	go_ids_chosen = []
	for x in goea_quiet_sig:
		go_ids_chosen.append(x.GO)
	gosubdag = GoSubDag(go_ids_chosen, obodag)
	go_id, go_term = max(gosubdag.go2obj.items(), key=lambda t: t[1].depth) 
	go_ids_chosen = go_term.get_all_parents() 
	go_ids_chosen.add(go_id)
	nts = [gosubdag.go2nt[go] for go in go_ids_chosen] 
	abc2gaf = {'eco': 'ecocyc.gaf'}
	abc2objanno = {s:get_objanno(f, 'gaf', godag=obodag) for s, f in abc2gaf.items()}
	abc2objtcnt = {s:TermCounts(obodag, o.get_id2gos_nss()) for s, o in abc2objanno.items()}
	fmt_str = ('{I:2}) {NS} {GO:10} {dcnt:4}  D{depth:02}   ' '{eco:6.3f}  ' '{GO_name}')
	print('                              |<tinfo>|') 
	print('IDX NS GO ID      dcnt Depth     eco    Name') 
	print('--- -- ---------- ---- ------ ------ ------ ----- ------------- -------')
	for idx, nt_go in enumerate(sorted(nts, key=lambda nt: nt.depth), 1):
		abc2tinfo = {s: get_info_content(nt_go.GO, o) for s, o in abc2objtcnt.items()}
		print(fmt_str.format(I=idx, **abc2tinfo, **nt_go._asdict())) 
	return goea_quiet_sig

def dunno(mod):
	#biomass ind is 925'th reaction in mod
	return

def s_prices(mod_KO, b_df, n):
	sol = mod_KO.optimize()
	print('Cluster knocked out: ', sol)
	stoich = cb.util.array.create_stoichiometric_matrix(mod_KO)
	#loc = np.where(stoich[inds,:])[0]
	#get biomass indices
	#b_fluxes = stoich[biomass['indices'].values, :] * np.transpose(sol.fluxes.values)
	#b_counts = b_fluxes.sum(axis = 1)
	l = [ x.replace('_', '') for x in sol.shadow_prices[b_df.indices].index]                    
	#millimoles per gram dry cell weight per hour
	b_df['shadow\_prices'] = pd.Series(sol.shadow_prices[b_df.indices].values)
	b_df['ids'] = l
	ax = sns.barplot(x = 'ids', y = 'shadow\_prices', data = b_df, hue = b_df['class'], dodge=False)  
	ax.set_xticklabels(labels = l, fontsize = 10, rotation=70) 
	#colors = sns.color_palette('Reds_d', n_colors=len(np.unique(b_df['class'])))
	#cmap = dict(zip(np.unique(b_df['class']), colors))
	#patches = [Patch(color=v, label=k) for k, v in cmap.items()]
	#plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, fontsize=15)
	plt.legend()
	#plt.show()
	#fig, ax1 = plt.subplots()
	#plt.plot(sol.shadow_prices.values, 'ko-')
	#plt.show()
	plt.savefig('./figs/shadow_prices/random/shadow_' + str(n) + '.pdf')
	return

#for each significant shadow price, find closest biomass component, do distance, and classify
#go back to pathways?

#dist1 = cdist(adj, adj)

def KOs(mod, gene_names, gene_dict, n, mets_full, b_df, dist1, fin_full):
	mod_KO = mod.copy()
	for x in gene_names[n]:
		mod_KO.genes[gene_dict[x]].knock_out()
	sol = mod_KO.optimize()
	print(sol)
	sp = sol.shadow_prices.values
	mets = np.where(np.abs(sp) > 0.1)[0]
	#inter = set(mets[0]).intersection(set(mets_full[n]))
	#for x in mets:
	#	ind = np.argmin(dist1[x,b_df.indices])
	#	print(b_df.iloc[ind]['class'])
	#	print(np.min(dist1[x, b_df.indices]))	
	#sns.distplot(sol.shadow_prices.values, kde=False)
	#plt.show()
	#for x in mets:
	#	r = list(mod.metabolites[x].reactions)
	#	for i in r:
	#		print(i.subsystem)
	over = fin_full[:, mets][mets,:].sum()/adj[:,mets].sum()
	print(over)
	return mod_KO, mets, sp, over

def random_KOs(gene_dict, n, mod, dist1, b_df, fin_full):
	i = random.sample(range(0, len(gene_dict)), n)
	mod_rand = mod.copy()
	k_ann = []
	for x in i:
		mod_rand.genes[x].knock_out()
		b = mod_rand.genes[x].id
		if b in kegg_df.index.values:
			k_ann.append(kegg_df.loc[b]['annotations'])
	k_terms = [x for l in k_ann for x in l]
	k_freqs = np.unique(k_terms, return_counts=True)
	sol = mod_rand.optimize()
	sp = sol.shadow_prices.values
	mets = np.where(np.abs(sp) > 0.1)[0]
	over = fin_full[:, mets][mets,:].sum()/adj[:,mets].sum()
	print(over)
#inter = set(mets[0]).intersection(set(mets_full[n]))
	#for x in mets:
	#	ind = np.argmin(dist1[x,b_df.indices])
	#	print(np.min(dist1[x, b_df.indices]))
	#	print(b_df.iloc[ind]['class'])
		

#	sns.distplot(sol.shadow_prices.values, kde=False)
#	plt.show()
	return mod_rand, sp, over, mets

def biomass():
	b_inds = pd.read_csv('biomass_inds.txt', delimiter = '\t', names = ['class', 'indices'])
	biomass = pd.read_csv('biomass.txt', delimiter = '\t', names = ['indices', 'names'])
	class_list = []
	class_inds = []
	for idx, row in b_inds.iterrows():
		il = list(map(int, b_inds.indices[idx].split(', ')))
		class_list.append(np.repeat(row['class'], len(il)))
		class_inds.append(il)
	c = [ x for l in class_list for x in l]
	c_inds = [int(x) for l in class_inds for x in l]
	c_df = pd.DataFrame(np.transpose([c, c_inds]), columns = ['class', 'indices'])
	c_df.indices = c_df.indices.astype(int)
	b_df = biomass.merge(c_df, on = 'indices')
	return b_df



if __name__ == '__main__':

	dna_inds = [628, 635, 676, 722]
	mod = cb.io.load_json_model('iAF1260.json') #load model
	adj, stoich = get_adj(mod) #get adjacency matrix for model
	adj_trim = find_outliers(adj) #find cutoff threshold for hubs
	hair_dict, inds_two = find_hairs(adj_trim) #finds hairs
	burn, burn_dict = find_branch_hairs(adj_trim, hair_dict, inds_two)
	full_makes, full_breaks, store = del_arcs(adj_trim, inds_two, burn)
	G, H, adj_new = draw_graph(adj_trim, hair_dict, inds_two, burn, full_makes, full_breaks)
	fin, i, adj_small = top_overlap(adj_new)
	fin_full, i_full, adj_full = top_overlap(adj)
	thresh = 4.5
	dn, Z, full = find_clusters(fin, 15, thresh)
	#G = plot_colours(adj_small, dn, fin)
	names = [x.name for x in mod.metabolites]
	go_df = pd.read_pickle('GO.pickle')
	kegg_df = pd.read_pickle('kegg_cleaned.pickle')
	full_clusts, paths_full, mets_full, funcs_full = define_clusters(dn, i, names, mod, go_df)
	#simp_adj, G, visual_style = connect_clusters(dn, adj_small, thresh, full)
	#plot_cairo(G, visual_style)
	gene_dict = defaultdict(None)
	for idx, val in enumerate(mod.genes):
		gene_dict[val.id] = idx
	#paths_full = cluster_func(dn, mod)
	green_subs = get_green_subgraphs(hair_dict, burn, adj) #need to add back in green_subs and store (these are the branches and arcs that were previously removed)
	new_mets = add_subs(mets_full, store, green_subs)
	clust_dict, clust_kegg, clust_path, genes, gene_names = get_clust_reacs(new_mets, stoich, mod, go_df, kegg_df)
	entropy = mutual_info(paths_full, mets_full)
	plot_freqs(clust_dict, thresh)
	x, y = shape_hierarchy(fin, Z)
	p = fit_clust(x, y)
	go = get_go_level(p, clust_dict, Z, x, 1, thresh)
	#genes = process_d_dict(genes)
	goea_quiet_sig = enrichment(genes)
  
	#v = clust.as_clustering()
