import networkx as nx
import cobra as cb
import igraph as ig
import seaborn as sns
import ast
import mod_code as mc
import pandas as pd
import pickle
from scipy.stats import entropy
import scipy.stats as st
import numpy as np
from ecoli import get_metabolism
from sklearn.cluster import AgglomerativeClustering
from cobamp.wrappers import KShortestEFMEnumeratorWrapper
from collections import defaultdict
from bioservices import KEGG
from colormap import rgb2hex
import random
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 20,10
from scipy.cluster import hierarchy


def define_clusters(fin, i, names, thresh):
	mets_full = []
	inds_full = defaultdict(None)
	inds_full_small = defaultdict(None)
	model = AgglomerativeClustering(n_clusters = None, affinity = 'euclidean', linkage  = 'complete', distance_threshold = thresh)
	clustering = model.fit(fin)
	for n in range(clustering.labels_.max() + 1):
		inds = np.where(clustering.labels_ == n)[0]
		inds_full[n] = i[0][inds]
		inds_full_small[n] = [inds]
		mets = names[i[0][inds]]
		mets_full.append(mets)
	m = clustering.distances_.max()
	labels = clustering.labels_
	col_list  = []
	for x in range(len(fin)):
		color = '#' + '%06x' % random.randint(0, 0xFFFFFF)
		col_list.append(color)
	return mets_full, inds_full, inds_full_small, m, labels, model, col_list


def plot_dendrogram(model, thresh, col_list, p):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
	hierarchy.set_link_color_palette(col_list)

    # Plot the corresponding dendrogram
	hierarchy.dendrogram(linkage_matrix, p = p, truncate_mode = 'level', color_threshold = thresh, no_labels = True)
	plt.savefig('toy_dend.pdf')
	return
		
		
#def define_clusters(dn

def get_clust_reacs(i, inds, stoich):
	reaction_inds = defaultdict(None)
	for key in inds:
		r = np.where(stoich[[inds[key]],:].sum(axis = 0))
		diff = np.where((stoich[[inds[key]],:].sum(axis = 0) - stoich.sum(axis = 0)) == 0)
		cols = set(diff[1]).intersection(set(r[1]))
		reaction_inds[key] = np.array(list(cols))
	return reaction_inds

def shape_hierarchy(fin, m):
	space = np.linspace(0, m, 100)
	full_n = []
	for x in space:
		clustering = AgglomerativeClustering(n_clusters = None, affinity = 'euclidean', linkage  = 'complete', distance_threshold = x).fit(fin)
		full_n.append(len(np.unique(clustering.labels_)))
		print(len(np.unique(clustering.labels_)))
	
	plt.plot(space, full_n, 'ko-') 	
	return space, full_n

def get_kegg_level(p, clust_kegg, m, x, n, thresh):
	plt.clf()
	nums = clust_kegg[n][1]
	x_data = np.linspace(1, max(x), len(nums))
	y_clust = np.exp(p[1]) * np.exp(p[0] * x_data)
	y_data = np.sort(nums)/nums.sum()
	y_data = y_data * ((max(y_clust) - min(y_clust))/max(y_data)) + min(y_clust)
	sns.lineplot(x = np.flipud(y_data), y = x_data, marker="o")
	plt.plot(y_clust, x_data, 'ko-')
	plt.show()
	p = np.argmin(np.abs(x_data - thresh))
	print(p)
	k = clust_kegg[n][0][p]
	return k


def get_gene_inds(cols, r, enz, mon, com):
	gene_inds = defaultdict(None)
	genes = defaultdict(None)
	for key in cols:
		id_list = []
		gene_list = []
		for i in cols[key]:
			r_id = r[i]
			if r_id in enz.keys():
				monomer = enz[r_id][0][:-3].replace('CPLX', 'MONOMER')
				if monomer in mon.index.values:
					inf = mon.loc[monomer]
					index = mon.index.get_loc(monomer)
					id_ = inf['id'].replace('_RNA', '')
					#id_ = inf.name.replace('-MONOMER', '')
					if id_ in com.keys():
						gene_list.append(com[id_][0])
		#			print(inf)
					id_list.append(index)
		gene_inds[key] = id_list
		genes[key] = gene_list
	return gene_inds, genes


def add_subs(inds_full, store, green_subs):
	small_list = []
	new_mets = inds_full.copy()
	for key in new_mets:
		for x in new_mets[key]:
			if x in store.keys():
				arcs = store[x]
				small_list = small_list + list(np.unique(arcs))
				#get the arc nodes and add to the cluster
			if x in green_subs.keys():
				hairs = green_subs[x]
				small_list = small_list + (list(hairs[0]))
				#get the hairs and add to the cluster
		#s = [int(x) for x in small_list]
		#print(s)
		new_mets[key] = np.append(new_mets[key], small_list)
		small_list = []
	for key in new_mets:
		new_mets[key] = [int(x) for x in new_mets[key]]
	return new_mets

def get_kegg_terms(genes, kegg_df):
	kegg_df = kegg_df.set_index('genes')
	kegg_clust = defaultdict(None)
	kegg_paths = defaultdict(None)
	for key in genes:
		terms_full = []
		paths_full = []
		for g in genes[key]:
			if g in kegg_df.index.values:
				terms = kegg_df.loc[g]['terms'] 
				paths = kegg_df.loc[g]['path']
				[terms_full.append(x) for x in terms]
				if paths:
					paths_full.append(paths)
		
		freqs = np.unique(terms_full, return_counts = True)
		order = np.argsort(freqs[1])
		path_freqs = np.unique(paths_full, return_counts = True)
		order_freqs = np.argsort(path_freqs[1])
		#print(key)
		kegg_clust[key] = (freqs[0][order], freqs[1][order])
		kegg_paths[key] = (path_freqs[0][order_freqs], path_freqs[1][order_freqs])
	
	return kegg_clust, kegg_paths

def get_col(inds_full):
	col_list = []
	for x in range(len(inds_full)):
		col = np.random.choice(range(256), size = 3)
		hexc = rgb2hex(col[0], col[1], col[2])
		col_list.append(hexc)
	return col_list


def connect_clusters(inds_full, adj_small, thresh, labels, col_list):
	#i = 0
	#i_dict = defaultdict(None)
	#for x in list(range(0, len(inds_full))):
	#	i_dict[x] = i
	#	i = i + 1
	simp_adj = np.zeros((len(inds_full), len(inds_full)))
	for key in inds_full:
		neighbours = np.where(adj_small[inds_full[key][0],:])[1]
		cols = labels[neighbours]
		c = np.unique(cols, return_counts=True)
		simp_adj[key, c[0]] = c[1]
		#simp_adj[cols, x] = 1

	np.fill_diagonal(simp_adj, 0)
	reac_graph = nx.from_numpy_matrix(simp_adj)
	weights = np.triu(simp_adj).flatten()
	visual_style = {}
	visual_style['edge_width'] = 10*(weights[np.where(weights)[0]]/max(weights))
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	G.vs['color'] = col_list[0:len(inds_full)] #get_col(inds_full)
	G.vs['size'] = 50
	G.vs['name'] = list(range(len(inds_full)))
	visual_style["vertex_label"] = G.vs["name"]
	visual_style['margins'] = 30
	layout = G.layout_kamada_kawai()
	fig = ig.plot(G, './figs/network_conc_clusts_sample_' + str(thresh) + '.pdf', layout = layout, **visual_style)
	return simp_adj, G, visual_style

def get_cluster_levels(fin, i, names):
	n = 12
	full_labels = np.zeros([n, len(fin)])
	for idx, val in enumerate(np.linspace(0, 11, n)):
		mets_full, inds_full, inds_full_small, m, labels, model, col_list = define_clusters(fin, i, names, val)
		full_labels[idx,:] = labels
	return full_labels

def get_competition(full_labels, inds_full, n, thresh, genes, mod):
	place = np.unique(full_labels[thresh + 1][inds_full[n]])
	above = np.where(full_labels[thresh + 1] == place)[0]
	clust = np.unique(full_labels[thresh + 1][above])

	#mets_full, inds_full, m, labels = define_clusters(fin, i, names, thresh)
	mets_up, inds_up, inds_full_small, m, labels, model, col_list = define_clusters(fin, i, names, thresh + 1)
	cols_up = get_clust_reacs(i, inds_up, mod.stoich)
	gene_ids, genes_up = get_gene_inds(cols_up, mod.reacs, mod.enzymes, monomer_df, mod.common_names)
	extra = list(set(genes_up[int(clust)]).difference(set(genes[n])))
	print(extra)
	print(genes)
	print(genes_up)
	return extra

def submodel(mets, reactions, mod):
	submodel = cb.Model('submodel')
	reactions = reactions.set_index('reaction id')
	
	r_inds = np.where(mod.stoich[mets, :])[1]	

	#define media from source_n (all of these need to be supplied into the module)
	#identify demand and sink metabolites from stoichiometric matrix
	for r in r_inds:
		reaction = cb.Reaction('r_' + str(r))
		r_name = mod.reacs[r]
		if r_name in reactions.index.values:
			reaction.name = r_name
			if reactions.loc[r_name]['is reversible']:
				reaction.lower_bound = -1000
			else:
				reaction.lower_bound = 0
			reaction.upper_bound = 1000
		#d = ast.literal_eval(reactions.loc[r_name]['stoichiometry'].replace('[', '_').replace('-', '_').replace(']', '').replace('_1', '-1'))
			d = ast.literal_eval(reactions.loc[r_name]['stoichiometry'])
			for key in d:
				n = key.replace('[', '').replace('-', '_')[:-2]
		#		if key in sink_n:
		#			n = n + '_c'
		#			vars()[n] = cb.Metabolite(n, compartment = 'c')
		#			submodel.add_metabolites({vars()[n] : d[key]})
		#			submodel.add_boundary(vars()[n], type = 'demand')
		#			sink_n.remove(key)
				n = n + '_c'
				vars()[n] = cb.Metabolite(n, compartment = 'c')
				reaction.add_metabolites({vars()[n] : d[key]}) 
			submodel.add_reaction(reaction)
	return submodel

def add_exchanges(submodel):
	submodel_new = submodel.copy()
	stoich = cb.util.array.create_stoichiometric_matrix(submodel_new)
	met_names = np.array([x.id for x in submodel_new.metabolites])
	#do all reactions first, then find sources and sinks from the cobra stoich, is there is a mis-match between the two matrices 
	#stoich = cb.util.array.create_stoichiometric matrix(model)
	use = np.where(stoich == -1)
	create = np.where(stoich == 1)
	
	sink = np.array(list(set(use[0]).difference(create[0]))) #in
	source = np.array(list(set(create[0]).difference(use[0]))) #out

	

	sink_n = list(met_names[source])
	source_n = list(met_names[sink])
	#source_r = r_inds[create[1][np.nonzero(np.in1d(create[0], source))[0]]]
	#sink_r = r_inds[use[1][np.nonzero(np.in1d(use[0], sink))[0]]]
	for x in source_n:
	#	n = x[:-2] + '_e'
	#	vars()[n] = cb.Metabolite(n, compartment = 'e')
		vars()[x] = cb.Metabolite(x, compartment = 'c')
		submodel_new.add_metabolites({vars()[x] : 1})
		submodel_new.add_boundary(vars()[x], type = 'sink')
		#reaction = cb.Reaction('r_' + x[:-2] + '_trns')
		#reaction.bounds = (-1000, 0)
		#reaction.add_metabolites({vars()[n]: -1, submodel_new.metabolites.get_by_id(x): 1})
		#submodel_new.add_reaction(reaction)
		#print(x) 
		#source_n.remove(x)

	for x in sink_n:
		#n = x[:-2] + '_e'
		#vars()[n] = cb.Metabolite(n, compartment = 'e')
		vars()[x] = cb.Metabolite(x, compartment = 'c')
		submodel_new.add_metabolites({vars()[x] : 1})
		submodel_new.add_boundary(vars()[x], type = 'demand')
		#reaction = cb.Reaction('r_' + x[:-2] + '_trns')
		#reaction.bounds = (0, 1000)
		#reaction.add_metabolites({vars()[n]: -1, submodel_new.metabolites.get_by_id(x): 1})
		#submodel_new.add_reaction(reaction) 
		#sink_n.remove(x)

	return submodel_new, sink, source



def add_hubs(submodel):
	ATP_e = cb.Metabolite('ATP_e', compartment = 'e')
	WATER_e = cb.Metabolite('WATER_e', compartment = 'e')
	PROTON_e = cb.Metabolite('PROTON_e', compartment = 'e') 
	submodel.add_metabolites({ATP_e: 1})
	submodel.add_metabolites({WATER_e: 1})
	submodel.add_metabolites({PROTON_e: 1})
	submodel.add_boundary(ATP_e, type = 'exchange')
	submodel.add_boundary(WATER_e, type = 'exchange')
	submodel.add_boundary(PROTON_e, type = 'exchange')
	reaction = cb.Reaction('r_0')
	reaction.bounds = (-1000, 1000)
	ATP_c = submodel.metabolites.ATP_c
	reaction.add_metabolites({ATP_e: -1, ATP_c: 1})
	submodel.add_reaction(reaction)
	reaction = cb.Reaction('r_1')
	reaction.bounds = (-1000, 1000)
	PROTON_c = submodel.metabolites.PROTON_c
	reaction.add_metabolites({PROTON_e: -1, PROTON_c: 1})
	submodel.add_reaction(reaction)
	reaction = cb.Reaction('r_2')
	reaction.bounds = (-1000, 1000)
	WATER_c = submodel.metabolites.WATER_c
	reaction.add_metabolites({WATER_e: -1, WATER_c: 1})
	submodel.add_reaction(reaction)


	return submodel

def bipartite(submodel, efm):
	submodel_small = submodel.copy()
	for r in submodel.reactions:
		if r.id not in efm.keys():
			submodel_small.remove_reactions([r.id]) 
	stoich = cb.util.array.create_stoichiometric_matrix(submodel_small)
	d = np.array([efm[key] for key in efm])
	stoich = stoich * d
	#order = np.argsort(stoich.sum(axis = 0)/np.abs(stoich).sum(axis = 0))
	#stoich = stoich[:, order]
	B = nx.DiGraph()
	r_names = np.array([x.id.replace('_', '-') for x in submodel_small.reactions])
	m_names = np.array([x.id.replace('_', '-') for x in submodel_small.metabolites])
	#m = np.shape(stoich)[0]
	#r = np.shape(stoich)[1]
	#B.add_nodes_from(list(range(m)), bipartite = 0)
	#B.add_nodes_from(list(range(m, m+r)), bipartite = 1)
	m = 0
	m_names_small = []
	for row in stoich:
		reacs = np.where(row)[0]
		for x in reacs:
			if row[x] == 1:
				B.add_edge(r_names[x], m_names[m], color = '#b0332a')
			elif row[x] == -1:
				B.add_edge(m_names[m], r_names[x], color = 'k')
			m_names_small.append(m_names[m])
		m = m + 1	
	
	#left, right = nx.bipartite.sets(B)
	
	pos = dict()
	pos.update( (n, (1, i + 1)) for i, n in enumerate(r_names) ) # put nodes from X at x=1
	pos.update( (n, (1.1, i + 1 )) for i, n in enumerate(np.unique(m_names_small)) ) # put nodes from Y at x=2

	pos_labels = dict()
	pos_labels.update( (n, (1, i + 1 - 0.35)) for i, n in enumerate(r_names) ) # put nodes from X at x=1
	pos_labels.update( (n, (1.1, i + 1 - 0.35)) for i, n in enumerate(np.unique(m_names_small)) ) # put nodes from Y at x=2

	colors = [B[u][v]['color'] for u,v in B.edges()]
	
	#node_shapes = dict()
	#node_shapes.update((n, 'h') for n in r_names)
	#node_shapes.update((n, 'o') for n in m_names)

	#nx.draw_networkx(B, arrows = True, arrowstyle = 'simple', edge_color = colors, with_labels = False, node_size = 300, pos=pos)
	nx.draw_networkx_nodes(B, node_size = 2000, pos = pos, node_color = '#dcf1f2', edgecolors = 'k')
	nx.draw_networkx_edges(B, pos = pos, width = 0.1, edge_color = colors)
	nx.draw_networkx_labels(B, font_size = 20, pos = pos_labels)
#	plt.show()
	plt.margins(x=0.3, y=0.1, tight=None)
	plt.savefig('bipartite_efm_val.pdf')
	return pos, B, m_names_small

def get_efm_inds(efm_list, mod, monomer_df, sm_new):
	keep = []
#	for efm in efm_list:
	for key in efm_list:
		r = sm_new.reactions.get_by_id(key)
		if r.name in mod.enzymes.keys():
			n = '-'.join(mod.enzymes[r.name][0].split('-')[:-1]) + '-MONOMER' #replace('CPLX', 'MONOMER')
			if n in monomer_df.index:
				print(n)
				ind = monomer_df.index.get_loc(n)
				keep.append(ind)
	return keep 


def sub_net(efm, sm_new):
	sub = cb.Model()
	for key in efm:
		reaction = sm_new.reactions.get_by_id(key)
		sub.add_reaction(reaction)
	return sub

def plot(submodel, sink, source):
	stoich = cb.util.array.create_stoichiometric_matrix(submodel)
	reac_graph = nx.from_numpy_matrix(np.matmul(stoich, stoich.transpose()))
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	for x in source:
		G.vs[x]['color'] = 'blue'

	for x in sink:
		G.vs[x]['color'] = 'pink'

	layout = G.layout_kamada_kawai()
	fig = ig.plot(G, 'sub_network_met.pdf', layout = layout) 
	return G



def efms(model, met):
	#model_url = 'http://bigg.ucsd.edu/static/models/e_coli_core.xml'
	#model_path, model_content = urllib.request.urlretrieve(model_url)	
	#model = cb.io.sbml.read_sbml_model(model_path)

	ksefm = KShortestEFMEnumeratorWrapper(
		model=model,
		non_consumed=[],
		non_produced=[],
		consumed=[],
		produced=[met],
		algorithm_type=KShortestEFMEnumeratorWrapper.ALGORITHM_TYPE_POPULATE,
		#algorithm_type=KShortestEFMEnumeratorWrapper.ALGORITHM_TYPE_ITERATIVE,
		stop_criteria=5000,
		solver = 'GUROBI'
		#solver='CPLEX'
		)

	enumerator = ksefm.get_enumerator()
	efm_list = []
	while len(efm_list) == 0:
		efm_list += next(enumerator)

	return efm_list

def plot_efm(submodel, efm):

	stoich = cb.util.array.create_stoichiometric_matrix(submodel)
	adj = np.matmul(np.transpose(stoich), stoich)

	col_list = []
	for idx, val in enumerate(submodel.reactions):
		if val.id in efm.keys():
			col_list.append('#e04a75')
		else:
			col_list.append('#e3dce0')


	reac_graph = nx.from_numpy_matrix(adj)
	G = ig.Graph()
	G.add_vertices(reac_graph.nodes)
	G.add_edges(reac_graph.edges)
	G.vs['color'] = col_list
	G.vs['size'] = 50
	visual_style = {}
	visual_style['margin'] = 100
	layout = G.layout_kamada_kawai()
	visual_style['layout'] = layout
	fig = ig.plot(G, 'efm_net.pdf', **visual_style)
	return

def plot_hist(adj):
	s = adj.sum(axis = 1)
	degree = s[s < 70]
	ks = st.kstest(s, 'expon')
	sns.histplot(degree, stat = 'probability', bins = 70)
	P = st.expon.fit(degree)
	#P = st.powerlaw.fit(degree)
	n = int(degree.min())
	m = int(degree.max())
	rX = np.linspace(n, m, m -1)
	rP = st.expon.pdf(rX, *P)
	#rP = st.powerlaw.pdf(rX, *P)
	plt.plot(rX, rP, 'k')
	plt.xlabel('Degree')
	plt.ylabel('Frequency')
	return

def powerlaw(adj)
	s = adj.sum(axis = 1)
	degree = s[s < 70]
	sns.histplot(degree, stat = 'probability', bins = 70)
	P = powerlaw.Fit(degree)
	n = int(degree.min())
	m = int(degree.max())
	rX = np.linspace(n, m, 100)
	a = P.alpha
	plt.plot(rX+P.xmin, rX**(-a), 'k')
	plt.xlabel('Degree')
	plt.ylabel('Frequency')
	return


	
if __name__ == '__main__':
	mod = pickle.load(open('ecoli_data.pickle', 'rb'))
	reactions = pd.read_csv('/home/sl13479/Documents/wholecell3/wcEcoli/reconstruction/ecoli/flat/reactions.tsv', delimiter = '\t')
	metabolites = pd.read_csv('/home/sl13479/Documents/wholecell3/wcEcoli/reconstruction/ecoli/flat/metabolites.tsv', delimiter = '\t')
	adj = np.matmul(mod.stoich, np.transpose(mod.stoich))
	adj[adj != 0] = 1
	adj_trim = mc.find_outliers(adj)
	hair_dict, inds_two = mc.find_hairs(adj_trim) #finds hairs
	burn, burn_dict = mc.find_branch_hairs(adj_trim, hair_dict, inds_two)
	full_makes, full_breaks, store = mc.del_arcs(adj_trim, inds_two, burn)
	G, H, adj_new = mc.draw_graph(adj_trim, hair_dict, inds_two, burn, full_makes, full_breaks)
	fin, i, adj_small = mc.top_overlap(adj_new)
	fin_full, i_full, adj_full = mc.top_overlap(adj)
	thresh = 6.5
	dn, Z, full = mc.find_clusters(fin, 30, thresh)
	#go_df = pd.read_pickle('GO.pickle')
	#kegg_df = pd.read_pickle('kegg_cleaned.pickle')
	kegg_df = pd.read_pickle('kegg_df.pickle')
	go_df = pd.read_pickle('GO_full.pickle')
	green_subs = mc.get_green_subgraphs(hair_dict, burn, adj) 
#	new_mets = mc.add_subs(mets_full, store, green_subs)
	names = np.array(list(mod.mets.keys()))
	mets_full, inds_full, inds_full_small, m, labels, model, col_list = define_clusters(fin, i, names, thresh)
	simp_adj, G, visual_style = connect_clusters(inds_full_small, adj_small, thresh, labels, col_list)
	plot_dendrogram(model, thresh, col_list, 15)
	mc.plot_cairo(G, visual_style)
	new_mets = add_subs(inds_full, store, green_subs)
	cols = get_clust_reacs(i, inds_full, mod.stoich)
	monomer_df = mod.r.copy()
	monomer_df = monomer_df.set_index('monomer_id')
	gene_ids, genes = get_gene_inds(cols, mod.reacs, mod.enzymes, monomer_df, mod.common_names)
	#goea_quiet_sig = mc.enrichment(genes[0])
	kegg_clust, kegg_paths = get_kegg_terms(genes, kegg_df)
	x, y = shape_hierarchy(fin, m)
	p = mc.fit_clust(x, y)
	#kegg = get_kegg_level(p, kegg_clust, m, x, 1, thresh)
	sm = submodel(inds_full[2], reactions, mod)
	sm_new, sink, source = add_exchanges(sm)
	efm_list = efms(sm_new, met)
	keep =  get_efm_inds(efm_list, mod, monomer_df, sm_new)
	KOs = set(gene_ids[31]).difference(keep)
	
	#use col indices to get reaction names (from mod.reacs)
	#get enzymes using reactions from mod.enzymes
	#get monomers from enzymes using mod.monomers --- the ones you need are in the monomer_id column, so you need to do set_index, and then I guess get the index and the gene name as well
	#get a list of indices and then KO
	#hooray!!




