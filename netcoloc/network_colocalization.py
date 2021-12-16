# -*- coding: utf-8 -*-

"""Functions for performing network colocalization
"""

import warnings
import logging

# External library imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance

logger = logging.getLogger(__name__)


def calculate_network_overlap(z_scores_1, z_scores_2, z_score_threshold=3,
                              z1_threshold=1.5, z2_threshold=1.5):
    """
    Function to determine which genes overlap. Returns a list of the
    overlapping genes

    :param z_scores_1: Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                       or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                       containing the z-scores of each gene following network
                       propagation. The index consists of gene names
    :type z_scores_1: :py:class:`pandas.Series` or :py:class: `numpy.ndarray` or :py:class: `pandas.DataFrame`
    :param z_scores_2: Similar to **z_scores_1**. This and **z_scores_1**
                       must contain the same genes (ie. come from the same
                       interactome network)
    :type z_scores_2: :py:class:`pandas.Series` or :py:class: `numpy.ndarray` or :py:class: `pandas.DataFrame`
    :param z_score_threshold: threshold to determine whether a gene is
        a part of the network overlap or not. Genes with combined z-scores
        below this threshold will be discarded
    :type z_score_threshold: float
    :param z1_threshold: individual z1-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z1-scores
        below this threshold will be discarded
    :type z1_threshold: float
    :param z2_threshold: individual z2-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z2-scores
        below this threshold will be discarded
    :type z2_threshold: float
    :return: genes in the network overlap (genes with high combined
            z-scores)
    :rtype: list
    """
    if isinstance(z_scores_1, pd.Series):
        z_scores_1 = z_scores_1.to_frame(name='z_scores_1')
        z_scores_2 = z_scores_2.to_frame(name='z_scores_2')
    elif isinstance(z_scores_1, np.ndarray):
        z_scores_1 = pd.DataFrame({"z_scores_1": z_scores_1})
        z_scores_2 = pd.DataFrame({"z_scores_2": z_scores_2})
    else:
        z_scores_1.columns = ["z_scores_1"]
        z_scores_2.columns = ["z_scores_2"]
    z_scores_joined = z_scores_1.join(z_scores_2)
    z_scores_combined = (z_scores_joined['z_scores_1']
                         * z_scores_joined['z_scores_2']
                         * (z_scores_joined['z_scores_1'] > 0)
                         * (z_scores_joined['z_scores_2'] > 0))
    # get rid of unlikely genes which have low scores in either z1 or z2
    high_z_score_genes = z_scores_combined[(z_scores_combined >= z_score_threshold)
                                           & (z_scores_joined['z_scores_1'] > z1_threshold)
                                           & (z_scores_joined['z_scores_2'] > z2_threshold)
                                           ].index.tolist()

    return high_z_score_genes


def calculate_network_overlap_subgraph(interactome, z_scores_1,
                                       z_scores_2, z_score_threshold=3,
                                       z1_threshold=1.5, z2_threshold=1.5):
    """
    Function to return subgraph of network intersection.

    Code to create subgraph is from
    `NetworkX subgraph documentation:
    <https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html>`__

    :param interactome: network whose subgraph will be returned
    :type interactome: :py:class:`networkx.Graph`
    :param z_scores_1: Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                       or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                       containing the z-scores of each gene following network
                       propagation. The index consists of gene names
    :type z_scores_1: :py:class:`pandas.Series`
    :param z_scores_2: Similar to **z_scores_1**. This and **z_scores_1**
                       must contain the same genes (ie. come from the same
                       interactome network)
    :type z_scores_2: :py:class:`pandas.Series`
    :param z_score_threshold: threshold to determine whether a gene is
        a part of the network overlap or not. Genes with combined z-scores
        below this threshold will be discarded
    :type z_score_threshold: float
    :param z1_threshold: individual z1-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z1-scores
        below this threshold will be discarded
    :type z1_threshold: float
    :param z2_threshold: individual z2-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z2-scores
        below this threshold will be discarded
    :type z2_threshold: float
    :return: Subgraph of the interactome containing only genes that
            are in the network intersection (genes with high combined z-scores)
    :rtype: :py:class:`networkx.Graph`
    """
    network_overlap = calculate_network_overlap(z_scores_1, z_scores_2, z_score_threshold=z_score_threshold,
                                                z1_threshold=z1_threshold, z2_threshold=z2_threshold)

    # Create subgraph that has the same type as original graph
    network_overlap_subgraph = interactome.__class__()
    network_overlap_subgraph.add_nodes_from((node, interactome.nodes[node]) for node in network_overlap)
    if network_overlap_subgraph.is_multigraph():
        network_overlap_subgraph.add_edges_from((node, neighbor, key, dictionary)
                                                for node, neighbors in interactome.adj.items() if
                                                node in network_overlap
                                                for neighbor, item in neighbors.items() if neighbor in network_overlap
                                                for key, dictionary in item.items())
    else:
        network_overlap_subgraph.add_edges_from((node, neighbor, dictionary)
                                                for node, neighbors in interactome.adj.items() if
                                                node in network_overlap
                                                for neighbor, dictionary in neighbors.items() if
                                                neighbor in network_overlap)
    network_overlap_subgraph.graph.update(interactome.graph)

    return network_overlap_subgraph


def calculate_expected_overlap(z_scores_1, z_scores_2, seed1=None, seed2=None,
                               z_score_threshold=3, z1_threshold=1.5, z2_threshold=1.5,
                               num_reps=1000, plot=False, overlap_control=None):
    """
    Determines size of expected network overlap by randomly
    shuffling gene names

    :param z_scores_1: Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                       or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                       containing the z-scores of each gene following network
                       propagation. The index consists of gene names
    :type z_scores_1: :py:class:`pandas.Series`
    :param z_scores_2: Similar to **z_scores_1**. This and **z_scores_1**
                       must contain the same genes (ie. come from the same
                       interactome network)
    :type z_scores_2: :py:class:`pandas.Series`
    :param seed1: List of seed genes for trait 1
    :param seed2: List of seed genes for trait 2
    :param z_score_threshold: threshold to determine whether a gene is
        a part of the network overlap or not. Genes with combined z-scores
        below this threshold will be discarded
    :type z_score_threshold: float
    :param z1_threshold: individual z1-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z1-scores
        below this threshold will be discarded
    :type z1_threshold: float
    :param z2_threshold: individual z2-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z2-scores
        below this threshold will be discarded
    :type z2_threshold: float
    :param num_reps:
    :param plot: If ``True``, distribution will be plotted
    :type plot: bool
    :param overlap_control: How should overlapping seed genes be accounted for when calculating expected overlap?
        None = no control
        "remove" = exclude overlapping seed genes from the analysis
        "bin" = bin all genes into "overlapping seed genes" and "all other genes" and calculate expectation for both
                bins separately
    :return:
    :rtype: float
    """
    assert type(z_scores_1) == type(z_scores_2), "z_scores_1 and z_scores_2 must be of the same type, " \
                                                 "either pd.Series or pd.DataFrame"
    # Build a distribution of expected network overlap sizes by shuffling node names
    if isinstance(z_scores_1, pd.DataFrame):  # merge dataframe z scores
        z1z2 = z_scores_1.join(z_scores_2, lsuffix="1", rsuffix="2")
        z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    elif isinstance(z_scores_1, pd.Series):  # merge series z scores
        z1z2 = pd.concat([z_scores_1, z_scores_2], axis=1)
    else:
        raise TypeError("z_scores must be either pd.Series or pd.DataFrame")
    # Account for overlaps in seed gene sets
    overlap_z1 = None
    overlap_z2 = None
    if overlap_control == "remove":  # remove overlapping seed genes from the analysis
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":  # remove overlapping seed genes from main set and keep as separate set
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        overlap_z2 = np.array(overlap_z1z2.z2)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)

    # Get observed network size
    network_overlap_size = len(calculate_network_overlap(z1z2.z1, z1z2.z2,
                                                         z_score_threshold=z_score_threshold,
                                                         z1_threshold=z1_threshold,
                                                         z2_threshold=z2_threshold))

    random_network_overlap_sizes = np.zeros(num_reps)
    for i in tqdm(range(num_reps)):
        # perm_z1z2 = np.zeros(len(z1))
        rn.shuffle(z1)
        perm_size = len(calculate_network_overlap(z1, z2,
                                                  z_score_threshold=z_score_threshold,
                                                  z1_threshold=z1_threshold,
                                                  z2_threshold=z2_threshold))
        if overlap_control == "bin":  # perform separate permutation for overlapping seed genes
            # overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            rn.shuffle(overlap_z1)
            perm_size_overlap = len(calculate_network_overlap(overlap_z1, overlap_z2,
                                                              z_score_threshold=z_score_threshold,
                                                              z1_threshold=z1_threshold,
                                                              z2_threshold=z2_threshold))

            perm_size += perm_size_overlap  # combine to get total conserved size

        random_network_overlap_sizes[i] = perm_size

    if plot:
        plt.figure(figsize=(5, 4))
        dfig = sns.histplot(random_network_overlap_sizes,
                            label='Expected network intersection size')
        plt.vlines(network_overlap_size, ymin=0, ymax=dfig.dataLim.bounds[3], color='r',
                   label='Observed network intersection size')
        plt.xlabel('Size of proximal subgraph, z > ' + str(z_score_threshold),
                   fontsize=16)
        plt.legend(fontsize=12)

    return network_overlap_size, random_network_overlap_sizes


def transform_edges(G, method='cosine_sim', edge_weight_threshold=0.95):
    """
    Transforms binary edges using selected method (currently only cosine similarity is implemented).
    Cosine similarity measures the similarity between neighbors of node pairs in the input network

    :param G: network whose edges will be transformed
    :type G: :py:class:`networkx.Graph`
    :param method: Method to use, only ``cosine_sim`` supported. Any other value will
                   cause this method to output a warning and immediately return
    :type method: str
    :param edge_weight_threshold: Transformed edges will be returned which have values greater than this
    :type edge_weight_threshold: float
    :return: Graph with nodes identical to input G, but with transformed edges (values > edge_weight_threshold)
    :rtype: :py:class:`networkx.Graph`
    """

    if method not in ['cosine_sim']:  # update this if we add more methods
        warnings.warn('Error: ' + method + ' method not yet implemented')
        return

    # compute the adjacency matrix
    logging.info('computing the adjacency matrix...')

    nodelist = list(G.nodes())
    # get graph as adjacency matrix
    graph_as_adj = nx.to_numpy_array(G)

    # add transpose to matrix to remove edge direction
    adj_temp = graph_as_adj + np.transpose(graph_as_adj)

    # compute the cosine similarity
    logging.info('computing the cosine similarity...')
    cos_pc = pd.DataFrame(np.zeros((adj_temp.shape[0],
                                    adj_temp.shape[1])),
                          index=nodelist)
    cos_pc.columns = nodelist

    for i in np.arange(0, len(nodelist) - 1):
        n1 = nodelist[i]
        # this node has no neighbors so set
        # cosine distance to maximum distance aka 1.0
        # in cos_pc and continue
        if max(adj_temp[i]) < 1.0:
            for j in np.arange(i + 1, len(nodelist)):
                n2 = nodelist[j]
                cos_pc.loc[n1][n2] = 1.0
                cos_pc.loc[n2][n1] = 1.0
            continue
        for j in np.arange(i + 1, len(nodelist)):
            n2 = nodelist[j]

            # make sure they have some neighbors
            if max(adj_temp[j]) > 0.0:
                cosine_distance = distance.cosine(adj_temp[i],
                                                  adj_temp[j])
            else:
                # no neighbors in neigh2 so set
                # cosine distance to maximum distance aka 1.0
                cosine_distance = 1.0

            cos_pc.loc[n1][n2] = cosine_distance
            cos_pc.loc[n2][n1] = cosine_distance

    # Rank transform 1-cos distance
    logger.info('rank transforming...')
    m1cos = 1 - cos_pc
    m1cos = m1cos.replace(np.nan, 0)
    sim_names = m1cos.index.tolist()
    sim_rank = m1cos.rank(0) / (m1cos.shape[0] - 1)
    sim_rank = pd.DataFrame((sim_rank.values + sim_rank.values.T) / 2.0,
                            columns=sim_names, index=sim_names)

    # remove self edges
    sim_rank.values[[np.arange(sim_rank.shape[0])] * 2] = 0

    sim_rank['gene_temp'] = sim_rank.index.tolist()
    sim_rank_EL = sim_rank.melt(id_vars=['gene_temp'])
    sim_rank_EL.columns = ['node1', 'node2', 'sim']
    sim_rank_EL = sim_rank_EL[sim_rank_EL['sim'] > edge_weight_threshold]
    G_transform = nx.Graph()
    G_transform.add_nodes_from(G)
    G_transform.add_weighted_edges_from(zip(sim_rank_EL['node1'],
                                            sim_rank_EL['node2'],
                                            sim_rank_EL['sim']))

    logger.info('number of transformed edges returned = ' +
                str(len(G_transform.edges())))

    return G_transform


def calculate_mean_z_score_distribution(z1, z2, num_reps, zero_double_negatives=False,
                                        overlap_control="remove", seed1=None, seed2=None):
    """
    Calculates the mean colocalization score (z1 * z2) over all genes for observed data and randomly permuted data.
    :param z1: z_scores for all genes for trait 1
    :type z1: :py:class: `pandas.DataFrame`
    :param z2: z_scores for all genes for trait 2
    :type z2: :py:class: `pandas.DataFrame`
    :param num_reps: Number of permutations to perform
    :type num_reps: int
    :param zero_double_negatives: Should genes receiving a negative zscore in both traits be zeroed out before
        calculating colocalization?
    :type zero_double_negatives: bool
    :param overlap_control: How should overlapping seed genes be accounted for when calculating expected overlap?
        None = no control
        "remove" = exclude overlapping seed genes from the analysis
        "bin" = bin all genes into "overlapping seed genes" and "all other genes" and calculate expectation for both
                bins separately
    :type overlap_control: str or None
    :param seed1: List of seed genes for trait 1
    :type seed1: list
    :param seed2: List of seed genes for trait 2
    :type seed2: list
    :returns: observed mean coloc score, array of means of permuted coloc scores
    """
    z1z2 = z1.join(z2, lsuffix="1", rsuffix="2")
    z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    overlap_z1 = None
    if overlap_control == "remove":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)
    if zero_double_negatives:
        for node in z1z2.index:
            if (z1z2.loc[node].z1 < 0) and (z1z2.loc[node].z2 < 0):
                z1z2.loc[node, 'zz'] = 0
    permutation_means = np.zeros(num_reps)
    for i in tqdm(range(num_reps)):
        perm_z1z2 = np.zeros(len(z1))
        np.random.shuffle(z1)

        for node in range(len(z1)):
            if not zero_double_negatives or not (z1[node] < 0 and z2[node] < 0):
                perm_z1z2[node] = z1[node] * z2[node]
            else:
                perm_z1z2[node] = 0
        if overlap_control == "bin":
            overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            np.random.shuffle(overlap_z1)
            for node in range(len(overlap_z1)):
                if zero_double_negatives and (overlap_z1[node] < 0 and z2[node] < 0):
                    overlap_perm_z1z2[node] = 0
                else:
                    overlap_perm_z1z2[node] = overlap_z1[node] * z2[node]
            perm_z1z2 = np.concatenate([perm_z1z2, overlap_perm_z1z2])

        permutation_means[i] = np.mean(perm_z1z2)
    return np.mean(z1z2.zz), permutation_means


def get_p_from_permutation_results(observed, permuted):
    """
    Calculates the significance of the observed mean relative to the empirical normal distribution of permuted means.
    :param observed: observed value
    :param permuted: vector of values from permutation
    :returns: p value of observed based on zscore of permutation distribution
    """
    p = norm.sf((observed-np.mean(permuted))/np.std(permuted))
    p = round(p, 4 - int(math.floor(math.log10(abs(p)))) - 1)
    return p
