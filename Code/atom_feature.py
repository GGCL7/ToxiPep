import numpy as np
from rdkit import Chem
from typing import List

aa_dict = {
    'a': 'C[C@H]([NH2:1])[C:1](=[O:1])[O:1]',
    'c': '[NH2:1][C@@H](CS)[C:1](=[O:1])[O:1]',
    'd': '[NH2:1][C@@H](CC(=O)O)[C:1](=[O:1])[OH:1]',
    'e': '[NH2:1][C@@H](CCC(=O)O)[C:1](=[O:1])[OH:1]',
    'f': '[NH2:1][C@@H](Cc1ccccc1)[C:1](=[O:1])[OH:1]',
    'g': '[NH2:1]C[C:1](=[O:1])[OH:1]',
    'h': '[NH2:1][C@@H](Cc1c[nH]cn1)[C:1](=[O:1])[OH:1]',
    'i': 'CC[C@H](C)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'k': 'NCCCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'l': 'CC(C)C[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'm': 'CSCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'n': 'NC(=O)C[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'p': '[O:1]=[C:1]([OH:1])[C@@H]1CCC[NH:1]1',
    'q': 'NC(=O)CC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'r': 'N=C(N)NCCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    's': '[NH2:1][C@@H](CO)[C:1](=[O:1])[OH:1]',
    't': 'C[C@@H](O)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'v': 'CC(C)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'w': '[NH2:1][C@@H](Cc1c[nH]c2ccccc12)[C:1](=[O:1])[OH:1]',
    'y': '[NH2:1][C@@H](Cc1ccc(O)cc1)[C:1](=[O:1])[OH:1]'
}


__all__ = ["pre_known_charges", "aa_dict"]

gasteiger_charges = {}

pre_known_charges = {
    'a': [-0.038941544836773818, 0.10498306785206031, -0.31852489852720955, 0.37128096695769647, -0.245609700240111,
          -0.245609700240111],
    'c': [-0.31760762075618459, 0.11671109552669053, 0.016444904280613366, -0.17711583286998214, 0.37240697866736511,
          -0.24556022418544537, -0.24556022418544537],
    'd': [-0.31795147828443121, 0.11365865612305733, 0.071241260785456353, 0.30503469709389208, -0.2523968033036173,
          -0.48117538323698905, 0.32057810707028217, -0.25058130351573815, -0.48007458571726869],
    'e': [-0.3184276525520085, 0.10368324144471773, -0.016649349421885624, 0.047372069789650653, 0.30291723427615275,
          -0.25248442381067765, -0.48122949062943815, 0.31998403007662279, -0.2506002043614457, -0.48008627987892583],
    'f': [-0.31816209755519703, 0.10718797936647743, -0.0017896024515534755, -0.045878604848919036,
          -0.058969174493864206, -0.06199304993033751, -0.06224871979673649, -0.06199304993033751,
          -0.058969174493864206, 0.32031550163956013, -0.25058631033747231, -0.48007770020344237],
    'g': [-0.32100388796548518, 0.091814239632494035, 0.31675597840558584, -0.25094762709006846, -0.48029713002439572],
    'h': [-0.31811218613827036, 0.10876025651267705, 0.017178897164274905, 0.060240393418915093, 0.023612279812735563,
          -0.35090363274243741, 0.092256578126356181, -0.24194850421171329, 0.32037773191321722, -0.25058513638855856,
          -0.48007696992659998],
    'i': [-0.065054342792623523, -0.051516109532843919, -0.01848141138509252, -0.060455687487424026, 0.1056921548136058,
          -0.31821728010260475, 0.3202466808084985, -0.25058798571573865, -0.48007873928516198],
    'k': [-0.33047349405550447, -0.0077263173078670703, -0.040819171235958823, -0.05005067194555176,
          -0.027103291386504948, 0.10312285864403094, -0.31844107224794488, 0.31996730024702658, -0.2506004737905258,
          -0.480086447570925],
    'l': [-0.062699721671212155, -0.045080166422972974, -0.062699721671212155, -0.024825714877267389,
          0.10336652779889306, -0.31843253370427399, 0.31997794662058598, -0.25060026439075556, -0.48008631732626356],
    'm': [-0.018398913027632899, -0.1653569719042017, -0.00497659329816642, -0.018028983910199952, 0.10391068699378787,
          -0.31841548940258046, 0.31999919762538026, -0.25059986574145887, -0.48008606933473558],
    'n': [-0.3696138646860766, 0.21908715699891673, -0.27528806213181317, 0.052212076976201664,
          0.11193696290241258, -0.31800778429812587, 0.32050790323073358, -0.25058264416049209,
          -0.48007541965947526],
    'p': [-0.25059127541388582, 0.32019187971860097, -0.48008076676695011, 0.1056602413452867,
          -0.025726752171922784, -0.038583656372423773, -0.0040133371139500825, -0.30441260474699489],
    'q': [-0.36967889475861154, 0.21698956994421906, -0.27537278910866281, 0.028297174649971519,
          -0.018362474406193877, 0.10361820954695432, -0.31842860380192411, 0.31998284457023513,
          -0.25060021491233908, -0.48008628646499379],
    'r': [-0.26962207123355619, 0.18520502814523329, -0.37016043799114567, -0.35676207127350623,
          0.017531796964524702, -0.03363639331536647, -0.025476863927366322, 0.10318774485000681,
          -0.31844010091909719, 0.31996851078193783, -0.25060046290166194, -0.48008644077399965],
    's': [-0.31651542552659373, 0.12646568376767636, 0.069022807911391801, -0.39413557675578653,
          0.32237114684868762, -0.25049509689415916, -0.4800213961433728],
    't': [-0.036810537847947578, 0.077045484390140126, -0.39117403578585813, 0.12904528397618095,
          -0.31628111187635122, 0.32266366773949673, -0.25048182094826726, -0.4800132023099517],
    'v': [-0.060711309783474096, -0.021064293477971893, -0.060711309783474096, 0.10543802912336016,
          -0.31822595394937808, 0.32023586583935859, -0.25058819622173711, -0.48007887022192064],
    'w': [-0.31816095861050087, 0.10726664689345959, 0.00032329871948459876, -0.020502848541995324,
          0.0049744588987986468, -0.36088879768036697, 0.045629818510211806, -0.038003021926457513,
          -0.060158911668000872, -0.061509233239626836, -0.052562691519528068, 0.0026344222033379563,
          0.32031692105916915, -0.25058629777133862, -0.48007769235935871],
    'y': [-0.31816209743565643, 0.10718801610357671, -0.0017850394584956058, -0.045610041081834043,
          -0.055294427338870203, -0.019954155177891012, 0.11509485950992565, -0.50796664517588408,
          -0.019954155177891012, -0.055294427338870203, 0.32031550178849594, -0.25058631033716872,
          -0.48007770020325236]
}
all_kinds_of_aminos = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w',
                       'y']



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_ring_info(atom):

    ring_info_feature = []
    for i in range(5, 7):
        ring_info_feature.append(1 if atom.IsInRingSize(i) else 0)
    return ring_info_feature

# 获取原子特征的函数
def atom_feature(atom) -> np.ndarray:

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [1, 2, 3]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) +
                    [atom.GetIsAromatic()] +
                    get_ring_info(atom) +
                    [int(atom.GetHybridization()) + 1])


def norm_Adj(adjacency):
    I = np.array(np.eye(adjacency.shape[0]))
    adj_hat = adjacency + I
    D_hat = np.diag(np.power(np.array(adj_hat.sum(1)), -0.5).flatten(), 0)  # Degree matrix
    adj_Norm = adj_hat.dot(D_hat).transpose().dot(D_hat)  # Normalized adjacency matrix
    return adj_Norm


def norm_fea(features):
    norm_fea_ = features / features.sum(1).reshape(-1, 1)  # 归一化节点特征矩阵
    return norm_fea_


def convert_to_graph_channel(seq: str, max_seq_len=50):

    seq = seq.lower()
    maxNumAtoms = 15
    graphFeaturesOneSeq = []
    seqSMILES = [aa_dict[b] for b in seq]

    for i_, aminoAcidSMILES in enumerate(seqSMILES):
        aaMol = Chem.MolFromSmiles(aminoAcidSMILES)


        AdjTmp = Chem.GetAdjacencyMatrix(aaMol)
        AdjNorm = norm_Adj(AdjTmp)


        numAtoms = AdjNorm.shape[0]
        if numAtoms <= maxNumAtoms:
            graphFeature = np.zeros((maxNumAtoms, 21))


            nodeFeatureTmp = [atom_feature(atom) for atom in aaMol.GetAtoms()]
            nodeFeatureNorm = norm_fea(np.asarray(nodeFeatureTmp))


            paddedNodeFeatureNorm = np.zeros((numAtoms, 20))
            paddedNodeFeatureNorm[:nodeFeatureNorm.shape[0], :nodeFeatureNorm.shape[1]] = nodeFeatureNorm


            for j_ in range(numAtoms):
                paddedNodeFeatureNorm[j_, -1] = pre_known_charges[seq[i_]][j_]


            graphFeature[0:numAtoms, 0:20] = np.dot(AdjNorm.T, paddedNodeFeatureNorm)


            for k in range(numAtoms):
                graphFeature[k, -1] = pre_known_charges[seq[i_]][k]

            graphFeaturesOneSeq.append(graphFeature)


    if len(graphFeaturesOneSeq) < max_seq_len:
        padding_length = max_seq_len - len(graphFeaturesOneSeq)
        padding_array = np.zeros((padding_length, maxNumAtoms, 21))
        graphFeaturesOneSeq = np.concatenate((graphFeaturesOneSeq, padding_array), axis=0)
    elif len(graphFeaturesOneSeq) > max_seq_len:

        graphFeaturesOneSeq = graphFeaturesOneSeq[:max_seq_len]


    graphFeaturesOneSeq = np.asarray(graphFeaturesOneSeq, dtype=np.float32)
    return graphFeaturesOneSeq

