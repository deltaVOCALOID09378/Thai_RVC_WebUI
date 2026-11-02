# ==============================================================================
# File: train-index.py
# Path: tools\infer\train-index.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Studio-Grade Audio Processing & Conversion Engine
"""
Format: direct cid index mapping
"""
import os
import logging

logger = logging.getLogger(__name__)

import faiss
import numpy as np

# Save raw features if needed
inp_root = r"E:\codes\py39\dataset\mi\2-co256"
npys = []
for name in sorted(list(os.listdir(inp_root))):
    phone = np.load("%s/%s" % (inp_root, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
logger.debug(big_npy.shape)  # (6196072, 192)#fp32#4.43G
np.save("infer/big_src_feature_mi.npy", big_npy)

##################train+add
# big_npy=np.load("/bili-coeus/jupyter/jupyterhub-liujing04/vits_ch/inference_f0/big_src_feature_mi.npy")
logger.debug(big_npy.shape)
index = faiss.index_factory(256, "IVF512,Flat")  # mi
logger.info("Training...")
index_ivf = faiss.extract_index_ivf(index)  #
index_ivf.nprobe = 9
index.train(big_npy)
faiss.write_index(index, "infer/trained_IVF512_Flat_mi_baseline_src_feat.index")
logger.info("Adding...")
index.add(big_npy)
faiss.write_index(index, "infer/added_IVF512_Flat_mi_baseline_src_feat.index")
"""
Size (FP32)
big_src_feature 2.95G
    (3098036, 256)
big_emb         4.43G
    (6196072, 192)
big_emb repeat pitch

"""
