from core.load_embeddings import load_all_emb_base, get_alpha_ldv_name, load_embeddings_ldv_hdf
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('embdir', type=str, help="directory where to find structure of embeddings")

args = parser.parse_args()

embdir = args.embdir

alphas, I0, Iu, Iud, Ius = load_all_emb_base(embdir)

I0_inv = np.linalg.inv(I0)
Iu_inv = np.linalg.inv(Iu)
Iud_inv = np.linalg.inv(Iud)
Ius_inv = np.linalg.inv(Ius)


# TEST ALL FINITE

thetas = ["u", "u+v"]
point_names = ["0", "u", "ud"]
I_inv = {
    "0" : I0_inv,
    "u" : Iu_inv,
    "ud" : Iud_inv,
}

for t in thetas:
    for p in point_names:
        for alpha in alphas:
            print(t, p, alpha)
            ldv_name = get_alpha_ldv_name(alpha, t+"_embeddings", p)
            ldv_path = os.path.join(embdir, ldv_name)
            ldv = load_embeddings_ldv_hdf(ldv_path)
            if not np.isfinite(ldv).all():
                print("LDV {:}-emb in point {:} found problem in alpha {:}".format(t, p, alpha))
            plog = np.matmul(I_inv[p], np.transpose(ldv)).transpose()
            if not np.isfinite(plog).all():
                print("PLOG {:}-emb in point {:} found problem in alpha {:}".format(t, p, alpha))

thetas = ["u+v"]
point_names = ["us"]
I_inv = {
    "us" : Iud_inv,
}

for t in thetas:
    for p in point_names:
        for alpha in alphas:
            print(t, p, alpha)
            ldv_name = get_alpha_ldv_name(alpha, t+"_embeddings", p)
            ldv_path = os.path.join(embdir, ldv_name)
            ldv = load_embeddings_ldv_hdf(ldv_path)
            if not np.isfinite(ldv).all():
                print("LDV {:}-emb in point {:} found problem in alpha {:}".format(t, p, alpha))
            plog = np.matmul(I_inv[p], np.transpose(ldv)).transpose()
            if not np.isfinite(plog).all():
                print("PLOG {:}-emb in point {:} found problem in alpha {:}".format(t, p, alpha))
