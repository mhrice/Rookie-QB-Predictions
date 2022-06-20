from fastai.tabular.all import *
dls = TabularDataLoaders.from_csv("combined.csv", path=".", 
                                   cont_names=["Cmp", "Att", "Yds", "TD", "Int", "Y/G", "Sk"],
                                   y_names=["Cmp2", "Att2", "Yds2", "TD2", "Int2", "Y/G2", "Sk2"],
                                   procs=[FillMissing, Normalize]
                                   )

dls.show_batch()