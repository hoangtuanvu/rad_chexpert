import numpy as np
from torchtext import vocab

embedder = vocab.GloVe(name='6B')
vecs = []

__labels = ['Pleural Effusion', #0
            'abnormal heart size', #1
            'Pneumothorax', #2
            'Pulmonary lung lesions', #3
            'Air space opacities', #4
            'Fracture', #5
            # 'Atelectasis', #6
            # 'Edema', #7
            # 'enlarged heart mediastinum', #8
            # 'Consolidation'
            ] #10

for c in __labels:

    vec = embedder.get_vecs_by_tokens(c.replace("_", " ").split(" "), lower_case_backup=True)
    print(c, ": ", [v.abs().sum() for v in vec])
    vecs.append(vec.mean(dim=0).numpy())

vecs = np.stack(vecs, axis=0)
print(vecs.shape)

np.save('diseases_embeddings', vecs)

# loads = np.load('diseases_embeddings.npy')
# print(loads.shape)