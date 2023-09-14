import numpy as np

test_points = np.array(
    [[3.506347656, 1.505371094, 0, 0.01171875],
     [-2.993652344, 1.505371094, 0, -6.48828125],
     [-2.993652344, 1.505371094, 1, -5.259794474],
     [-2.993652344, 1.505371094, 2, -137.7707057],
     [-2.993652344, 1.505371094, 3, 0.4421354949],
     [9.993652344, 1.494628906, 0, 6.48828125],
     [0.2563476562, -1.244628906, 0, -5.98828125],
     [0.2563476562, -1.244628906, 1, -1.292753458],
     [0.2563476562, -1.244628906, 2, -2.394333479],
     [0.2563476562, -1.244628906, 3, 0.4902477799],
     [0.2563476562, 4.255371094, 0, -0.48828125],
     [0.2563476562, 4.255371094, 1, -17.85183549],
     [0.2563476562, 4.255371094, 2, -28.88886473],
     [0.2563476562, 4.255371094, 3, 3.990795622],
     [0.2563476562, -3.994628906, 0, -8.73828125],
     [0.2563476562, -3.994628906, 1, -15.70071244],
     [0.2563476562, -3.994628906, 2, -25.44706785],
     [0.2563476562, -3.994628906, 3, 0.2545840319],
     [-1.368652344, -2.619628906, 0, -8.98828125],
     [-1.368652344, -2.619628906, 1, -8.23110795],
     [-1.368652344, -2.619628906, 2, -23.79879004],
     [-1.368652344, -2.619628906, 3, 0.283941977],
     [-2.993652344, -2.619628906, 0, -10.61328125],
     [-2.993652344, -2.619628906, 1, -9.85610795],
     [-2.993652344, -2.619628906, 2, -145.1248073],
     [-2.993652344, -2.619628906, 3, 0.2274168903],
     [0.2436523438, 1.462402344, 0, -3.293945312],
     [0.2436523438, 1.462402344, 1, -1.894968271],
     [0.2436523438, 1.462402344, 2, -3.349469093],
     [0.2436523438, 1.462402344, 3, 1.06688394],
     [-2.181152344, -3.307128906, 0, -10.48828125],
     [-2.181152344, -3.307128906, 1, -13.11825395],
     [-2.181152344, -3.307128906, 2, -69.38271197],
     [-2.181152344, -3.307128906, 3, 0.2256709677],
     [1.957519531, -2.727050781, 0, -5.76953125],
     [1.957519531, -2.727050781, 1, -5.479286432],
     [1.957519531, -2.727050781, 2, 25.60603674],
     [1.779785156, -3.983886719, 0, -7.204101562],
     [1.779785156, -3.983886719, 1, -14.09156823],
     [1.779785156, -3.983886719, 2, 2.794385148],
     [1.894042969, -1.330566406, 0, -4.436523438],
     [1.894042969, -1.330566406, 1, 0.1236360073],
     [1.005371094, -3.274902344, 0, -7.26953125],
     [1.005371094, -3.274902344, 1, -9.719614267],
     [1.005371094, -3.274902344, 2, -12.07897667],
     [1.005371094, -3.274902344, 3, 0.318029036],
     [1.119628906, -1.975097656, 0, -5.85546875],
     [1.119628906, -1.975097656, 1, -2.781381845],
     [1.119628906, -1.975097656, 2, 0.7760426105],
     [-2.181152344, -3.994628906, 0, -11.17578125],
     [-2.181152344, -3.994628906, 1, -18.13821244],
     [-2.181152344, -3.994628906, 2, -77.41464556],
     [-2.181152344, -3.994628906, 3, 0.2007715046],
     [0.2309570312, -2.630371094, 0, -7.399414062],
     [0.2309570312, -2.630371094, 1, -6.68789506],
     [0.2309570312, -2.630371094, 2, -11.00856578],
     [0.2309570312, -2.630371094, 3, 0.3445904384],
     [-1.635253906, 0.00146484375, 0, -6.633789062],
     [-1.635253906, 0.00146484375, 1, -1.635256052],
     [-1.635253906, 0.00146484375, 2, -21.86380062],
     [-1.635253906, 0.00146484375, 3, 0.4631976208],
     [1.817871094, -3.231933594, 0, -6.4140625],
     [1.817871094, -3.231933594, 1, -8.627523661],
     [1.817871094, -3.231933594, 2, 13.32455524],
     [1.779785156, -1.889160156, 0, -5.109375],
     [1.779785156, -1.889160156, 1, -1.78914094],
     [1.779785156, -1.889160156, 2, 22.47826881],
     [2.097167969, -0.1596679688, 0, -3.0625],
     [2.097167969, -0.1596679688, 1, 2.071674109],
     [1.817871094, -0.9653320312, 0, -4.147460938],
     [1.817871094, -0.9653320312, 1, 0.8860051632],
     [2.224121094, 1.440917969, 0, -1.334960938],
     [2.224121094, 1.440917969, 1, 0.1478765011],
     [0.8911132812, -0.4711914062, 0, -4.580078125],
     [0.8911132812, -0.4711914062, 1, 0.6690919399],
     [4.369628906, 5.265136719, 0, 4.634765625],
     [6.743652344, -1.255371094, 0, 0.48828125],
     [6.756347656, -3.994628906, 0, -2.23828125],
     [6.756347656, -3.994628906, 1, -9.200712442],
     [6.756347656, -3.994628906, 2, 1516.545386],
     [2.846191406, -1.964355469, 0, -4.118164062],
     [2.846191406, -1.964355469, 1, -1.012501001],
     [2.846191406, -1.964355469, 2, 109.1083074],
     [-2.688964844, 0.2700195312, 0, -7.418945312],
     [-2.688964844, 0.2700195312, 1, -2.761875391],
     [-2.688964844, 0.2700195312, 2, -97.32988769],
     [-2.688964844, 0.2700195312, 3, 0.3883026333]
     ], dtype=np.double)
