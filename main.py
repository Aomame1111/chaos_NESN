from NESN import NESN, RLS

n_u = 3
n_x = 300
n_y = 3

for i in range(3):
    lorenz = NESN(n_u, n_x, n_y, 'dataset/RK_lorenz_3D.csv', 0.7)
    lorenz.train_one_step(RLS(n_x, n_y, 0.9995, 0.001), i)
    lorenz.test('step')
    lorenz.train_free_run(RLS(n_x, n_y, 0.9995,  0.001), i)
    lorenz.test('free')
    lorenz.graph(i)
