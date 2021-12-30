import sys
import traceback
import csv


class LorenzAttractorRungeKutta:
    h = 0.01
    step = 4000
    x_0, y_0, z_0 = -8.0, 8.0, 27.0

    def __init__(self):
        self.res = [[], [], []]

    def exec(self):
        try:
            outfile = open("RK_lorenz_3D.csv", "w", newline="")
            writer = csv.writer(outfile)
            writer.writerow(["step", "x", "y", "z"])
            xyz = [self.x_0, self.y_0, self.z_0]

            for j in range(self.step):
                writer.writerow([j, xyz[0], xyz[1], xyz[2]])

                k_0 = self.__lorenz(xyz)

                k_1 = self.__lorenz([
                    x + k * self.h / 2 for x, k in zip(xyz, k_0)
                ])

                k_2 = self.__lorenz([
                    x + k * self.h / 2 for x, k in zip(xyz, k_1)
                ])

                k_3 = self.__lorenz([
                    x + k * self.h for x, k in zip(xyz, k_2)
                ])

                for i in range(3):
                    xyz[i] += (k_0[i] + 2 * k_1[i] + 2 * k_2[i] + k_3[i]) * self.h / 6.0
                    self.res[i].append(xyz[i])
            outfile.close()

        except Exception as e:
            raise

    def __lorenz(self, xyz, p=10, r=28, b=8/3.0):

        try:
            return [
                -p * xyz[0] + p * xyz[1],
                -xyz[0] * xyz[2] + r * xyz[0] - xyz[1],
                xyz[0] * xyz[1] - b * xyz[2]
            ]
        except Exception as e:
            raise


if __name__ == '__main__':

    try:
        obj = LorenzAttractorRungeKutta()
        obj.exec()

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
