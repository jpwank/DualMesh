import dualmesh as dm
import matplotlib.pyplot as plt
import meshio
import meshzoo
import numpy as np

# create the mesh

x, y = 1, 1
n = 0
disc_x, disc_y = int(x / y) * 10 * 2 ** n + 1, int(x) * 10 * 2 ** n + 1
X, Y = np.linspace(0.0, x, disc_x), np.linspace(0.0, y, disc_y)
variant = "up"
points, cells = meshzoo.rectangle_tri(X, Y, variant=variant)

mesh = meshio.Mesh(points, {"triangle": cells})

mesh.write("ex_rect_plate" + ".msh")
msh = meshio.read("ex_rect_plate.msh")

dual_msh = dm.get_dual(msh, order=True)

fig, ax = plt.subplots()
ax.triplot(msh.points[:, 0], msh.points[:, 1], msh.cells[0].data)

for cell in dual_msh.cells[0].data:
    # We get the coordinates of the current polygon.
    # We repeat one of the points, so that when plotted, we get a complete polygon
    cell_points = dual_msh.points[cell + [cell[0]]]
    # print(cell_points)

    # We plot the points of the current polygon
    ax.plot(cell_points[:, 0], cell_points[:, 1], "-", color="black")


# We add a legend to the figure
fig.legend([ax.lines[0], ax.lines[-1]], ["Mesh", "Dual mesh"])
# We save the resulting figure
# fig.savefig("ex_rect_plate.png")
# plt.show()

print(np.array(dual_msh.cells[0].data))

## Info
# Elemente in Dualmesh sind nach Knoten in Ausgangsmesh geordnet
# Elmente 0 in Dualmesh gehört zu Knoten 0 in Ausgangsmesh

# Sortiert nach kleinster x- und dann kleinster y-Koordinate
