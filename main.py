import taichi as ti

ti.init(arch=ti.cuda)

res = (512, 512)
# pixes = ti.Vector.field(3, ti.f32, shape=res)

grid = (512, 512)
Vx = ti.field(ti.f32, shape=grid)
Vy = ti.field(ti.f32, shape=grid)
Vx_buffer = ti.field(ti.f32, shape=grid)
Vy_buffer = ti.field(ti.f32, shape=grid)
force = ti.Vector.field(2, ti.f32, shape=grid)
pressure = ti.field(ti.f32, shape=grid)

# tracer
n_tracer = 200000
tracer = ti.Vector.field(2, ti.f32, shape=n_tracer)

last_curser = ti.Vector.field(2, ti.f32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())

@ti.kernel
def init():
    for i in range(n_tracer):
        tracer[i] = [ti.random(dtype=ti.f32) * 512, ti.random(dtype=ti.f32) * 512]

    clear_force()

@ti.func
def clear_force():
    for i, j in ti.ndrange(grid[0], grid[1]):
        force[i, j] = ti.Vector([0, 0])

@ti.kernel
def curse_force():
    dir = curser[None] - last_curser[None]

    for i, j in ti.ndrange(grid[0], grid[1]):
        if i < grid[0] / 2:
            force[i, j] += dir

@ti.func
def add_force(df, i: ti.i32, j: ti.i32):
    force[i, j] += df

@ti.func
def apply_force(t: ti.f32, i: ti.i32, j: ti.i32):
    Vx[i, j] += force[i, j][0] * t
    Vy[i, j] += force[i, j][1] * t

@ti.func
def advect(t: ti.f32, i: ti.i32, j: ti.i32):
    pass

@ti.func
def diffuse(t: ti.f32):
    pass

@ti.func
def project(t: ti.f32):
    pass

@ti.func
def updateTracers(t: ti.f32):
    for i in range(n_tracer):
        x = t * sample_bilinear(Vx, tracer[i])
        y = t * sample_bilinear(Vy, tracer[i])
        tracer[i] = [tracer[i][0] + x, tracer[i][1] + y]
        tracer[i] = retile_point(tracer[i])

@ti.kernel
def update(t: ti.f32):
    for i, j in ti.ndrange(grid[0], grid[1]):
        apply_force(t, i, j)
    for i, j in ti.ndrange(grid[0], grid[1]):
        advect(t, i, j)
    diffuse(t)
    project(t)

    #clear_force()

    updateTracers(t)


@ti.func
def semi_lagrangian(x, new_x, dt):


@ti.func
def pos2Index(p):
    return [ti.cast(p[0], ti.i32), ti.cast(p[1], ti.i32)]

@ti.func
def retile_point(p):
    if p[0] > 512:
        p[0] -= 512
    if p[1] > 512:
        p[1] -= 512
    if p[0] < 0:
        p[0] += 512
    if p[1] < 0:
        p[1] += 512
    return p

@ti.func
def sample_bilinear(x, p):
    mid_point = ti.cast(p + ti.Vector([0.5, 0.5]), ti.i32)
    g = p - ti.Vector([0.5, 0.5]) - ti.floor(p - ti.Vector([0.5, 0.5]))
    small_point = retile_point(mid_point - ti.Vector([0.5, 0.5]))
    I = pos2Index(small_point)
    f = 1 - g

    return x[I] * (g[0] * g[1]) + x[retile_point(I + ti.Vector([1, 0]))] * (
            f[0] * g[1]) + x[retile_point(I + ti.Vector([0, 1]))] * (
            g[0] * f[1]) + x[retile_point(I + ti.Vector([1, 1]))] * (f[0] * f[1])

if __name__ == "__main__":
    init()
    gui = ti.GUI("Euler fluid", res, background_color=0x0)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()

        if gui.is_pressed(ti.GUI.LMB):
            last_curser[None] = curser[None]
            curser[None] = gui.get_cursor_pos()
            if picking[None] == 0:
                picking[None] = 1
                last_curser[None] = curser[None]
            curse_force()
        else:
            picking[None] = 0

        for i in range(5):
            update(0.016)
        gui.circles(tracer.to_numpy() / 512, radius=1, color=0x00FF00)
        gui.show()
