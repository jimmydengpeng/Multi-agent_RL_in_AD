import ray, time
ray.init()

@ray.remote
def f(x):
    return x

def g(x):
    return x

all_f = []
all_g = []

st = time.time()
for i in range(10000):
    futures = f.remote(i) 
    all_f.append(ray.get(futures))
end = time.time()

dt = end - st
print(f"remote: {dt}")

st = time.time()
for i in range(10000):
    all_g.append(g(i))
end = time.time()

dt = end - st
print(f"local: {dt}")
