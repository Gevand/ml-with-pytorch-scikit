import multiprocessing as mp
import numpy as np


def square(x):
    return np.square(x)


x = np.arange(64)

print(x)

print(mp.cpu_count())
pool = mp.Pool(8)
squared = pool.map(square, [x[8 * i:8*i + 8] for i in range(8)])
print(squared)


def square_with_queue(i, x, queue):
    print(f"In process {i}")
    queue.put(np.square(x))


processes = []
queue = mp.Queue()
x = np.arange(64)
for i in range(8):
    start_index = 8 * i
    proc = mp.Process(target=square_with_queue, args=(
        i, x[start_index:start_index+8], queue))
    proc.start()
    processes.append(proc)

for proc in processes:
    proc.join()

for proc in processes:
    proc.terminate()

results = []
while not queue.empty():
    results.append(queue.get())

print(results)
