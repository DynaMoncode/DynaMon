import sys

sys.path.append("/root/Archived-DHT/DHT_revise")

from ConsisSketch_withmax import PrioritySketch
from utils import generate_zipf_distribution

sketch = PrioritySketch(10240, 100, 100)

print(sketch.highSketch)
print(sketch.lowSketch)

N = 1000

work = generate_zipf_distribution(1.5, N, 100000)
for item in range(N):
    for _ in range(work[item]):
        sketch.insert(item, bool(item % 2 == 0), 1, 100)

oddCnt = 0
oddAAE = 0
oddARE = 0

evenCnt = 0
evenAAE = 0
evenARE = 0

for item in range(N):
    trueValue = work[item]
    if item % 2 == 0:
        res, _, _ = sketch.query(item, True)
        oddCnt += 1
        oddAAE += abs(res - trueValue)
        oddARE += abs(res - trueValue) / trueValue
    else:
        res, _, _ = sketch.query(item, False)
        evenCnt += 1
        evenAAE += abs(res - trueValue)
        evenARE += abs(res - trueValue) / trueValue

print("Expected result should show that the odd has smaller error than the even")
print(oddAAE / oddCnt)
print(oddARE / oddCnt)
print(evenAAE / evenCnt)
print(evenARE / evenCnt)