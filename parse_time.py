import sys
from collections import defaultdict
filename = sys.argv[1]

def str2second(time):
    time = time.split(":")
    if len(time) == 3:
        return int(time[0])*3600 + int(time[1])*60 + int(time[2])
    elif len(time) == 2:
        return int(time[0])*60 + int(time[1])

results = {}

with open(filename) as f:
    cur_error_rate = 0
    for line in f:
        if "error" in line:
            cur_error_rate = float(line.split(" ")[-1])
            if cur_error_rate in results:
                print(results)
                results = {}
            results[cur_error_rate] = []
        if "]Timeout" in line or "100%" in line:
            start_id = line.find("[")
            end_id = line.find(",")
            time = line[start_id+1:end_id]
            time_1, time_2 = time.split("<")
            if time_2 == "?":
                time_2 = "4:0:0"
            # convert time to seconds
            time_1 = str2second(time_1)
            time_2 = str2second(time_2)
            results[cur_error_rate].append(time_2 + time_1)

print(results)