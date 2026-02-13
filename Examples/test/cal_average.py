import re
import numpy as np

# اینجا متن‌ها رو کپی کن
log_text = """
Average NewDetectCommonRegions: 10.4304475
Average CorrectLoop: 8.752875
Average LoopClosing: 17.958205
Average NewDetectCommonRegions: 10.960443333333332
Average CorrectLoop: 16.62796666666667
Average LoopClosing: 26.717399999999998
Average NewDetectCommonRegions: 10.425664285714287
Average CorrectLoop: 7.529657142857142
Average LoopClosing: 16.63857857142857
Average NewDetectCommonRegions: 10.9617
Average CorrectLoop: 14.891649999999998
Average LoopClosing: 24.43415
Average NewDetectCommonRegions: 10.55716
Average CorrectLoop: 4.495263636363637
Average LoopClosing: 13.154767272727272
"""

# پیدا کردن همه اعداد با regex
ndcr_values = [float(x) for x in re.findall(r'NewDetectCommonRegions:\s*([\d.]+)', log_text)]
cl_values = [float(x) for x in re.findall(r'CorrectLoop:\s*([\d.]+)', log_text)]
lc_values = [float(x) for x in re.findall(r'LoopClosing:\s*([\d.]+)', log_text)]

# محاسبه میانگین
print("Average NewDetectCommonRegions:", np.mean(ndcr_values))
print("Average CorrectLoop:", np.mean(cl_values))
print("Average LoopClosing:", np.mean(lc_values))



# Average NewDetectCommonRegions: 10.4304475
# Average CorrectLoop: 8.752875
# Average LoopClosing: 17.958205
# Average NewDetectCommonRegions: 10.960443333333332
# Average CorrectLoop: 16.62796666666667
# Average LoopClosing: 26.717399999999998
# Average NewDetectCommonRegions: 10.425664285714287
# Average CorrectLoop: 7.529657142857142
# Average LoopClosing: 16.63857857142857
# Average NewDetectCommonRegions: 10.9617
# Average CorrectLoop: 14.891649999999998
# Average LoopClosing: 24.43415
# Average NewDetectCommonRegions: 10.55716
# Average CorrectLoop: 4.495263636363637
# Average LoopClosing: 13.154767272727272