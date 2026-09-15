import os
import ctypes
import glob
import time

site_packages = r'C:\Users\Admin\Downloads\EBT_Project_2026\venv\Lib\site-packages'
files = glob.glob(site_packages + '/**/*.pyd', recursive=True) + glob.glob(site_packages + '/**/*.dll', recursive=True)

print(f'Found {len(files)} dll/pyd files')

max_retries = 15
for attempt in range(max_retries):
    failed = 0
    for f in files:
        try:
            ctypes.CDLL(f)
        except OSError as e:
            if 'Application Control policy' in str(e) or 'blocked' in str(e) or '126' in str(e) or '193' in str(e):
                # some errors might just be missing dlls, but we only care about if they are decreasing
                failed += 1
        except Exception:
            pass
    print(f'Attempt {attempt + 1}: {failed} files failed to load')
    if failed == 0:
        print('All clear!')
        break
    time.sleep(2)
