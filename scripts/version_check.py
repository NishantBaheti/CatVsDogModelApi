
import re 

def get_version(in_str):
    regex_str = '\\v?(\d+)\.(\d+)\.(\d+)'
    match = re.search(regex_str, in_str)
    if match:
        return match.group()
    else:
        return '0.0.0'                                                                        

l = [
    'nishant',
    'baheti',
    'hello',
    'model-v2.0.1',
    'model-1.5.0',
    'model-0.0.1',
    'nishnat-1.0.0'
]

print(sorted(l,key=get_version))
