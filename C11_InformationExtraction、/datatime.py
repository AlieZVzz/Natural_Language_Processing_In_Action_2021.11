# 提取GPS信息
import re

lat = r'([-]?[0-9]?[0-9][0-9][.][0-9]{2,10})'
lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2.10})'
sep = r'[./ ]{1,3}'
re_gps = re.compile(lat + sep + lon)

# 美国日期表达式
us = r'((([01]?\d)[-/]([0123]?\d))([-/]([0123]\d)\d\d)?)'
eu = r'((([0123]?\d[-/]([01]?\d))([-/]([0123]\d)?\d\d)?)'
mdy = re.findall(us, 'Santa came 12/25/2017. An elf appeared 12/12')
print(mdy)
