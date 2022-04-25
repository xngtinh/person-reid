

a = r"static/Market-1501-v15.09.15/pytorch\query\0002\2_1_112.jpg"

b = r"static/Market-1501-v15.09.15/pytorch/query/0002/2_1_112.jpg"

print(a)

if a == b:
    print('yes')
else:
    print('no')

# txt = "static/Market-1501-v15.09.15/pytorch/query\0002\2_1_112.jpg"
#
# x = txt.replace()
#
# print(x)
filename = "10_2_258.jpg"
id = filename.split('_')[0]
print(id)