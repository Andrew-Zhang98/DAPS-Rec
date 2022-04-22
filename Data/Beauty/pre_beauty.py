
# preprocess Steam
# add rate and timestamp
# from xxlimited import new


f = open("/home/ruoyan/code/SASRec/data/Beauty.txt","r") 
new_lines = []
lines = f.readlines() #读取全部内容 ，并以列表方式返回
num = 1
for line in lines:
    line = line.strip('\n').split(' ')
    line.append('10')
    line.append(str(num))
    new_lines.append(line)
    num += 1

fileObject = open('Beauty.txt', 'w')
for line in new_lines:
    # print(line)
    # import pdb; pdb.set_trace()
    # print(" ".join(line))
    fileObject.write(" ".join(line))
    # print(line)
    # break
    # for char in line:
    #     fileObject.write(str(char))
    #     fileObject.write(' ')
    fileObject.write('\n')
fileObject.close()

        