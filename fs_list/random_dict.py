import glob

vlists = glob.glob('voc*shot.txt')

for seed in range(1,6):
    for v in vlists:
        d=open(v).readlines()
        with open(v.split('.txt')[0]+'_seed'+str(seed)+'.txt','a') as f:
            res = []
            for line in d:
                res.append(line.replace('voclist','voclist'+str(seed)))
            for i in res:
                f.write(i)
