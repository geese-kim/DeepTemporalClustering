import matplotlib.pyplot as plt
import csv


epoch=range(30)
c_val_acc=[]
b_val_acc=[]

dataset_list=['cairo', 'kyoto11', 'kyoto7', 'kyoto8', 'milan']
target=dataset_list[4]
# use clustering
with open('biLSTM-{}-20201117-062534.csv'.format(target), 'r') as csvfile:
    reader=csv.reader(csvfile)
    idx=0
    for row in reader:
        if idx==0:
            idx+=1
            continue
        c_val_acc.append(float(row[3]))
        idx+=1

# baseline
with open('biLSTM-{}-20201116-123034.csv'.format(target), 'r') as csvfile:
    reader=csv.reader(csvfile)
    idx=0
    for row in reader:
        if idx==0:
            idx+=1
            continue
        b_val_acc.append(float(row[3]))
        idx+=1

plt.title(target)
plt.plot(epoch, c_val_acc, label='clustering')
plt.plot(epoch, b_val_acc, label='baseline (fixed)')
plt.yticks([float(i/10) for i in range(4,11)])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val_accuracy')
plt.savefig('./fig/{}_val_acc.png'.format(target))