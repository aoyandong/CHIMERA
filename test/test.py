#!/usr/bin/env python
import sys,os,csv,numpy
from subprocess import Popen
from sklearn.metrics import adjusted_rand_score as ARI

def main():
    cwd = os.getcwd()+'/'+sys.argv[0].replace('test.py','')
    sys.stdout.write("Testing... this may take a few minutes.\n")
    sys.stdout.flush()
    process = Popen("CHIMERA -i "+ cwd+"/test_data.csv -r "+ cwd+"/output.txt " +\
             "-k 2 -m 20 -N 3 -e 0.01", shell=True)
    process.communicate()

    with open(cwd+'/output.txt') as f:
        out_label = numpy.asarray(list(csv.reader(f,delimiter='\t')))
    idx = numpy.nonzero(out_label[0]=="Cluster")[0]
    out_label = out_label[1:,idx].flatten().astype(numpy.int)

    true_label = numpy.append(numpy.ones(250),numpy.ones(250)*2)
    
    measure = ARI(true_label,out_label)
    sys.stdout.write("Test Complete, output labels in test/ folder.\n")
    sys.stdout.write("Clustering test samples yields an adjusted rand index of %.3f with ground truth labels.\n" % measure)
    if measure>=0.9: sys.stdout.write("Test is successful.\n")
    
if __name__ == '__main__': main()
