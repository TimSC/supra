import pickle

if __name__=="__main__":
	p = pickle.load(open("prof.dat"))

	pl = [(x[2], x) for x in p.func_stats]
	pl.sort()
	
	for x in pl:
		print x[1]

