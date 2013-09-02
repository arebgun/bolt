#!/usr/bin/env python

import sys
sys.path.insert(1,"..")

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from automain import automain
import numpy as np
import shelve


def running_avg(arr,window):
		return  [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]

def view_graphs(save_file, window_size):

	f = shelve.open(save_file)

	if 'landmark_answers' in f and len(f['landmark_answers']) > 0:
		correct_answers, got_correct = zip(*f['landmark_answers'])
		
		avg_correct = running_avg(np.array(got_correct),window_size)
		x = range(len(avg_correct))
		plt.figure()
		plt.plot(x, avg_correct, linewidth=3, label='All')
		plt.ylim([0,1])
		leg = plt.legend()
		leg.draggable(state=True)
		plt.show()
		
		possible_answers = set(correct_answers)
		for possible_answer in possible_answers:
			masked = np.ma.masked_array(
									got_correct, 
									np.array([answer!=possible_answer 
											  for answer in correct_answers],
											 dtype=float))
			avg_masked = running_avg(masked, window_size)
			plt.figure()
			plt.plot(x, avg_masked, linewidth=3, label=possible_answer)
			plt.ylim([0,1])
			leg = plt.legend()
			leg.draggable(state=True)
			plt.show()

	if 'relation_answers' in f and len(f['relation_answers']) > 0:
		data, got_correct = zip(*f['relation_answers'])
		relations,landmarks,landmark_guesses = zip(*data)
		
		avg_correct = running_avg(np.array(got_correct),window_size)
		x = range(len(avg_correct))
		plt.figure()
		plt.plot(x, avg_correct, linewidth=3, label='All')
		plt.ylim([0,1])
		leg = plt.legend()
		leg.draggable(state=True)
		plt.show()
		
		possible_relations = set(relations)
		for possible_relation in possible_relations:
			masked = np.ma.masked_array(
									got_correct, 
									np.array([relation!=possible_relation
											  for relation in relations],
											 dtype=float))
			avg_masked = running_avg(masked, window_size)
			plt.figure()
			plt.plot(x, avg_masked, linewidth=3, label=possible_relation)
			plt.ylim([0,1])
			leg = plt.legend()
			leg.draggable(state=True)
			plt.show()

	f.close()


@automain
def main():
	parser = ArgumentParser()
	parser.add_argument('graphs_file', type=str)
	parser.add_argument('-w','--window-size', type=int, default=100)
	args = parser.parse_args()
	
	view_graphs(args.graphs_file, args.window_size)
